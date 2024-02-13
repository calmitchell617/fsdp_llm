import inspect
import math
import os
import pickle
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import destroy_process_group, init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from models.nanogpt.model import CausalSelfAttention, MLP, GPT, GPTConfig

import config.nanogpt_config as fsdp_config

import time

# io
out_dir = "out"
eval_interval = 2000
log_interval = 2
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume' or 'gpt2*'

# wandb
wandb_log = False
wandb_project = "owt"
wandb_run_name = "gpt2"

# data
data_dir = "data"
dataset = "openwebtext"
gradient_accumulation_steps = 1
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
use_flash22_bf16 = False
use_flash22_fp16 = False
use_flash_pytorch_sdpa = True

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# DDP settings
backend = "nccl"
# system
device = ("cuda")

compile = True
pt2_compile = True
iters_to_run = 50000
num_epochs = 2

batch_size = 8
vocab_size = 50304  # use 65 for shakespeare, GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency

# FSDP specific
use_mixed_precision: bool = True
wrapping_policy = ModuleWrapPolicy({CausalSelfAttention, MLP})
model_sharding_strategy = ShardingStrategy.FULL_SHARD

# optimizer overlap
use_optimizer_overlap: bool = True

# stats - dynamic, not set by user
current_model_params: int = 0

# Init TP
_multi_gpu = int(os.environ.get("RANK", -1)) != -1  # verify distributed run
assert _multi_gpu, "this config assumes distributed setup - multi-gpu not ready here."


init_process_group(backend=backend)
_rank = int(os.environ["RANK"])
_local_rank = int(os.environ["LOCAL_RANK"])

world_size = int(os.environ["WORLD_SIZE"])
device = f"cuda:{_local_rank}"
torch.cuda.set_device(device)
master_process = _rank == 0
seed_offset = _rank

# wrapper to avoid cluttering with if rank==0...
def rank_print(x):
    if _rank == 0:
        print(x)


if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = "cuda"

# poor man's data loader
data_dir = os.path.join(data_dir, dataset)
rank_print(f"{data_dir=}")
train_data = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

batch_size = batch_size

def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y

def get_vocab_size():
    meta_path = os.path.join(data_dir, "meta.pkl")
    meta_vocab_size = None
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        meta_vocab_size = meta["vocab_size"]
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

    assert (
        meta_vocab_size is not None
    ), "Failed to determine vocab size"
    return meta_vocab_size


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

rank_print("Initializing a new model from scratch")

rank_print("Before Model Initialization")

if vocab_size is None:
    vocab_size = get_vocab_size()

n_layer: int = 32
n_head: int = 16
n_embd: int = 1024

model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    vocab_size=vocab_size,
    dropout=dropout,
    use_flash22_fp16=use_flash22_fp16,
    use_flash22_bf16=use_flash22_bf16,
)

gpt_conf = GPTConfig(**model_args)
model = GPT(gpt_conf, rank=_rank)
current_model_params = model.get_num_params()

rank_print("Model Initialization done")

# we need this or else calcing mfu in fsdp = sharded model size...
fsdp_pg = None

from config.mixed_precision import get_mixed_precision_policy

mixed_precision_policy = None
if use_mixed_precision:
    mixed_precision_policy = get_mixed_precision_policy()

model = FSDP(
    model,
    sharding_strategy=model_sharding_strategy,
    auto_wrap_policy=wrapping_policy,
    mixed_precision=mixed_precision_policy,
    device_id=device,
    process_group=fsdp_pg,
    use_orig_params=True,
)


# ---- debug print gpu's in use by FSDP
shard_g_size = model.process_group.size()
shard_rank = model.process_group.rank()
replicate_g = None  # model._inter_node_state.process_group

dist.barrier()
print(f"{shard_g_size=}, {shard_rank=}\n")
dist.barrier()

# optimizer
# new PyTorch nightly has a new 'fused' option for AdamW that is much faster

use_fused = (device_type == "cuda") and (
    "fused" in inspect.signature(torch.optim.AdamW).parameters
)

rank_print(f"Optimizer = using fused AdamW: {use_fused}")
extra_args = dict(fused=True) if use_fused else dict()


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, **extra_args)
if compile:
    rank_print("compiling the model with PT2 compiler... (takes a ~minute)")
    start = time.perf_counter()
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0
    stop = time.perf_counter()
    compile_time = round(stop - start, 5)
    rank_print(f"compilation complete.  Time on compile: {compile_time}")

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            print("after get batch ", k)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# training loop
X, Y = get_batch("train")  # fetch the very first batch

local_iter_num = 0  # number of iterations in the lifetime of this process
eval_interval = 1
warmup = 5
iter_time_accumulator = 0.0
iter_count = 0

while local_iter_num < iters_to_run:
    t0 = time.perf_counter()
    logits, loss = model(X, Y)
    X, Y = get_batch("train")
    loss.backward()
    optimizer.step()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)
    torch.distributed.barrier()

    # timing and logging
    t1 = time.perf_counter()
    dt = t1 - t0

    if iter_num >= warmup:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        # if local_iter_num >= 3:  # let the training loop settle a bit

        rank_print(
            f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms%"
        )
        iter_time_accumulator += dt
        iter_count += 1
    iter_num += 1
    local_iter_num += 1
    rank_print(f"iter {iter_num} completed...")

    # termination conditions
    if iter_num > max_iters:
        break

dist.barrier()
rank_print(
    f"\nTraining completed."
)
destroy_process_group()

from dataclasses import dataclass

from .base_config import base_config, fsdp_checkpointing_base

# wrap model into FSDP container
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.fsdp import (
    ShardingStrategy,
)

def build_model(cfg, tp_mesh=None, rank=None):
    """load model config and return built model (from scratch)"""
    if model_name == "10.5M":
        # baby GPT model :)
        n_layer = 6
        n_head = 6
        n_embd = 384

    elif model_name == "124M":
        # block_size: int = 1024
        # vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
        n_layer: int = 24
        n_head: int = 24
        n_embd: int = 768

    elif model_name == "201M":
        n_layer: int = 16
        n_head: int = 16
        n_embd: int = 1024

    elif model_name == "1B":
        n_layer: int = 32
        n_head: int = 16
        n_embd: int = 1024

    elif model_name == "1.5B":
        n_layer: int = 46
        n_head: int = 20
        n_embd: int = 1600

    elif model_name == "13B":
        n_layer: int = 44
        n_head: int = 40
        n_embd: int = 5120

    elif model_name == "20B":
        n_layer: int = 60
        n_head: int = 48
        n_embd: int = 6144

    else:
        assert False, f"model {model_name} not supported yet."

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=cfg.block_size,
        bias=cfg.use_bias,
        vocab_size=cfg.vocab_size,
        dropout=cfg.dropout,
        use_flash22_fp16=cfg.use_flash22_fp16,  # pass Triton option in
        use_flash22_bf16=cfg.use_flash22_bf16,
    )

    assert not (
        cfg.use_flash22_fp16 and cfg.use_flash22_bf16
    ), f"both fp16 and bf16 set to True...please use only one at a time."

    gpt_conf = GPTConfig(**model_args)
    model = GPT(tp_mesh, gpt_conf, rank=rank)
    cfg.current_model_params = model.get_num_params()
    return model, gpt_conf





def apply_checkpointing_policy(model):
    return fsdp_checkpointing_base(model, (MLP, CausalSelfAttention))

"""Model / tokenizer loading for T4 (fp16, SDPA) and architecture introspection helpers."""

from __future__ import annotations

import glob
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

log = logging.getLogger("spark")

DTYPES = {"float16": torch.float16, "fp16": torch.float16, "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
          "float32": torch.float32, "fp32": torch.float32}


def load_tokenizer(model_id: str, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = padding_side
    return tok


def load_model(model_id: str, device: str | int | None = "cuda:0", dtype: str = "float16",
               attn: str = "sdpa", quant: str | None = None, device_map: Any = None):
    """Load Spark (trust_remote_code). quant='nf4' -> bitsandbytes 4-bit (QLoRA)."""
    from transformers import AutoModelForCausalLM

    torch_dtype = DTYPES[dtype]
    kwargs: dict[str, Any] = dict(trust_remote_code=True, torch_dtype=torch_dtype, low_cpu_mem_usage=True)
    if quant == "nf4":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16 if torch_dtype != torch.float32 else torch.float32)
    if device_map is not None:
        kwargs["device_map"] = device_map
    elif device is not None and str(device) != "cpu":
        kwargs["device_map"] = {"": device}
    last_err = None
    for impl in [attn, "sdpa", "eager"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation=impl, **kwargs)
            log.info("loaded %s (dtype=%s, attn=%s, quant=%s)", model_id, dtype, impl, quant or "none")
            break
        except (ValueError, ImportError, TypeError) as e:  # attn impl unsupported by remote code
            last_err = e
            log.warning("attn_implementation=%s failed: %s", impl, str(e).splitlines()[0][:200])
    else:
        raise RuntimeError(f"could not load {model_id}") from last_err
    if device is not None and str(device) == "cpu" and device_map is None:
        model.to("cpu")
    model.config.use_cache = True
    return model


# ----------------------------------------------------------------------------- introspection

def text_config(model_or_config):
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "text_config", None) or cfg


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    for path in ("model.layers", "model.model.layers", "base_model.model.model.layers", "transformer.h",
                 "model.decoder.layers", "gpt_neox.layers"):
        cur: Any = model
        ok = True
        for part in path.split("."):
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if ok and isinstance(cur, nn.ModuleList):
            return cur
    lists = [m for m in model.modules() if isinstance(m, nn.ModuleList)]
    if not lists:
        raise RuntimeError("no decoder layer list found")
    return max(lists, key=len)


def hidden_size(model) -> int:
    cfg = text_config(model)
    return int(getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model"))


def kv_bytes_per_token(model, dtype_bytes: int = 2) -> int:
    """Upper bound on KV-cache bytes per token (treats sliding layers as full)."""
    cfg = text_config(model)
    n_layers = int(getattr(cfg, "num_hidden_layers", len(decoder_layers(model))))
    n_heads = int(getattr(cfg, "num_attention_heads", 16))
    n_kv = int(getattr(cfg, "num_key_value_heads", None) or n_heads)
    head_dim = int(getattr(cfg, "head_dim", None) or hidden_size(model) // n_heads)
    return 2 * n_layers * n_kv * head_dim * dtype_bytes


def get_decoder_and_head(model):
    """(decoder module returning last_hidden_state, lm_head weight, softcap or None) through PEFT wrappers."""
    base = model
    for _ in range(4):  # unwrap PeftModel / LoraModel / DDP
        if hasattr(base, "get_decoder") and hasattr(base, "get_output_embeddings"):
            break
        base = getattr(base, "base_model", None) or getattr(base, "model", None) or getattr(base, "module")
    decoder = base.get_decoder()
    head = base.get_output_embeddings()
    cfg = text_config(base)
    softcap = getattr(cfg, "final_logit_softcapping", None)
    scale = getattr(cfg, "logit_scale", None)
    return decoder, head, softcap, scale


ATTN_OUT_NAMES = ("o_proj", "out_proj", "wo", "dense", "c_proj")
MLP_OUT_NAMES = ("down_proj", "w2", "fc2", "dense_4h_to_h", "c_proj")


def residual_writers(layer: nn.Module, hidden: int) -> dict[str, list[tuple[str, nn.Linear]]]:
    """Linear modules of one decoder layer that write into the residual stream.

    Returns {'attn': [...], 'mlp': [...]} matched by name, validated by out_features == hidden.
    """
    found: dict[str, list[tuple[str, nn.Linear]]] = {"attn": [], "mlp": []}
    for name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear) and not hasattr(mod, "weight"):
            continue
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 2 or w.shape[0] != hidden:
            continue
        leaf = name.split(".")[-1]
        lname = name.lower()
        if "lora" in lname or "base_layer" in lname:
            continue
        if leaf in ATTN_OUT_NAMES and ("attn" in lname or "attention" in lname):
            found["attn"].append((name, mod))
        elif leaf in MLP_OUT_NAMES and ("mlp" in lname or "ffn" in lname or "feed_forward" in lname):
            found["mlp"].append((name, mod))
    return found


def enable_gradient_checkpointing(model) -> None:
    """HF checkpointing if supported, else wrap each decoder layer manually (non-reentrant)."""
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if getattr(model, "is_gradient_checkpointing", True):
            log.info("gradient checkpointing: transformers")
            return
    except (ValueError, AttributeError, TypeError) as e:
        log.warning("HF gradient checkpointing unavailable (%s); wrapping layers manually", e)
    from torch.utils.checkpoint import checkpoint

    for layer in decoder_layers(model):
        if getattr(layer, "_spark_ckpt", False):
            continue
        orig = layer.forward

        def fwd(*args, _orig=orig, _layer=layer, **kwargs):
            if torch.is_grad_enabled() and _layer.training:
                return checkpoint(_orig, *args, use_reentrant=False, **kwargs)
            return _orig(*args, **kwargs)

        layer.forward = fwd
        layer._spark_ckpt = True
    log.info("gradient checkpointing: manual per-layer")


def copy_remote_code(src: str, dst: str) -> None:
    """Copy modeling/config python files + chat template so the saved model loads with trust_remote_code."""
    src_dir = resolve_local_dir(src)
    if src_dir is None:
        return
    Path(dst).mkdir(parents=True, exist_ok=True)
    patterns = ["*.py", "chat_template.jinja", "generation_config.json", "tokenizer*", "special_tokens_map.json",
                "*.tiktoken", "vocab*", "merges.txt"]
    for pat in patterns:
        for f in glob.glob(os.path.join(src_dir, pat)):
            target = os.path.join(dst, os.path.basename(f))
            if not os.path.exists(target):
                shutil.copy2(f, target)


def resolve_local_dir(model_id: str) -> str | None:
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_id, allow_patterns=["*.py", "*.json", "*.jinja", "*.txt", "*.model",
                                                           "*.tiktoken", "tokenizer*"])
    except Exception as e:
        log.warning("could not resolve local snapshot for %s: %s", model_id, e)
        return None


def load_for_inference(model_id: str, adapters: list[str] | None, device: str, hw_cfg: dict):
    """fp16 model on one GPU with LoRA adapters merged in order (in memory)."""
    dtype = hw_cfg.get("dtype", "float16") if str(device) != "cpu" else "float32"
    model = load_model(model_id, device=device, dtype=dtype, attn=hw_cfg.get("attn_implementation", "sdpa"))
    for a in adapters or []:
        if not a:
            continue
        if not os.path.isdir(a):
            raise FileNotFoundError(f"adapter not found: {a}")
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, a).merge_and_unload()
        log.info("merged adapter %s", a)
    model.eval()
    return model


def free_gb(device: int | str = 0) -> float:
    if not torch.cuda.is_available():
        return 8.0
    free, _ = torch.cuda.mem_get_info(torch.device(device))
    return free / 2**30

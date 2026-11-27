"""Merge LoRA adapters into the base checkpoint in fp32 on CPU, save bf16 safetensors.

Merging in fp32 matters: adding a small LoRA delta directly to bf16 weights rounds part of it away.
Kaggle T4x2 VMs have ~29 GB RAM; a 4B model in fp32 is ~16.5 GB.

    python -m spark_kaggle.merge --config configs/default.yaml
"""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path

import torch

from .config import base_parser, cfg_from_args
from .env import hf_login, setup_env, setup_logging, write_json
from .modeling import DTYPES, copy_remote_code, load_model, load_tokenizer

log = logging.getLogger("spark")


def merge_adapters(model_id: str, adapters: list[str], out_dir: str, save_dtype: str = "bfloat16") -> str:
    from peft import PeftModel

    model = load_model(model_id, device="cpu", dtype="float32", attn="eager")
    used = []
    for a in adapters:
        if not a:
            continue
        if not os.path.isdir(a):
            log.warning("adapter %s not found - skipped", a)
            continue
        model = PeftModel.from_pretrained(model, a, torch_dtype=torch.float32).merge_and_unload()
        used.append(a)
        log.info("merged %s", a)
        gc.collect()
    model = model.to(DTYPES[save_dtype])
    model.config.use_cache = True
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="2GB")
    tok = load_tokenizer(model_id)
    tok.save_pretrained(out_dir)
    copy_remote_code(model_id, out_dir)
    write_json(Path(out_dir) / "merge_meta.json", {"base": model_id, "adapters": used, "dtype": save_dtype})
    log.info("merged model saved to %s", out_dir)
    del model
    gc.collect()
    return out_dir


def main() -> None:
    p = base_parser("Merge LoRA adapters")
    p.add_argument("--adapters", default=None, help="comma-separated (default cfg.merge.adapters)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging()
    hf_login()
    adapters = args.adapters.split(",") if args.adapters else list(cfg.merge.adapters)
    merge_adapters(cfg.model_id, adapters, args.out or cfg.merge.out_dir, cfg.merge.save_dtype)


if __name__ == "__main__":
    main()

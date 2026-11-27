"""Environment helpers: Kaggle detection, HF login, GPU report, seeding, JSONL IO, distributed init."""

from __future__ import annotations

import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator

log = logging.getLogger("spark")


def setup_logging(rank: int = 0) -> None:
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        level=level,
        format=f"%(asctime)s [r{rank}] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


def setup_env() -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if is_kaggle():
        os.environ.setdefault("HF_HOME", "/tmp/hf_home")  # root disk is larger than /kaggle/working


def is_kaggle() -> bool:
    return os.path.isdir("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def hf_login() -> str | None:
    """Log into the HF Hub using a Kaggle secret named HF_TOKEN, or the HF_TOKEN env var."""
    token = os.environ.get("HF_TOKEN")
    if not token and is_kaggle():
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            token = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception as e:  # secret not attached
            log.warning("No HF_TOKEN Kaggle secret (%s). Gated datasets/models will fail.", e)
    if token:
        os.environ["HF_TOKEN"] = token
        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False)
        except Exception as e:
            log.warning("huggingface_hub.login failed: %s", e)
    return token


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def gpu_report() -> list[dict]:
    import torch

    out = []
    if not torch.cuda.is_available():
        log.warning("CUDA not available - running on CPU (tests only).")
        return out
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        info = {
            "index": i,
            "name": p.name,
            "capability": f"{p.major}.{p.minor}",
            "total_gb": round(total / 2**30, 2),
            "free_gb": round(free / 2**30, 2),
        }
        out.append(info)
        log.info("GPU %d: %s sm_%d%d  %.1f/%.1f GB free", i, p.name, p.major, p.minor, info["free_gb"], info["total_gb"])
        if (p.major, p.minor) < (8, 0):
            log.info("  sm_%d%d: no bf16 tensor cores, no FlashAttention-2 -> fp16 + SDPA path.", p.major, p.minor)
    return out


def disk_report(paths: Iterable[str]) -> None:
    for p in paths:
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            u = shutil.disk_usage(p)
            log.info("disk %s: %.1f GB free / %.1f GB", p, u.free / 2**30, u.total / 2**30)
        except OSError as e:
            log.warning("disk %s: %s", p, e)


# ---------------------------------------------------------------- JSONL

def read_jsonl(path: str | os.PathLike) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | os.PathLike) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | os.PathLike, rows: Iterable[dict], append: bool = False) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "a" if append else "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_json(path: str | os.PathLike, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path: str | os.PathLike, default: Any = None) -> Any:
    if not os.path.exists(path):
        return default
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------- distributed (torchrun)

def dist_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    return rank, world, local


def init_distributed() -> tuple[int, int, int]:
    import torch
    import torch.distributed as dist

    rank, world, local = dist_info()
    if world > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
        dist.init_process_group(backend=backend)
    return rank, world, local


def barrier() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


class Timer:
    def __init__(self, budget_hours: float | None = None):
        self.t0 = time.time()
        self.budget = budget_hours * 3600 if budget_hours else None

    def elapsed(self) -> float:
        return time.time() - self.t0

    def expired(self) -> bool:
        return self.budget is not None and self.elapsed() > self.budget

"""Shared training machinery for SFT / DPO / GRPO on 2x T4.

Design choices for Kaggle T4s:
  * LoRA on every linear layer of a frozen fp16 base (or NF4 with quant=nf4)
  * fp16 autocast + GradScaler (no bf16 on sm_75); LoRA weights stay fp32
  * chunked lm_head + cross-entropy under activation checkpointing: never materialises [T, vocab] logits
  * data parallel across GPUs with ONE flat all-reduce of LoRA grads per optimizer step (no DDP wrapper,
    so activation checkpointing / PEFT / remote code interact with nothing)
  * resumable checkpoints + wall-clock budget so a 12 h Kaggle session can stop and continue
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .modeling import enable_gradient_checkpointing, get_decoder_and_head, load_model

log = logging.getLogger("spark")


# ============================================================================ model

def build_trainable(model_id: str, stage_cfg, lora_cfg, init_adapters: list[str], device: str, hw_cfg,
                    resume_adapter: str | None = None):
    """Load base (fp16 or nf4), merge `init_adapters`, attach a fresh LoRA (or resume one)."""
    from peft import LoraConfig, PeftModel, get_peft_model

    cuda = str(device).startswith("cuda")
    quant = stage_cfg.get("quant", "none")
    quant = None if quant in (None, "none") else quant
    dtype = hw_cfg.get("dtype", "float16") if cuda else "float32"
    model = load_model(model_id, device=device, dtype=dtype, attn=hw_cfg.get("attn_implementation", "sdpa"),
                       quant=quant)
    for a in init_adapters or []:
        if a and os.path.isdir(a):
            model = PeftModel.from_pretrained(model, a).merge_and_unload()
            log.info("merged init adapter %s%s", a, " (into 4-bit weights: lossy)" if quant else "")
        elif a:
            raise FileNotFoundError(f"init adapter missing: {a}")
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    enable_gradient_checkpointing(model)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if resume_adapter and os.path.isdir(resume_adapter):
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
        log.info("resumed adapter %s", resume_adapter)
    else:
        lc = LoraConfig(r=int(lora_cfg.r), lora_alpha=int(lora_cfg.alpha), lora_dropout=float(lora_cfg.dropout),
                        target_modules=lora_cfg.target_modules, use_rslora=bool(lora_cfg.get("use_rslora", False)),
                        bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
    for n, p in model.named_parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()  # GradScaler cannot unscale fp16 grads
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("trainable LoRA params: %.1fM", n_train / 1e6)
    model.train()
    return model


# ============================================================================ losses

def _chunk_logp(h, t, w, b, softcap, scale):
    logits = F.linear(h, w, b).float()
    if scale:
        logits = logits * scale
    if softcap:
        logits = softcap * torch.tanh(logits / softcap)
    return -F.cross_entropy(logits, t, reduction="none")


def token_logps(model, input_ids, attention_mask, labels, chunk: int = 1024):
    """Log-probs of every labelled next token. Returns (logp [N] fp32, row index [N])."""
    decoder, head, softcap, scale = get_decoder_and_head(model)
    out = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    h = out[0]
    tgt = labels[:, 1:]
    mask = tgt != -100
    hs = h[:, :-1][mask]
    ts = tgt[mask]
    rows = torch.arange(labels.size(0), device=labels.device).unsqueeze(1).expand_as(tgt)[mask]
    w = head.weight
    b = getattr(head, "bias", None)
    pieces = []
    for i in range(0, hs.size(0), chunk):
        if torch.is_grad_enabled():
            pieces.append(checkpoint(_chunk_logp, hs[i:i + chunk], ts[i:i + chunk], w, b, softcap, scale,
                                     use_reentrant=False))
        else:
            pieces.append(_chunk_logp(hs[i:i + chunk], ts[i:i + chunk], w, b, softcap, scale))
    logp = torch.cat(pieces) if pieces else hs.new_zeros(0, dtype=torch.float32)
    return logp, rows


def seq_logps(model, input_ids, attention_mask, labels, chunk: int = 1024):
    """Sum of label log-probs per row + label counts per row."""
    logp, rows = token_logps(model, input_ids, attention_mask, labels, chunk)
    B = input_ids.size(0)
    sums = torch.zeros(B, device=logp.device, dtype=torch.float32).index_add(0, rows, logp)
    counts = torch.zeros(B, device=logp.device, dtype=torch.float32).index_add(0, rows, torch.ones_like(logp))
    return sums, counts


# ============================================================================ batching

def segments_to_example(segments: list[dict], max_len: int) -> dict | None:
    ids, labels = [], []
    for s in segments:
        ids += s["ids"]
        labels += s["ids"] if s["train"] else [-100] * len(s["ids"])
    if len(ids) > max_len or not any(l != -100 for l in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def pad_batch(examples: list[dict], pad_id: int, device) -> dict:
    L = max(len(e["input_ids"]) for e in examples)
    ids = torch.full((len(examples), L), pad_id, dtype=torch.long)
    lab = torch.full((len(examples), L), -100, dtype=torch.long)
    att = torch.zeros((len(examples), L), dtype=torch.long)
    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"])
        lab[i, :n] = torch.tensor(e["labels"])
        att[i, :n] = 1
    return {"input_ids": ids.to(device), "labels": lab.to(device), "attention_mask": att.to(device)}


def token_budget_batches(examples: list[Any], length_fn, budget: int, seed: int) -> list[list[Any]]:
    """Length-sorted greedy packing into padded-token budgets, then shuffled."""
    order = sorted(range(len(examples)), key=lambda i: length_fn(examples[i]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = length_fn(examples[i])
        if cur and max(cur_max, L) * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(examples[i])
        cur_max = max(cur_max, L)
    if cur:
        batches.append(cur)
    random.Random(seed).shuffle(batches)
    return batches


def plan_steps(n_micro: int, world: int, tokens_per_micro: float, global_tokens: int, epochs: int = 1):
    """Same plan on every rank: micro-batches per optimizer step and number of steps."""
    per_rank = n_micro // world
    accum = max(1, round(global_tokens / max(world * tokens_per_micro, 1)))
    steps = (per_rank // accum) * epochs
    return accum, steps, per_rank


def memory_probe(model, batches: list[dict], device, chunk: int, label: str = "") -> None:
    """Forward+backward the largest padded batch once before training so an OOM shows up in minute one,
    not hours into a Kaggle session."""
    if not str(device).startswith("cuda") or not batches:
        return
    big = max(batches, key=lambda b: b["input_ids"].numel())
    torch.cuda.reset_peak_memory_stats(device)
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            s, c = seq_logps(model, big["input_ids"], big["attention_mask"], big["labels"], chunk)
        (-(s.sum()) / c.sum().clamp(min=1)).backward()
    except torch.cuda.OutOfMemoryError as e:
        raise SystemExit(f"OOM on the largest {label} micro-batch {tuple(big['input_ids'].shape)}: lower max_seq_len / "
                         "micro_batch_tokens, or use quant=nf4") from e
    finally:
        for p in model.parameters():
            p.grad = None
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    total = torch.cuda.get_device_properties(device).total_memory / 2**30
    log.info("memory probe (%s, batch %s): peak %.1f / %.1f GB", label, tuple(big["input_ids"].shape), peak, total)
    torch.cuda.empty_cache()


# ============================================================================ optimizer / step

class LoraOptimizer:
    def __init__(self, model, lr: float, total_steps: int, warmup_ratio: float = 0.03, min_lr_ratio: float = 0.1,
                 weight_decay: float = 0.0, max_grad_norm: float = 1.0):
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.cuda = self.params[0].is_cuda
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
        warm = max(1, int(total_steps * warmup_ratio))

        def lr_lambda(step):
            if step < warm:
                return (step + 1) / warm
            prog = min(1.0, (step - warm) / max(1, total_steps - warm))
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * prog))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.cuda, init_scale=2.0**14)
        self.max_grad_norm = max_grad_norm
        self.step_idx = 0

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.float16, enabled=self.cuda)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def step(self, grad_multiplier: float = 1.0) -> dict:
        for p in self.params:
            if p.grad is None:
                p.grad = torch.zeros_like(p)
        grads = [p.grad for p in self.params]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            flat = torch._utils._flatten_dense_tensors(grads)
            dist.all_reduce(flat)
            for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
                g.copy_(s)
        if grad_multiplier != 1.0:
            for g in grads:
                g.mul_(grad_multiplier)
        self.scaler.unscale_(self.opt)
        norm = torch.nn.utils.clip_grad_norm_(self.params, self.max_grad_norm)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.opt.zero_grad(set_to_none=True)
        self.sched.step()
        self.step_idx += 1
        return {"grad_norm": float(norm), "lr": self.sched.get_last_lr()[0],
                "loss_scale": float(self.scaler.get_scale()) if self.cuda else 1.0}

    def state_dict(self) -> dict:
        return {"opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "scaler": self.scaler.state_dict(),
                "step": self.step_idx}

    def load_state_dict(self, sd: dict) -> None:
        self.opt.load_state_dict(sd["opt"])
        self.sched.load_state_dict(sd["sched"])
        self.scaler.load_state_dict(sd["scaler"])
        self.step_idx = sd["step"]


def all_reduce_sum(x: float, device) -> float:
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        t = torch.tensor([x], device=device, dtype=torch.float64)
        dist.all_reduce(t)
        return float(t.item())
    return x


def all_reduce_vec(vals: Iterable[float], device) -> list[float]:
    t = torch.tensor(list(vals), device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t)
    return t.tolist()


# ============================================================================ checkpoints

def save_checkpoint(model, opt: LoraOptimizer, out_dir: str, extra: dict, rank: int) -> None:
    if rank != 0:
        return
    ck = Path(out_dir) / "checkpoint"
    tmp = Path(out_dir) / "checkpoint.tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(tmp)
    torch.save({"optim": opt.state_dict(), "extra": extra, "rng": random.getstate()}, tmp / "trainer_state.pt")
    if ck.exists():
        import shutil

        shutil.rmtree(ck)
    tmp.rename(ck)
    log.info("checkpoint saved at step %d -> %s", opt.step_idx, ck)


def load_checkpoint(opt: LoraOptimizer, out_dir: str) -> dict | None:
    f = Path(out_dir) / "checkpoint" / "trainer_state.pt"
    if not f.exists():
        return None
    sd = torch.load(f, map_location="cpu", weights_only=False)
    opt.load_state_dict(sd["optim"])
    random.setstate(sd["rng"])
    log.info("resumed trainer state at step %d", opt.step_idx)
    return sd["extra"]


def save_final(model, out_dir: str, rank: int, meta: dict) -> None:
    if rank != 0:
        return
    final = Path(out_dir) / "final"
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    (final / "training_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    log.info("final adapter -> %s", final)


class MetricLog:
    def __init__(self, path: str, rank: int):
        self.path, self.rank = path, rank
        self.t0 = time.time()
        if rank == 0:
            Path(path).parent.mkdir(parents=True, exist_ok=True)

    def write(self, row: dict) -> None:
        if self.rank != 0:
            return
        row = {"time_s": round(time.time() - self.t0, 1), **row}
        with open(self.path, "a") as f:
            f.write(json.dumps(row) + "\n")

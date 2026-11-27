"""Batched HF-transformers generation sized for a 15 GB T4.

vLLM is not used: Spark2_5ForCausalLM is not in vLLM's registry (needs an out-of-tree plugin) and
vLLM's Turing (sm_75) support is limited. Plain `model.generate` with length-sorted, memory-budgeted
batches gets roughly 500-1200 tok/s per T4 for a 4B fp16 model.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import torch

from .modeling import free_gb, get_decoder_and_head, kv_bytes_per_token, text_config

log = logging.getLogger("spark")


@contextlib.contextmanager
def last_position_logits(model):
    """Make lm_head see only the last position (avoids [B, T, vocab] prefill logits on remote code
    that lacks `logits_to_keep`)."""
    _, head, _, _ = get_decoder_and_head(model)
    handle = head.register_forward_pre_hook(lambda mod, args: (args[0][:, -1:, :],) + tuple(args[1:]))
    try:
        yield
    finally:
        handle.remove()


def eos_ids(model, tok) -> list[int]:
    ids: set[int] = set()
    gc = getattr(model, "generation_config", None)
    e = getattr(gc, "eos_token_id", None) if gc is not None else None
    if isinstance(e, int):
        ids.add(e)
    elif isinstance(e, (list, tuple)):
        ids.update(int(x) for x in e)
    if tok.eos_token_id is not None:
        ids.add(int(tok.eos_token_id))
    return sorted(ids)


class Generator:
    def __init__(self, model, tok, hw_cfg: dict | None = None, device: Any = None):
        self.model = model
        self.tok = tok
        hw = hw_cfg or {}
        self.max_batch_tokens = int(hw.get("gen_max_batch_tokens", 98304))
        self.max_batch_size = int(hw.get("gen_max_batch_size", 48))
        self.kv_fraction = float(hw.get("kv_mem_fraction", 0.55))
        self.device = device if device is not None else next(model.parameters()).device
        # upper bound from the config unless preflight measured the real cost (sliding-window layers keep less)
        self.kv_per_tok = int(hw.get("kv_bytes_per_token") or kv_bytes_per_token(model))
        self.eos = eos_ids(model, tok)
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else self.eos[0]
        self.oom_scale = 1.0
        self.vocab = int(getattr(text_config(model), "vocab_size", 150000))

    def _batch_limit(self, seq_len: int) -> int:
        by_tokens = self.max_batch_tokens // max(seq_len, 1)
        if torch.cuda.is_available() and self.device.type == "cuda":
            free = free_gb(self.device) * 2**30 * self.kv_fraction
            # KV cache + one step of fp32 sampling logits per row
            by_mem = int(free // (seq_len * self.kv_per_tok + self.vocab * 8))
        else:
            by_mem = self.max_batch_size
        return max(1, int(min(self.max_batch_size, by_tokens, by_mem) * self.oom_scale))

    @torch.no_grad()
    def generate(self, prompts: list[list[int]], max_new_tokens: int, temperature: float = 1.0,
                 top_p: float = 0.95, top_k: int = 0, progress: bool = False) -> list[dict]:
        n = len(prompts)
        order = sorted(range(n), key=lambda i: -len(prompts[i]))
        results: list[dict | None] = [None] * n
        i = 0
        was_training = self.model.training
        self.model.eval()
        try:
            while i < n:
                seq_len = len(prompts[order[i]]) + max_new_tokens
                bs = min(self._batch_limit(seq_len), n - i)
                idxs = order[i:i + bs]
                try:
                    outs = self._gen_batch([prompts[j] for j in idxs], max_new_tokens, temperature, top_p, top_k)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if bs == 1:
                        log.warning("OOM at batch 1 (seq %d) - marking as failed", seq_len)
                        outs = [{"ids": [], "finish": "oom"}]
                    else:
                        self.oom_scale *= 0.6
                        log.warning("OOM with batch %d (seq %d); scaling batch by %.2f", bs, seq_len, self.oom_scale)
                        continue
                for j, o in zip(idxs, outs):
                    results[j] = o
                i += len(outs)
                if progress:
                    log.info("generated %d/%d", i, n)
        finally:
            if was_training:
                self.model.train()
        return results  # type: ignore[return-value]

    def _gen_batch(self, batch: list[list[int]], max_new_tokens: int, temperature: float, top_p: float,
                   top_k: int) -> list[dict]:
        L = max(len(p) for p in batch)
        input_ids = torch.full((len(batch), L), self.pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), L), dtype=torch.long)
        for r, p in enumerate(batch):
            input_ids[r, L - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[r, L - len(p):] = 1
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens, pad_token_id=self.pad_id, eos_token_id=self.eos, use_cache=True)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=float(top_p),
                              top_k=int(top_k) if top_k else None)
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        with last_position_logits(self.model):
            out = self.model.generate(input_ids=input_ids, attention_mask=attn, **gen_kwargs)
        new = out[:, L:].tolist()
        res = []
        eos = set(self.eos)
        for row in new:
            finish, stop_id = "length", None
            ids = []
            for t in row:
                if t in eos:
                    finish, stop_id = "stop", t
                    break
                ids.append(t)
            res.append({"ids": ids, "finish": finish, "stop_id": stop_id})
        return res

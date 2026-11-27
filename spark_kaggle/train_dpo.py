"""LoRA DPO (+ RPO-style NLL on the chosen answer) on on-policy pass/fail pairs, 2x T4.

The SFT adapter is merged into the base first; that merged model is the frozen reference
(computed once up front with the new adapter disabled), so no second model copy is needed.

    torchrun --nproc_per_node 2 -m spark_kaggle.train_dpo --config configs/default.yaml
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import base_parser, cfg_from_args
from .env import (Timer, cleanup_distributed, hf_login, init_distributed, iter_jsonl, seed_everything, setup_env,
                  setup_logging)
from .modeling import load_tokenizer
from .train_common import (LoraOptimizer, MetricLog, all_reduce_vec, build_trainable, load_checkpoint,
                           memory_probe, pad_batch, save_checkpoint, save_final, seq_logps, token_budget_batches)

log = logging.getLogger("spark")


def pair_examples(row: dict, max_len: int) -> list[dict] | None:
    p = row["prompt_ids"]
    out = []
    for key in ("chosen_ids", "rejected_ids"):
        c = row[key]
        if len(p) + len(c) > max_len or not c:
            return None
        out.append({"input_ids": p + c, "labels": [-100] * len(p) + c})
    return out


def main() -> None:
    p = base_parser("LoRA DPO")
    p.add_argument("--data", default=None)
    p.add_argument("--tag", default="base")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--time-budget-hours", type=float, default=9.0)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    rank, world, local = init_distributed()
    setup_logging(rank)
    hf_login()
    seed_everything(cfg.seed)
    dc = cfg.dpo
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    data_path = args.data or str(Path(cfg.run_dir) / "collect" / args.tag / "dpo.jsonl")
    tok = load_tokenizer(cfg.model_id, padding_side="right")

    pairs = [x for x in (pair_examples(r, int(dc.max_seq_len)) for r in iter_jsonl(data_path)) if x]
    log.info("DPO pairs: %d", len(pairs))
    groups = token_budget_batches(pairs, lambda pr: max(len(e["input_ids"]) for e in pr),
                                  int(dc.max_seq_len), cfg.seed)  # ~1-3 pairs per micro-batch
    per_rank = len(groups) // world
    my = groups[rank::world][:per_rank]
    pairs_per_micro = len(pairs) / max(len(groups), 1)
    accum = max(1, round(int(dc.pairs_per_step) / (world * pairs_per_micro)))
    steps = (per_rank // accum) * int(dc.epochs)
    if args.max_steps:
        steps = min(steps, args.max_steps)
    if steps < 1:
        raise SystemExit("not enough DPO pairs for one step; lower dpo.pairs_per_step")
    log.info("micro/rank=%d accum=%d steps=%d", per_rank, accum, steps)

    ck_dir = Path(dc.out_dir) / "checkpoint"
    model = build_trainable(cfg.model_id, dc, cfg.lora, list(dc.init_adapters), device, cfg.hardware,
                            resume_adapter=str(ck_dir) if ck_dir.exists() else None)

    # ---- reference log-probs (adapter disabled == merged SFT policy), cached per micro-batch
    ref: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad(), model.disable_adapter():
        for g in my:
            batch = pad_batch([e for pr in g for e in pr], tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                s, _ = seq_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"],
                                 int(dc.loss_chunk))
            ref.append(s.float().cpu())
    model.train()
    log.info("reference log-probs cached for %d micro-batches", len(ref))

    longest = max(my, key=lambda g: max(len(e["input_ids"]) for pr in g for e in pr) * len(g))
    memory_probe(model, [pad_batch([e for pr in longest for e in pr], tok.pad_token_id, device)], device,
                 int(dc.loss_chunk), "DPO")
    opt = LoraOptimizer(model, float(dc.lr), steps, float(dc.warmup_ratio), 0.0, 0.0, float(dc.max_grad_norm))
    load_checkpoint(opt, dc.out_dir)
    metrics = MetricLog(str(Path(dc.out_dir) / "metrics.jsonl"), rank)
    timer = Timer(args.time_budget_hours)
    beta, sft_w = float(dc.beta), float(dc.sft_weight)
    scale_const = float(dc.pairs_per_step)

    for step in range(opt.step_idx, steps):
        acc = {"loss": 0.0, "pairs": 0.0, "correct": 0.0, "margin": 0.0, "chosen_nll": 0.0}
        for k in range(accum):
            j = (step * accum + k) % len(my)
            g = my[j]
            batch = pad_batch([e for pr in g for e in pr], tok.pad_token_id, device)
            with opt.autocast():
                s, cnt = seq_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"],
                                   int(dc.loss_chunk))
            r = ref[j].to(s.device)
            pc, pr_, rc, rr = s[0::2], s[1::2], r[0::2], r[1::2]
            logits = beta * ((pc - rc) - (pr_ - rr))
            nll = -pc / cnt[0::2].clamp(min=1)
            loss_vec = -F.logsigmoid(logits) + sft_w * nll
            opt.backward(loss_vec.sum() / scale_const)
            acc["loss"] += float(loss_vec.sum().detach())
            acc["pairs"] += len(g)
            acc["correct"] += float((logits > 0).float().sum())
            acc["margin"] += float(logits.detach().sum() / beta)
            acc["chosen_nll"] += float(nll.detach().sum())
        tot = dict(zip(acc, all_reduce_vec(acc.values(), device)))
        info = opt.step(grad_multiplier=scale_const / max(tot["pairs"], 1.0))
        if rank == 0 and (step % 5 == 0 or step == steps - 1):
            n = max(tot["pairs"], 1)
            row = {"step": step + 1, "steps": steps, "loss": tot["loss"] / n, "reward_acc": tot["correct"] / n,
                   "margin": tot["margin"] / n, "chosen_nll": tot["chosen_nll"] / n, **info}
            log.info("step %d/%d loss %.4f acc %.3f margin %.3f nll %.3f", step + 1, steps, row["loss"],
                     row["reward_acc"], row["margin"], row["chosen_nll"])
            metrics.write(row)
        if (step + 1) % int(dc.save_every_steps) == 0:
            save_checkpoint(model, opt, dc.out_dir, {"step": step + 1}, rank)
        if timer.expired():
            save_checkpoint(model, opt, dc.out_dir, {"step": step + 1}, rank)
            log.warning("time budget reached - stopped at step %d (rerun to resume)", step + 1)
            cleanup_distributed()
            return
    save_final(model, dc.out_dir, rank, {"steps": steps, "pairs": len(pairs), "dpo": dict(dc)})
    cleanup_distributed()


if __name__ == "__main__":
    main()

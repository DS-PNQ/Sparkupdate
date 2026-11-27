"""LoRA SFT on verified self-generated data (RFT / self-distillation), 2x T4 data parallel.

    torchrun --nproc_per_node 2 -m spark_kaggle.train_sft --config configs/default.yaml
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import torch

from .config import base_parser, cfg_from_args
from .env import (Timer, cleanup_distributed, hf_login, init_distributed, iter_jsonl, seed_everything, setup_env,
                  setup_logging)
from .modeling import load_tokenizer
from .train_common import (LoraOptimizer, MetricLog, all_reduce_vec, build_trainable, load_checkpoint,
                           memory_probe, pad_batch, plan_steps, save_checkpoint, save_final, segments_to_example,
                           seq_logps, token_budget_batches)

log = logging.getLogger("spark")


def main() -> None:
    p = base_parser("LoRA SFT")
    p.add_argument("--data", default=None, help="sft.jsonl (default: run_dir/collect/<tag>/sft.jsonl)")
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
    sc = cfg.sft
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"
    data_path = args.data or str(Path(cfg.run_dir) / "collect" / args.tag / "sft.jsonl")

    tok = load_tokenizer(cfg.model_id, padding_side="right")
    pad_id = tok.pad_token_id
    examples, dropped = [], 0
    for row in iter_jsonl(data_path):
        ex = segments_to_example(row["segments"], int(sc.max_seq_len))
        if ex is None:
            dropped += 1
        else:
            examples.append(ex)
    log.info("SFT examples: %d (dropped %d longer than %d tokens)", len(examples), dropped, sc.max_seq_len)
    if not examples:
        raise SystemExit("no SFT examples")

    batches = token_budget_batches(examples, lambda e: len(e["input_ids"]), int(sc.micro_batch_tokens), cfg.seed)
    tokens_per_micro = sum(len(e["input_ids"]) for e in examples) / len(batches)
    accum, steps, per_rank = plan_steps(len(batches), world, tokens_per_micro, int(sc.global_batch_tokens),
                                        int(sc.epochs))
    if args.max_steps:
        steps = min(steps, args.max_steps)
    if steps < 1:
        raise SystemExit(f"not enough data for one optimizer step ({len(batches)} micro-batches, accum {accum}); "
                         "lower sft.global_batch_tokens")
    my_batches = batches[rank::world][:per_rank]
    log.info("micro-batches/rank=%d accum=%d steps=%d (~%.0f tokens/micro)", per_rank, accum, steps,
             tokens_per_micro)

    ck_dir = Path(sc.out_dir) / "checkpoint"
    model = build_trainable(cfg.model_id, sc, cfg.lora, [], device, cfg.hardware,
                            resume_adapter=str(ck_dir) if ck_dir.exists() else None)
    longest = max(my_batches, key=lambda b: max(len(e["input_ids"]) for e in b) * len(b))
    memory_probe(model, [pad_batch(longest, pad_id, device)], device, int(sc.loss_chunk), "SFT")
    opt = LoraOptimizer(model, float(sc.lr), steps, float(sc.warmup_ratio), float(sc.min_lr_ratio),
                        float(sc.weight_decay), float(sc.max_grad_norm))
    extra = load_checkpoint(opt, sc.out_dir) or {}
    start_step = opt.step_idx
    metrics = MetricLog(str(Path(sc.out_dir) / "metrics.jsonl"), rank)
    timer = Timer(args.time_budget_hours)
    budget = float(sc.micro_batch_tokens)
    t_last, tok_count = time.time(), 0.0

    for step in range(start_step, steps):
        loss_sum = n_tok = 0.0
        for k in range(accum):
            mb = my_batches[(step * accum + k) % len(my_batches)]
            batch = pad_batch(mb, pad_id, device)
            with opt.autocast():
                sums, counts = seq_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"],
                                         int(sc.loss_chunk))
                loss = -sums.sum() / budget
            opt.backward(loss)
            loss_sum += float(-sums.sum().detach())
            n_tok += float(counts.sum())
            tok_count += float(batch["attention_mask"].sum())
        g_loss, g_tok = all_reduce_vec([loss_sum, n_tok], device)
        info = opt.step(grad_multiplier=budget / max(g_tok, 1.0))
        if rank == 0 and (step % 5 == 0 or step == steps - 1):
            dt = time.time() - t_last
            row = {"step": step + 1, "steps": steps, "loss": g_loss / max(g_tok, 1), "label_tokens": g_tok,
                   "tok_per_s_rank0": tok_count / max(dt, 1e-6), **info}
            log.info("step %d/%d loss %.4f lr %.2e gnorm %.3f %.0f tok/s", step + 1, steps, row["loss"], info["lr"],
                     info["grad_norm"], row["tok_per_s_rank0"])
            metrics.write(row)
            t_last, tok_count = time.time(), 0.0
        if (step + 1) % int(sc.save_every_steps) == 0:
            save_checkpoint(model, opt, sc.out_dir, {"step": step + 1}, rank)
        if timer.expired():
            log.warning("time budget reached at step %d - saving and stopping (rerun to resume)", step + 1)
            save_checkpoint(model, opt, sc.out_dir, {"step": step + 1}, rank)
            cleanup_distributed()
            return
    save_final(model, sc.out_dir, rank, {"steps": steps, "examples": len(examples), "data": data_path,
                                          "lora": dict(cfg.lora), "sft": dict(sc)})
    cleanup_distributed()


if __name__ == "__main__":
    main()

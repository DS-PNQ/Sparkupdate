"""Optional: GRPO-style RL with verifiable rewards (ToolRL fine-grained tool reward / unit tests).

On-policy, one gradient step per batch (so the PPO ratio is 1 and clipping is a no-op), group-mean
baseline, Dr.GRPO length normalisation (constant denominator), groups with zero reward variance
skipped (DAPO dynamic sampling). LoRA RL works at small rank (Thinking Machines, "LoRA Without Regret").

Expensive on T4s: ~3-4 min per step with 2x8 prompts x 6 samples. Enable with grpo.enabled=true.

    torchrun --nproc_per_node 2 -m spark_kaggle.train_grpo --config configs/default.yaml --set grpo.enabled=true
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import torch

from .chat import ChatFormatter
from .config import base_parser, cfg_from_args
from .engine import Generator
from .env import (Timer, cleanup_distributed, hf_login, init_distributed, read_json, read_jsonl, seed_everything,
                  setup_env, setup_logging)
from .modeling import load_tokenizer
from .train_common import (LoraOptimizer, MetricLog, all_reduce_vec, build_trainable, load_checkpoint, pad_batch,
                           save_checkpoint, save_final, token_budget_batches, token_logps)
from .verify import verify_code_batch, verify_tool

log = logging.getLogger("spark")


def main() -> None:
    p = base_parser("GRPO (optional)")
    p.add_argument("--tag", default="base", help="collection whose hard_tasks.json defines the curriculum")
    p.add_argument("--time-budget-hours", type=float, default=9.0)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    gc = cfg.grpo
    if not gc.enabled:
        print("grpo.enabled=false - skipping (pass --set grpo.enabled=true)")
        return
    setup_env()
    rank, world, local = init_distributed()
    setup_logging(rank)
    hf_login()
    seed_everything(cfg.seed + rank)
    device = f"cuda:{local}" if torch.cuda.is_available() else "cpu"

    tasks = []
    for t in gc.task_types:
        f = Path(cfg.tasks.out_dir) / f"{t}_train.jsonl"
        if f.exists():
            tasks += read_jsonl(f)
    if gc.only_hard:
        hard = set(read_json(Path(cfg.run_dir) / "collect" / args.tag / "hard_tasks.json", []) or [])
        if hard:
            tasks = [t for t in tasks if t["id"] in hard]
    random.Random(cfg.seed).shuffle(tasks)
    tasks = tasks[rank::world]
    if not tasks:
        raise SystemExit("no GRPO tasks")
    log.info("GRPO tasks on this rank: %d", len(tasks))

    tok = load_tokenizer(cfg.model_id)
    fmt = ChatFormatter(tok, cfg.chat)
    ck_dir = Path(gc.out_dir) / "checkpoint"
    model = build_trainable(cfg.model_id, {"quant": "none"}, cfg.lora, list(gc.init_adapters), device,
                            cfg.hardware, resume_adapter=str(ck_dir) if ck_dir.exists() else None)
    gen = Generator(model, tok, cfg.hardware, torch.device(device))
    opt = LoraOptimizer(model, float(gc.lr), int(gc.steps), 0.0, 1.0, 0.0, 1.0)
    load_checkpoint(opt, gc.out_dir)
    metrics = MetricLog(str(Path(gc.out_dir) / "metrics.jsonl"), rank)
    timer = Timer(args.time_budget_hours)
    G, P, mode = int(gc.group_size), int(gc.prompts_per_step), gc.mode
    max_new = int(gc.max_new_tokens)
    norm_const = float(P * G * max_new)  # Dr.GRPO constant normaliser (per rank)
    sampling = dict(temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p, top_k=cfg.sampling.top_k)

    for step in range(opt.step_idx, int(gc.steps)):
        batch_tasks = [tasks[(step * P + i) % len(tasks)] for i in range(P)]
        prompts = []
        for t in batch_tasks:
            msgs = t["messages"] if t["type"] == "tool" else [{"role": "user", "content": t["prompt"]}]
            prompts.append(fmt.prompt_ids(msgs, t.get("tools") if t["type"] == "tool" else None, mode))
        outs = gen.generate([pp for pp in prompts for _ in range(G)], max_new, **sampling)
        texts = [fmt.decode_completion(o["ids"]) for o in outs]
        rewards = [0.0] * len(outs)
        code_idx = [i for i in range(len(outs)) if batch_tasks[i // G]["type"] == "code"]
        for i in range(len(outs)):
            t = batch_tasks[i // G]
            if t["type"] == "tool":
                rewards[i] = verify_tool(t, texts[i], fmt, mode)["reward"]
        if code_idx:
            res = verify_code_batch([batch_tasks[i // G] for i in code_idx], [texts[i] for i in code_idx], fmt,
                                    [mode] * len(code_idx))
            for i, r in zip(code_idx, res):
                rewards[i] = r["reward"]
        for i, o in enumerate(outs):
            if o["finish"] != "stop":
                rewards[i] = min(rewards[i], 0.0)  # overlong -> no credit

        examples, n_used = [], 0
        for g in range(P):
            rs = torch.tensor(rewards[g * G:(g + 1) * G])
            if float(rs.std()) < 1e-6:
                continue
            adv = rs - rs.mean()
            if gc.scale_rewards:
                adv = adv / (rs.std() + 1e-4)
            n_used += 1
            for k in range(G):
                i = g * G + k
                comp = outs[i]["ids"] + ([outs[i]["stop_id"]] if outs[i].get("stop_id") is not None else [])
                if comp:
                    examples.append({"input_ids": prompts[g] + comp, "labels": [-100] * len(prompts[g]) + comp,
                                     "adv": float(adv[k])})
        model.train()
        if examples:
            for mb in token_budget_batches(examples, lambda e: len(e["input_ids"]), int(cfg.sft.micro_batch_tokens),
                                           step):
                batch = pad_batch(mb, tok.pad_token_id, device)
                adv = torch.tensor([e["adv"] for e in mb], device=device)
                with opt.autocast():
                    logp, rows = token_logps(model, batch["input_ids"], batch["attention_mask"], batch["labels"],
                                             int(gc.loss_chunk))
                    loss = -(adv[rows] * logp).sum() / norm_const
                    if float(gc.beta_kl) > 0:
                        with torch.no_grad(), model.disable_adapter():
                            ref, _ = token_logps(model, batch["input_ids"], batch["attention_mask"],
                                                 batch["labels"], int(gc.loss_chunk))
                        d = ref - logp
                        loss = loss + float(gc.beta_kl) * (torch.exp(d) - d - 1).sum() / norm_const
                opt.backward(loss)
        stats = all_reduce_vec([sum(rewards), len(rewards), sum(r >= 1.0 for r in rewards), n_used, P], device)
        info = opt.step(grad_multiplier=1.0 / world)
        if rank == 0:
            row = {"step": step + 1, "mean_reward": stats[0] / stats[1], "solve_rate": stats[2] / stats[1],
                   "groups_with_signal": stats[3] / stats[4], **info}
            log.info("step %d/%d reward %.3f solved %.3f signal %.2f", step + 1, gc.steps, row["mean_reward"],
                     row["solve_rate"], row["groups_with_signal"])
            metrics.write(row)
        if (step + 1) % int(gc.save_every_steps) == 0:
            save_checkpoint(model, opt, gc.out_dir, {"step": step + 1}, rank)
        if timer.expired():
            save_checkpoint(model, opt, gc.out_dir, {"step": step + 1}, rank)
            cleanup_distributed()
            return
    save_final(model, gc.out_dir, rank, {"grpo": dict(gc)})
    cleanup_distributed()


if __name__ == "__main__":
    main()

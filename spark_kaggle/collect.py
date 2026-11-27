"""Rejection-sampling self-distillation (RFT) + on-policy preference pairs.

The model samples K answers per verifiable task (thinking or no-think mode), every answer is checked
(tool-call AST match / unit tests / agent episode tests), and:
  * verified answers   -> sft.jsonl  (exact prompt/completion token ids - no template re-rendering)
  * pass vs fail pairs -> dpo.jsonl
  * per-task pass rates -> collect_stats.json, hard_tasks.json (curriculum for GRPO)

One worker per GPU (data parallel), resumable, time-budgeted:
    python -m spark_kaggle.collect --config configs/default.yaml --launch
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from .agent import run_agent_episodes
from .chat import ChatFormatter
from .config import base_parser, cfg_from_args
from .engine import Generator
from .env import (Timer, hf_login, iter_jsonl, read_jsonl, seed_everything, setup_env, setup_logging, write_json,
                  write_jsonl)
from .modeling import load_for_inference, load_tokenizer
from .verify import verify_code_batch, verify_tool

log = logging.getLogger("spark")


def task_mode(task_id: str, think_fraction: float) -> str:
    h = int(hashlib.md5(("mode" + task_id).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "think" if h < think_fraction else "nothink"


def collect_dir(cfg, tag: str) -> Path:
    return Path(cfg.run_dir) / "collect" / tag


# ============================================================================ worker

def run_worker(cfg, shard: int, num_shards: int, device: str, model_path: str, adapters: list[str], tag: str):
    out_dir = collect_dir(cfg, tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = out_dir / f"results_shard{shard}.jsonl"
    done = {r["task_id"] for r in iter_jsonl(res_path)} if res_path.exists() else set()
    seed_everything(cfg.seed + 1000 * shard + len(done))
    c = cfg.collect
    timer = Timer(c.time_budget_hours)

    tok = load_tokenizer(model_path)
    fmt = ChatFormatter(tok, cfg.chat)
    model = load_for_inference(model_path, adapters, device, cfg.hardware)
    gen = Generator(model, tok, cfg.hardware, torch.device(device))
    sampling = dict(temperature=cfg.sampling.temperature, top_p=cfg.sampling.top_p, top_k=cfg.sampling.top_k)
    K = int(c.samples_per_task)
    tdir = Path(cfg.tasks.out_dir)

    for ttype in c.task_types:
        path = tdir / f"{ttype}_train.jsonl"
        if not path.exists():
            log.warning("missing %s - run spark_kaggle.tasks first", path)
            continue
        tasks = [t for i, t in enumerate(read_jsonl(path)) if i % num_shards == shard and t["id"] not in done]
        log.info("[%s] shard %d: %d tasks to do (%d already done)", ttype, shard, len(tasks), len(done))
        chunk = 48 if ttype != "agent" else 16
        for start in range(0, len(tasks), chunk):
            if timer.expired():
                log.warning("time budget reached - stopping collection (resumable)")
                return
            part = tasks[start:start + chunk]
            t0 = time.time()
            if ttype == "agent":
                rows = _collect_agent(part, gen, fmt, c, sampling, K)
            else:
                rows = _collect_single(part, ttype, gen, fmt, c, sampling, K)
            write_jsonl(res_path, rows, append=True)
            npass = sum(any(s["pass"] for s in r.get("samples", r.get("episodes", []))) for r in rows)
            log.info("[%s] shard %d: %d/%d  (%.0fs, %d/%d tasks with >=1 pass, elapsed %.2fh)", ttype, shard,
                     start + len(part), len(tasks), time.time() - t0, npass, len(rows), timer.elapsed() / 3600)


def _collect_single(part, ttype, gen, fmt, c, sampling, K):
    by_mode = defaultdict(list)
    for t in part:
        by_mode[task_mode(t["id"], c.think_fraction)].append(t)
    rows = []
    for mode, ts in by_mode.items():
        max_new = c.max_new_tokens_think if mode == "think" else c.max_new_tokens_nothink
        prompts = []
        for t in ts:
            if ttype == "tool":
                p = fmt.prompt_ids(t["messages"], t["tools"], mode)
            else:
                p = fmt.prompt_ids([{"role": "user", "content": t["prompt"]}], None, mode)
            prompts.append(p)
        flat = [p for p in prompts for _ in range(K)]
        outs = gen.generate(flat, max_new, **sampling)
        texts = [fmt.decode_completion(o["ids"]) for o in outs]
        if ttype == "tool":
            ver = [verify_tool(ts[i // K], texts[i], fmt, mode) for i in range(len(outs))]
        else:
            ver = verify_code_batch([ts[i // K] for i in range(len(outs))], texts, fmt, [mode] * len(outs))
        for j, (t, p) in enumerate(zip(ts, prompts)):
            samples = []
            for k in range(K):
                o, v = outs[j * K + k], ver[j * K + k]
                passed = bool(v["pass"]) and o["finish"] == "stop"
                samples.append({
                    "completion_ids": o["ids"] + ([o["stop_id"]] if o.get("stop_id") is not None else []),
                    "finish": o["finish"], "pass": passed, "reward": float(v.get("reward", 0.0)),
                    "fail_reason": None if passed else v.get("fail_reason", o["finish"])})
            rows.append({"task_id": t["id"], "type": ttype, "category": t.get("category", t.get("source")),
                         "mode": mode, "prompt_ids": p, "samples": samples})
    return rows


def _collect_agent(part, gen, fmt, c, sampling, K):
    by_mode = defaultdict(list)
    for t in part:
        by_mode[task_mode(t["id"], c.think_fraction)].append(t)
    rows = []
    for mode, ts in by_mode.items():
        eps = run_agent_episodes(gen, fmt, ts, K, mode, c.agent_max_turns, c.agent_max_new_tokens,
                                 c.agent_max_context, sampling, keep_turns="pass")
        by_task = defaultdict(list)
        for e in eps:
            by_task[e["task_id"]].append(e)
        for t in ts:
            rows.append({"task_id": t["id"], "type": "agent", "category": t["category"], "mode": mode,
                         "episodes": by_task[t["id"]]})
    return rows


# ============================================================================ building training sets

def _merge_turns(turns: list[dict], max_len: int | None = None) -> list[list[dict]]:
    """Merge consecutive agent turns into one example when the next prompt extends prompt+completion
    (and the merged example stays within max_len); otherwise start a new example at that turn."""
    examples: list[list[dict]] = []
    cur: list[dict] | None = None
    cur_ids: list[int] = []
    for t in turns:
        p, comp = list(t["prompt_ids"]), list(t["completion_ids"])
        fits = max_len is None or len(p) + len(comp) <= max_len
        if cur is not None and fits and len(p) >= len(cur_ids) and p[:len(cur_ids)] == cur_ids:
            cur.append({"ids": p[len(cur_ids):], "train": False})
        else:
            if cur:
                examples.append(cur)
            cur = [{"ids": p, "train": False}]
        cur.append({"ids": comp, "train": True})
        cur_ids = p + comp
    if cur:
        examples.append(cur)
    return examples


def _gold_example(task: dict, fmt: ChatFormatter) -> list[dict] | None:
    msgs = task["messages"] + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": f"call_{i}", "type": "function", "function": {"name": g["name"], "arguments": g["arguments"]}}
        for i, g in enumerate(task["gold_calls"])]}]
    try:
        prompt = fmt.encode(fmt.prompt(task["messages"], task["tools"], "nothink"))
        full = fmt.encode(fmt.render(msgs, task["tools"], "nothink", add_generation_prompt=False))
    except Exception:
        return None
    if len(full) <= len(prompt) or full[:len(prompt)] != prompt:
        return None
    return [{"ids": prompt, "train": False}, {"ids": full[len(prompt):], "train": True}]


def build_sets(cfg, tag: str, tok=None) -> dict:
    out_dir = collect_dir(cfg, tag)
    rows = []
    for f in sorted(out_dir.glob("results_shard*.jsonl")):
        rows += read_jsonl(f)
    if not rows:
        raise SystemExit(f"no collection results in {out_dir}")
    c = cfg.collect
    rng = random.Random(cfg.seed)
    tasks_by_id = {}
    for ttype in ("tool",):
        p = Path(cfg.tasks.out_dir) / f"{ttype}_train.jsonl"
        if p.exists():
            tasks_by_id.update({t["id"]: t for t in iter_jsonl(p)})
    fmt = None
    if c.gold_fallback_fraction > 0 and tasks_by_id:
        fmt = ChatFormatter(tok or load_tokenizer(cfg.model_id), cfg.chat)

    sft, dpo, hard = [], [], []
    stats: dict = defaultdict(lambda: defaultdict(float))
    for r in rows:
        key = f"{r['type']}/{r['mode']}"
        items = r.get("samples") or r.get("episodes") or []
        passes = [s for s in items if s["pass"]]
        fails = [s for s in items if not s["pass"]]
        st = stats[key]
        st["tasks"] += 1
        st["samples"] += len(items)
        st["passes"] += len(passes)
        st["solved_any"] += bool(passes)
        st["solved_all"] += len(passes) == len(items) and bool(items)
        for s in fails:
            st["fail:" + str(s.get("fail_reason") or s.get("finish"))] += 1
        if len(passes) < len(items):
            hard.append(r["task_id"])
        # ---- SFT
        keep = passes
        if passes and len(passes) == len(items) and rng.random() > c.easy_task_keep_prob:
            keep = []
        rng.shuffle(keep)
        keep = keep[: (1 if len(passes) == len(items) else c.max_sft_per_task)]
        for s in keep:
            if r["type"] == "agent":
                for ex in _merge_turns(s["turns"], int(cfg.sft.max_seq_len)):
                    sft.append({"task_id": r["task_id"], "type": "agent", "mode": r["mode"], "segments": ex})
            else:
                sft.append({"task_id": r["task_id"], "type": r["type"], "mode": r["mode"], "segments": [
                    {"ids": r["prompt_ids"], "train": False}, {"ids": s["completion_ids"], "train": True}]})
        if not passes and r["type"] == "tool" and fmt and rng.random() < c.gold_fallback_fraction:
            task = tasks_by_id.get(r["task_id"])
            ex = _gold_example(task, fmt) if task else None
            if ex:
                sft.append({"task_id": r["task_id"], "type": "tool_gold", "mode": "nothink", "segments": ex})
                st["gold_fallback"] += 1
        # ---- DPO (single-turn only)
        if r["type"] != "agent" and passes:
            rej = [s for s in fails if s["finish"] == "stop"]
            rej.sort(key=lambda s: -s["reward"])  # near misses first
            ch = passes[:]
            rng.shuffle(ch)
            for a, b in list(zip(ch, rej))[: c.max_pairs_per_task]:
                dpo.append({"task_id": r["task_id"], "type": r["type"], "mode": r["mode"],
                            "prompt_ids": r["prompt_ids"], "chosen_ids": a["completion_ids"],
                            "rejected_ids": b["completion_ids"]})
    rng.shuffle(sft)
    rng.shuffle(dpo)
    n_sft = write_jsonl(out_dir / "sft.jsonl", sft)
    n_dpo = write_jsonl(out_dir / "dpo.jsonl", dpo)
    summary = {}
    for k, st in stats.items():
        summary[k] = dict(st)
        summary[k]["pass@1"] = round(st["passes"] / max(st["samples"], 1), 4)
        summary[k]["pass@k"] = round(st["solved_any"] / max(st["tasks"], 1), 4)
    summary["_totals"] = {"sft_examples": n_sft, "dpo_pairs": n_dpo, "hard_tasks": len(hard),
                          "sft_tokens": sum(len(s["ids"]) for e in sft for s in e["segments"])}
    write_json(out_dir / "collect_stats.json", summary)
    write_json(out_dir / "hard_tasks.json", hard)
    log.info("built %d SFT examples, %d DPO pairs, %d hard tasks", n_sft, n_dpo, len(hard))
    for k, v in summary.items():
        log.info("  %-16s %s", k, {kk: vv for kk, vv in v.items() if not str(kk).startswith("fail:")})
    return summary


# ============================================================================ launcher

def launch(cfg_args: list[str], n_gpus: int, extra: list[str]) -> None:
    procs = []
    log_dir = Path("/tmp/spark_logs")
    log_dir.mkdir(exist_ok=True)
    for i in range(n_gpus):
        cmd = [sys.executable, "-m", "spark_kaggle.collect", "--worker", "--shard", str(i), "--num-shards",
               str(n_gpus), "--device", f"cuda:{i}" if torch.cuda.is_available() else "cpu"] + cfg_args + extra
        stdout = None if i == 0 else open(log_dir / f"collect_shard{i}.log", "w")
        procs.append(subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT if stdout else None,
                                      env={**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}))
    codes = [p.wait() for p in procs]
    for i in range(1, n_gpus):
        tail = (log_dir / f"collect_shard{i}.log").read_text().splitlines()[-15:]
        log.info("---- shard %d log tail ----\n%s", i, "\n".join(tail))
    if any(codes):
        raise SystemExit(f"collection workers exited with {codes}")


def main() -> None:
    p = base_parser("RFT collection")
    p.add_argument("--worker", action="store_true")
    p.add_argument("--launch", action="store_true", help="spawn one worker per visible GPU, then build sets")
    p.add_argument("--build-only", action="store_true")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--model", default=None, help="model path (default cfg.model_id)")
    p.add_argument("--adapters", default="", help="comma-separated LoRA adapters merged in order")
    p.add_argument("--tag", default="base", help="collection name (e.g. base, sft_onpolicy)")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging(args.shard if args.worker else 0)
    hf_login()
    model_path = args.model or cfg.model_id
    adapters = [a for a in args.adapters.split(",") if a]
    if args.worker:
        run_worker(cfg, args.shard, args.num_shards, args.device, model_path, adapters, args.tag)
        return
    if args.launch:
        cfg_args = (["--config", args.config] if args.config else []) + sum([["--set", o] for o in args.overrides], [])
        extra = ["--tag", args.tag] + (["--model", args.model] if args.model else []) + (
            ["--adapters", args.adapters] if args.adapters else [])
        n = max(1, torch.cuda.device_count())
        launch(cfg_args, n, extra)
    if args.launch or args.build_only:
        build_sets(cfg, args.tag)
        return
    run_worker(cfg, 0, 1, args.device, model_path, adapters, args.tag)
    build_sets(cfg, args.tag)


if __name__ == "__main__":
    main()

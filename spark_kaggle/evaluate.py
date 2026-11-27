"""Evaluation: tool calling (BFCL-style categories), code pass@1, agentic repo tasks, refusal / over-refusal.

    python -m spark_kaggle.evaluate --tag base
    python -m spark_kaggle.evaluate --tag sft_dpo --adapters run/adapters/sft/final,run/adapters/dpo/final
    python -m spark_kaggle.evaluate --tag final --model run/final_abliterated
    python -m spark_kaggle.evaluate --compare
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import torch

from .abliterate import is_refusal, load_prompts
from .agent import run_agent_episodes
from .chat import ChatFormatter, split_reasoning
from .config import base_parser, cfg_from_args
from .engine import Generator
from .env import hf_login, read_json, read_jsonl, seed_everything, setup_env, setup_logging, write_json
from .modeling import load_for_inference, load_tokenizer
from .verify import verify_code_batch, verify_tool

log = logging.getLogger("spark")


def _gen_kwargs(cfg) -> dict:
    t = float(cfg.eval.temperature)
    return dict(temperature=t, top_p=cfg.sampling.top_p, top_k=cfg.sampling.top_k)


def eval_tool(gen, fmt, tasks, mode, max_new, gk) -> dict:
    prompts = [fmt.prompt_ids(t["messages"], t["tools"], mode) for t in tasks]
    outs = gen.generate(prompts, max_new, **gk)
    by_cat: dict[str, list] = defaultdict(list)
    fails: dict[str, int] = defaultdict(int)
    for t, o in zip(tasks, outs):
        v = verify_tool(t, fmt.decode_completion(o["ids"]), fmt, mode)
        ok = bool(v["pass"]) and o["finish"] == "stop"
        by_cat[t["category"]].append((ok, v["format_ok"], v["name_f1"]))
        if not ok:
            fails[o["finish"] if o["finish"] in ("oom", "too_long") else (v.get("fail_reason") or o["finish"])] += 1
    res = {"n": len(tasks), "accuracy": sum(x[0] for v in by_cat.values() for x in v) / max(len(tasks), 1)}
    res["by_category"] = {c: round(sum(x[0] for x in v) / len(v), 4) for c, v in sorted(by_cat.items())}
    res["format_ok"] = sum(x[1] for v in by_cat.values() for x in v) / max(len(tasks), 1)
    res["failures"] = dict(fails)
    return res


def eval_code(gen, fmt, tasks, mode, max_new, gk) -> dict:
    prompts = [fmt.prompt_ids([{"role": "user", "content": t["prompt"]}], None, mode) for t in tasks]
    outs = gen.generate(prompts, max_new, **gk)
    ver = verify_code_batch(tasks, [fmt.decode_completion(o["ids"]) for o in outs], fmt, [mode] * len(tasks))
    by_src: dict[str, list] = defaultdict(list)
    for t, v in zip(tasks, ver):
        by_src[t["source"]].append(bool(v["pass"]))
    return {"n": len(tasks), "pass@1": sum(v["pass"] for v in ver) / max(len(tasks), 1),
            "oom_or_too_long": sum(o["finish"] in ("oom", "too_long") for o in outs),
            "by_source": {s: round(sum(v) / len(v), 4) for s, v in by_src.items()}}


def eval_agent(gen, fmt, tasks, mode, cfg, gk) -> dict:
    c = cfg.collect
    eps = run_agent_episodes(gen, fmt, tasks, 1, mode, c.agent_max_turns, c.agent_max_new_tokens,
                             c.agent_max_context, gk, keep_turns="none")
    n = max(len(eps), 1)
    by_cat: dict[str, list] = defaultdict(list)
    cat = {t["id"]: t["category"] for t in tasks}
    for e in eps:
        by_cat[cat[e["task_id"]]].append(e["tests_pass"])
    finishes: dict[str, int] = defaultdict(int)
    for e in eps:
        finishes[e["finish"]] += 1
    return {"n": len(eps), "resolved": sum(e["tests_pass"] for e in eps) / n,
            "resolved_and_finished": sum(e["pass"] for e in eps) / n,
            "by_category": {k: round(sum(v) / len(v), 4) for k, v in by_cat.items()},
            "avg_turns": sum(e["n_turns"] for e in eps) / n,
            "tool_error_rate": sum(e["n_tool_errors"] for e in eps) / max(sum(e["n_calls"] for e in eps), 1),
            "format_error_turns": sum(e["n_format_errors"] for e in eps) / max(sum(e["n_turns"] for e in eps), 1),
            "finish": dict(finishes)}


def eval_refusal(gen, fmt, cfg, mode, max_new) -> dict:
    ac = cfg.abliterate
    n = int(cfg.eval.max_refusal)
    out = {}
    for name, spec, split in [("harmful", ac.harmful, "test"), ("harmless", ac.harmless, "test")]:
        prompts = load_prompts(spec, split, 0 if name == "harmful" else 200, n)
        ids = [fmt.prompt_ids([{"role": "user", "content": p}], None, mode) for p in prompts]
        outs = gen.generate(ids, max_new, temperature=0.0)
        flags = []
        for o in outs:
            r, a, term = split_reasoning(fmt.decode_completion(o["ids"]), mode, fmt.prompt_opens_think(mode))
            flags.append(is_refusal(a if term else r[-600:]))
        out[f"{name}_refusal_rate"] = sum(flags) / max(len(flags), 1)
    return out


def run_eval(cfg, model_path: str, adapters: list[str], tag: str, suites: list[str], modes: list[str],
             device: str) -> dict:
    ec = cfg.eval
    seed_everything(cfg.seed)
    tok = load_tokenizer(model_path)
    fmt = ChatFormatter(tok, cfg.chat)
    model = load_for_inference(model_path, adapters, device, cfg.hardware)
    gen = Generator(model, tok, cfg.hardware, torch.device(device))
    gk = _gen_kwargs(cfg)
    tdir = Path(cfg.tasks.out_dir)
    res: dict = {"tag": tag, "model": model_path, "adapters": adapters, "temperature": float(ec.temperature)}
    for mode in modes:
        mx = int(ec.max_new_tokens_think if mode == "think" else ec.max_new_tokens_nothink)
        r: dict = {}
        if "tool" in suites and (tdir / "tool_eval.jsonl").exists():
            r["tool"] = eval_tool(gen, fmt, read_jsonl(tdir / "tool_eval.jsonl")[: int(ec.max_tool)], mode, mx, gk)
            log.info("[%s/%s] tool accuracy %.4f %s", tag, mode, r["tool"]["accuracy"], r["tool"]["by_category"])
        if "code" in suites and (tdir / "code_eval.jsonl").exists():
            r["code"] = eval_code(gen, fmt, read_jsonl(tdir / "code_eval.jsonl")[: int(ec.max_code)], mode, mx, gk)
            log.info("[%s/%s] code pass@1 %.4f %s", tag, mode, r["code"]["pass@1"], r["code"]["by_source"])
        if "agent" in suites and (tdir / "agent_eval.jsonl").exists():
            r["agent"] = eval_agent(gen, fmt, read_jsonl(tdir / "agent_eval.jsonl")[: int(ec.max_agent)], mode, cfg,
                                    gk)
            log.info("[%s/%s] agent resolved %.4f", tag, mode, r["agent"]["resolved"])
        if "refusal" in suites:
            try:
                r["refusal"] = eval_refusal(gen, fmt, cfg, mode, 256 if mode == "nothink" else 1024)
                log.info("[%s/%s] refusal %s", tag, mode, r["refusal"])
            except Exception as e:
                log.warning("refusal eval skipped: %s", e)
        res[mode] = r
        out = Path(ec.out_dir) / f"{tag}__{mode}.json"  # one file per mode: parallel runs never collide
        write_json(out, {**{k: v for k, v in res.items() if k not in ("nothink", "think")}, "mode": mode, mode: r})
        log.info("results -> %s", out)
    return res


def compare(cfg) -> str:
    rows = []
    files = sorted(Path(cfg.eval.out_dir).glob("*__*.json"), key=lambda f: f.stat().st_mtime)
    for f in files:
        r = read_json(f)
        for mode in ("nothink", "think"):
            m = r.get(mode)
            if not m:
                continue
            rows.append([r["tag"], mode,
                         m.get("tool", {}).get("accuracy"), m.get("tool", {}).get("by_category", {}).get("irrelevance"),
                         m.get("code", {}).get("pass@1"), m.get("agent", {}).get("resolved"),
                         m.get("refusal", {}).get("harmful_refusal_rate"),
                         m.get("refusal", {}).get("harmless_refusal_rate")])
    head = ["tag", "mode", "tool acc", "irrelevance", "code pass@1", "agent resolved", "harmful refusal",
            "harmless refusal"]
    fmt = lambda v: "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))
    md = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    md += "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in rows)
    Path(cfg.eval.out_dir).mkdir(parents=True, exist_ok=True)
    (Path(cfg.eval.out_dir) / "summary.md").write_text(md + "\n")
    print(md)
    return md


def _spawn_per_mode(args, modes: list[str]) -> None:
    import subprocess
    import sys

    procs = []
    for i, mode in enumerate(modes):
        cmd = [sys.executable, "-m", "spark_kaggle.evaluate", "--tag", args.tag, "--modes", mode,
               "--device", f"cuda:{i % torch.cuda.device_count()}", "--adapters", args.adapters]
        cmd += (["--model", args.model] if args.model else []) + (["--suites", args.suites] if args.suites else [])
        cmd += (["--config", args.config] if args.config else []) + sum([["--set", o] for o in args.overrides], [])
        procs.append(subprocess.Popen(cmd))
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"eval workers exited with {codes}")


def main() -> None:
    p = base_parser("Evaluate")
    p.add_argument("--model", default=None)
    p.add_argument("--adapters", default="")
    p.add_argument("--tag", default="base")
    p.add_argument("--suites", default=None, help="comma list: tool,code,agent,refusal")
    p.add_argument("--modes", default=None, help="comma list: nothink,think")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--compare", action="store_true")
    p.add_argument("--parallel", action="store_true", help="one process per mode, each on its own GPU")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging()
    if args.compare:
        compare(cfg)
        return
    hf_login()
    device = args.device if torch.cuda.is_available() else "cpu"
    suites = args.suites.split(",") if args.suites else list(cfg.eval.suites)
    modes = args.modes.split(",") if args.modes else list(cfg.eval.modes)
    if args.parallel and torch.cuda.device_count() >= 2 and len(modes) >= 2:
        _spawn_per_mode(args, modes)
        return
    run_eval(cfg, args.model or cfg.model_id, [a for a in args.adapters.split(",") if a], args.tag, suites, modes,
             device)


if __name__ == "__main__":
    main()

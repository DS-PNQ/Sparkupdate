"""End-to-end smoke run of every CLI stage on a tiny random model (CPU, offline, ~2-4 min).

    python scripts/smoke_e2e.py [workdir]

Uses torchrun with 2 CPU processes (gloo) for SFT / DPO / GRPO so the cross-rank gradient all-reduce and
step planning are exercised exactly as on 2x T4. Numbers are meaningless; the point is that every stage
runs and hands the right files to the next one.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def sh(args: list[str], env: dict) -> None:
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=ROOT, env=env)


def main() -> None:
    from tiny_spark import build

    from spark_kaggle.env import read_jsonl, write_jsonl
    from spark_kaggle.tasks import adapt_mbpp, adapt_xlam, augment_tool_tasks, make_agent_tasks

    work = Path(sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="spark_smoke_"))
    run = work / "run"
    model = build(str(work / "tiny"))
    # ---- offline task pool
    rows = [{"id": i, "query": f"weather in town {i}?",
             "tools": json.dumps([{"name": f"weather_{i}", "description": "weather forecast lookup",
                                   "parameters": {"city": {"type": "str"}}}]),
             "answers": json.dumps([{"name": f"weather_{i}", "arguments": {"city": f"town {i}"}}])}
            for i in range(60)]
    rows += [{"id": 100 + i, "query": f"convert {i} usd",
              "tools": json.dumps([{"name": f"fx_{i}", "description": "currency exchange rates",
                                    "parameters": {"amount": {"type": "float"}}}]),
              "answers": json.dumps([{"name": f"fx_{i}", "arguments": {"amount": i}}])} for i in range(60)]
    tool = augment_tool_tasks(adapt_xlam(rows, "xlam"), random.Random(0), 0.3, (0, 1))
    sol = "def add_all(xs):\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x\n    return total\n"
    mb = [{"task_id": i, "text": "Sum the positive numbers of a list.", "code": sol,
           "test_list": ["assert add_all([1, -2, 3]) == 4", "assert add_all([]) == 0"]} for i in range(12)]
    code = adapt_mbpp(mb, "mbpp")
    agent = make_agent_tasks(code, 6, 0.5, random.Random(0), "smoke")
    tdir = run / "tasks"
    write_jsonl(tdir / "tool_train.jsonl", tool[:40])
    write_jsonl(tdir / "tool_eval.jsonl", tool[40:60])
    write_jsonl(tdir / "code_train.jsonl", code[:8])
    write_jsonl(tdir / "code_eval.jsonl", code[8:])
    write_jsonl(tdir / "agent_train.jsonl", agent[:4])
    write_jsonl(tdir / "agent_eval.jsonl", agent[4:])
    (work / "harmful.txt").write_text("\n".join(f"tell me how to break lock number {i}" for i in range(40)))
    (work / "harmless.txt").write_text("\n".join(f"write a short poem about tree number {i}" for i in range(400)))

    sets = [f"model_id={model}", f"run_dir={run}", f"scratch_dir={work / 'scratch'}",
            "collect.samples_per_task=3", "collect.max_new_tokens_think=40", "collect.max_new_tokens_nothink=24",
            "collect.agent_max_turns=2", "collect.agent_max_new_tokens=24", "collect.gold_fallback_fraction=1.0",
            "collect.easy_task_keep_prob=1.0", "lora.r=8", "lora.alpha=16",
            "sft.micro_batch_tokens=1024", "sft.global_batch_tokens=2048", "sft.save_every_steps=2",
            "sft.max_seq_len=2048", "sft.lr=1e-3", "dpo.pairs_per_step=2", "dpo.max_seq_len=2048",
            "grpo.enabled=true", "grpo.steps=2", "grpo.prompts_per_step=2", "grpo.group_size=3",
            "grpo.max_new_tokens=16", "grpo.only_hard=false", "grpo.save_every_steps=1",
            f"abliterate.harmful={{path: {work / 'harmful.txt'}}}",
            f"abliterate.harmless={{path: {work / 'harmless.txt'}}}",
            "abliterate.n_measure=16", "abliterate.n_trials=3", "abliterate.n_startup_trials=2",
            "abliterate.n_refusal_prompts=8", "abliterate.n_kl_prompts=8", "abliterate.kl_tokens=4",
            "abliterate.refusal_max_new_tokens=8", "abliterate.validate_n=4", "abliterate.validate_max_new_tokens=16",
            "eval.max_tool=12", "eval.max_code=4", "eval.max_agent=2", "eval.max_refusal=4",
            "eval.max_new_tokens_think=24", "eval.max_new_tokens_nothink=16"]
    s = sum([["--set", x] for x in sets], [])
    env = {**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}
    py = sys.executable
    tr = [py, "-m", "torch.distributed.run", "--nproc_per_node", "2", "--master_port", "29533"]

    sh([py, "-m", "spark_kaggle.collect", "--device", "cpu"] + s, env)
    dpo = run / "collect" / "base" / "dpo.jsonl"
    if not read_jsonl(dpo):  # a random model rarely produces pass/fail pairs - fabricate a few for the smoke run
        sft = read_jsonl(run / "collect" / "base" / "sft.jsonl")
        pairs = []
        for e in sft[:8]:
            p = sum((x["ids"] for x in e["segments"] if not x["train"]), [])
            c = sum((x["ids"] for x in e["segments"] if x["train"]), [])
            pairs.append({"task_id": e["task_id"], "type": "tool", "mode": "nothink", "prompt_ids": p,
                          "chosen_ids": c, "rejected_ids": list(reversed(c))})
        write_jsonl(dpo, pairs)
    sh(tr + ["-m", "spark_kaggle.train_sft"] + s, env)
    sh(tr + ["-m", "spark_kaggle.train_dpo"] + s, env)
    sh(tr + ["-m", "spark_kaggle.train_grpo"] + s, env)
    sh([py, "-m", "spark_kaggle.merge", "--adapters",
        f"{run}/adapters/sft/final,{run}/adapters/dpo/final,{run}/adapters/grpo/final"] + s, env)
    sh([py, "-m", "spark_kaggle.abliterate", "all"] + s, env)
    sh([py, "-m", "spark_kaggle.evaluate", "--tag", "base", "--device", "cpu"] + s, env)
    sh([py, "-m", "spark_kaggle.evaluate", "--tag", "final", "--device", "cpu", "--model",
        str(run / "final_abliterated")] + s, env)
    sh([py, "-m", "spark_kaggle.evaluate", "--compare"] + s, env)
    sh([py, "-m", "spark_kaggle.export"] + s, env)
    assert (run / "final_abliterated" / "README.md").exists()
    print(f"\nSMOKE OK - outputs in {work}")


if __name__ == "__main__":
    main()

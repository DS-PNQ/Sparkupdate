"""Generate notebooks/spark_x26_t4x2.ipynb (Kaggle, GPU T4 x2).

    python scripts/make_notebook.py
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

cells: list[tuple[str, str]] = []


def md(s: str) -> None:
    cells.append(("markdown", s.strip("\n")))


def code(s: str) -> None:
    cells.append(("code", s.strip("\n")))


md("""
# Spark-X2.6-4B → agentic coding / tool calling fine-tune + abliteration (Kaggle **GPU T4 x2**)

Pipeline: verifiable tasks → rejection-sampled self-distillation (RFT) → LoRA SFT → on-policy DPO → (optional GRPO)
→ merge → norm-preserving biprojected abliteration (Optuna-searched) → eval → export.

**Before running**
1. *Settings → Accelerator*: **GPU T4 x2**. *Settings → Internet*: **on** (phone-verified account).
2. *Add-ons → Secrets*: add `HF_TOKEN` (read access is enough to train, write access to push). On huggingface.co accept the
   terms of the gated datasets you keep in the config (`Salesforce/xlam-function-calling-60k`, `Salesforce/APIGen-MT-5k`).
3. A Kaggle session is capped at 12 h, so the run is split in two sessions. Session A runs the stages up to SFT.
   Then use **Save Version → Save & Run All** (or save the output), attach that notebook's output as an input in Session B,
   and set `RESUME_FROM`.

| Session | Stages | Rough time on 2x T4 |
|---|---|---|
| A | setup, preflight, tasks, baseline eval, RFT collection, SFT | 9-11 h |
| B | DPO, eval, merge, abliterate, final eval, export | 4-6 h |
""")

code("""
# ---- knobs -------------------------------------------------------------------------------------------
REPO_URL   = "https://github.com/ds-pnq/sparkupdate.git"   # or attach the repo as a Kaggle dataset and set REPO_DIR
REPO_BRANCH = "claude/peaceful-goldberg-9s6v0p"
REPO_DIR   = "/kaggle/working/sparkupdate"
RUN_DIR    = "/kaggle/working/spark_run"
RESUME_FROM = ""          # e.g. "/kaggle/input/spark-session-a/spark_run" in session B
HUB_REPO   = ""           # e.g. "your-name/Spark-X2.6-4B-agentic-abliterated" (needs a write token)

STAGES = {                # session A: preflight..sft ; session B: dpo..export
    "preflight": True, "tasks": True, "eval_base": True, "collect": True, "sft": True,
    "dpo": False, "grpo": False, "eval_trained": False, "merge": False, "abliterate": False,
    "eval_final": False, "export": False,
}
EXTRA = [                 # any config override, e.g. "collect.time_budget_hours=4", "tasks.agent_tasks=400"
]
""")

code("""
import os, subprocess, sys, shutil, time
os.environ["SPARK_RUN_DIR"] = RUN_DIR
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if not os.path.isdir(REPO_DIR):
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR], check=True)
os.chdir(REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-kaggle.txt"], check=True)

if RESUME_FROM and os.path.isdir(RESUME_FROM) and not os.path.isdir(RUN_DIR):
    shutil.copytree(RESUME_FROM, RUN_DIR)
    print("resumed run dir from", RESUME_FROM)

SETS = sum([["--set", s] for s in EXTRA], [])

def run(cmd, check=True):
    \"\"\"Stream a command's output into the notebook log (works in interactive and Save & Run All).\"\"\"
    print("\\n$ " + " ".join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        print(line, end="", flush=True)
    code = p.wait()
    print(f"[exit {code} after {(time.time() - t0) / 60:.1f} min]")
    if check and code:
        raise RuntimeError(f"failed: {' '.join(cmd)}")

PY = [sys.executable, "-m"]
TORCHRUN = [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", "2"]
import torch
print(torch.__version__, [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
""")

md("""
## 1. Preflight
Loads the real checkpoint and checks: chat-template variables (look for the **thinking-level** knob and copy it into
`chat.think_kwargs` via `EXTRA`), tool-call rendering (`<arg_key>/<arg_value>`), architecture + abliteration targets,
fp16 activation overflow on T4, chunked-loss correctness, a real tool call, and the measured KV-cache cost per token.
""")
code("""
if STAGES["preflight"]:
    run(PY + ["spark_kaggle.preflight"] + SETS)
    import json; pf = json.load(open(f"{RUN_DIR}/preflight.json"))
    print({k: pf.get(k) for k in ["template_variables", "tool_args_mode", "tool_roundtrip_has_xml", "max_abs_residual",
                                  "chunked_loss_abs_diff", "measured_kv_bytes_per_token", "config_kv_bytes_per_token"]})
    # If measured_kv_bytes_per_token is far below the config bound, add to EXTRA:
    #   f"hardware.kv_bytes_per_token={int(pf['measured_kv_bytes_per_token'] * 1.25)}"
""")

md("## 2. Verifiable task pool (tool calls with gold answers, unit-tested code, synthetic repo tasks)")
code("""
if STAGES["tasks"]:
    run(PY + ["spark_kaggle.tasks"] + SETS)
    print(open(f"{RUN_DIR}/tasks/stats.json").read())
""")

md("## 3. Baseline evaluation (thinking + no-think in parallel, one per GPU)")
code("""
if STAGES["eval_base"]:
    run(PY + ["spark_kaggle.evaluate", "--tag", "base", "--parallel"] + SETS)
    run(PY + ["spark_kaggle.evaluate", "--compare"] + SETS)
""")

md("""
## 4. Rejection-sampling self-distillation (one generation worker per T4)
Every sample is verified; passes → SFT data, pass/fail pairs → DPO data. Resumable and time-budgeted
(`collect.time_budget_hours`).
""")
code("""
if STAGES["collect"]:
    run(PY + ["spark_kaggle.collect", "--launch", "--tag", "base"] + SETS)
    print(open(f"{RUN_DIR}/collect/base/collect_stats.json").read())
""")

md("## 5. LoRA SFT on verified samples (2x T4 data parallel, fp16, chunked loss)")
code("""
if STAGES["sft"]:
    run(TORCHRUN + ["-m", "spark_kaggle.train_sft", "--time-budget-hours", "9"] + SETS)
""")

md("""
---
# Session B
Attach session A's output, set `RESUME_FROM`, flip the stage flags, and run all.

## 6. On-policy DPO (reference = SFT-merged model, cached log-probs)
""")
code("""
if STAGES["dpo"]:
    run(TORCHRUN + ["-m", "spark_kaggle.train_dpo"] + SETS)
""")

md("## 7. (Optional, slow) GRPO with verifiable rewards on the hard tasks")
code("""
if STAGES["grpo"]:
    run(TORCHRUN + ["-m", "spark_kaggle.train_grpo", "--set", "grpo.enabled=true"] + SETS)
""")

md("## 8. Evaluate the trained adapters")
code("""
ADAPTERS = [f"{RUN_DIR}/adapters/sft/final", f"{RUN_DIR}/adapters/dpo/final"]
if STAGES["grpo"]:
    ADAPTERS.append(f"{RUN_DIR}/adapters/grpo/final")
ADAPTERS = [a for a in ADAPTERS if os.path.isdir(a)]
if STAGES["eval_trained"]:
    run(PY + ["spark_kaggle.evaluate", "--tag", "trained", "--parallel", "--adapters", ",".join(ADAPTERS)] + SETS)
    run(PY + ["spark_kaggle.evaluate", "--compare"] + SETS)
""")

md("## 9. Merge adapters (fp32 on CPU → bf16) into /tmp (not persisted)")
code("""
if STAGES["merge"]:
    run(PY + ["spark_kaggle.merge", "--adapters", ",".join(ADAPTERS)] + SETS)
""")

md("""
## 10. Abliteration
Measure refusal directions (thinking + no-think prompts) → Optuna search over the per-layer ablation profile
(attention and MLP separately), minimising refusals and KL on harmless prompts → bake the best edit into the weights
(fp32 on CPU) → validate refusals before/after.
""")
code("""
if STAGES["abliterate"]:
    run(PY + ["spark_kaggle.abliterate", "all"] + SETS)
    print(open(f"{RUN_DIR}/final_abliterated/abliteration_validation.json").read()[:2000])
""")

md("## 11. Final evaluation + comparison table")
code("""
if STAGES["eval_final"]:
    run(PY + ["spark_kaggle.evaluate", "--tag", "final", "--parallel", "--model", f"{RUN_DIR}/final_abliterated"] + SETS)
    run(PY + ["spark_kaggle.evaluate", "--compare"] + SETS)
""")

md("## 12. Export (model card, HF Hub push, optional GGUF)")
code("""
if STAGES["export"]:
    sets = SETS + (["--set", f"export.hub_repo={HUB_REPO}"] if HUB_REPO else [])
    run(PY + ["spark_kaggle.export"] + sets)
""")

md("## Output size check (Kaggle keeps at most 20 GB of /kaggle/working)")
code("""
run(["du", "-sh", RUN_DIR] + [os.path.join(RUN_DIR, d) for d in os.listdir(RUN_DIR)], check=False)
""")


def build() -> dict:
    nb_cells = []
    for kind, src in cells:
        lines = src.splitlines(keepends=True)
        c = {"cell_type": kind, "metadata": {}, "source": lines}
        if kind == "code":
            c.update(execution_count=None, outputs=[])
        nb_cells.append(c)
    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out = ROOT / "notebooks" / "spark_x26_t4x2.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(build(), indent=1, ensure_ascii=False) + "\n")
    print(out)

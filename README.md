# Spark-X2.6-4B: agentic coding + tool calling fine-tune, then abliteration (Kaggle 2x T4)

A resumable pipeline that improves [XHToken/Spark-X2.6-4B](https://github.com/XHToken/Spark-X2.6) on tool calling and
agentic coding, abliterates it (removes refusals), and evaluates every step. It runs on a free Kaggle **GPU T4 x2**
notebook.

The reasoning behind each choice is in **[docs/RESEARCH.md](docs/RESEARCH.md)**. The short version: the model is
already strongly post-trained for agents, so naive SFT on public function-calling data would likely make it worse.
The pipeline trains it on **its own verified outputs**, then on pass/fail preference pairs, and abliterates the merged
result with a KL-guarded search.

```
tasks ─► collect (sample K, verify) ─► SFT (LoRA) ─► DPO (LoRA) ─► [GRPO] ─► merge ─► abliterate ─► eval ─► export
  │            │                                                                       │
  │            ├─ tool calls: AST match vs gold (spark25 XML-KV format)                ├─ measure refusal dirs (think + no-think)
  │            ├─ code: unit tests in a sandbox                                        ├─ Optuna: refusals vs KL, per-layer kernel
  │            └─ agent: multi-turn repo repair/implement with bash/edit tools         └─ norm-preserving biprojected weight edit
  └─ xLAM / Hermes FC / APIGen-MT (+irrelevance, distractor tools), MBPP / KodCode / HumanEval, synthetic repos
```

## Run it on Kaggle

1. New notebook → *File → Import notebook* → `notebooks/spark_x26_t4x2.ipynb` (or clone this repo inside one).
2. *Settings*: Accelerator **GPU T4 x2**, Internet **on**. *Add-ons → Secrets*: `HF_TOKEN`. On Hugging Face, accept the
   terms of the gated sets you use: `Salesforce/xlam-function-calling-60k`, `Salesforce/APIGen-MT-5k`.
3. **Session A**: keep the default `STAGES` (preflight → tasks → baseline eval → collect → SFT), then *Save Version*.
   This takes roughly 9-11 h.
4. **Session B**: attach session A's output as input, set `RESUME_FROM`, and enable dpo → … → export. This takes
   roughly 4-6 h. Set `HUB_REPO` to push the model.

Run the preflight cell first and read its output. It loads the real checkpoint and reports:
* the template's thinking-level variable, which you copy into `chat.think_kwargs`;
* whether tool calls render as `<arg_key>/<arg_value>`;
* the abliteration target modules, fp16 overflow risk and chunked-loss correctness;
* a real tool call;
* the measured KV-cache bytes per token, which you can feed back as `hardware.kv_bytes_per_token` for bigger batches.

## Stages (CLI)

Every stage reads `configs/default.yaml`; override anything with `--set dotted.key=value`.

| Stage | Command | Notes |
|---|---|---|
| Preflight | `python -m spark_kaggle.preflight` | Run first, on the real model |
| Tasks | `python -m spark_kaggle.tasks` | Writes `run_dir/tasks/*.jsonl` |
| Eval | `python -m spark_kaggle.evaluate --tag base --parallel` | think/no-think on separate GPUs |
| Collect | `python -m spark_kaggle.collect --launch` | One worker per GPU; resumable; `collect.time_budget_hours` |
| SFT | `torchrun --nproc_per_node 2 -m spark_kaggle.train_sft` | LoRA r32 all-linear, fp16, chunked CE |
| DPO | `torchrun --nproc_per_node 2 -m spark_kaggle.train_dpo` | ref = SFT-merged model, cached log-probs, RPO NLL |
| GRPO | `torchrun --nproc_per_node 2 -m spark_kaggle.train_grpo --set grpo.enabled=true` | Optional; ~3-4 min/step |
| Merge | `python -m spark_kaggle.merge` | fp32 on CPU → bf16 |
| Abliterate | `python -m spark_kaggle.abliterate all` | `measure` / `search` / `apply` / `validate` |
| Compare | `python -m spark_kaggle.evaluate --compare` | Writes `eval/summary.md` |
| Export | `python -m spark_kaggle.export --set export.hub_repo=you/name` | Model card, Hub push, optional GGUF |

### T4-specific choices
* **fp16 + SDPA.** T4s have no bf16 tensor cores and FlashAttention-2 needs sm_80. LoRA weights stay fp32 under
  GradScaler.
* **Chunked lm_head + cross-entropy** under activation checkpointing, so `[tokens × vocab]` logits are never built.
* **Data parallel without DDP**: one flat all-reduce of LoRA gradients per step. Gradient checkpointing, PEFT and the
  model's remote code interact with nothing.
* **HF `generate` instead of vLLM.** `Spark2_5ForCausalLM` needs a vLLM plugin, and vLLM's Turing support is limited.
  Batches are sized from free VRAM, and an lm_head hook avoids full-vocabulary prefill logits.
* Merged weights live in `/tmp`, because `/kaggle/working` keeps at most 20 GB.
* SFT and DPO run a **memory probe** first: forward and backward on the largest micro-batch. An OOM then shows up in
  the first minute with a suggested fix, instead of hours into a session.

## Layout

```
spark_kaggle/   config, env, chat (template + spark25 parser), toolmatch, sandbox, engine, agent, tasks, verify,
                collect, train_common, train_sft, train_dpo, train_grpo, merge, abliterate, evaluate, preflight, export
configs/default.yaml
notebooks/spark_x26_t4x2.ipynb     (generated by scripts/make_notebook.py)
docs/RESEARCH.md                   research report + sources
tests/                             CPU unit tests on a tiny Spark-style model
scripts/smoke_e2e.py               runs every CLI stage end to end on CPU (~2 min)
```

## Testing

```bash
pip install torch "transformers==4.57.1" "huggingface_hub<1.0" "datasets<4.3" peft optuna pyyaml pytest
```

```bash
python -m pytest -q tests
```

```bash
python scripts/smoke_e2e.py
```

The tests use a tiny random Llama with a tokenizer and chat template that copy Spark's markup and tool format. They
cover the logic: parsing, scoring, masking, distributed steps, merge equivalence, ablation geometry and stage
hand-offs. **They do not exercise the real 4B checkpoint**, which was not reachable from the development
environment. That is what the preflight stage is for.

## Responsible use

An abliterated model no longer refuses harmful requests. The model card that `export` writes says so. Put your own
safeguards in front of anything user-facing. Spark-X2.6 is Apache-2.0; check the licences of the datasets you train
on (several are CC-BY or have usage terms).

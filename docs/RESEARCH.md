# Improving Spark-X2.6-4B for agentic coding and tool calling, then abliterating it, on Kaggle 2x T4

Research notes behind the pipeline in this repo. Researched November 2026. Facts read from primary sources (the
model's GitHub READMEs, the llama.cpp converter, the SGLang tool parser, the vendor's LLaMA-Factory fork) are marked
as such. Third-party benchmark claims are marked unverified.

## TL;DR

* **Spark-X2.6-4B is already heavily post-trained for agents.** The vendor reports state-of-the-art results at its size
  and says it matches ~9B models. A naive LoRA SFT on public function-calling data (xLAM, Glaive, Hermes) is the most
  likely way to make it *worse*. That data is off-policy, has no reasoning traces, and uses different tool-call
  formats, so it would overwrite the model's own `spark25` format and its thinking behaviour.
* **What does fit a 2x T4 budget**, roughly in order of gain per GPU-hour:
  1. **Inference-side fixes, free.** Use the right chat template and `spark25` parser. Pick the thinking level per
     task. Constrain or validate tool-call syntax. Vote over several sampled tool calls. Generate code and run the
     tests.
  2. **Rejection-sampled self-distillation (RFT).** The model attempts verifiable tasks, and only verified answers
     are trained on. The data stays on-policy, which limits forgetting.
  3. **On-policy DPO** on pass/fail pairs from the same prompts.
  4. **GRPO with verifiable, fine-grained rewards**, small LoRA rank. Optional: it works, but it is slow on T4s.
* **Abliterate last**, after fine-tuning, on the merged model. Use norm-preserving biprojected weight edits. Search the
  per-layer strength with a refusal-rate + KL objective (Heretic-style). Measure directions in both thinking and
  no-think templates, because reasoning models refuse partly inside the chain of thought. Check that tool-calling
  "irrelevance" behaviour (declining when no tool fits) survives.

## 1. What Spark-X2.6-4B is

| Item | Value | Source |
|---|---|---|
| Size / license | ~4.11B params, Apache-2.0 | X2.6 README; third-party summaries |
| Architecture | **Identical to X2.5**: `Spark2_5ForCausalLM`, `model_type: spark2_5`. Gains come from post-training only | X2.6 README |
| Attention | Hybrid: 1 full-attention layer per 3 sliding-window layers. Native 1M context | X2.5 README |
| Attention detail | Fused `self_attn.q_k_v_proj`; **head-wise sigmoid output gate** `self_attn.g_proj` (one gate per head); partial RoPE with separate parameters for full vs sliding layers | llama.cpp `conversion/spark2_5.py` |
| MLP | GELU (`hidden_act: gelu`) | llama.cpp converter |
| Thinking | Levels none / low / high. `enable_thinking` in the template; `qwen3` reasoning parser (`<think>…</think>`) | X2.6 README |
| Tool calls | GLM-style XML key/values, SGLang parser `spark25`: `<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value>\n</tool_call>`. Values are typed by the tool's JSON schema | SGLang `spark25_detector.py` |
| Chat markup | `<|System|>…<｜end▁of▁sentence｜>`, `<｜start▁of▁sentence｜><|User|>…<｜end▁of▁sentence｜><｜start▁of▁sentence｜><|Bot|>`, tool results as `<|Tool|><tool_response>…</tool_response>`. An **empty thought is a bare `</think>`** | XHToken/LlamaFactory `template.py` (`spark` template) |
| Runtime pins | `transformers==4.57.1`, `trust_remote_code=True`. Recommended sampling: T=1.0, top_p=0.95, top_k off | X2.6 README |
| Fine-tuning route | Vendor recommends its LLaMA-Factory fork | X2.6 README |
| Serving | SGLang (spark25 + qwen3 parsers), llama.cpp ≥ b10828 (native `spark2_5`), Ollama, LM Studio, MLX. vLLM: `Spark2_5ForCausalLM` is **not in vLLM's registry**, so it needs an out-of-tree plugin | READMEs; Ttimms NVFP4 card |

**Benchmarks.** The X2.5-4B table (thinking mode) gives BFCL-V4 65.1, τ²-bench 75.1, MCP-Atlas 54.6, SWE-bench
Verified 41.6, SWE-bench Pro 44.4 and IFEval 93.0. For X2.6-4B, a third-party summary reports **BFCL-V4 71.8**. The
official X2.6 numbers are in SVG charts I did not parse, so treat 71.8 as unverified. The only hands-on agentic coding
reports I found cover X2.5, not X2.6. Hugging Face was unreachable from the research environment, so exact `config.json`
values (layers, hidden size, vocab, window) were **not** read; `spark_kaggle.preflight` prints them on Kaggle.

**Consequences for this project.**
* Training must keep the model's own tool format and its thinking/no-think conventions. The pipeline therefore
  trains on the model's own completions, as exact token ids (no template re-rendering), plus a few gold tool calls
  rendered by the checkpoint's own template.
* The head-wise sigmoid gate sits *before* `o_proj`, so `o_proj` and `down_proj` remain the only matrices writing into
  the residual stream, which is what abliteration needs. Gated attention is also reported to suppress massive
  activations and attention sinks (Qiu et al. 2025). That is good news for fp16 on T4s, but preflight still measures
  the peak residual magnitude.

## 2. Kaggle "GPU T4 x2" constraints

| Constraint | Consequence in the pipeline |
|---|---|
| 2x T4, ~15 GB each, sm_75 | **fp16** compute; no bf16 tensor cores. **No FlashAttention-2** (needs sm_80) → PyTorch SDPA. A 4B fp16 model (~8.2 GB) fits on one T4 with room for LoRA training at 4-6k tokens |
| vLLM: Spark2_5 needs a plugin; Turing support limited (forum reports of FA/sliding-window and shared-memory limits on sm_75) | Generation uses plain `transformers` `generate`, batched by memory budget, one worker per GPU |
| Session limit 12 h (Kaggle docs, per a Sept 2026 guide); weekly GPU quota often quoted as 30 h (unverified) | The run is split into two sessions; every stage is resumable and time-budgeted |
| `/kaggle/working` keeps ≤ 20 GB | Merged weights go to `/tmp`; the final model is pushed to the Hub |
| ~4 CPU cores | Code verification runs 4 pytest processes in parallel |
| Possible fp16 overflow | `preflight` checks peak residual magnitude and finite logits; fallback is fp32 split across both GPUs, or NF4 |

Throughput estimates, from first principles rather than measurements. Decode with a 4B fp16 model is memory-bound
(~8 GB read per step at ~300 GB/s), which gives roughly 500-1,200 generated tokens/s per T4 at batch 16-48. LoRA
training with checkpointing costs ~8x params FLOPs per token, roughly 400-500 tokens/s per T4 at realistic fp16
utilisation. That works out to about **3M training tokens/hour on 2 GPUs**.

## 3. Ways to improve agentic coding and tool calling, ranked for this budget

### 3.1 Inference-time (no training; do these regardless)
* **Template and parser correctness.** A wrong parser silently fails on Spark's XML-KV calls; use `spark25` (SGLang)
  or llama.cpp's native support. The `preflight` stage renders a round-trip tool call to confirm the template emits
  `<arg_key>/<arg_value>`.
* **Thinking level per call.** Use no-think or low for routing and simple calls, high for code. The pipeline trains
  and evaluates both modes so neither degrades.
* **Grammar-constrained decoding** of the `<tool_call>` block, e.g. GBNF in llama.cpp built from the tool schemas. This
  removes format and hallucinated-name errors, which are a large share of small-model failures. Our eval breaks
  failures down (`format`, `hallucinated_tool`, `wrong_call`, `no_call`, `irrelevance_called`) so you can see what is
  left.
* **Validate and retry.** Parse the call, check it against the JSON schema, and on failure feed the error back once.
* **Self-consistency for tool calls.** Sample N calls and take the majority after normalisation (the same normaliser
  as `toolmatch.py`). For code, sample N and run the tests.
* **Harness.** Small models do better with a minimal tool set (bash + file edit, mini-swe-agent style) and short
  contexts. The 1M-token context is nominal; the vendor's own long-context score (AA-LCR 56.3 for X2.5) shows the
  degradation.

### 3.2 Rejection-sampling self-distillation (RFT): the core of this pipeline
* Sample K answers per verifiable task; keep only verified ones; LoRA-SFT on them. Tasks come in three kinds: tool
  tasks with gold calls, code with unit tests, and repository tasks with tests.
* The data is the model's own distribution, so SFT changes less of what already works. Self-distillation is reported
  to cut catastrophic forgetting versus SFT on external data: Yang et al. 2024 evaluate tool use, code and safety
  among others; Shenfeld et al. 2026 propose on-policy SDFT.
* Selection rules: down-weight tasks solved K/K (little signal). Train on near-misses' successes. For tool tasks never
  solved, add the gold call rendered by the checkpoint's own template (no-think, `gold_fallback_fraction`).
* **Augmentations for precision.** *Irrelevance* tasks offer only unrelated tools, so the model must answer without
  calling one (BFCL "irrelevance", NVIDIA When2Call). *Distractor tools* are added to the list (BFCL "multiple";
  related to Hammer's function masking).

### 3.3 Preference optimisation on own failures
* DPO on (verified, failed) pairs from the same prompt. These are on-policy pairs, with near-miss rejections first.
* Add an NLL term on the chosen answer (RPO-style, Pang et al. 2024). It guards against the known DPO failure where
  both likelihoods fall.
* The reference is the SFT-merged model. Its log-probs are cached once, so no second model copy is held on a 15 GB GPU.

### 3.4 RL with verifiable rewards (optional on T4)
* Tool-N1 (NVIDIA) uses binary format+correctness rewards with GRPO. ToolRL finds that *fine-grained* rewards
  (format / name / parameter keys / values) train more stably. `toolmatch.py` provides both: exact = 1.0, else a
  0.1-format + 0.3-name-F1 + 0.6-argument decomposition.
* Use Dr.GRPO-style constant length normalisation and no std scaling. Skip zero-variance groups (DAPO dynamic
  sampling).
* LoRA is enough for RL. Thinking Machines' "LoRA Without Regret" finds LoRA matches full fine-tuning for RL even at
  very small rank, and that the best LoRA LR is about 10x the full-fine-tuning LR.
* Cost is the issue. Each step generates P×G completions with HF `generate` (no vLLM), at ~3-4 min/step for
  16 prompts × 6 samples, so the pipeline defaults to 60 steps on the hard tasks only.

### 3.5 Agentic coding data
* Public SWE trajectory sets come from much larger teachers and use 30-100k-token contexts: SWE-smith (5k
  trajectories → SWE-agent-LM), nvidia/Open-SWE-Traces (OpenHands, SWE-agent, mini-swe-agent, opencode, claude-code
  harnesses), SWE-ZERO/HERO (300k execution-free + 13k execution-based trajectories from Qwen3-Coder-480B), and the
  Nemotron SWE pool via MetaCurate. Training on them at full length does not fit a T4 (activation memory beyond ~8k
  tokens with LoRA + fp16).
* Kimi-Dev argues "agentless" skills (localise → repair → test) are a prior that transfers to agents.
* What fits: **short, executable, synthetic repositories.** Bugs are injected into verified solutions by AST
  mutation, or functions are stubbed. The model works with `bash` / `read_file` / `write_file` / `str_replace` tools,
  and the reward is the hidden test run (tests are restored before grading, so editing them does not help). The
  trajectories are on-policy and multi-turn, and they teach the inspect → edit → run tests → summarise loop.
* If you have more budget, mix in a few thousand of the shortest Open-SWE-Traces / SWE-smith trajectories (≤ 8k
  tokens) through the `openai_jsonl` adapter. Watch that their tool-call format is converted to Spark's.

### 3.6 LoRA settings
* All linear layers (`all-linear`: fused QKV, gate, o, MLP), r=32, alpha=32, LR 1e-4 for SFT, 5e-6 for DPO, 2e-5 for
  GRPO. A community X2.5 LoRA used r=16 / alpha=32, which is a reasonable lower setting.
* Merge in **fp32**. Adding a small LoRA delta directly to bf16 weights rounds part of it away.

## 4. Abliteration

### 4.1 Lineage
1. **Arditi et al. 2024.** Refusal is mediated by one residual-stream direction: the mean activation on harmful
   prompts minus the mean on harmless ones. Orthogonalising every residual-writing matrix against it removes refusal.
   FailSpy and mlabonne popularised this as "abliteration".
2. **grimjim 2025 (projected → norm-preserving biprojected).** The refusal direction is orthogonalised against the
   harmless mean, both at the measurement layer and again at each destination layer (that is the "bi" in
   biprojected). Each weight column (the vector a neuron writes into the residual) keeps its norm; only its direction
   loses the refusal component. The author reports better reasoning scores and stronger refusal removal than plain
   abliteration (self-reported). Reference implementation: `jim-plus/llm-abliteration`. Our `ablate_matrix` follows
   its column-norm-preserving update.
3. **Heretic (p-e-w).** It searches ablation parameters with Optuna TPE to co-minimise the refusal count on harmful
   prompts and the KL divergence on harmless prompts. The parametrisation interpolates the direction across layers
   (float index) and uses separate attention/MLP weight kernels (peak position, max/min weight, span). The default
   KL is first-token KL on 100 harmless prompts. Heretic 2.x defaults to an "ARA" LoRA-based modifier.
4. **Comparative study (arXiv 2512.13655, 16 models, 4 tools).** Math is the most sensitive capability: GSM8K changes
   ranged from +1.5 to −18.8 points depending on tool and architecture. Heretic ran on 16/16 models; the single-pass
   tools preserved GSM8K best on the subset tested. **Capability loss is real and model-dependent, so the pipeline
   measures it** (KL during the search; tool, code and agent evals after).

### 4.2 Reasoning models refuse partly inside the chain of thought
* "Where Do Reasoning Models Refuse?" (ICML 2026) finds the refusal decision is often set by the opening of the CoT.
  Ablating linear refusal directions works less reliably than on non-reasoning models and costs some capability.
* CoT-Hijacking extracts Qwen3-14B's direction at the 4th token from the end of the prompt. Another paper notes that
  refusals follow the `<think>` + blank-line pattern.
* "Beyond a Single Direction" finds refusal in reasoning models is encoded jointly in activations and CoT. On
  GPT-OSS-20B, one paper finds no single direction ablates refusal on held-out data.

**Design response.**
* Directions are measured on prompts rendered in **both** thinking and no-think templates (`measure_modes`), at
  configurable positions.
* The search optimises no-think refusals (cheap). Final validation regenerates with **thinking on** (1,024 tokens). If
  the model has not closed `</think>`, the refusal check reads the reasoning.
* If thinking-mode refusals stay high, raise the search budget (`abliterate.n_trials`), add more positions
  (`measure_positions: [-1, -2, -3, -4]`), or set `abliterate.search_mode=think`.

### 4.3 Ordering and interaction with fine-tuning
* **Abliterate after fine-tuning.** The direction is computed on, and removed from, the exact weights you ship. The
  alternative, abliterating first and then fine-tuning, can let training partly rebuild the refusal direction. The
  training data here has no refusal examples, but it isn't measured.
* Healing: if KL or the evals show damage, the standard fix is a short DPO/SFT pass on the abliterated model. Here
  that is the same RFT data, which has no refusal content. You can do this with `train_sft` using
  `model_id=<abliterated dir>`.
* Watch **irrelevance accuracy**. Declining to call a tool when none fits is not a safety refusal, but heavy ablation
  can disturb it. The eval table shows it before and after.

### 4.4 Our objective
* `refusal_rate`: marker match on the first ~320 characters of the answer, on held-out harmful prompts not used for
  measurement.
* `KL`: **teacher-forced KL over the original model's first 24 greedy answer tokens** on harmless prompts. Heretic
  uses first-token KL, but with a thinking model the first token is often fixed, so it says little.
* Multi-objective TPE over Heretic's parameter space. The final pick is the lowest KL with refusal ≤ 5%, otherwise
  the lowest refusal. The first trial is classic full-strength abliteration, as an anchor.

## 5. What is and isn't verified

* **Verified here (CPU).** Every stage runs end to end on a tiny random Llama with a Spark-style tokenizer and
  template (`scripts/smoke_e2e.py`): collection with real verification, 2-process `torchrun` SFT/DPO/GRPO with the
  gradient all-reduce (gloo), merge, Optuna abliteration, eval and export. Unit tests cover parsing, scoring, sandbox,
  task adapters, chunked loss vs full logits, LoRA learning, merge equivalence and the ablation geometry: outputs ⟂ r
  and column norms preserved (`pytest tests`, 20 tests).
* **Not verified.** The real 4B checkpoint was never loaded: Hugging Face was blocked from this environment and there
  is no GPU. Unknowns, each checked by `preflight` on Kaggle: exact module names (expected `o_proj`/`down_proj`), the
  thinking-level template variable, whether the template wants tool arguments as dict or JSON string (auto-detected),
  fp16 safety, and the sliding-window KV footprint.
* The dataset adapters (xLAM, Hermes FC, APIGen-MT, MBPP, HumanEval, KodCode) were written from their public schemas
  and log loudly when a source yields 0 tasks.

## Sources

Model and tooling (read directly):
* Spark-X2.6 README: https://raw.githubusercontent.com/XHToken/Spark-X2.6/X2.6/README.md
* Spark-X2.5 README (architecture, benchmark table): https://raw.githubusercontent.com/XHToken/Spark-X2.5/main/README.md
* llama.cpp Spark2_5 converter: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/conversion/spark2_5.py
* SGLang spark25 tool parser: https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/function_call/spark25_detector.py
* XHToken LLaMA-Factory `spark` template: https://raw.githubusercontent.com/XHToken/LlamaFactory/main/src/llamafactory/data/template.py
* jim-plus/llm-abliteration (norm-preserving / projected code): https://github.com/jim-plus/llm-abliteration
* Heretic (README, config, modifiers, scorers): https://github.com/p-e-w/heretic

Third-party coverage (unverified claims):
* https://mer.vin/news/spark-x2-6-4b-4b-on-device-agent-model-benchmarks-and-local-setup/
* https://www.mindstudio.ai/blog/spark-x26-4b-on-device-agent-model
* https://alphasignal.ai/news/sparkllm-s-spark-x2-6-4b-runs-local-agents-on-3-54-gb-with-adjustable-reasoning
* vLLM registry note (X2.5 NVFP4 card): https://huggingface.co/Ttimms/Spark-X2.5-4B-NVFP4
* Community X2.5 LoRA (r16/alpha32): https://huggingface.co/antibyte/AuraGo-Spark
* X2.5 agentic-coding GGUF test: https://huggingface.co/Myric/Spark-X2.5-4B-GGUF

Abliteration:
* grimjim, Norm-Preserving Biprojected Abliteration: https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration
* grimjim, Projected Abliteration: https://huggingface.co/blog/grimjim/projected-abliteration
* mlabonne, Uncensor any LLM with abliteration: https://huggingface.co/blog/mlabonne/abliteration
* Comparative Analysis of LLM Abliteration Methods (arXiv 2512.13655): https://arxiv.org/abs/2512.13655
* Where Do Reasoning Models Refuse? (ICML 2026): https://icml.cc/virtual/2026/79376
* Chain-of-Thought Hijacking: https://arxiv.org/pdf/2510.26418
* Beyond a Single Direction: CoT Disrupts Simple Steering of Refusal: https://arxiv.org/abs/2605.26772
* Think-token logit suppression: https://arxiv.org/pdf/2505.23848.pdf

Tool calling / agents / training:
* ToolRL: https://arxiv.org/abs/2504.13958 · Nemotron-Research-Tool-N1: https://arxiv.org/abs/2505.00024
* ToolACE: https://arxiv.org/abs/2406.18518 · When2Call: https://huggingface.co/datasets/nvidia/When2Call ·
  Toucan: https://research.ibm.com/blog/toucan-for-tool-calling
* SWE-smith: https://arxiv.org/abs/2504.21798 · Kimi-Dev: https://arxiv.org/abs/2509.23045 ·
  SWE-ZERO/HERO: https://arxiv.org/abs/2604.01496 · Open-SWE-Traces: https://huggingface.co/datasets/nvidia/Open-SWE-Traces ·
  MetaCurate: https://arxiv.org/pdf/2610.23134
* LoRA Without Regret: https://thinkingmachines.ai/blog/lora/
* Self-distillation: https://arxiv.org/abs/2402.13669 (Yang 2024), https://arxiv.org/abs/2601.19897 (on-policy SDFT),
  https://arxiv.org/pdf/2601.20802 (SDPO)

Kaggle / hardware:
* Kaggle session limits (Sept 2026 guide): https://llcuda.github.io/guides/kaggle-setup
* Free GPU overview 2026: https://gpuperhour.com/blog/free-cloud-gpus-and-credits
* vLLM on T4 / Turing threads: https://discuss.vllm.ai/t/gemma3-on-a-t4-gpu/643,
  https://discuss.vllm.ai/t/title-outofresources-shared-memory-98304-65536-on-turing-sm75-global-attention-layers-with-head-dim-512/2990

Background references (from prior knowledge, not re-fetched in this session):
* Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, arXiv 2406.11717
* Qiu et al., Gated Attention for Large Language Models, arXiv 2505.06708
* Pang et al., Iterative Reasoning Preference Optimization (DPO + NLL), arXiv 2404.19733
* Yu et al., DAPO, arXiv 2503.14476 · Liu et al., Understanding R1-Zero-Like Training (Dr. GRPO), arXiv 2503.20783

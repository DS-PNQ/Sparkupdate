"""Export: model card, push to the HF Hub, optional GGUF (llama.cpp has native spark2_5 support, b10828+).

    python -m spark_kaggle.export --config configs/default.yaml --set export.hub_repo=you/Spark-X2.6-4B-agentic-abliterated
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import base_parser, cfg_from_args
from .env import hf_login, read_json, setup_env, setup_logging

log = logging.getLogger("spark")


def model_card(cfg, model_dir: Path) -> str:
    meta = read_json(model_dir / "abliteration_meta.json", {}) or {}
    val = read_json(model_dir / "abliteration_validation.json", {}) or {}
    summary = Path(cfg.eval.out_dir) / "summary.md"
    table = summary.read_text() if summary.exists() else "_run `python -m spark_kaggle.evaluate --compare`_"
    best = meta.get("best", {})
    return f"""---
license: apache-2.0
base_model: {cfg.model_id}
tags: [spark2_5, tool-calling, agentic-coding, abliterated, lora, rft, dpo]
pipeline_tag: text-generation
---

# {cfg.export.hub_repo or 'Spark-X2.6-4B agentic abliterated'}

Fine-tune of [{cfg.model_id}](https://huggingface.co/{cfg.model_id}) for agentic coding and tool calling,
then abliterated. Trained on Kaggle 2x T4.

## Recipe
1. **Verifiable task pool**: function-calling tasks with gold calls (xLAM / Hermes FC / APIGen-MT, plus
   irrelevance and distractor-tool augmentation), unit-tested coding problems (MBPP / KodCode), and synthetic
   repository repair/implement tasks solved with `bash` / `read_file` / `write_file` / `str_replace` tools.
2. **Rejection-sampling self-distillation (RFT)**: the model samples answers (thinking and no-think modes);
   only answers that pass AST-match / unit tests are kept. LoRA SFT on those (r={cfg.lora.r}, all linear layers).
3. **On-policy DPO** on pass-vs-fail pairs from the same prompts (beta={cfg.dpo.beta}, RPO NLL weight
   {cfg.dpo.sft_weight}).
4. **Abliteration**: {meta.get('method', 'directional ablation')}; projected={meta.get('projected')},
   norm_preserve={meta.get('norm_preserve')}. Layer profile searched with Optuna to minimise
   refusals and KL divergence on harmless prompts. Best trial: refusal {best.get('refusal')},
   KL {best.get('kl')}.

## Results (held-out tasks)
{table}

Refusal validation: {val.get('before_nothink_refusal')} -> {val.get('after_nothink_refusal')} (no-think),
{val.get('before_think_refusal')} -> {val.get('after_think_refusal')} (thinking).

## Usage
Same as the base model: `trust_remote_code=True`, `transformers==4.57.1`, tool-call parser `spark25`,
reasoning parser `qwen3`; recommended sampling temperature 1.0, top_p 0.95.

## Warning
Abliteration removes the model's refusal behaviour. It will follow harmful instructions. You are
responsible for how it is deployed; add your own safeguards for anything user-facing.
"""


def to_gguf(cfg, model_dir: Path, out_dir: Path) -> list[Path]:
    ref = cfg.export.llama_cpp_ref
    lc = Path("/tmp/llama.cpp")
    if not lc.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, "https://github.com/ggml-org/llama.cpp",
                        str(lc)], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(lc / "gguf-py")], check=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    types = [t.lower() for t in cfg.export.gguf_types]
    direct = [t for t in types if t in ("f16", "bf16", "q8_0")]
    for t in direct:
        f = out_dir / f"{model_dir.name}-{t.upper()}.gguf"
        subprocess.run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(f),
                        "--outtype", t], check=True)
        made.append(f)
    k_quants = [t for t in types if t not in direct]
    if k_quants:
        f16 = out_dir / f"{model_dir.name}-F16.gguf"
        if not f16.exists():
            subprocess.run([sys.executable, str(lc / "convert_hf_to_gguf.py"), str(model_dir), "--outfile", str(f16),
                            "--outtype", "f16"], check=True)
        subprocess.run(["cmake", "-B", str(lc / "build"), "-S", str(lc), "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"],
                       check=True)
        subprocess.run(["cmake", "--build", str(lc / "build"), "--target", "llama-quantize", "-j",
                        str(os.cpu_count() or 4)], check=True)
        for t in k_quants:
            f = out_dir / f"{model_dir.name}-{t.upper()}.gguf"
            subprocess.run([str(lc / "build" / "bin" / "llama-quantize"), str(f16), str(f), t.upper()], check=True)
            made.append(f)
    return made


def main() -> None:
    p = base_parser("Export")
    p.add_argument("--model-dir", default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging()
    token = hf_login()
    model_dir = Path(args.model_dir or cfg.abliterate.out_dir)
    (model_dir / "README.md").write_text(model_card(cfg, model_dir))
    ggufs = to_gguf(cfg, model_dir, Path(cfg.scratch_dir) / "gguf") if cfg.export.gguf else []
    if not cfg.export.hub_repo:
        log.info("export.hub_repo not set - model card written to %s; nothing pushed", model_dir)
        return
    if not token:
        raise SystemExit("HF_TOKEN needed to push (add it as a Kaggle secret)")
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(cfg.export.hub_repo, private=bool(cfg.export.private), exist_ok=True)
    api.upload_folder(folder_path=str(model_dir), repo_id=cfg.export.hub_repo,
                      commit_message="Spark-X2.6-4B agentic fine-tune + abliteration")
    log.info("pushed %s -> https://huggingface.co/%s", model_dir, cfg.export.hub_repo)
    if ggufs:
        gguf_repo = cfg.export.hub_repo + "-GGUF"
        api.create_repo(gguf_repo, private=bool(cfg.export.private), exist_ok=True)
        for f in ggufs:
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name, repo_id=gguf_repo)
            log.info("pushed %s -> %s", f.name, gguf_repo)


if __name__ == "__main__":
    main()

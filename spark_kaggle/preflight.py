"""Preflight on the real checkpoint before spending GPU hours.

Checks: GPUs, chat-template variables (thinking-level knob), tool-call rendering mode, think-prefill,
architecture summary + abliteration targets, fp16 activation overflow, chunked-loss == full-logits loss,
and one real tool-calling generation parsed with the spark25 format.

    python -m spark_kaggle.preflight --config configs/default.yaml
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from .chat import ChatFormatter, detect_tool_args_mode, template_variables
from .config import base_parser, cfg_from_args
from .engine import Generator
from .env import disk_report, gpu_report, hf_login, setup_env, setup_logging, write_json
from .modeling import (decoder_layers, get_decoder_and_head, hidden_size, kv_bytes_per_token, load_model,
                       load_tokenizer, residual_writers, text_config)
from .train_common import seq_logps

log = logging.getLogger("spark")

DEMO_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
                                      "parameters": {"type": "object", "properties": {
                                          "city": {"type": "string", "description": "City name"},
                                          "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
                                          "required": ["city"]}}},
    {"type": "function", "function": {"name": "search_flights", "description": "Search flights between two airports.",
                                      "parameters": {"type": "object", "properties": {
                                          "origin": {"type": "string"}, "destination": {"type": "string"},
                                          "date": {"type": "string", "description": "YYYY-MM-DD"},
                                          "passengers": {"type": "integer"}},
                                          "required": ["origin", "destination", "date"]}}},
]


def main() -> None:
    p = base_parser("Preflight checks")
    p.add_argument("--model", default=None)
    p.add_argument("--skip-model", action="store_true", help="tokenizer/template checks only")
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging()
    hf_login()
    model_id = args.model or cfg.model_id
    report: dict = {"model_id": model_id, "gpus": gpu_report()}
    disk_report([cfg.run_dir, cfg.scratch_dir])

    tok = load_tokenizer(model_id)
    tvars = template_variables(tok)
    report["template_variables"] = tvars
    log.info("chat template variables: %s", tvars)
    knobs = [v for v in tvars if any(k in v for k in ("think", "reason", "effort", "budget", "level"))]
    log.info("thinking-related knobs: %s  (set chat.think_kwargs / nothink_kwargs accordingly)", knobs)
    report["tool_args_mode"] = detect_tool_args_mode(tok)
    fmt = ChatFormatter(tok, cfg.chat)
    msgs = [{"role": "user", "content": "What's the weather in Hanoi in celsius?"}]
    for mode in ("think", "nothink"):
        txt = fmt.prompt(msgs, DEMO_TOOLS, mode)
        report[f"prompt_tail_{mode}"] = txt[-160:]
        log.info("---- %s prompt (tail) ----\n%s", mode, txt[-400:])
    conv = msgs + [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_0", "type": "function", "function": {"name": "get_weather",
                                                          "arguments": {"city": "Hanoi", "unit": "celsius"}}}]},
                   {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "{\"temp\": 31}"}]
    rendered = fmt.render(conv, DEMO_TOOLS, "nothink", add_generation_prompt=True)
    report["tool_roundtrip_has_xml"] = "<arg_key>city</arg_key>" in rendered
    log.info("---- rendered tool round trip (tail) ----\n%s", rendered[-600:])
    log.info("tool args mode: %s | XML-KV rendered: %s | think prefilled in prompt: %s", report["tool_args_mode"],
             report["tool_roundtrip_has_xml"], fmt.prompt_opens_think("think"))
    if not report["tool_roundtrip_has_xml"]:
        log.warning("template did not render <arg_key>/<arg_value> - check chat.tool_args_mode and the template")

    if args.skip_model:
        write_json(Path(cfg.run_dir) / "preflight.json", report)
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = cfg.hardware.dtype if torch.cuda.is_available() else "float32"
    model = load_model(model_id, device=device, dtype=dtype, attn=cfg.hardware.attn_implementation)
    tc = text_config(model)
    layers = decoder_layers(model)
    H = hidden_size(model)
    lt = getattr(tc, "layer_types", None)
    arch = {k: getattr(tc, k, None) for k in ("model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
                                              "num_attention_heads", "num_key_value_heads", "head_dim", "vocab_size",
                                              "sliding_window", "tie_word_embeddings", "max_position_embeddings",
                                              "hidden_act", "final_logit_softcapping")}
    arch["layer_types"] = dict(Counter(lt)) if lt else None
    arch["kv_bytes_per_token_upper_bound"] = kv_bytes_per_token(model)
    arch["params_b"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 3)
    w0 = residual_writers(layers[0], H)
    arch["abliteration_targets_layer0"] = {k: [n for n, _ in v] for k, v in w0.items()}
    report["arch"] = arch
    log.info("architecture: %s", arch)
    if not w0["attn"] or not w0["mlp"]:
        log.warning("could not find attention/MLP output projections by name - abliteration needs a fix: %s",
                    [n for n, _ in layers[0].named_modules()][:40])

    # fp16 overflow probe
    maxes: dict[int, float] = {}

    def hook(i):
        def fn(mod, a, out):
            h = out[0] if isinstance(out, tuple) else out
            maxes[i] = max(maxes.get(i, 0.0), float(h.detach().abs().float().max()))
        return fn

    handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
    texts = [fmt.prompt(msgs, DEMO_TOOLS, "think"),
             fmt.prompt([{"role": "user", "content": "Write a Python function that merges two sorted lists. " * 8}],
                        None, "nothink")]
    with torch.no_grad():
        for t in texts:
            ids = torch.tensor([fmt.encode(t)], device=device)
            logits = model(input_ids=ids).logits
            report.setdefault("logits_finite", True)
            report["logits_finite"] &= bool(torch.isfinite(logits).all())
    for h in handles:
        h.remove()
    peak = max(maxes.values())
    report["max_abs_residual"] = peak
    log.info("max |residual| per layer: %s", " ".join(f"{i}:{v:.0f}" for i, v in sorted(maxes.items())))
    if not report["logits_finite"] or peak > 3.0e4 or math.isnan(peak):
        log.error("fp16 overflow risk (peak %.0f, finite=%s): set hardware.dtype=float32 and use "
                  "device_map across both GPUs for inference, quant=nf4 for training", peak, report["logits_finite"])
    else:
        log.info("fp16 OK: peak residual %.0f (fp16 max 65504)", peak)

    # chunked loss == full logits loss
    ids = torch.tensor([fmt.encode(texts[0])], device=device)
    labels = ids.clone()
    labels[:, : ids.shape[1] // 2] = -100
    with torch.no_grad():
        full = model(input_ids=ids).logits.float()
        ref = -F.cross_entropy(full[0, :-1][labels[0, 1:] != -100], labels[0, 1:][labels[0, 1:] != -100],
                               reduction="sum")
        mine, _ = seq_logps(model, ids, torch.ones_like(ids), labels, chunk=7)
    diff = abs(float(ref) - float(mine[0]))
    report["chunked_loss_abs_diff"] = diff
    log.info("chunked loss check: full %.4f vs chunked %.4f (diff %.2e)", float(ref), float(mine[0]), diff)
    if diff > 1e-2 * max(1.0, abs(float(ref))):
        log.error("chunked loss mismatch - the decoder/lm_head split is wrong for this model")

    # one real generation
    gen = Generator(model, tok, cfg.hardware)
    for mode in ("nothink", "think"):
        out = gen.generate([fmt.prompt_ids(msgs, DEMO_TOOLS, mode)], 1024 if mode == "think" else 256,
                           temperature=0.0)[0]
        text = fmt.decode_completion(out["ids"])
        parsed = fmt.parse(text, DEMO_TOOLS, mode)
        report[f"demo_{mode}"] = {"text": text[-800:], "calls": parsed["calls"], "errors": parsed["errors"],
                                  "finish": out["finish"]}
        log.info("demo %s -> calls %s errors %s finish %s\n%s", mode, parsed["calls"], parsed["errors"],
                 out["finish"], text[-500:])
    # measured KV-cache cost per token (sliding-window layers make the config bound pessimistic)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        long_prompt = fmt.prompt_ids([{"role": "user", "content": "Summarise this log line by line:\n" +
                                       "\n".join(f"line {i}: ok status=200 latency={i % 97}ms" for i in range(600))}],
                                     None, "nothink")
        B, new = 4, 256
        gen.generate([long_prompt] * B, new, temperature=1.0)
        peak = torch.cuda.max_memory_allocated(device) - base
        per_tok = int(peak / (B * (len(long_prompt) + new)))
        report["measured_kv_bytes_per_token"] = per_tok
        report["config_kv_bytes_per_token"] = kv_bytes_per_token(model)
        log.info("KV+activation bytes/token measured %d vs config upper bound %d (prompt %d tok). If much lower, "
                 "set --set hardware.kv_bytes_per_token=%d for bigger generation batches", per_tok,
                 kv_bytes_per_token(model), len(long_prompt), int(per_tok * 1.25))
    write_json(Path(cfg.run_dir) / "preflight.json", report)
    log.info("preflight report -> %s", Path(cfg.run_dir) / "preflight.json")


if __name__ == "__main__":
    main()

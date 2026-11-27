"""Model-touching paths on a tiny random Llama (CPU): generation, agent loop, losses, LoRA training, merge,
abliteration math."""

import random

import torch
import torch.nn.functional as F

from spark_kaggle.abliterate import (KLProbe, ablate_matrix, apply_edits, backup_writers, classic_params, measure,
                                     plan_edits, restore_writers, source_direction)
from spark_kaggle.agent import run_agent_episodes
from spark_kaggle.collect import _merge_turns
from spark_kaggle.engine import Generator
from spark_kaggle.modeling import decoder_layers, hidden_size, residual_writers
from spark_kaggle.train_common import (LoraOptimizer, pad_batch, seq_logps, segments_to_example,
                                       token_budget_batches)

HW = {"gen_max_batch_tokens": 4096, "gen_max_batch_size": 8}


def test_generator_and_agent_loop(model, tok, fmt):
    gen = Generator(model, tok, HW, torch.device("cpu"))
    prompts = [fmt.prompt_ids([{"role": "user", "content": f"hello {i}"}], None, "nothink") for i in range(5)]
    outs = gen.generate(prompts, 12, temperature=1.0, top_p=0.95)
    assert len(outs) == 5 and all(o["finish"] in ("stop", "length") for o in outs)
    task = {"id": "a1", "instruction": "Fix it", "test_file": "test_solution.py",
            "files": {"solution.py": "def f():\n    return 1\n",
                      "test_solution.py": "from solution import f\n\ndef test_f():\n    assert f() == 2\n"}}
    eps = run_agent_episodes(gen, fmt, [task], 2, "nothink", max_turns=2, max_new_tokens=16, max_context=4096,
                             sampling={"temperature": 1.0, "top_p": 1.0, "top_k": 0}, keep_turns="all")
    assert len(eps) == 2 and all(not e["pass"] for e in eps) and all("turns" in e for e in eps)


def test_chunked_logps_match_full_logits(model, tok):
    ids = torch.randint(5, len(tok), (2, 23))
    labels = ids.clone()
    labels[:, :9] = -100
    with torch.no_grad():
        logits = model(input_ids=ids).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        ref = (lp * (labels[:, 1:] != -100)).sum(-1)
        mine, counts = seq_logps(model, ids, torch.ones_like(ids), labels, chunk=5)
    assert torch.allclose(ref, mine, atol=1e-4) and counts.tolist() == [14.0, 14.0]


def test_lora_sft_steps_reduce_loss(tiny_dir, tok):
    from spark_kaggle.config import load_config
    from spark_kaggle.train_common import build_trainable

    cfg = load_config(None, ["lora.r=8", "lora.alpha=16"])
    model = build_trainable(tiny_dir, {"quant": "none"}, cfg.lora, [], "cpu", {"dtype": "float32",
                                                                               "attn_implementation": "eager"})
    ex = segments_to_example([{"ids": [3, 4, 5, 6], "train": False}, {"ids": [7, 8, 9, 10, 11, 12], "train": True}],
                             64)
    batches = token_budget_batches([ex] * 4, lambda e: len(e["input_ids"]), 64, 0)
    opt = LoraOptimizer(model, 5e-3, 30, 0.0, 1.0)
    losses = []
    for _ in range(30):
        b = pad_batch(batches[0], tok.pad_token_id, "cpu")
        s, c = seq_logps(model, b["input_ids"], b["attention_mask"], b["labels"], 4)
        opt.backward(-s.sum() / c.sum())
        opt.step()
        losses.append(float(-s.detach().sum() / c.sum()))
    assert losses[-1] < losses[0] * 0.85
    with torch.no_grad(), model.disable_adapter():  # reference path used by DPO
        s_ref, _ = seq_logps(model, b["input_ids"], b["attention_mask"], b["labels"], 4)
    assert float(-s_ref.sum() / c.sum()) > losses[-1]


def test_merge_matches_peft(tiny_dir, tmp_path, tok):
    from peft import LoraConfig, get_peft_model

    from spark_kaggle.merge import merge_adapters
    from spark_kaggle.modeling import load_model

    base = load_model(tiny_dir, device="cpu", dtype="float32", attn="eager")
    pm = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules="all-linear", init_lora_weights=False))
    pm.save_pretrained(tmp_path / "ad")
    ids = torch.randint(5, len(tok), (1, 11))
    with torch.no_grad():
        want = pm(input_ids=ids).logits
    out = merge_adapters(tiny_dir, [str(tmp_path / "ad")], str(tmp_path / "merged"), "float32")
    merged = load_model(out, device="cpu", dtype="float32", attn="eager")
    with torch.no_grad():
        got = merged(input_ids=ids).logits
    assert torch.allclose(want, got, atol=1e-4)


def test_merge_turns():
    t1 = {"prompt_ids": [1, 2], "completion_ids": [3, 9]}
    t2 = {"prompt_ids": [1, 2, 3, 9, 4], "completion_ids": [5, 9]}
    t3 = {"prompt_ids": [1, 2, 7], "completion_ids": [8, 9]}  # history re-rendered differently
    ex = _merge_turns([t1, t2, t3])
    assert len(ex) == 2
    assert [s["train"] for s in ex[0]] == [False, True, False, True] and ex[0][2]["ids"] == [4]
    capped = _merge_turns([t1, t2, t3], max_len=5)  # t2 alone is 7 tokens -> not merged into t1's example
    assert len(capped) == 3


def test_ablate_matrix_properties():
    torch.manual_seed(0)
    W = torch.randn(16, 40)
    r = F.normalize(torch.randn(16), dim=0)
    plain = ablate_matrix(W, r, 1.0, norm_preserve=False)
    assert torch.allclose(r @ plain, torch.zeros(40), atol=1e-5)
    npres = ablate_matrix(W, r, 1.0, norm_preserve=True)
    assert torch.allclose(r @ npres, torch.zeros(40), atol=1e-5)
    assert torch.allclose(npres.norm(dim=0), W.norm(dim=0), atol=1e-4)
    half = ablate_matrix(W, r, 0.5, norm_preserve=False)
    assert torch.allclose(r @ half, 0.5 * (r @ W), atol=1e-5)


def test_abliteration_pipeline_on_tiny(model, tok, fmt):
    gen = Generator(model, tok, HW, torch.device("cpu"))
    harmful = [f"how do I break thing {i}" for i in range(6)]
    harmless = [f"write a poem about item {i}" for i in range(6)]
    dirs = measure(model, fmt, harmful, harmless, ["think", "nothink"], [-1], batch_size=4)
    L, H = len(decoder_layers(model)), hidden_size(model)
    assert dirs["refusal"].shape == (L, H)
    probe = KLProbe(model, gen, fmt, harmless[:3], n_tokens=4)
    assert probe.kl() < 1e-3  # base log-probs are stored in fp16
    params = classic_params(L)
    params.update(attn_span=10.0, mlp_span=10.0)
    edits = plan_edits(model, dirs, params, projected=False)
    assert len(edits) == 2 * L  # o_proj + down_proj per layer
    bk = backup_writers(model)
    apply_edits(edits, norm_preserve=False)
    r = source_direction(dirs, params["direction_index"], projected=False).float()
    w = residual_writers(decoder_layers(model)[1], H)
    for comp in ("attn", "mlp"):
        W = w[comp][0][1].weight
        assert torch.allclose(r @ W, torch.zeros(W.shape[1]), atol=1e-4)
    assert probe.kl() > 0
    restore_writers(model, bk)
    assert probe.kl() < 1e-3

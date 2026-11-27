"""Tool-call verification (BFCL-AST-style matching) and ToolRL-style fine-grained rewards."""

from __future__ import annotations

import math
from typing import Any

from .chat import schema_properties, tool_index


def _num(x: Any) -> float | None:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except ValueError:
            return None
    return None


def values_equal(pred: Any, gold: Any) -> bool:
    if gold is None or pred is None:
        return gold is pred or (gold in ("", None) and pred in ("", None))
    if isinstance(gold, bool) or isinstance(pred, bool):
        if isinstance(pred, str):
            pred = pred.strip().lower() in ("true", "1")
        if isinstance(gold, str):
            gold = gold.strip().lower() in ("true", "1")
        return bool(pred) == bool(gold)
    gn, pn = _num(gold), _num(pred)
    if isinstance(gold, (int, float)) and pn is not None:
        return math.isclose(pn, gn, rel_tol=1e-6, abs_tol=1e-9)
    if isinstance(gold, str) and isinstance(pred, (int, float)):
        return gn is not None and math.isclose(float(pred), gn, rel_tol=1e-6)
    if isinstance(gold, str) and isinstance(pred, str):
        return _norm_str(pred) == _norm_str(gold)
    if isinstance(gold, list) and isinstance(pred, list):
        return len(gold) == len(pred) and all(values_equal(p, g) for p, g in zip(pred, gold))
    if isinstance(gold, dict) and isinstance(pred, dict):
        return set(gold) == set(pred) and all(values_equal(pred[k], gold[k]) for k in gold)
    if isinstance(gold, (list, dict)) and isinstance(pred, str):
        import json

        try:
            return values_equal(json.loads(pred), gold)
        except ValueError:
            return False
    return pred == gold


def _norm_str(s: str) -> str:
    return " ".join(s.strip().split())


def _defaults(params: dict) -> dict:
    out = {}
    for k, spec in schema_properties(params).items():
        if isinstance(spec, dict) and "default" in spec:
            out[k] = spec["default"]
    return out


def call_matches(pred: dict, gold: dict, params: dict) -> tuple[bool, float]:
    """Exact match of one call + a partial score in [0,1] for reward shaping."""
    if pred.get("name") != gold.get("name"):
        return False, 0.0
    pa, ga = pred.get("arguments") or {}, gold.get("arguments") or {}
    defaults = _defaults(params)
    required = set(params.get("required") or [])
    keys = set(ga) | set(pa)
    if not keys:
        return True, 1.0
    good = 0
    ok = True
    for k in keys:
        if k in ga and k in pa:
            if values_equal(pa[k], ga[k]):
                good += 1
            else:
                ok = False
        elif k in pa:  # extra arg: fine only if it equals the declared default
            if k in defaults and values_equal(pa[k], defaults[k]):
                good += 1
            else:
                ok = False
        else:  # gold arg missing from prediction: fine only if optional and gold == default
            if k not in required and k in defaults and values_equal(ga[k], defaults[k]):
                good += 1
            else:
                ok = False
    return ok, good / len(keys)


def score_tool_calls(pred_calls: list[dict], gold_calls: list[dict], tools: list[dict],
                     parse_errors: list[str] | None = None, terminated: bool = True) -> dict:
    """Order-insensitive multiset matching. gold_calls == [] means 'must not call a tool'."""
    idx = tool_index(tools)
    names = set(idx)
    format_ok = terminated and not any(e.startswith(("unterminated", "unparseable", "empty")) for e in parse_errors or [])
    valid_names = all(c.get("name") in names for c in pred_calls)
    out = {"format_ok": format_ok, "valid_names": valid_names, "n_pred": len(pred_calls), "n_gold": len(gold_calls)}

    if not gold_calls:
        exact = format_ok and not pred_calls
        out.update(exact=exact, name_f1=1.0 if not pred_calls else 0.0, arg_score=1.0 if exact else 0.0)
        out["reward"] = 1.0 if exact else 0.0
        return out

    # greedy best matching (n is tiny)
    unused = list(range(len(pred_calls)))
    exact_hits, partial = 0, 0.0
    for g in gold_calls:
        best_j, best = None, (-1.0, False)
        for j in unused:
            ok, s = call_matches(pred_calls[j], g, idx.get(g.get("name"), {}))
            if (s, ok) > best:
                best_j, best = j, (s, ok)
        if best_j is not None and best[0] > 0:
            unused.remove(best_j)
            partial += best[0]
            exact_hits += int(best[1])
    gold_names = [g["name"] for g in gold_calls]
    pred_names = [c.get("name") for c in pred_calls]
    common = sum(min(gold_names.count(n), pred_names.count(n)) for n in set(gold_names))
    prec = common / len(pred_names) if pred_names else 0.0
    rec = common / len(gold_names)
    name_f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    exact = format_ok and exact_hits == len(gold_calls) and len(pred_calls) == len(gold_calls)
    arg_score = partial / max(len(gold_calls), len(pred_calls))
    # ToolRL-style decomposed reward: format 0.1 + names 0.3 + args 0.6, exact = 1.0
    reward = 1.0 if exact else (0.1 * format_ok + 0.3 * name_f1 + 0.6 * arg_score) * 0.9
    out.update(exact=exact, name_f1=name_f1, arg_score=arg_score, reward=reward)
    return out

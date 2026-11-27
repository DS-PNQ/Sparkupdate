"""Build verifiable tasks: tool calling (gold calls), code (unit tests), agentic repo tasks.

    python -m spark_kaggle.tasks --config configs/default.yaml

Outputs (cfg.tasks.out_dir): tool_{train,eval}.jsonl, code_{train,eval}.jsonl, agent_{train,eval}.jsonl

Dataset schemas were written from the public dataset cards; every adapter validates rows and logs
how many it skipped, so a schema change shows up as a loud warning rather than silent garbage.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import logging
import random
import re
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .chat import as_openai_tool, tool_name
from .config import base_parser, cfg_from_args
from .env import setup_env, setup_logging, write_json, write_jsonl, hf_login
from .sandbox import run_tests_many

log = logging.getLogger("spark")

# ============================================================================ helpers


def _hash_frac(key: str) -> float:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def _loads(s: Any) -> Any:
    if not isinstance(s, str):
        return s
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def load_hf(src: dict) -> Iterable[dict]:
    from datasets import load_dataset

    kw: dict[str, Any] = {"split": src.get("split", "train")}
    if src.get("config"):
        kw["name"] = src["config"]
    if src.get("limit") and src.get("streaming", True):
        ds = load_dataset(src["hf_id"], streaming=True, **kw)
        if src.get("shuffle", True):
            ds = ds.shuffle(seed=1234, buffer_size=int(src.get("shuffle_buffer", 5000)))
        return ds
    return load_dataset(src["hf_id"], **kw)


def _take(rows: Iterable[dict], limit: int | None, oversample: float = 1.0) -> list[dict]:
    out = []
    cap = int(limit * oversample) if limit else None
    for r in rows:
        out.append(dict(r))
        if cap and len(out) >= cap:
            break
    return out


# ============================================================================ tool tasks

_XLAM_TYPES = {"str": "string", "string": "string", "int": "integer", "integer": "integer", "float": "number",
               "number": "number", "bool": "boolean", "boolean": "boolean", "list": "array", "tuple": "array",
               "set": "array", "dict": "object", "object": "object"}


def _xlam_type(t: str) -> tuple[str, bool]:
    t = (t or "string").strip()
    optional = "optional" in t.lower()
    base = t.split(",")[0].strip()
    head = re.split(r"[\[\(]", base)[0].strip().lower()
    return _XLAM_TYPES.get(head, "string"), optional


def xlam_tool_to_openai(tool: dict) -> dict:
    params = tool.get("parameters") or {}
    if isinstance(params, dict) and params.get("type") == "object" and "properties" in params:
        schema = params
    else:
        props, required = {}, []
        for pname, spec in (params or {}).items():
            spec = spec if isinstance(spec, dict) else {"type": str(spec)}
            jtype, optional = _xlam_type(str(spec.get("type", "string")))
            prop: dict[str, Any] = {"type": jtype}
            if spec.get("description"):
                prop["description"] = spec["description"]
            if "default" in spec and spec["default"] not in ("", None):
                prop["default"] = spec["default"]
            props[pname] = prop
            if not optional and "default" not in spec:
                required.append(pname)
        schema = {"type": "object", "properties": props, "required": required}
    return as_openai_tool({"name": tool["name"], "description": tool.get("description", ""), "parameters": schema})


def _call(obj: dict) -> dict:
    args = obj.get("arguments", obj.get("parameters", {}))
    args = _loads(args) if isinstance(args, str) and args.strip() else (args or {})
    return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


def adapt_xlam(rows: list[dict], src: str) -> list[dict]:
    tasks = []
    for r in rows:
        try:
            tools = [xlam_tool_to_openai(t) for t in _loads(r["tools"])]
            gold = [_call(a) for a in _loads(r["answers"])]
            q = r["query"]
        except Exception:
            continue
        if not tools or not q:
            continue
        tasks.append({"id": f"{src}-{r.get('id', len(tasks))}", "type": "tool", "source": src,
                      "category": "parallel" if len(gold) > 1 else "simple", "tools": tools,
                      "messages": [{"role": "user", "content": q}], "gold_calls": gold})
    return tasks


_TOOLS_RE = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
_TC_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_TR_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.DOTALL)


def _hermes_tools(system: str) -> list[dict]:
    m = _TOOLS_RE.search(system or "")
    if not m:
        return []
    body = m.group(1).strip()
    try:
        tools = _loads(body)
    except Exception:
        tools = [_loads(x) for x in re.findall(r"\{.*?\}\s*(?=\{|$)", body, re.DOTALL)]
    tools = tools if isinstance(tools, list) else [tools]
    return [as_openai_tool(t) for t in tools if isinstance(t, dict)]


def _conv_to_tasks(conv: list[tuple[str, Any]], tools: list[dict], system: str | None, src: str, base_id: str,
                   multi_turn_label: bool = True) -> list[dict]:
    """conv: list of (role, payload) with role in user/assistant_text/assistant_calls/tool."""
    tasks, history = [], []
    if system:
        history.append({"role": "system", "content": system})
    seen_assistant = False
    for role, payload in conv:
        if role == "user":
            history.append({"role": "user", "content": payload})
        elif role == "assistant_text":
            history.append({"role": "assistant", "content": payload})
            seen_assistant = True
        elif role == "assistant_calls":
            calls = payload
            if any(m["role"] == "user" for m in history):
                cat = "parallel" if len(calls) > 1 else "simple"
                if seen_assistant and multi_turn_label:
                    cat = "multi_turn"
                tasks.append({"id": f"{base_id}-s{len(tasks)}", "type": "tool", "source": src, "category": cat,
                              "tools": tools, "messages": copy.deepcopy(history), "gold_calls": calls})
            history.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for i, c in enumerate(calls)]})
            seen_assistant = True
        elif role == "tool":
            history.append({"role": "tool", "content": payload})
    return tasks


def adapt_hermes(rows: list[dict], src: str) -> list[dict]:
    tasks = []
    for k, r in enumerate(rows):
        conv_raw = r.get("conversations") or []
        sys_msg = next((c["value"] for c in conv_raw if c.get("from") == "system"), "")
        tools = _hermes_tools(sys_msg)
        if not tools:
            continue
        conv: list[tuple[str, Any]] = []
        try:
            for c in conv_raw:
                f, v = c.get("from"), c.get("value", "")
                if f == "human":
                    conv.append(("user", v))
                elif f == "gpt":
                    blocks = _TC_RE.findall(v)
                    if blocks:
                        conv.append(("assistant_calls", [_call(_loads(b)) for b in blocks]))
                    else:
                        conv.append(("assistant_text", v.strip()))
                elif f == "tool":
                    resp = _TR_RE.findall(v)
                    for x in resp or [v]:
                        conv.append(("tool", x.strip()))
        except Exception:
            continue
        # Hermes system prompts describe a JSON tool-call format that conflicts with Spark's XML one -> dropped
        tasks += _conv_to_tasks(conv, tools, None, src, f"{src}-{r.get('id', k)}")
    return tasks


def adapt_apigen_mt(rows: list[dict], src: str) -> list[dict]:
    tasks = []
    for k, r in enumerate(rows):
        try:
            tools = [as_openai_tool(t) for t in _loads(r["tools"])]
            conv_raw = r["conversations"]
        except Exception:
            continue
        conv: list[tuple[str, Any]] = []
        pending: list[dict] = []
        try:
            for c in conv_raw:
                f, v = c.get("from"), c.get("value", "")
                if f != "function_call" and pending:
                    conv.append(("assistant_calls", pending))
                    pending = []
                if f == "human":
                    conv.append(("user", v))
                elif f == "gpt":
                    conv.append(("assistant_text", v))
                elif f == "function_call":
                    pending.append(_call(_loads(v)))
                elif f == "observation":
                    conv.append(("tool", v if isinstance(v, str) else json.dumps(v)))
            if pending:
                conv.append(("assistant_calls", pending))
        except Exception:
            continue
        tasks += _conv_to_tasks(conv, tools, r.get("system") or None, src, f"{src}-{k}")
    return tasks


def adapt_openai_jsonl(rows: list[dict], src: str) -> list[dict]:
    tasks = []
    for k, r in enumerate(rows):
        tools = [as_openai_tool(t) for t in r.get("tools") or []]
        conv: list[tuple[str, Any]] = []
        system = None
        for m in r.get("messages", []):
            role = m.get("role")
            if role == "system":
                system = m.get("content")
            elif role == "user":
                conv.append(("user", m.get("content", "")))
            elif role == "assistant" and m.get("tool_calls"):
                conv.append(("assistant_calls", [_call(tc.get("function", tc)) for tc in m["tool_calls"]]))
            elif role == "assistant":
                conv.append(("assistant_text", m.get("content", "")))
            elif role == "tool":
                conv.append(("tool", m.get("content", "")))
        tasks += _conv_to_tasks(conv, tools, system, src, f"{src}-{r.get('id', k)}")
    return tasks


TOOL_ADAPTERS = {"xlam": adapt_xlam, "hermes": adapt_hermes, "apigen_mt": adapt_apigen_mt,
                 "openai_jsonl": adapt_openai_jsonl}


def _words(t: dict) -> set[str]:
    fn = t.get("function", t)
    text = f"{fn.get('name', '')} {fn.get('description', '')}".lower().replace("_", " ")
    return {w for w in re.findall(r"[a-z]{4,}", text)}


def augment_tool_tasks(tasks: list[dict], rng: random.Random, irrelevance_fraction: float,
                       distractors: tuple[int, int]) -> list[dict]:
    pool = [t for task in tasks for t in task["tools"]]
    uniq: dict[str, dict] = {}
    for t in pool:
        uniq.setdefault(tool_name(t), t)
    pool = list(uniq.values())
    out = []
    for task in tasks:
        task = copy.deepcopy(task)
        own = {tool_name(t) for t in task["tools"]}
        own_words = set().union(*[_words(t) for t in task["tools"]]) | set(
            re.findall(r"[a-z]{4,}", json.dumps(task["messages"][-1].get("content", "")).lower()))
        if task["category"] in ("simple", "parallel") and rng.random() < irrelevance_fraction and len(pool) > 50:
            cands = []
            for _ in range(60):
                t = rng.choice(pool)
                w = _words(t)
                if tool_name(t) not in own and len(w & own_words) <= 1:
                    cands.append(t)
                if len(cands) >= rng.randint(1, 4):
                    break
            if cands:
                task.update(id=task["id"] + "-irr", category="irrelevance", tools=cands, gold_calls=[])
                out.append(task)
                continue
        lo, hi = distractors
        k = rng.randint(lo, hi) if hi > 0 else 0
        if k and len(pool) > k + 5:
            extra = [t for t in rng.sample(pool, k + 5) if tool_name(t) not in own][:k]
            task["tools"] = task["tools"] + extra
            rng.shuffle(task["tools"])
            if task["category"] == "simple" and extra:
                task["category"] = "multiple"
        out.append(task)
    return out


def build_tool_tasks(cfg, rng: random.Random) -> tuple[list[dict], list[dict]]:
    train, evals = [], []
    for src in cfg.tasks.tool_sources:
        name = src["name"]
        try:
            if src.get("path"):
                from .env import read_jsonl

                rows = read_jsonl(src["path"])[: src.get("limit") or None]
            else:
                rows = _take(load_hf(src), src.get("limit"), oversample=1.0 + cfg.tasks.eval_fraction * 2)
        except Exception as e:
            log.warning("tool source %s (%s) failed to load: %s", name, src.get("hf_id") or src.get("path"), e)
            continue
        tasks = TOOL_ADAPTERS[name](rows, name + (f"_{src['config']}" if src.get("config") else ""))
        log.info("tool source %-12s rows=%5d -> tasks=%5d %s", name, len(rows), len(tasks),
                 "" if tasks or not rows else "  <-- 0 tasks: schema mismatch? check the dataset card")
        tr = [t for t in tasks if _hash_frac(t["id"].split("-s")[0]) >= cfg.tasks.eval_fraction]
        ev = [t for t in tasks if _hash_frac(t["id"].split("-s")[0]) < cfg.tasks.eval_fraction]
        train += tr
        evals += ev[: cfg.tasks.max_eval_per_source]
    d = cfg.tasks.distractor_tools
    train = augment_tool_tasks(train, rng, cfg.tasks.irrelevance_fraction, (int(d[0]), int(d[1])))
    evals = augment_tool_tasks(evals, random.Random(7), cfg.tasks.irrelevance_fraction, (int(d[0]), int(d[1])))
    return train, evals


# ============================================================================ code tasks

CODE_INSTR = ("\n\nWrite a complete, self-contained Python solution. Put the final code in a single ```python code "
              "block.")


def _mbpp_tests(r: dict) -> str:
    setup = r.get("test_setup_code") or ""
    imports = "\n".join(r.get("test_imports") or [])
    asserts = "\n".join("    " + a for a in r["test_list"])
    return f"from solution import *\n{imports}\n{setup}\n\n\ndef test_main():\n{asserts}\n"


def adapt_mbpp(rows: list[dict], src: str) -> list[dict]:
    out = []
    for r in rows:
        text = r.get("text") or r.get("prompt")
        if not text or not r.get("test_list"):
            continue
        prompt = f"{text}\nYour code should pass this test:\n{r['test_list'][0]}" + CODE_INSTR
        out.append({"id": f"{src}-{r['task_id']}", "type": "code", "source": src, "prompt": prompt,
                    "tests": _mbpp_tests(r), "spec": text, "canonical": r.get("code")})
    return out


def adapt_humaneval(rows: list[dict], src: str) -> list[dict]:
    out = []
    for r in rows:
        prompt = ("Complete the following Python function. Return the whole function (with imports).\n"
                  f"```python\n{r['prompt']}```" + CODE_INSTR)
        tests = f"from solution import *\n\n{r['test']}\n\n\ndef test_main():\n    check({r['entry_point']})\n"
        out.append({"id": f"{src}-{r['task_id'].replace('/', '_')}", "type": "code", "source": src, "prompt": prompt,
                    "tests": tests, "entry_point": r["entry_point"], "prompt_code": r["prompt"],
                    "spec": r["prompt"], "canonical": r["prompt"] + r["canonical_solution"]})
    return out


def adapt_kodcode(rows: list[dict], src: str) -> list[dict]:
    out = []
    for k, r in enumerate(rows):
        q, sol, test = r.get("question"), r.get("solution"), r.get("test")
        if not (q and sol and test) or "from solution import" not in test:
            continue
        decls = []
        for ti in r.get("test_info") or []:
            if isinstance(ti, dict) and ti.get("function_declaration"):
                decls.append(ti["function_declaration"].strip())
        extra = ("\n\nYour solution must define: " + "; ".join(f"`{d}`" for d in decls)) if decls else ""
        out.append({"id": f"{src}-{r.get('question_id', k)}", "type": "code", "source": src,
                    "prompt": q + extra + CODE_INSTR, "tests": test, "spec": q + extra, "canonical": sol})
    return out


CODE_ADAPTERS = {"mbpp": adapt_mbpp, "humaneval": adapt_humaneval, "kodcode": adapt_kodcode}


# ---------------------------------------------------------------------------- mutation (bug injection)

_SWAPS = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
          ast.NotEq: ast.Eq, ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mult,
          ast.And: ast.Or, ast.Or: ast.And, ast.Is: ast.IsNot, ast.IsNot: ast.Is, ast.In: ast.NotIn,
          ast.NotIn: ast.In}


class _Sites(ast.NodeVisitor):
    def __init__(self):
        self.sites: list[tuple[str, ast.AST, int]] = []

    def generic_visit(self, node):
        if isinstance(node, ast.Compare):
            for i, op in enumerate(node.ops):
                if type(op) in _SWAPS:
                    self.sites.append(("cmp", node, i))
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _SWAPS:
            self.sites.append(("binop", node, 0))
        elif isinstance(node, ast.BoolOp) and type(node.op) in _SWAPS:
            self.sites.append(("boolop", node, 0))
        elif isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool) \
                and abs(node.value) < 1000:
            self.sites.append(("const", node, 0))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            self.sites.append(("not", node, 0))
        super().generic_visit(node)


def mutate(code: str, rng: random.Random) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    v = _Sites()
    v.visit(tree)
    funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    inside = {id(n) for f in funcs for n in ast.walk(f)}
    sites = [s for s in v.sites if id(s[1]) in inside]
    if not sites:
        return None
    kind, node, i = rng.choice(sites)
    if kind == "cmp":
        node.ops[i] = _SWAPS[type(node.ops[i])]()
    elif kind in ("binop", "boolop"):
        node.op = _SWAPS[type(node.op)]()
    elif kind == "const":
        node.value = node.value + rng.choice([-1, 1])
    elif kind == "not" and not _drop_not(tree, node):  # `not x` -> `x`
        return None
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _drop_not(tree: ast.AST, target: ast.UnaryOp) -> bool:
    for parent in ast.walk(tree):
        for field, value in ast.iter_fields(parent):
            if value is target:
                setattr(parent, field, target.operand)
                return True
            if isinstance(value, list):
                for j, item in enumerate(value):
                    if item is target:
                        value[j] = target.operand
                        return True
    return False


def make_stub(code: str) -> str | None:
    """Keep imports/classes/signatures, replace function bodies by docstring + NotImplementedError."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    changed = False
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node)
            body: list[ast.stmt] = []
            if doc:
                body.append(ast.Expr(ast.Constant(doc)))
            body.append(ast.parse("raise NotImplementedError('TODO')").body[0])
            node.body = body
            changed = True
    return ast.unparse(tree) if changed else None


AGENT_SYSTEM = ("You are an autonomous coding agent working inside a Python repository. Use the provided tools to "
                "inspect files, edit code and run the tests. Work step by step and verify your change by running the "
                "tests. When the task is complete, reply with a short summary and do not call any tool.")


def make_agent_tasks(code_tasks: list[dict], n: int, repair_fraction: float, rng: random.Random,
                     tag: str) -> list[dict]:
    cands = [t for t in code_tasks if t.get("canonical")]
    rng.shuffle(cands)
    proposals = []
    for t in cands[: int(n * 1.6) + 10]:
        canonical = t["canonical"]
        repair = rng.random() < repair_fraction
        broken = mutate(canonical, rng) if repair else make_stub(canonical)
        if broken is None or broken.strip() == canonical.strip():
            broken, repair = make_stub(canonical), False
        if broken is None:
            continue
        readme = f"# Task\n\n{t['spec']}\n"
        if repair:
            instr = ("The test suite in `test_solution.py` is failing. Find and fix the bug in `solution.py` so that "
                     "all tests pass. Do not modify the tests.")
        else:
            instr = ("Implement the missing functionality in `solution.py` as described in `README.md`. "
                     "The tests are in `test_solution.py`; do not modify them.")
        proposals.append({
            "id": f"agent-{tag}-{'repair' if repair else 'impl'}-{t['id']}", "type": "agent", "source": t["source"],
            "category": "repair" if repair else "implement", "instruction": instr,
            "files": {"solution.py": broken, "test_solution.py": t["tests"], "README.md": readme},
            "canonical": canonical, "test_file": "test_solution.py"})
    # validate: canonical passes, broken fails
    jobs = []
    for p in proposals:
        jobs.append({"files": {"solution.py": p["canonical"], "test_solution.py": p["files"]["test_solution.py"]}})
        jobs.append({"files": {"solution.py": p["files"]["solution.py"],
                               "test_solution.py": p["files"]["test_solution.py"]}})
    res = run_tests_many(jobs, timeout=30)
    good = []
    for k, p in enumerate(proposals):
        ok_canon, ok_broken = res[2 * k][0], res[2 * k + 1][0]
        if ok_canon and not ok_broken:
            good.append(p)
    log.info("agent tasks (%s): %d proposals -> %d valid (canonical passes, broken fails)", tag, len(proposals),
             len(good))
    return good[:n]


def build_code_tasks(cfg, rng: random.Random) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    def load(sources):
        out = []
        for src in sources:
            try:
                rows = _take(load_hf(src), src.get("limit"))
            except Exception as e:
                log.warning("code source %s failed to load: %s", src.get("hf_id"), e)
                continue
            tasks = CODE_ADAPTERS[src["name"]](rows, src["name"])
            log.info("code source %-10s rows=%5d -> tasks=%5d", src["name"], len(rows), len(tasks))
            out += tasks
        return out

    code_train = load(cfg.tasks.code_sources)
    code_eval = load(cfg.tasks.code_eval_sources)
    agent_train = make_agent_tasks(code_train, int(cfg.tasks.agent_tasks), cfg.tasks.repair_fraction, rng, "train")
    agent_eval = make_agent_tasks(code_eval, int(cfg.tasks.agent_eval_tasks), cfg.tasks.repair_fraction,
                                  random.Random(11), "eval")
    return code_train, code_eval, agent_train, agent_eval


def main() -> None:
    p = base_parser("Build verifiable tool/code/agent tasks")
    p.add_argument("--only", choices=["tool", "code"], default=None)
    args = p.parse_args()
    cfg = cfg_from_args(args)
    setup_env()
    setup_logging()
    hf_login()
    rng = random.Random(cfg.seed)
    out = Path(cfg.tasks.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, Any] = {}
    if args.only in (None, "tool"):
        tr, ev = build_tool_tasks(cfg, rng)
        write_jsonl(out / "tool_train.jsonl", tr)
        write_jsonl(out / "tool_eval.jsonl", ev)
        stats["tool_train"] = dict(Counter(t["category"] for t in tr))
        stats["tool_eval"] = dict(Counter(t["category"] for t in ev))
    if args.only in (None, "code"):
        ct, ce, at, ae = build_code_tasks(cfg, rng)
        for name, rows in [("code_train", ct), ("code_eval", ce), ("agent_train", at), ("agent_eval", ae)]:
            write_jsonl(out / f"{name}.jsonl", rows)
            stats[name] = len(rows)
    write_json(out / "stats.json", stats)
    log.info("tasks written to %s: %s", out, json.dumps(stats))


if __name__ == "__main__":
    main()

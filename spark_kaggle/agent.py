"""Batched multi-turn agent loop over synthetic repositories (bash/read/write/str_replace tools)."""

from __future__ import annotations

import logging

from .chat import ChatFormatter
from .engine import Generator
from .sandbox import AGENT_TOOLS, PYTEST_CMD, Workspace, run_command
from .tasks import AGENT_SYSTEM

log = logging.getLogger("spark")


def run_agent_episodes(gen: Generator, fmt: ChatFormatter, tasks: list[dict], samples: int, mode: str,
                       max_turns: int, max_new_tokens: int, max_context: int, sampling: dict,
                       keep_turns: str = "pass") -> list[dict]:
    """Returns one dict per episode. keep_turns: 'pass' (token turns only for solved episodes), 'all', 'none'."""
    eps = []
    for t in tasks:
        for s in range(samples):
            eps.append({
                "task": t, "sample": s, "ws": Workspace(t["files"]), "turns": [], "done": False, "finish": None,
                "n_calls": 0, "n_tool_errors": 0, "n_format_errors": 0,
                "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": t["instruction"]}],
            })
    try:
        for _turn in range(max_turns):
            active = [e for e in eps if not e["done"]]
            if not active:
                break
            batch = []
            for e in active:
                p = fmt.prompt_ids(e["messages"], AGENT_TOOLS, mode)
                if len(p) + max_new_tokens > max_context:
                    e["done"], e["finish"] = True, "context"
                else:
                    batch.append((e, p))
            if not batch:
                break
            outs = gen.generate([p for _, p in batch], max_new_tokens, **sampling)
            for (e, p), o in zip(batch, outs):
                text = fmt.decode_completion(o["ids"])
                parsed = fmt.parse(text, AGENT_TOOLS, mode)
                comp = o["ids"] + ([o["stop_id"]] if o.get("stop_id") is not None else [])
                e["turns"].append({"prompt_ids": p, "completion_ids": comp})
                if o["finish"] != "stop" or not parsed["terminated"]:
                    e["done"], e["finish"] = True, "truncated"
                    continue
                if parsed["errors"]:
                    e["n_format_errors"] += 1
                e["messages"].append(fmt.assistant_message(parsed))
                if not parsed["calls"]:
                    e["done"], e["finish"] = True, "answered"
                    continue
                for i, c in enumerate(parsed["calls"]):
                    result = e["ws"].call(c["name"], c["arguments"])
                    e["n_tool_errors"] += int(result.startswith("error"))
                    e["messages"].append({"role": "tool", "tool_call_id": f"call_{i}", "name": c["name"],
                                          "content": result})
                e["n_calls"] += len(parsed["calls"])
        results = []
        for e in eps:
            if not e["done"]:
                e["finish"] = "max_turns"
            t = e["task"]
            ws: Workspace = e["ws"]
            ws.write(t["test_file"], t["files"][t["test_file"]])  # undo any edits to the tests
            code, out = run_command(PYTEST_CMD + [t["test_file"]], ws.dir, timeout=40)
            solved = code == 0
            changed = ws.read("solution.py") != t["files"]["solution.py"] if ws.exists("solution.py") else False
            r = {"task_id": t["id"], "sample": e["sample"], "tests_pass": solved and changed,
                 "pass": solved and changed and e["finish"] == "answered", "finish": e["finish"],
                 "n_turns": len(e["turns"]), "n_calls": e["n_calls"], "n_tool_errors": e["n_tool_errors"],
                 "n_format_errors": e["n_format_errors"]}
            if keep_turns == "all" or (keep_turns == "pass" and r["pass"]):
                r["turns"] = e["turns"]
            results.append(r)
        return results
    finally:
        for e in eps:
            e["ws"].cleanup()

"""Verifiers shared by collection, evaluation and GRPO."""

from __future__ import annotations

import re

from .chat import ChatFormatter
from .sandbox import run_tests_many
from .toolmatch import score_tool_calls

_CODE_BLOCK = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.DOTALL)


def verify_tool(task: dict, completion: str, fmt: ChatFormatter, mode: str) -> dict:
    parsed = fmt.parse(completion, task["tools"], mode)
    s = score_tool_calls(parsed["calls"], task["gold_calls"], task["tools"], parsed["errors"], parsed["terminated"])
    s["pass"] = bool(s["exact"])
    s["pred_calls"] = parsed["calls"]
    if not parsed["terminated"]:
        s["fail_reason"] = "unterminated_thinking"
    elif not s["format_ok"]:
        s["fail_reason"] = "format"
    elif not s["valid_names"]:
        s["fail_reason"] = "hallucinated_tool"
    elif not s["exact"]:
        s["fail_reason"] = "irrelevance_called" if not task["gold_calls"] else (
            "no_call" if not parsed["calls"] else "wrong_call")
    return s


def extract_code(answer: str) -> str | None:
    blocks = _CODE_BLOCK.findall(answer)
    if blocks:
        return max(blocks[-3:], key=len) if len(blocks) > 1 and "def " not in blocks[-1] else blocks[-1]
    if "def " in answer and "```" not in answer:
        return answer
    return None


def _solution_file(task: dict, code: str) -> str:
    if task.get("entry_point") and task.get("prompt_code"):
        if f"def {task['entry_point']}" not in code:
            code = task["prompt_code"] + code
        imports = [ln for ln in task["prompt_code"].splitlines() if ln.startswith(("import ", "from "))]
        code = "\n".join(imports) + "\n" + code
    return code


def verify_code_batch(tasks: list[dict], completions: list[str], fmt: ChatFormatter, modes: list[str],
                      timeout: float = 30) -> list[dict]:
    jobs, idx, results = [], [], []
    for k, (t, comp, mode) in enumerate(zip(tasks, completions, modes)):
        parsed = fmt.parse(comp, None, mode)
        r = {"pass": False, "reward": 0.0, "terminated": parsed["terminated"]}
        if not parsed["terminated"]:
            r["fail_reason"] = "unterminated_thinking"
        else:
            code = extract_code(parsed["answer"])
            if code is None:
                r["fail_reason"] = "no_code"
            else:
                jobs.append({"files": {"solution.py": _solution_file(t, code), "test_solution.py": t["tests"]}})
                idx.append(k)
        results.append(r)
    for k, (ok, out) in zip(idx, run_tests_many(jobs, timeout=timeout)):
        results[k]["pass"] = ok
        results[k]["reward"] = 1.0 if ok else 0.1  # small credit for well-formed code
        if not ok:
            results[k]["fail_reason"] = "tests_failed"
            results[k]["test_output"] = out[-800:]
    return results

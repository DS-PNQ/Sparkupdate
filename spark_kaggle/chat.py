"""Chat-template rendering and Spark-X2.5/2.6 tool-call parsing.

Spark-X2.6 wire format (SGLang `spark25` detector, GLM-style XML key/values):

    <tool_call>function_name
    <arg_key>key</arg_key><arg_value>value</arg_value>
    </tool_call>

Values are plain text; non-string parameters are converted with the tool's JSON-schema type.
Reasoning uses <think> ... </think>; in no-think mode the template ends the prompt with a bare
</think> (see XHToken/LlamaFactory `SparkReasoningTemplate`).

Nothing here hard-codes the turn markup: prompts are always produced by the checkpoint's own
`chat_template.jinja` through `tokenizer.apply_chat_template`.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_BEGIN = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
ARG_KEY_BEGIN, ARG_KEY_END = "<arg_key>", "</arg_key>"
ARG_VALUE_BEGIN, ARG_VALUE_END = "<arg_value>", "</arg_value>"

_TOOL_BLOCK = re.compile(re.escape(TOOL_CALL_BEGIN) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL)
_ARG_PAIR = re.compile(
    re.escape(ARG_KEY_BEGIN) + r"(.*?)" + re.escape(ARG_KEY_END) + r"\s*"
    + re.escape(ARG_VALUE_BEGIN) + r"(.*?)" + re.escape(ARG_VALUE_END),
    re.DOTALL,
)


# ----------------------------------------------------------------------------- tools / schemas

def tool_name(tool: dict) -> str:
    return (tool.get("function") or tool).get("name", "")


def tool_params(tool: dict) -> dict:
    fn = tool.get("function") or tool
    params = fn.get("parameters") or {}
    return params if isinstance(params, dict) else {}


def tool_index(tools: list[dict] | None) -> dict[str, dict]:
    return {tool_name(t): tool_params(t) for t in tools or []}


def schema_properties(params: dict) -> dict:
    props = params.get("properties")
    return props if isinstance(props, dict) else {}


def param_type(tools_idx: dict[str, dict], fn: str, key: str) -> str:
    spec = schema_properties(tools_idx.get(fn, {})).get(key)
    if isinstance(spec, dict):
        t = spec.get("type")
        if isinstance(t, str):
            return t
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if non_null:
                return non_null[0]
    return "string"


def as_openai_tool(tool: dict) -> dict:
    """Accept {'name','description','parameters'} or OpenAI {'type':'function','function':{...}}."""
    if tool.get("type") == "function" and "function" in tool:
        return tool
    return {"type": "function", "function": {k: v for k, v in tool.items() if k in ("name", "description", "parameters")}}


# ----------------------------------------------------------------------------- value conversion

def convert_value(raw: str, ptype: str) -> Any:
    """Mirror of sglang spark25_detector._convert_value."""
    value = raw
    if value.lower() == "null":
        return None
    t = (ptype or "string").lower()
    try:
        if t in {"string", "str", "text"}:
            return value
        if t in {"integer", "int"}:
            return int(value)
        if t in {"number", "float"}:
            num = float(value)
            return int(num) if num.is_integer() else num
        if t in {"boolean", "bool"}:
            v = value.strip().lower()
            if v not in {"true", "1", "false", "0"}:
                raise ValueError(value)
            return v in {"true", "1"}
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return value


def value_to_text(v: Any) -> str:
    """Inverse of convert_value: how a value is written inside <arg_value>."""
    if isinstance(v, str):
        return v
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    return json.dumps(v, ensure_ascii=False)


# ----------------------------------------------------------------------------- parsing

def split_reasoning(text: str, mode: str, prompt_opened_think: bool = False) -> tuple[str, str, bool]:
    """Return (reasoning, answer, terminated). terminated=False -> thinking never closed."""
    if THINK_END in text:
        reasoning, answer = text.split(THINK_END, 1)
        reasoning = reasoning.replace(THINK_START, "", 1).strip()
        return reasoning, answer.strip(), True
    stripped = text.lstrip()
    if mode == "think" and (prompt_opened_think or stripped.startswith(THINK_START)):
        return stripped.replace(THINK_START, "", 1).strip(), "", False
    return "", text.strip(), True


def parse_tool_calls(answer: str, tools: list[dict] | None) -> tuple[list[dict], str, list[str]]:
    """Parse every <tool_call> block. Returns (calls, remaining_text, errors)."""
    idx = tool_index(tools)
    calls: list[dict] = []
    errors: list[str] = []
    for m in _TOOL_BLOCK.finditer(answer):
        body = m.group(1)
        first = body.find(ARG_KEY_BEGIN)
        name = (body if first < 0 else body[:first]).strip()
        if not name:
            errors.append("empty function name")
            continue
        if name.startswith("{"):
            # JSON-style call (Hermes/Qwen format) - accept but flag
            try:
                obj = json.loads(name)
                calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {}) or {}})
                errors.append("json-style tool call")
            except json.JSONDecodeError:
                errors.append("unparseable json tool call")
            continue
        args: dict[str, Any] = {}
        for km in _ARG_PAIR.finditer(body):
            key, raw = km.group(1).strip(), km.group(2)
            if not key:
                continue
            if key in args:
                errors.append(f"duplicate argument {key}")
            args[key] = convert_value(raw, param_type(idx, name, key))
        calls.append({"name": name, "arguments": args})
    if TOOL_CALL_BEGIN in answer and answer.count(TOOL_CALL_BEGIN) != answer.count(TOOL_CALL_END):
        errors.append("unterminated tool_call block")
    remaining = _TOOL_BLOCK.sub("", answer).strip()
    return calls, remaining, errors


def format_tool_calls_xml(calls: list[dict]) -> str:
    out = []
    for c in calls:
        parts = [TOOL_CALL_BEGIN + c["name"] + "\n"]
        for k, v in (c.get("arguments") or {}).items():
            parts.append(f"{ARG_KEY_BEGIN}{k}{ARG_KEY_END}{ARG_VALUE_BEGIN}{value_to_text(v)}{ARG_VALUE_END}\n")
        parts.append(TOOL_CALL_END)
        out.append("".join(parts))
    return "\n".join(out)


# ----------------------------------------------------------------------------- template rendering

class ChatFormatter:
    """Wraps tokenizer.apply_chat_template with the run's thinking/tool settings."""

    def __init__(self, tokenizer, chat_cfg: dict | None = None):
        self.tok = tokenizer
        cfg = chat_cfg or {}
        self.think_kwargs = dict(cfg.get("think_kwargs") or {"enable_thinking": True})
        self.nothink_kwargs = dict(cfg.get("nothink_kwargs") or {"enable_thinking": False})
        self.system_prompt = cfg.get("system_prompt")
        mode = cfg.get("tool_args_mode", "auto")
        self.tool_args_mode = detect_tool_args_mode(tokenizer) if mode == "auto" else mode
        self._prefill_cache: dict[str, bool] = {}

    def kwargs(self, mode: str) -> dict:
        return dict(self.think_kwargs if mode == "think" else self.nothink_kwargs)

    def prepare_messages(self, messages: list[dict]) -> list[dict]:
        msgs = []
        if self.system_prompt and not (messages and messages[0].get("role") == "system"):
            msgs.append({"role": "system", "content": self.system_prompt})
        for m in messages:
            m = copy.deepcopy(m)
            if m.get("role") == "assistant" and m.get("tool_calls"):
                m["tool_calls"] = [self._tc(tc, i) for i, tc in enumerate(m["tool_calls"])]
                m.setdefault("content", "")
            if m.get("content") is None:
                m["content"] = ""
            msgs.append(m)
        return msgs

    def _tc(self, tc: dict, i: int) -> dict:
        fn = tc.get("function") or tc
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        if self.tool_args_mode == "json_string" and not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        return {"id": tc.get("id", f"call_{i}"), "type": "function", "function": {"name": fn["name"], "arguments": args}}

    def render(self, messages: list[dict], tools: list[dict] | None, mode: str, add_generation_prompt: bool) -> str:
        kw = self.kwargs(mode)
        return self.tok.apply_chat_template(
            self.prepare_messages(messages),
            tools=[as_openai_tool(t) for t in tools] if tools else None,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            **kw,
        )

    def prompt(self, messages: list[dict], tools: list[dict] | None, mode: str) -> str:
        return self.render(messages, tools, mode, add_generation_prompt=True)

    def prompt_ids(self, messages: list[dict], tools: list[dict] | None, mode: str) -> list[int]:
        return self.encode(self.prompt(messages, tools, mode))

    def encode(self, text: str) -> list[int]:
        # the template already contains BOS/special tokens
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def prompt_opens_think(self, mode: str) -> bool:
        if mode not in self._prefill_cache:
            p = self.prompt([{"role": "user", "content": "hi"}], None, mode)
            tail = p.rstrip()
            self._prefill_cache[mode] = tail.endswith(THINK_START)
        return self._prefill_cache[mode]

    def decode_completion(self, ids: list[int]) -> str:
        text = self.tok.decode(ids, skip_special_tokens=False)
        for t in {self.tok.eos_token, self.tok.pad_token}:
            if t:
                text = text.replace(t, "")
        return text

    def parse(self, completion: str, tools: list[dict] | None, mode: str) -> dict:
        reasoning, answer, terminated = split_reasoning(completion, mode, self.prompt_opens_think(mode))
        calls, content, errors = parse_tool_calls(answer, tools)
        return {"reasoning": reasoning, "answer": answer, "content": content, "calls": calls,
                "errors": errors, "terminated": terminated}

    def assistant_message(self, parsed: dict) -> dict:
        msg: dict[str, Any] = {"role": "assistant", "content": parsed["content"]}
        if parsed.get("reasoning"):
            msg["reasoning_content"] = parsed["reasoning"]
        if parsed["calls"]:
            msg["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for i, c in enumerate(parsed["calls"])
            ]
        return msg


_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather forecast",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
            "required": ["city"],
        },
    },
}


def detect_tool_args_mode(tokenizer) -> str:
    """Render one tool call with dict and with JSON-string arguments; keep whichever renders XML-KV."""
    results = {}
    for mode in ("dict", "json_string"):
        args: Any = {"city": "Paris", "days": 3}
        if mode == "json_string":
            args = json.dumps(args)
        msgs = [
            {"role": "user", "content": "Weather in Paris for 3 days?"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": args}}]},
            {"role": "tool", "tool_call_id": "call_0", "name": "get_weather", "content": "sunny"},
        ]
        try:
            text = tokenizer.apply_chat_template(msgs, tools=[_PROBE_TOOL], tokenize=False)
        except Exception:  # template rejects this argument type
            results[mode] = -1
            continue
        score = 0
        if f"{ARG_KEY_BEGIN}city{ARG_KEY_END}" in text:
            score += 2
        if f"{ARG_VALUE_BEGIN}Paris{ARG_VALUE_END}" in text:
            score += 2
        if "get_weather" in text:
            score += 1
        results[mode] = score
    return max(results, key=lambda k: (results[k], k == "dict"))


def template_variables(tokenizer) -> list[str]:
    """Undeclared variables of the chat template (e.g. enable_thinking, thinking level knobs)."""
    tpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tpl, dict):
        tpl = tpl.get("default") or next(iter(tpl.values()))
    if not tpl:
        return []
    try:
        import jinja2
        from jinja2 import meta

        env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
        return sorted(meta.find_undeclared_variables(env.parse(tpl)))
    except Exception:
        return sorted(set(re.findall(r"\b([a-z_]+(?:thinking|effort|budget|level)[a-z_]*)\b", tpl)))

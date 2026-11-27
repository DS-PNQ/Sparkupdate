import json
import random

from spark_kaggle.sandbox import Workspace, run_tests
from spark_kaggle.tasks import (adapt_apigen_mt, adapt_hermes, adapt_mbpp, adapt_xlam, augment_tool_tasks,
                                make_agent_tasks, make_stub, mutate)

SOL = "def add_all(xs):\n    total = 0\n    for x in xs:\n        if x > 0:\n            total += x\n    return total\n"
TEST = "from solution import *\n\ndef test_main():\n    assert add_all([1, -2, 3]) == 4\n    assert add_all([]) == 0\n"


def test_sandbox_runs_and_limits(tmp_path):
    assert run_tests({"solution.py": SOL, "test_solution.py": TEST})[0]
    assert not run_tests({"solution.py": SOL.replace("x > 0", "x >= 0").replace("+= x", "-= x"),
                          "test_solution.py": TEST})[0]
    ok, out = run_tests({"solution.py": "while True:\n    pass\n", "test_solution.py": TEST}, timeout=3)
    assert not ok and "timeout" in out
    ws = Workspace({"a.txt": "hello world"})
    try:
        assert ws.call("str_replace", {"path": "a.txt", "old_str": "world", "new_str": "there"}) == "edited a.txt"
        assert ws.read("a.txt") == "hello there"
        assert ws.call("read_file", {"path": "../../etc/passwd"}).startswith("error")
        assert "exit_code: 0" in ws.call("bash", {"command": "ls"})
        assert ws.call("str_replace", {"path": "a.txt", "old_str": "zzz", "new_str": ""}).startswith("error")
    finally:
        ws.cleanup()


def test_mutation_and_stub_break_tests():
    rng = random.Random(0)
    broken = [m for m in (mutate(SOL, rng) for _ in range(20)) if m]
    assert broken and any(not run_tests({"solution.py": b, "test_solution.py": TEST})[0] for b in broken)
    stub = make_stub(SOL)
    assert "NotImplementedError" in stub and not run_tests({"solution.py": stub, "test_solution.py": TEST})[0]


def test_agent_task_validation():
    code = [{"id": "c1", "source": "mbpp", "spec": "Sum positives", "canonical": SOL, "tests": TEST}] * 3
    tasks = make_agent_tasks(code, 3, 0.5, random.Random(1), "t")
    assert tasks and all(t["files"]["solution.py"] != SOL for t in tasks)


def test_xlam_and_augmentation():
    rows = [{"id": i, "query": f"weather in city {i}",
             "tools": json.dumps([{"name": f"weather_{i}", "description": "weather forecast lookup",
                                   "parameters": {"city": {"type": "str", "description": "c"},
                                                  "days": {"type": "int, optional", "default": 1}}}]),
             "answers": json.dumps([{"name": f"weather_{i}", "arguments": {"city": "Paris"}}])} for i in range(80)]
    rows += [{"id": 100 + i, "query": f"convert currency {i}",
              "tools": json.dumps([{"name": f"fx_{i}", "description": "currency exchange rates",
                                    "parameters": {"amount": {"type": "float"}}}]),
              "answers": json.dumps([{"name": f"fx_{i}", "arguments": {"amount": 3.5}}])} for i in range(80)]
    tasks = adapt_xlam(rows, "xlam")
    assert len(tasks) == 160
    schema = tasks[0]["tools"][0]["function"]["parameters"]
    assert schema["properties"]["days"]["type"] == "integer" and schema["required"] == ["city"]
    aug = augment_tool_tasks(tasks, random.Random(0), 0.3, (1, 2))
    cats = {t["category"] for t in aug}
    assert "irrelevance" in cats and "multiple" in cats
    own = {t["id"]: t["tools"][0]["function"]["name"] for t in tasks}
    for t in aug:
        if t["category"] == "irrelevance":
            orig = own[t["id"][: -len("-irr")]]
            assert t["gold_calls"] == [] and orig not in {x["function"]["name"] for x in t["tools"]}


def test_hermes_and_apigen_adapters():
    sysmsg = ("You are a function calling AI model. <tools> [{'type': 'function', 'function': {'name': 'get_stock', "
              "'description': 'price', 'parameters': {'type': 'object', 'properties': {'sym': {'type': 'string'}}, "
              "'required': ['sym']}}}] </tools> Use <tool_call> JSON.")
    row = {"id": "h1", "conversations": [
        {"from": "system", "value": sysmsg}, {"from": "human", "value": "AAPL price?"},
        {"from": "gpt", "value": "<tool_call>\n{'arguments': {'sym': 'AAPL'}, 'name': 'get_stock'}\n</tool_call>"},
        {"from": "tool", "value": "<tool_response>\n{'price': 1}\n</tool_response>"},
        {"from": "gpt", "value": "It is 1."}, {"from": "human", "value": "and MSFT?"},
        {"from": "gpt", "value": "<tool_call>\n{\"name\": \"get_stock\", \"arguments\": {\"sym\": \"MSFT\"}}\n</tool_call>"}]}
    tasks = adapt_hermes([row], "hermes")
    assert len(tasks) == 2 and tasks[1]["category"] == "multi_turn"
    assert tasks[1]["gold_calls"] == [{"name": "get_stock", "arguments": {"sym": "MSFT"}}]
    assert all(m["role"] != "system" for m in tasks[0]["messages"])
    assert tasks[1]["messages"][2]["role"] == "tool"
    mt = {"system": "policy", "tools": json.dumps([{"name": "find_user", "description": "d", "parameters": {
        "type": "object", "properties": {"id": {"type": "string"}}}}]), "conversations": [
        {"from": "human", "value": "hi I am u1"},
        {"from": "function_call", "value": json.dumps({"name": "find_user", "arguments": {"id": "u1"}})},
        {"from": "function_call", "value": json.dumps({"name": "find_user", "arguments": {"id": "u2"}})},
        {"from": "observation", "value": "ok"}, {"from": "gpt", "value": "found"}]}
    t2 = adapt_apigen_mt([mt], "apigen")
    assert len(t2) == 1 and len(t2[0]["gold_calls"]) == 2 and t2[0]["messages"][0]["content"] == "policy"


def test_mbpp_adapter_tests_run():
    r = {"task_id": 1, "text": "Sum positives", "code": SOL,
         "test_list": ["assert add_all([1, -2, 3]) == 4"], "test_setup_code": "", "challenge_test_list": []}
    t = adapt_mbpp([r], "mbpp")[0]
    assert run_tests({"solution.py": SOL, "test_solution.py": t["tests"]})[0]

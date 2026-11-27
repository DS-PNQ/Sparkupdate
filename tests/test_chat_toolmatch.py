from spark_kaggle.chat import (ChatFormatter, detect_tool_args_mode, format_tool_calls_xml, parse_tool_calls,
                               split_reasoning, template_variables)
from spark_kaggle.config import load_config
from spark_kaggle.toolmatch import score_tool_calls

TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {
        "city": {"type": "string"}, "days": {"type": "integer"}, "metric": {"type": "boolean", "default": True},
        "tags": {"type": "array"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "search", "parameters": {"type": "object", "properties": {
        "q": {"type": "string"}}, "required": ["q"]}}},
]


def test_parse_spark_xml_with_schema_types():
    text = ("Sure.<tool_call>get_weather\n<arg_key>city</arg_key><arg_value>Hà Nội</arg_value>\n"
            "<arg_key>days</arg_key><arg_value>3</arg_value>\n<arg_key>metric</arg_key><arg_value>false</arg_value>\n"
            "<arg_key>tags</arg_key><arg_value>[\"a\", 1]</arg_value>\n</tool_call>"
            "<tool_call>search<arg_key>q</arg_key><arg_value>null</arg_value></tool_call>")
    calls, rest, errors = parse_tool_calls(text, TOOLS)
    assert calls[0] == {"name": "get_weather", "arguments": {"city": "Hà Nội", "days": 3, "metric": False,
                                                             "tags": ["a", 1]}}
    assert calls[1] == {"name": "search", "arguments": {"q": None}}
    assert rest == "Sure." and not errors


def test_parse_unterminated_and_roundtrip():
    calls, _, errors = parse_tool_calls("<tool_call>search<arg_key>q</arg_key><arg_value>x</arg_value>", TOOLS)
    assert calls == [] and any("unterminated" in e for e in errors)
    gold = [{"name": "get_weather", "arguments": {"city": "Paris", "days": 2, "tags": ["x"]}}]
    calls, _, errors = parse_tool_calls(format_tool_calls_xml(gold), TOOLS)
    assert calls == gold and not errors


def test_split_reasoning_modes():
    assert split_reasoning("<think>plan</think>\nanswer", "think") == ("plan", "answer", True)
    assert split_reasoning("<think>still going", "think") == ("still going", "", False)
    assert split_reasoning("direct", "think") == ("", "direct", True)
    assert split_reasoning("still going", "think", prompt_opened_think=True)[2] is False
    assert split_reasoning("answer", "nothink") == ("", "answer", True)


def test_template_detection(tok, fmt):
    assert "enable_thinking" in template_variables(tok)
    assert detect_tool_args_mode(tok) == "dict"
    p_think = fmt.prompt([{"role": "user", "content": "hi"}], None, "think")
    p_no = fmt.prompt([{"role": "user", "content": "hi"}], None, "nothink")
    assert p_no.endswith("</think>") and not p_think.endswith("</think>")


def test_score_tool_calls():
    gold = [{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}}]
    ok = score_tool_calls([{"name": "get_weather", "arguments": {"city": " Paris", "days": 3.0, "metric": True}}],
                          gold, TOOLS)
    assert ok["exact"] and ok["reward"] == 1.0  # whitespace, int/float, default-valued extra arg
    bad = score_tool_calls([{"name": "get_weather", "arguments": {"city": "Paris", "days": 4}}], gold, TOOLS)
    assert not bad["exact"] and 0 < bad["reward"] < 1
    miss = score_tool_calls([{"name": "get_weather", "arguments": {"days": 3}}], gold, TOOLS)
    assert not miss["exact"]
    par_gold = gold + [{"name": "search", "arguments": {"q": "x"}}]
    par = score_tool_calls(list(reversed([{"name": "get_weather", "arguments": {"city": "Paris", "days": 3}},
                                          {"name": "search", "arguments": {"q": "x"}}])), par_gold, TOOLS)
    assert par["exact"]
    assert score_tool_calls([], [], TOOLS)["exact"]
    assert not score_tool_calls(gold, [], TOOLS)["exact"]
    assert not score_tool_calls(gold, gold, TOOLS, terminated=False)["exact"]


def test_config_interpolation_and_overrides(tmp_path):
    cfg = load_config(None, ["run_dir=/x/run", "sft.lr=3e-5", "lora.r=8"])
    assert cfg.sft.out_dir == "/x/run/adapters/sft" and cfg.sft.lr == 3e-5 and cfg.lora.r == 8
    assert cfg.dpo.init_adapters == ["/x/run/adapters/sft/final"]
    assert cfg.abliterate.source_model == cfg.scratch_dir + "/merged"


def test_formatter_assistant_message_roundtrip(fmt):
    parsed = fmt.parse("<think>r</think>ok<tool_call>search<arg_key>q</arg_key><arg_value>cats</arg_value>"
                       "</tool_call>", TOOLS, "think")
    msg = fmt.assistant_message(parsed)
    assert msg["reasoning_content"] == "r" and msg["tool_calls"][0]["function"]["arguments"] == {"q": "cats"}
    text = fmt.render([{"role": "user", "content": "find cats"}, msg], TOOLS, "think", add_generation_prompt=False)
    assert "<arg_key>q</arg_key><arg_value>cats</arg_value>" in text

"""Build a tiny random Llama with a Spark-X2.6-style tokenizer + chat template for CPU smoke tests.

The template reproduces Spark's turn markup from XHToken/LlamaFactory (`<|User|>`, `<|Bot|>`, `<|Tool|>`,
`<tool_response>`, bare `</think>` for no-think) and the spark25 XML-KV tool-call format. It is a stand-in:
the real checkpoint ships its own chat_template.jinja.

    python tests/tiny_spark.py /tmp/tiny_spark
"""

from __future__ import annotations

import sys
from pathlib import Path

BOS, EOS, SOS = "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜start▁of▁sentence｜>"
SPECIAL = [BOS, EOS, SOS, "<|System|>", "<|User|>", "<|Bot|>", "<|Tool|>", "<think>", "</think>", "<tool_call>",
           "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>", "<tool_response>",
           "</tool_response>", "<pad>"]

CHAT_TEMPLATE = (
    "{{- bos_token -}}"
    "{%- set ns = namespace(system='\\nyou are a helpful assistant.', last_user=-1) -%}"
    "{%- for m in messages %}{% if m.role == 'user' %}{% set ns.last_user = loop.index0 %}{% endif %}{% endfor -%}"
    "{%- if messages and messages[0].role == 'system' %}{% set ns.system = messages[0].content %}{% endif -%}"
    "<|System|>{{ ns.system }}"
    "{%- if tools %}\n\n# Tools\nAvailable functions:\n<tools>"
    "{%- for t in tools %}\n{{ t | tojson }}{%- endfor %}\n</tools>\n"
    "Call a function with:\n<tool_call>function_name\n<arg_key>key</arg_key><arg_value>value</arg_value>\n</tool_call>"
    "{%- endif %}" + EOS +
    "{%- for m in messages %}"
    "{%- if m.role == 'user' %}" + SOS + "<|User|>{{ m.content }}" + EOS + SOS + "<|Bot|>"
    "{%- elif m.role == 'assistant' %}"
    "{%- if loop.index0 > ns.last_user and m.reasoning_content %}<think>{{ m.reasoning_content }}</think>"
    "{%- else %}</think>{%- endif %}"
    "{{- m.content if m.content else '' }}"
    "{%- if m.tool_calls %}{%- for tc in m.tool_calls %}<tool_call>{{ tc.function.name }}\n"
    "{%- for k, v in tc.function.arguments.items() %}<arg_key>{{ k }}</arg_key><arg_value>"
    "{{- v if v is string else (v | tojson) }}</arg_value>\n{%- endfor %}</tool_call>{%- endfor %}{%- endif %}"
    + EOS +
    "{%- elif m.role == 'tool' %}" + SOS + "<|Tool|><tool_response>{{ m.content }}</tool_response>" + EOS + SOS
    + "<|Bot|>"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt and enable_thinking is defined and not enable_thinking %}</think>{%- endif %}"
)

CORPUS = [
    "def add(a, b):\n    return a + b\n", "The weather in Paris is sunny today.", "import os\nprint(os.getcwd())",
    "Search flights from SFO to JFK on 2024-05-01 for 2 passengers.", "You are a helpful assistant.",
    "python -m pytest -q test_solution.py", "exit_code: 0\n1 passed in 0.01s", "Fix the bug in solution.py",
    "{\"city\": \"Hanoi\", \"days\": 3, \"unit\": \"celsius\"}", "I'm sorry, but I can't help with that.",
] * 20


def build(out_dir: str, layers: int = 4, hidden: int = 64, vocab_size: int = 600) -> str:
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, trainer)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token=BOS, eos_token=EOS, pad_token="<pad>")
    tok.chat_template = CHAT_TEMPLATE
    tok.save_pretrained(out)

    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16384,
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg)
    model.generation_config.eos_token_id = tok.eos_token_id
    model.save_pretrained(out)
    return str(out)


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny_spark"))

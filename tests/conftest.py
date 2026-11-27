import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(scope="session")
def tiny_dir(tmp_path_factory):
    from tiny_spark import build

    return build(str(tmp_path_factory.mktemp("tiny_spark")))


@pytest.fixture(scope="session")
def tok(tiny_dir):
    from spark_kaggle.modeling import load_tokenizer

    return load_tokenizer(tiny_dir)


@pytest.fixture(scope="session")
def fmt(tok):
    from spark_kaggle.chat import ChatFormatter

    return ChatFormatter(tok, {})


@pytest.fixture()
def model(tiny_dir):
    from spark_kaggle.modeling import load_model

    return load_model(tiny_dir, device="cpu", dtype="float32", attn="eager")

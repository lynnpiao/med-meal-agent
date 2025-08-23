# tests/conftest.py
# -*- coding: utf-8 -*-
import importlib
import sys
from pathlib import Path
import pytest
import os 
from dotenv import load_dotenv

# 优先加载项目根目录下的 .env
load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env", override=False)

def pytest_configure(config):
    # 注册 live 标记，避免 UnknownMark 警告
    config.addinivalue_line("markers", "live: run tests that call real LLM endpoints")

def pytest_report_header(config):
    # 在测试报告头部打印一个可见提示，确认是否读到 KEY
    return f"OPENAI_API_KEY present: {bool(os.getenv('OPENAI_API_KEY'))}"


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_DIR = ROOT / "i18n" / "base_dir"

@pytest.fixture()
def zh_lexicon_mod():
    """
    重载 i18n.zh_lexicon，使用真实 base_dir 词库。
    """
    import i18n.zh_lexicon as zl
    zl = importlib.reload(zl)
    stats = zl.init_lexicons_from_json(str(BASE_DIR))
    assert stats["conditions"] > 0
    assert stats["allergens"] > 0
    assert stats["ingredients"] > 0
    return zl

def pytest_addoption(parser):
    parser.addoption(
        "--llm-live", action="store_true",
        help="run tests that hit real LLM endpoints"
    )

# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--llm-live"):
#         return
#     skip_marker = pytest.mark.skip(reason="use --llm-live to enable real LLM tests")
#     for item in items:
#         if "live" in item.keywords:  # 标记名统一用 live
#             item.add_marker(skip_marker)
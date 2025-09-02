# tests/test_services_vision_inventory.py
from pathlib import Path
import os
import pytest

from services.vision_inventory import scan_fridge_from_image

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
CANDIDATES = ["fridge_vision.jpg", "fridge_vision.jpeg", "fridge_vision.png"]

IMG_PATH = next((ASSETS_DIR / n for n in CANDIDATES if (ASSETS_DIR / n).exists()), None)

# 如果没有测试图，就跳过整个文件的收集，避免 FileNotFoundError
if IMG_PATH is None:
    pytest.skip(
        "Test image not found. Put a fridge photo at tests/assets/fridge_vision.jpg (or .jpeg/.png).",
        allow_module_level=True,
    )

FRIDGE_IMG_BYTES = IMG_PATH.read_bytes()


def _print_items(items, title: str):
    print("\n" + title)
    for it in items:
        print(f"- {it.name} {it.quantity} {it.unit.value}")

def test_ovd_smoke_prints():
    """基于 OWL-ViT 的开放词汇检测（零样本、文本类目）。仅打印，不作严格断言。"""
    try:
        import ultralytics  # noqa: F401
    except Exception:
        pytest.skip("ultralytics not installed; skipping ovd smoke test")

    items = scan_fridge_from_image(
        FRIDGE_IMG_BYTES, 
        backend="ovd", 
        language="zh")
    _print_items(items, "[OVD] Detected inventory")
    assert isinstance(items, list)  
    
    
# ==========================================================
# =============== LLM（single / consensus / tiled）=========
# ==========================================================
@pytest.mark.smoke
def test_llm_single_smoke_prints():
    """LLM 单次推理（需要 OPENAI_API_KEY）。仅打印，不作严格断言。"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM smoke test")

    items = scan_fridge_from_image(
        FRIDGE_IMG_BYTES,
        backend="llm",
        model="gpt-4o-mini",
        language='zh',
        mode="single",     # 单次、稳定
        debug=False,        # 也可不打印内部 debug，这里主要看我们自己的打印
    )
    _print_items(items, "[LLM single] Detected inventory")
    assert isinstance(items, list)


@pytest.mark.smoke
def test_llm_consensus_smoke_prints():
    """LLM 多次共识（需要 OPENAI_API_KEY）。打印每轮与共识。"""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set; skipping LLM smoke test")

    consensus, runs = scan_fridge_from_image(
        FRIDGE_IMG_BYTES,
        backend="llm",
        model="gpt-4o-mini",
        language='zh',
        mode="consensus",     # 同图多次 → 取中位数等规则聚合
        debug=False,          # 我们自己打印每一轮
        return_runs=True,     # 要回每一轮的结果
    )

    # 打印每一轮
    for i, r in enumerate(runs, 1):
        _print_items(r, f"[LLM consensus] Run #{i}")

    # 打印共识结果
    _print_items(consensus, "[LLM consensus] Aggregated")

    # 基本检查
    assert isinstance(runs, list) and len(runs) >= 1
    assert isinstance(consensus, list)





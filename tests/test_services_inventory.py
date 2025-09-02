# tests/test_services_inventory.py
# -*- coding: utf-8 -*-
import math
import pytest
import io
import os
import shutil
import sys
import traceback
import pytest

from services.inventory import (
    parse_inventory_line,
    scan_fridge_from_text,
    scan_fridge_from_lines,
    merge_inventory_items,
    scan_fridge_from_image_recipe
)
from models.schemas import Unit, InventoryItem


def _approx(a, b, rel=1e-3, abs_tol=1e-6):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_tol)


# ---------------------------
# 单行解析：中/英单位
# ---------------------------
def test_parse_inventory_line_cn_units():
    it = parse_inventory_line("西兰花 300g")
    assert it.name == "西兰花"
    assert it.unit == Unit.g
    assert _approx(it.quantity, 300.0)

    it = parse_inventory_line("牛奶 1 升")
    assert it.name == "牛奶"
    assert it.unit == Unit.ml
    assert _approx(it.quantity, 1000.0)

    it = parse_inventory_line("鸡蛋 半个")
    assert it.name == "鸡蛋"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 0.5)

    # “半两”= 25 g
    it = parse_inventory_line("花生  半两")
    assert it.name == "花生"
    assert it.unit == Unit.g
    assert _approx(it.quantity, 25.0)

    # 中文件数量词（pcs）
    it = parse_inventory_line("土豆 3袋")
    assert it.name == "土豆"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 3.0)
    
     # 中文件数量词（pcs）反过来
    it = parse_inventory_line("3袋 土豆")
    assert it.name == "土豆"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 3.0)


def test_parse_inventory_line_en_units():
    it = parse_inventory_line("olive oil 1 tbsp")
    assert it.name == "olive oil"
    assert it.unit == Unit.ml
    assert _approx(it.quantity, 15.0)

    # 用 0.5 cup（避免 1/2 分数解析差异）
    it = parse_inventory_line("rice 0.5 cup")
    assert it.name == "rice"
    assert it.unit == Unit.ml
    assert _approx(it.quantity, 120.0)  # 0.5 * 240 ml

    it = parse_inventory_line("flour 2 oz")
    assert it.name == "flour"
    assert it.unit == Unit.g
    assert _approx(it.quantity, 56.699)  # 2 * 28.3495
    
    it = parse_inventory_line("2 oz flour")
    assert it.name == "flour"
    assert it.unit == Unit.g
    assert _approx(it.quantity, 56.699)  # 2 * 28.3495
    

    it = parse_inventory_line("spinach 1 bunch")
    assert it.name == "spinach"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 1.0)

    it = parse_inventory_line("basil 2 leaves")
    assert it.name == "basil"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 2.0)


# ---------------------------
# 单行解析：x2/2x/仅数字
# ---------------------------
def test_parse_inventory_line_x2_and_num_only():
    it = parse_inventory_line("鸡蛋 x2")
    assert it.name == "鸡蛋"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 2.0)

    it = parse_inventory_line("2x 可乐")
    assert it.name == "可乐"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 2.0)

    it = parse_inventory_line("苹果 3")
    assert it.name == "苹果"
    assert it.unit == Unit.pcs
    assert _approx(it.quantity, 3.0)


# ---------------------------
# 文本块解析与合并
# ---------------------------
def test_scan_fridge_from_text_merge_and_convert():
    text = "鸡蛋 2个；鸡蛋 x2；鸡蛋 1 个；牛奶 200 毫升；牛奶 1 cup；"
    items = scan_fridge_from_text(text)

    # 应有两类：鸡蛋（pcs 合并 2+2+1=5），牛奶（ml 合并 200+240=440）
    eggs = [it for it in items if it.name == "鸡蛋" and it.unit == Unit.pcs]
    milk = [it for it in items if it.name == "牛奶" and it.unit == Unit.ml]

    assert len(eggs) == 1
    assert _approx(eggs[0].quantity, 5.0)

    assert len(milk) == 1
    assert _approx(milk[0].quantity, 440.0)
    
    
    text = "2个 鸡蛋；2x 鸡蛋；1 个 鸡蛋；200毫升 牛奶；1 cup 牛奶；"
    items = scan_fridge_from_text(text)

    # 应有两类：鸡蛋（pcs 合并 2+2+1=5），牛奶（ml 合并 200+240=440）
    eggs = [it for it in items if it.name == "鸡蛋" and it.unit == Unit.pcs]
    milk = [it for it in items if it.name == "牛奶" and it.unit == Unit.ml]

    assert len(eggs) == 1
    assert _approx(eggs[0].quantity, 5.0)

    assert len(milk) == 1
    assert _approx(milk[0].quantity, 440.0)
    
     
    text = "鸡蛋 2个；2x 鸡蛋；1 个 鸡蛋；牛奶 200毫升；1 cup 牛奶；"
    items = scan_fridge_from_text(text)

    # 应有两类：鸡蛋（pcs 合并 2+2+1=5），牛奶（ml 合并 200+240=440）
    eggs = [it for it in items if it.name == "鸡蛋" and it.unit == Unit.pcs]
    milk = [it for it in items if it.name == "牛奶" and it.unit == Unit.ml]

    assert len(eggs) == 1
    assert _approx(eggs[0].quantity, 5.0)

    assert len(milk) == 1
    assert _approx(milk[0].quantity, 440.0)
    

def test_scan_fridge_from_lines():
    lines = [
        "tomatoes 2 cups",
        "tomatoes 1 cup",
        "洋葱 半个",
        "洋葱 0.5个",
    ]
    items = scan_fridge_from_lines(lines)

    # tomatoes → 2 cups + 1 cup = 3 cups = 720 ml
    toms = [it for it in items if it.name == "tomatoes" and it.unit == Unit.ml]
    assert len(toms) == 1
    assert _approx(toms[0].quantity, 720.0)

    onions = [it for it in items if it.name == "洋葱" and it.unit == Unit.pcs]
    assert len(onions) == 1
    assert _approx(onions[0].quantity, 1.0)  # 0.5 + 0.5

    lines = [
        "2 cups tomatoes",
        "1 cup tomatoes",
        "半个 洋葱",
        "0.5个 洋葱",
    ]
    items = scan_fridge_from_lines(lines)

    # tomatoes → 2 cups + 1 cup = 3 cups = 720 ml
    toms = [it for it in items if it.name == "tomatoes" and it.unit == Unit.ml]
    assert len(toms) == 1
    assert _approx(toms[0].quantity, 720.0)

    onions = [it for it in items if it.name == "洋葱" and it.unit == Unit.pcs]
    assert len(onions) == 1
    assert _approx(onions[0].quantity, 1.0)  # 0.5 + 0.5

    lines = [
        "tomatoes 2 cups",
        "1 cup tomatoes",
        "洋葱 半个",
        "0.5个 洋葱",
    ]
    items = scan_fridge_from_lines(lines)

    # tomatoes → 2 cups + 1 cup = 3 cups = 720 ml
    toms = [it for it in items if it.name == "tomatoes" and it.unit == Unit.ml]
    assert len(toms) == 1
    assert _approx(toms[0].quantity, 720.0)

    onions = [it for it in items if it.name == "洋葱" and it.unit == Unit.pcs]
    assert len(onions) == 1
    assert _approx(onions[0].quantity, 1.0)  # 0.5 + 0.5

def test_merge_inventory_items_rounding():
    # 同名同单位合并，并四舍五入到 3 位
    items = [
        InventoryItem(name="Salt", quantity=0.3334, unit=Unit.g),
        InventoryItem(name="salt", quantity=0.3334, unit=Unit.g),
        InventoryItem(name="salt", quantity=0.3334, unit=Unit.g),
    ]
    merged = merge_inventory_items(items)
    assert len(merged) == 1
    assert merged[0].name == "salt"
    assert merged[0].unit == Unit.g
    # 0.3334 * 3 = 1.0002 → round(…, 3) = 1.0
    assert _approx(merged[0].quantity, 1.0, abs_tol=1e-3)


# ---------------------------
# （可选）图片 → OCR → 清单
# 若缺少依赖/本机未装 tesseract，则自动跳过
# ---------------------------
def _tesseract_ready() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    return shutil.which("tesseract") is not None

def _render_lines(lines, font=None, size=(800, 280)):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(im)
    y = 24
    for line in lines:
        draw.text((24, y), line, fill="black", font=font or ImageFont.load_default())
        y += 80
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    from pathlib import Path
    Path("debug").mkdir(exist_ok=True)
    im.save("debug/ocr_input.png")   # 保存你画的原图
    return buf.getvalue()

@pytest.mark.skipif(not _tesseract_ready(), reason="tesseract 不可用：请安装可执行文件并确保在 PATH 中")
def test_scan_recipe_image_english_tesseract():
    # 仅英文：不需要中文字体
    img_bytes = _render_lines([
        "eggs 3",
        "milk 200 ml",
    ])

    # 只用 tesseract，语言设为 eng；psm=6 + oem=1
    items = scan_fridge_from_image_recipe(
        img_bytes,
        ocr="tesseract",
        tesseract_lang="eng",
        config_str="--oem 1 --psm 6",
    )

    # 结果放宽校验：识别至少一条；且包含 eggs(pcs) 或 milk(ml)
    assert isinstance(items, list) and len(items) >= 1
    names = {(it.name.lower(), it.unit.value) for it in items}
    assert ("eggs", "pcs") in names or ("milk", "ml") in names

@pytest.mark.skipif(not _tesseract_ready(), reason="tesseract 不可用：请安装并加入 PATH")
def test_scan_recipe_image_chinese_tesseract():
    from PIL import ImageFont
    # 需要中文字体：通过环境变量 CN_FONT 指定 .ttf/.otf/.ttc
    font_path = os.getenv("CN_FONT")
    if not font_path:
        pytest.skip("未找到可用的中文字体，设置环境变量 CN_FONT 指向字体文件")

    try:
        font = ImageFont.truetype(font_path, 36)
    except Exception:
        pytest.skip(f"中文字体加载失败：{font_path}")

    img_bytes = _render_lines([
        "鸡蛋 3个",
        "牛奶 200 ml",
    ], font=font)

    # chi_sim 语言数据可能未安装；若抛错则跳过并给出提示
    try:
        items = scan_fridge_from_image_recipe(
            img_bytes,
            ocr="tesseract",
            tesseract_lang="chi_sim+eng",
            config_str="--oem 1 --psm 6",
        )
    except Exception as e:
        msg = str(e)
        if "chi_sim" in msg or "traineddata" in msg:
            pytest.skip("未安装 chi_sim 语言包。macOS 可参考：将 chi_sim.traineddata 放到 $(brew --prefix)/share/tessdata/")
        raise

    assert isinstance(items, list) and len(items) >= 1
    names = {(it.name, it.unit.value) for it in items}
    assert ("鸡蛋", "pcs") in names or ("牛奶", "ml") in names
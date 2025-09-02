# services/inventory.py
from __future__ import annotations
import io
import os
import re
import base64
from typing import List, Dict, Tuple, Optional, Literal

from i18n.units import parse_qty_unit, _normalize_line_for_units        
from models.schemas import InventoryItem, Unit


# ---- 可选依赖（本地 OCR & 预处理）----
try:
    import pytesseract  # 可选：pip install pytesseract
except Exception:
    pytesseract = None

try:
    import cv2  # 可选：pip install opencv-python
except Exception:
    cv2 = None

try:
    from PIL import Image, ImageOps, ImageFilter  # pip install pillow
except Exception:
    Image = None


__all__ = [
    "parse_inventory_line",
    "scan_fridge_from_text",
    "scan_fridge_from_lines",
    "scan_fridge_from_image_recipe",
    "merge_inventory_items",
]


# ------------------------------
# 单行解析：返回 InventoryItem
# ------------------------------
def parse_inventory_line(line: str) -> InventoryItem:
    """
    解析一行如：
      - '鸡蛋 6个' / '西兰花 300g' / '牛奶 1 L' / '洋葱 半个' / '大蒜 3 瓣'
      - 'tomatoes 2 cups' / 'olive oil 1 tbsp' / 'rice 1/2 cup' / 'flour 2 oz'
      - '鸡蛋 x2' / '2x 可乐' / '苹果 3'
      - 且支持反向（“数量 单位 名称”，如 “3袋 土豆 / 200 ml 牛奶 / 200ml 牛奶”）
      - 正反混用时也能正确合并与换算
    输出统一为：g / ml / pcs 的 InventoryItem，并保持 name 原样（仅去首尾空白）。
    """
    if not line or not line.strip():
        return InventoryItem(name="", quantity=0.0, unit=Unit.g)

    # 预处理让仅数字/“x2/2x”可落到 pcs
    s = _normalize_line_for_units(line)

    # 复用 i18n.units.parse_qty_unit（已支持中英单位、半个、中文量词等）
    name, qty, base = parse_qty_unit(s)

    try:
        unit = Unit(base)
    except Exception:
        # 极端兜底（理论上不会到这里，因为 parse_qty_unit 已经只返回 g/ml/pcs）
        unit = Unit.g

    return InventoryItem(name=name.strip(), quantity=float(qty), unit=unit)


# ------------------------------
# 文本块解析：返回合并去重后的清单
# ------------------------------
def scan_fridge_from_text(text: str) -> List[InventoryItem]:
    """
    输入例：
      '鸡蛋 6个；西兰花 300g；牛奶 1L；番茄 2 个；洋葱 0.5个；大蒜 3 瓣；'
      'tomatoes 2 cups; olive oil 1 tbsp; rice 1/2 cup'
    切分符：中文/英文分号、句号、换行、顿号。
    返回：同名同单位合并后的 InventoryItem 列表（数量四舍五入到 3 位）
    """
    chunks = re.split(r"[；;。.\n、]+", text or "")
    items: List[InventoryItem] = []

    for ch in chunks:
        ch = ch.strip()
        if not ch:
            continue
        it = parse_inventory_line(ch)
        # 空名/零量跳过
        if not it.name or it.quantity <= 0:
            continue
        items.append(it)

    return merge_inventory_items(items)


def scan_fridge_from_lines(lines: List[str]) -> List[InventoryItem]:
    """
    当 OCR/前端把每个条目已经拆成行时使用。
    """
    items: List[InventoryItem] = []
    for ln in (lines or []):
        it = parse_inventory_line(ln)
        if not it.name or it.quantity <= 0:
            continue
        items.append(it)
    return merge_inventory_items(items)


# ------------------------------
# 合并去重
# ------------------------------
def merge_inventory_items(items: List[InventoryItem]) -> List[InventoryItem]:
    """
    按 (name.lower().strip(), unit) 合并数量；数量保留 3 位小数。
    """
    agg: Dict[tuple, float] = {}
    for it in items:
        key = (it.name.strip().lower(), it.unit.value)
        agg[key] = agg.get(key, 0.0) + float(it.quantity)

    out: List[InventoryItem] = []
    for (name, unit), qty in agg.items():
        out.append(InventoryItem(name=name, quantity=round(qty, 3), unit=Unit(unit)))
    return out


# ==============================
#        图片 → OCR → 清单
# ==============================
def _pil_preprocess(img_bytes: bytes) -> "Image.Image":
    """
    PIL 预处理：转灰度、增强对比度、轻度锐化、二值化。
    对收据/冰箱拍照的文字有一定帮助（不依赖 OpenCV）。
    """
    if Image is None:
        raise RuntimeError("Pillow 未安装，请 pip install pillow")
    im = Image.open(io.BytesIO(img_bytes))
    im = ImageOps.exif_transpose(im)  # 处理拍照方向
    im = im.convert("L")              # 灰度
    im = ImageOps.autocontrast(im, cutoff=2)
    im = im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=150, threshold=3))
    # 简单阈值化（对收据背景有效）
    im = im.point(lambda p: 255 if p > 180 else 0)
    return im


def _ocr_tesseract(img_bytes: bytes, lang: str = "chi_sim+eng", config: str="--oem 1 --psm 6" ) -> str:
    """
    本地 OCR：优先使用（稳定、无额度依赖）。
    """
    if pytesseract is None or Image is None:
        raise RuntimeError("pytesseract 或 pillow 未安装。可 pip install pytesseract pillow")
    im = _pil_preprocess(img_bytes)
    text = pytesseract.image_to_string(im, lang=lang, config=config)
    return text or ""


def _ocr_openai_vision(image_bytes: bytes, model: str = "gpt-4o-mini") -> str:
    """
    备选：OpenAI 视觉 OCR。需要 OPENAI_API_KEY。
    只输出“每行一条 食材 数量 单位”的纯文本；忽略价格/店名/时间等非食材信息。
    """
    from openai import OpenAI  # 仅在需要时导入
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY 未设置，无法使用 OpenAI 视觉 OCR。")

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    client = OpenAI()
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "这是一张收据/冰箱食材照片。请仅抄录食材行项目为纯文本，每行一种："
                        "格式尽量为“名称 数量 单位”，例如：'鸡蛋 6 个'、'西兰花 300 g'、'olive oil 1 tbsp'。"
                        "忽略价格、店名、日期、条形码、联系电话等非食材信息。不要解释。"
                    ),
                },
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}" }},
            ],
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0
    )
    txt = resp.choices[0].message.content if resp and resp.choices else ""
    return txt or ""


def scan_fridge_from_image_recipe(
    image_bytes: bytes,
    *,
    ocr: Literal["auto", "tesseract", "llm"] = "auto",
    tesseract_lang: str = "chi_sim+eng",
    config_str: str="--oem 1 --psm 6",
    llm_model: str = "gpt-4o-mini"
) -> List[InventoryItem]:
    """
    将图片直接转为清单：
    1) ocr='tesseract' ：仅用本地 OCR（推荐，稳定）
    2) ocr='llm'       ：仅用 OpenAI 视觉（需要额度）
    3) ocr='auto'      ：优先 tesseract，失败则回退 llm
    """
    text = ""
    err_tess: Optional[Exception] = None
    err_llm: Optional[Exception] = None

    if ocr in ("tesseract", "auto"):
        try:
            text = _ocr_tesseract(image_bytes, lang=tesseract_lang, config=config_str)
        except Exception as e:
            err_tess = e
            if ocr == "tesseract":
                raise

    if not text and ocr in ("llm", "auto"):
        try:
            text = _ocr_openai_vision(image_bytes, model=llm_model)
        except Exception as e:
            err_llm = e
            if ocr == "llm":
                raise

    if not text:
        # 两个后端都失败，抛出最可诊断的错误
        if err_tess and err_llm:
            raise RuntimeError(f"OCR 失败：tesseract={err_tess}；llm={err_llm}")
        raise RuntimeError("OCR 失败：未获得任何文本。")

    return scan_fridge_from_text(text)
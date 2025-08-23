from typing import Tuple, Optional
import re

# —— 实验室单位换算 ——（统一到 mg/dL、ng/mL 等）
def glucose_mgdl_from_mmol(glu_mmol_l: float) -> float:
    return glu_mmol_l * 18.0

def chol_mgdl_from_mmol(chol_mmol_l: float) -> float:
    return chol_mmol_l * 38.67

def tg_mgdl_from_mmol(tg_mmol_l: float) -> float:
    return tg_mmol_l * 88.57

def creat_mgdl_from_umol(creat_umol_l: float) -> float:
    return creat_umol_l / 88.4

def uric_mgdl_from_umol(uric_umol_l: float) -> float:
    return uric_umol_l / 59.48

def vitd_ngml_from_nmol(vitd_nmol_l: float) -> float:
    return vitd_nmol_l / 2.496

# —— 中文数量单位 → 统一到 g/ml/pcs ——（MVP 规则）
ZH_UNIT_TO_BASE = {
    "克": ("g", 1.0),
    "g": ("g", 1.0),
    "千克": ("g", 1000.0), "公斤": ("g", 1000.0), "kg": ("g", 1000.0),
    "斤": ("g", 500.0),
    "两": ("g", 50.0),
    "毫升": ("ml", 1.0), "ml": ("ml", 1.0),
    "升": ("ml", 1000.0), "l": ("ml", 1000.0),
    "个": ("pcs", 1.0), "枚": ("pcs", 1.0), "只": ("pcs", 1.0), "块": ("pcs", 1.0),
}

def parse_zh_qty_unit(line: str) -> Tuple[str, float, str]:
    """
    输入示例：'西兰花 300克' / '鸡蛋 2个' / '牛奶 200 ml' / '鸡胸肉 半斤'
    返回：(中文名称, 统一数量, 统一单位[g/ml/pcs])
    """
    s = line.strip()
    # 半斤、半两等
    s = s.replace("半斤", "0.5 斤").replace("半两", "0.5 两")

    m = re.search(r"(.+?)\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z\u4e00-\u9fa5]+)", s)
    if not m:
        # 没解析到数量/单位，默认 0 g
        return (s, 0.0, "g")

    name = m.group(1).strip()
    qty = float(m.group(2))
    raw_unit = m.group(3).strip().lower()

    if raw_unit in ZH_UNIT_TO_BASE:
        base_unit, factor = ZH_UNIT_TO_BASE[raw_unit]
        qty_std = qty * factor
        return (name, qty_std, base_unit)

    # 未知单位 → 默认 g
    return (name, qty, "g")

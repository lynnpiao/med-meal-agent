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
    "升": ("ml", 1000.0), "l": ("ml", 1000.0)
}

# —— 中文“件数”量词：统一按 pcs 处理（每个=1 件）
PCS_TOKENS = {"个","枚","颗","只","袋","盒","罐","根","片","块","杯","串","条","朵","把","头","粒","瓶"}
ZH_UNIT_TO_BASE.update({u: ("pcs", 1.0) for u in PCS_TOKENS})

# —— 常见中文体积
ZH_UNIT_TO_BASE.update({
    "茶匙": ("ml", 5.0),
    "小勺": ("ml", 5.0),
    "汤匙": ("ml", 15.0),
    "大勺": ("ml", 15.0),
    "杯":   ("ml", 240.0),  # 如果你把“杯”归入 pcs，就别开这行
})

# —— 英文厨房计量单位 → 统一到 g/ml/pcs（带单复数/缩写；全部小写）——
# 约定：cup=240 ml（美制），tbsp=15 ml，tsp=5 ml；
# oz 默认按“重量盎司”换算为克（28.3495 g）；若遇到 fl oz（液体盎司）再换算为 ml（29.5735 ml）
EN_UNIT_TO_BASE = {
    # 体积勺
    "tbsp": ("ml", 15.0), "tbs": ("ml", 15.0), "tbsp.": ("ml", 15.0),
    "tablespoon": ("ml", 15.0), "tablespoons": ("ml", 15.0),
    "tsp": ("ml", 5.0), "tsp.": ("ml", 5.0),
    "teaspoon": ("ml", 5.0), "teaspoons": ("ml", 5.0),

    # 杯
    "cup": ("ml", 240.0), "cups": ("ml", 240.0),

    # 份
    "serving": ("pcs", 1.0), "servings": ("pcs", 1.0),

    # 蒜瓣/整头
    "clove": ("pcs", 1.0), "cloves": ("pcs", 1.0),
    "head": ("pcs", 1.0), "heads": ("pcs", 1.0),

    # 把/束
    "bunch": ("pcs", 1.0), "bunches": ("pcs", 1.0),

    # 片
    "slice": ("pcs", 1.0), "slices": ("pcs", 1.0),

    # 罐
    "can": ("pcs", 1.0), "cans": ("pcs", 1.0),

    # 少许
    "dash": ("ml", 0.62), "dashes": ("ml", 0.62),      # ~ 1/8 tsp
    "pinch": ("ml", 0.31), "pinches": ("ml", 0.31),    # ~ 1/16 tsp

    # 一把（近似，视食材而定；MVP 给个体积量便于换算）
    "handful": ("ml", 60.0), "handfuls": ("ml", 60.0), # ~ 1/4 cup 的中间值

    # 半个（表达成 0.5 个）
    "halve": ("pcs", 0.5), "halves": ("pcs", 0.5),

    # 叶
    "leaf": ("pcs", 1.0), "leaves": ("pcs", 1.0),
    "leave": ("pcs", 1.0),  # 常见拼写错误，等同 leaf

    # 重量
    "oz": ("g", 28.3495), "ounce": ("g", 28.3495), "ounces": ("g", 28.3495),
    "lb": ("g", 453.592), "lbs": ("g", 453.592),
    "pound": ("g", 453.592), "pounds": ("g", 453.592),

    # 液体盎司（显式写法）
    "fl oz": ("ml", 29.5735), "floz": ("ml", 29.5735), "fl-oz": ("ml", 29.5735),
}

EN_UNIT_TO_BASE.update({
    "pcs": ("pcs", 1.0), "piece": ("pcs", 1.0), "pieces": ("pcs", 1.0),
    "bag": ("pcs", 1.0), "bags": ("pcs", 1.0),
    "bottle": ("pcs", 1.0), "bottles": ("pcs", 1.0),
    "carton": ("pcs", 1.0), "cartons": ("pcs", 1.0),
    "slice": ("pcs", 1.0), "slices": ("pcs", 1.0),
    "bunch": ("pcs", 1.0), "bunches": ("pcs", 1.0),
    "can": ("pcs", 1.0), "cans": ("pcs", 1.0),
    "container": ("pcs", 1.0), "containers": ("pcs", 1.0),
})


# # 英文单位 → 中文名（全部小写键）
# EN_UNIT_ZH_NAME = {
#     "tbsp": "汤匙(大勺)", "tbs": "汤匙(大勺)", "tbsp.": "汤匙(大勺)",
#     "tablespoon": "汤匙(大勺)", "tablespoons": "汤匙(大勺)",
#     "tsp": "茶匙(小勺)", "tsp.": "茶匙(小勺)",
#     "teaspoon": "茶匙(小勺)", "teaspoons": "茶匙(小勺)",
#     "cup": "杯", "cups": "杯",
#     "serving": "份", "servings": "份",
#     "clove": "瓣", "cloves": "瓣",
#     "head": "头", "heads": "头",
#     "bunch": "把/束", "bunches": "把/束",
#     "slice": "片", "slices": "片",
#     "can": "罐", "cans": "罐",
#     "dash": "少许", "dashes": "少许",
#     "pinch": "少许(一撮)", "pinches": "少许(一撮)",
#     "handful": "一把(约60ml)", "handfuls": "一把(约60ml)",
#     "halve": "半个", "halves": "半个",
#     "leaf": "叶", "leaves": "叶", "leave": "叶",
#     "oz": "盎司(重)", "ounce": "盎司(重)", "ounces": "盎司(重)",
#     "lb": "磅", "lbs": "磅", "pound": "磅", "pounds": "磅",
#     "fl oz": "液体盎司", "floz": "液体盎司", "fl-oz": "液体盎司",
# }

# ===== 行文本预规范化（供库存解析用）=====
# 支持解析分数
_FRACTION_MAP = {"½": 0.5, "¼": 0.25, "¾": 0.75}

def _parse_number(num_str: str) -> float:
    """解析数量：
       - 支持 '1 1/2'、'1/2'、'½/¼/¾'、'1.5' 等。
    """
    s = num_str.strip()

    # Unicode 分数（可能是 "1 ½"）
    for sym, val in _FRACTION_MAP.items():
        if sym in s and not re.search(r"\d+/\d+", s):
            parts = s.split()
            if len(parts) == 2 and parts[1] in _FRACTION_MAP:
                return float(parts[0]) + _FRACTION_MAP[parts[1]]
            return float(val)

    # 形如 "1 1/2"
    m = re.match(r"^\s*(\d+)\s+(\d+)/(\d+)\s*$", s)
    if m:
        return float(m.group(1)) + float(m.group(2)) / float(m.group(3))

    # 形如 "1/2"
    m = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", s)
    if m:
        return float(m.group(1)) / float(m.group(2))

    # 普通小数/整数
    return float(s)

# ========= 预编译通用正则 =========

# 支持：整数/小数/真分数/Unicode 分数（½¼¾）
_NUM_RE  = r"(?:\d+(?:\.\d+)?|\d+\s+\d+/\d+|\d+/\d+|[½¼¾])"
_NUM_ONLY_RE = re.compile(rf"^\s*(.+?)\s+({_NUM_RE})\s*$")
_X2_RIGHT_RE = re.compile(r"^\s*(.+?)\s*[x×]\s*(\d+(?:\.\d+)?)\s*$", flags=re.IGNORECASE)
_X2_LEFT_RE  = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[x×]\s*(.+?)\s*$", flags=re.IGNORECASE)


# ===== 构造“单位候选”的正则 ========
def _build_unit_alt_pattern() -> str:
    """用已知单位词表构造 alternation；含 fl oz 多写法，按长度降序避免 'l' 抢先匹配 'ml'。"""
    zh_units = set(ZH_UNIT_TO_BASE.keys())
    en_units = set(EN_UNIT_TO_BASE.keys())

    # 多词/变体（不转义，保留空格/点/连字符）
    specials = [
        r"fl\.?\s*-?\s*oz",       # fl oz / fl. oz / fl-oz / floz
        r"fluid\s+ounce(?:s)?",   # fluid ounce(s)
    ]
    escaped = sorted((re.escape(u) for u in (zh_units | en_units)), key=len, reverse=True)
    return r"(?:%s)" % "|".join(specials + escaped)

_UNIT_ALT = _build_unit_alt_pattern()

# 反向：qty unit name（空格版 + 紧邻版）
_REV_SP   = re.compile(rf"^\s*(?P<qty>{_NUM_RE})\s*(?P<unit>{_UNIT_ALT})\s*(?P<name>.+?)\s*$", re.IGNORECASE)
_REV_TIGHT= re.compile(rf"^\s*(?P<qty>{_NUM_RE})(?P<unit>{_UNIT_ALT})(?P<name>.+?)\s*$", re.IGNORECASE)

# 正向：name qty unit（空格版 + 紧邻版）
_FWD_SP   = re.compile(rf"^(?P<name>.+?)\s*(?P<qty>{_NUM_RE})\s*(?P<unit>{_UNIT_ALT})\s*$", re.IGNORECASE)
_FWD_TIGHT= re.compile(rf"^(?P<name>.+?)\s*(?P<qty>{_NUM_RE})(?P<unit>{_UNIT_ALT})\s*$", re.IGNORECASE)

# “名称 半{件数量词}”
_HALF_NAME_RE = re.compile(rf"^\s*(.+?)\s*半([{''.join(PCS_TOKENS)}])\s*$")

# 名称 + 可选数字（兜底：只有名称或“名称 数字”）
_NAME_NUM_OPT = re.compile(r"^(?P<name>[\s\S]*?)\s*(?P<num>\d+(?:\.\d+)?)?\s*$")

def _norm_en_unit_key(u: str) -> str:
    # 统一小写、去句点、折叠空白
    u = (u or "").strip().lower().replace(".", "")
    u = re.sub(r"\s+", " ", u)
    return u

def _normalize_line_for_units(line: str) -> str:
    """把“鸡蛋 3 / 鸡蛋 x2 / 2x 可乐”标准化成 “<name> <num> 个”，便于落到 pcs。"""
    s = (line or "").strip()
    if not s:
        return s
    m = _NUM_ONLY_RE.match(s)
    if m:
        name, num = m.group(1).strip(), m.group(2)
        return f"{name} {num} 个"
    m = _X2_RIGHT_RE.match(s)
    if m:
        name, num = m.group(1).strip(), m.group(2)
        return f"{name} {num} 个"
    m = _X2_LEFT_RE.match(s)
    if m:
        num, name = m.group(1), m.group(2).strip()
        return f"{name} {num} 个"
    return s


def _match_unit(raw_unit: str):
    """
    统一做单位查表/归一：
    返回 (base_unit, factor) 或 None。
    - 先查中文单位（含紧凑写法）
    - PCS 量词：直接 ('pcs', 1.0)
    - 再查英文单位（含 fl oz 变体）
    """
    ru = (raw_unit or "").strip().lower()
    ru_compact = ru.replace(" ", "")

    if ru in ZH_UNIT_TO_BASE:
        return ZH_UNIT_TO_BASE[ru]
    if ru_compact in ZH_UNIT_TO_BASE:
        return ZH_UNIT_TO_BASE[ru_compact]

    # “小袋/大瓶”等后缀或列在 PCS_TOKENS 里 → pcs
    if (ru_compact in PCS_TOKENS) or any(ru_compact.endswith(tok) for tok in PCS_TOKENS):
        return ("pcs", 1.0)

    key = _norm_en_unit_key(ru)
    if key in {"floz", "fl oz", "fl-oz", "fluid ounce", "fluid ounces"}:
        return EN_UNIT_TO_BASE["fl oz"]
    if key in EN_UNIT_TO_BASE:
        return EN_UNIT_TO_BASE[key]

    return None

# def _match_as_pcs(u: str) -> bool:
#     """中文单位后缀是否像件数：如 '小袋'、'大瓶' 等。"""
#     return any(u.endswith(tok) for tok in PCS_TOKENS)

def parse_qty_unit(line: str, default_unit: str = "g") -> Tuple[str, float, str]:
    """
    等价于你现有实现，但性能更好、结构更清晰：
      1) 处理“半*”（半斤/半两/半个/半袋/半盒）
      2) 反向：数量 + 单位 + 名称（支持 200 ml 牛奶 / 200ml牛奶 / 3袋 土豆 / 3袋土豆）
      3) 正向：名称 + 数量 + 单位（支持 牛奶 200 ml / 牛奶 200ml）
      4) 兜底：只有名称或“名称 数字”→ 默认 pcs
    统一输出： (name, quantity_in_base, base_unit in {'g','ml','pcs'})
    """
    s = (line or "").strip()
    if not s:
        return ("", 0.0, default_unit)

    # “半*”归一（半斤/半两 + 半{件数量词}）
    s = s.replace("半斤", "0.5 斤").replace("半两", "0.5 两")
    for u in PCS_TOKENS:
        s = s.replace(f"半{u}", f"0.5 {u}")

    m_half = _HALF_NAME_RE.match(line or "")
    if m_half:
        name = m_half.group(1).strip()
        return (name, 0.5, "pcs")

    # ① 反向：qty unit name
    m = _REV_SP.match(s) or _REV_TIGHT.match(s)
    if m:
        qty = _parse_number(m.group("qty"))
        unit = m.group("unit")
        name = m.group("name").strip()
        mu = _match_unit(unit)
        if mu:
            base, k = mu
            return (name, qty * k, base)
        return (name, qty, default_unit)

    # ② 正向：name qty unit
    m = _FWD_SP.match(s) or _FWD_TIGHT.match(s)
    if m:
        name = m.group("name").strip()
        qty  = _parse_number(m.group("qty"))
        unit = m.group("unit")
        mu = _match_unit(unit)
        if mu:
            base, k = mu
            return (name, qty * k, base)
        return (name, qty, default_unit)

    # ③ 兜底：只有“名称”或“名称 数字” → 默认 pcs；若仅“名称”数量取 1
    m = _NAME_NUM_OPT.match(s)
    if m:
        name = (m.group("name") or "").strip()
        qty  = _parse_number(m.group("num")) if m.group("num") else 1.0
        return (name, qty, "pcs")

    # 理论到不了
    return (s, 0.0, default_unit)

# 向后兼容：旧函数名
def parse_zh_qty_unit(line: str) -> Tuple[str, float, str]:
    return parse_qty_unit(line)

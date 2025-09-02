"""
services/vision_inventory.py
============================

目的
----
从“冰箱内部”照片中抽取结构化的食材清单：`List[InventoryItem]`，
数量与单位最终归一到 `g/ml/pcs`（见 `i18n.units.parse_qty_unit`）。

当前支持的后端
--------------
1) OVD（开放词汇检测：OWL-ViT）
   - 零样本目标检测：把一组英文类目（默认来自内置词表 + i18n/ingredients.json）作为查询词
   - 对检测框按分数阈值 `score` 过滤，并按类目进行计数（单位恒为 `pcs`）
   - 可选 `language="zh"` 时，名称会通过映射表显示为中文（内部计算仍使用英文）

2) LLM（视觉 LLM 行文本解析）
   - 让模型输出逐行的“名称 数量 单位”文本（中/英提示词）
   - 预处理：去行首编号（如 "1)"/"•"）、英文单位→中文单位（如 pcs/bag/bottle… → 个/袋/瓶）
   - 标准化 & 解析：`_normalize_line_for_units` → `parse_qty_unit` → `g/ml/pcs`
   - `mode="single"`：单次、快速；`mode="consensus"`：多次取中位数，更稳（成本更高）

核心入口
--------
- `scan_fridge_from_image(image_bytes, backend="ovd" | "llm", **kwargs)`
  顶层 API，返回 `List[InventoryItem]`，或在 `return_runs=True` 时返回
  `(final_items, [run1_items, run2_items, ...])`（仅 LLM）。

常用参数
--------
- OVD:
  - `labels: Optional[List[str]]`  查询词；默认 `_default_food_labels()`（含 i18n/ingredients.json 扩展）
  - `score: float = 0.25`          置信度阈值
  - `language: str = "zh"`         返回中文/英文名称显示

- LLM:
  - `model: str = "gpt-4o-mini"`   模型名（需要环境变量 `OPENAI_API_KEY`）
  - `language: str = "zh"`         提示词语言（影响 LLM 输出风格）
  - `mode: "single" | "consensus"` 单次或多次共识
  - `seed: Optional[int] = 1`      固定采样（若 SDK 支持）
  - `labels: Optional[List[str]]`  可选词表，用于一致性聚合时的轻度名称对齐
  - `debug: bool = False`          打印标准化与解析后的结构化行
  - `return_runs: bool = False`    返回每轮中间结果（仅 LLM）

依赖与资源
----------
- OVD：`transformers` 会首次下载 OWL-ViT 权重（~数百 MB）
- LLM：`langchain-openai`，需要 `OPENAI_API_KEY`
- 解析：`i18n.units.parse_qty_unit` 会将单位标准化到 `g/ml/pcs`
- 解析：`i18n.units.parse_qty_unit` 会将单位标准化到 `g/ml/pcs`
"""
from __future__ import annotations
import base64
import io
from typing import List, Dict, Optional
from pathlib import Path
from functools import lru_cache
import json
import re
from PIL import Image                 

from models.schemas import InventoryItem, Unit
from services.inventory import merge_inventory_items
from i18n.units import parse_qty_unit, _normalize_line_for_units  # 你已暴露的工具

# -----------------------------------------------------------------------------
# 行首编号 / 项目符号剥离
# 说明：
# - 一些 LLM 英文输出会在每行前自动加“1) / 2. / • / (1)”等编号或项目符号
# - 为与其他条件下的输出保持一致，我们将对其进行剥离
# - 正则各分支涵盖：数字 + )/./、/冒号；(1) / （1）；a)；i)/iv)；项目符号 • - – —
# -----------------------------------------------------------------------------
_LEADING_INDEX_RE = re.compile(
    r"""^\s*(
        # 1) 纯数字序号 + 右括号/点/顿号/冒号： 1)  1.  1、  1:
        \d{1,3}\s*[\)\.、:：-]
        |
        # 2) 括号数字： (1) （1）  1）
        [\(\（]\s*\d{1,3}\s*[\)\）]
        |
        # 3) 小写字母 a)  b. 等（偶发）
        [a-zA-Z]\)
        |
        # 4) 罗马数字 i)  iv) 等（极少见）
        [ivxlcdm]{1,5}\)
        |
        # 5) 常见项目符号：• - – —
        [•\-\–\—]
    )\s*""",
    re.IGNORECASE | re.VERBOSE,
)

def _strip_leading_index(s: str) -> str:
    """去掉行首的编号/项目符号（只影响 *行首*，不会误伤 'watermelon 1 slice' 这种）"""
    return _LEADING_INDEX_RE.sub("", (s or "").strip())


# =============================================================================
#                     默认“冰箱食材”类目 & 词表扩展
# =============================================================================
def _extract_canonical_from_ingredients(obj) -> List[str]:
    """
    从 ingredients.json 中尽可能提取 canonical 英文名。
    兼容结构：
      1) {"entries": [{"canonical_en": "...", ...}, ...]}
      2) {"ingredients": [{"canonical_en"/"en"/"name_en"/"name": "..."}]}
      3) 顶层就是 list[dict/str]
      4) 简单映射 dict（如 zh->en 或 en->zh），尽量抓到英文值/键
    """
    out: List[str] = []

    def add(n: str | None):
        if not n:
            return
        s = str(n).strip()
        if s:
            out.append(s)

    def looks_english(s: str) -> bool:
        return any("a" <= c.lower() <= "z" for c in s)

    if isinstance(obj, dict):
        if isinstance(obj.get("entries"), list):
            for it in obj["entries"]:
                if isinstance(it, dict):
                    add(it.get("canonical_en") or it.get("en") or it.get("name_en") or it.get("name"))
        elif isinstance(obj.get("ingredients"), list):
            for it in obj["ingredients"]:
                if isinstance(it, dict):
                    add(it.get("canonical_en") or it.get("en") or it.get("name_en") or it.get("name"))
        else:
            # 映射字典（如 zh->en / en->zh）
            for k, v in obj.items():
                if isinstance(v, str) and looks_english(v):
                    add(v)
                if isinstance(k, str) and looks_english(k):
                    add(k)
    elif isinstance(obj, list):
        for it in obj:
            if isinstance(it, dict):
                add(it.get("canonical_en") or it.get("en") or it.get("name_en") or it.get("name"))
            elif isinstance(it, str) and it.strip():
                add(it)

    # 去重（不区分大小写，保序）
    seen = set()
    deduped: List[str] = []
    for s in out:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    return deduped

@lru_cache(maxsize=1)
def _default_food_labels() -> List[str]:
    """
    返回默认食材词表：
    - 优先包含常见生鲜 & 肉蛋奶；兼容包装容器类（便于后续 OCR/条码细化）
    - 若存在 i18n/base_dir/ingredients.json，则将其中的 canonical_en 追加到词表（去重）
    """
    labels = [
        # 生鲜常见
        "egg", "milk", "yogurt", "cheese", "butter", "tofu",
        "broccoli", "cucumber", "tomato", "bell pepper", "carrot", "mushroom",
        "lettuce", "spinach", "grape", "orange", "apple", "banana", "watermelon",
        # 肉蛋奶海鲜
        "chicken", "beef", "pork", "fish", "shrimp",
        # 包装类（便于与 OCR/条码二次细化）
        "bottle", "carton", "cup",
    ]
    seen = {s.lower() for s in labels}

    # 改为从 i18n/base_dir/ingredients.json 读取
    json_path = Path(__file__).resolve().parents[1] / "i18n" / "base_dir" / "ingredients.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            extra = _extract_canonical_from_ingredients(data)
            for c in extra:
                if c and c.lower() not in seen:
                    labels.append(c)
                    seen.add(c.lower())
        except Exception:
            # 读/解析失败：忽略，返回基础列表
            pass

    return labels
    
# =============================================================================
#                   OWL-ViT（开放词汇，零样本检测，支持中文显示）
# =============================================================================
@lru_cache(maxsize=1)
def _en2zh_map() -> Dict[str, str]:
    """
    构造 英文 → 中文 的显示映射：
    - 优先使用 i18n/base_dir/ingredients.json 的 zh_aliases（取第一个非空）
    - 同时兼容 en_aliases
    - 缺失时用 fallback 兜底
    说明：仅用于“显示语言=中文”时的名称转换，内部计算仍用英文类目。
    """
    mapping: Dict[str, str] = {}

    # 1) 从 ingredients.json 读取
    json_path = Path(__file__).resolve().parents[1] / "i18n" / "base_dir" / "ingredients.json"
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            entries = []
            if isinstance(data, dict):
                entries = data.get("entries") or data.get("ingredients") or []
            elif isinstance(data, list):
                entries = data
            # 统一处理 entries
            if isinstance(entries, list):
                for it in entries:
                    if not isinstance(it, dict):
                        continue
                    en = it.get("canonical_en") or it.get("en") or it.get("name_en") or it.get("name")
                    zh_aliases = it.get("zh_aliases") or it.get("zh") or []
                    if isinstance(zh_aliases, str):
                        zh_aliases = [zh_aliases]
                    en_aliases = it.get("en_aliases") or []
                    if isinstance(en_aliases, str):
                        en_aliases = [en_aliases]

                    def _choose_zh(arr):
                        for z in arr or []:
                            zs = str(z).strip()
                            if zs:
                                return zs
                        return None

                    zh = _choose_zh(zh_aliases)
                    if en and zh:
                        mapping[en.strip().lower()] = zh
                    for a in en_aliases:
                        if a and zh:
                            mapping[str(a).strip().lower()] = zh
        except Exception:
            pass

    # 2) 常见名词 fallback（仅在 JSON 缺失时使用）
    fallback = {
        "milk": "牛奶",
        "yogurt": "酸奶",
        "greek yogurt": "希腊酸奶",
        "cheese": "奶酪",
        "butter": "黄油",
        "egg": "鸡蛋",
        "eggs": "鸡蛋",
        "tofu": "豆腐",
        "broccoli": "西兰花",
        "cucumber": "黄瓜",
        "tomato": "番茄",
        "bell pepper": "彩椒",
        "carrot": "胡萝卜",
        "mushroom": "蘑菇",
        "lettuce": "生菜",
        "spinach": "菠菜",
        "grape": "葡萄",
        "orange": "橙子",
        "apple": "苹果",
        "banana": "香蕉",
        "watermelon": "西瓜",
        "chicken": "鸡肉",
        "beef": "牛肉",
        "pork": "猪肉",
        "fish": "鱼",
        "shrimp": "虾",
        "bottle": "瓶",
        "carton": "纸盒",
        "cup": "杯",
        "potato": "土豆",
        "pear": "梨",
        "tomatoes": "番茄",
        "grapes": "葡萄",
    }
    # 仅补不存在的键
    for k, v in fallback.items():
        mapping.setdefault(k.lower(), v)

    return mapping

def _en_to_zh(name_en: str) -> str:
    """将英文标签转为中文显示，若未知则原样返回。"""
    if not name_en:
        return name_en
    return _en2zh_map().get(name_en.strip().lower(), name_en)

@lru_cache(maxsize=1)
def _load_owlv2():
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
    model.eval()
    return proc, model

def _scan_with_ovd(
    image_bytes: bytes,
    labels: Optional[List[str]] = None,
    score: float = 0.25,
    language: str = "zh",   # 新增语言参数，默认中文
) -> List[InventoryItem]:
    """
    使用 OWL-ViT（开放词汇零样本）进行检测：
    - 将 _default_food_labels() 作为查询词（可覆盖 labels）
    - 对每个检测框按 score 阈值过滤，按类目计数
    - 显示名称根据 language 可切换中文/英文，内部始终以 pcs 计数
    """
    import torch
    import io
    if not labels:
        labels = _default_food_labels()

    proc, model = _load_owlv2()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = proc(text=[labels], images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])  # (h, w)
    results = proc.post_process_object_detection(outputs=outputs, target_sizes=target_sizes)[0]
    scores, labels_idx = results["scores"], results["labels"]

    counts: Dict[str, int] = {}
    for s, li in zip(scores, labels_idx):
        if s.item() < score:
            continue
        name = labels[int(li)]
        # 根据需求输出中文或英文
        disp = _en_to_zh(name) if language.lower().startswith("zh") else name
        counts[disp] = counts.get(disp, 0) + 1

    items = [InventoryItem(name=n, quantity=float(c), unit=Unit.pcs) for n, c in counts.items()]
    return merge_inventory_items(items)

# =============================================================================
#                                视觉 LLM（行文本解析）
# =============================================================================
def _canon_name(name: str) -> str:
    return (name or "").strip().lower()

def _build_allowed_names(labels: Optional[List[str]] = None, cap: int = 256) -> List[str]:
    """构造稳定的可选名称列表：去重、小写去重、按字母序、截断到 cap。"""
    lab = labels or _default_food_labels()
    seen = set()
    out = []
    for x in sorted(lab, key=lambda s: s.lower()):
        k = x.strip()
        if not k:
            continue
        lk = k.lower()
        if lk in seen:
            continue
        seen.add(lk)
        out.append(k)
    return out[:cap]

# # 英文/中文里常见的单位词尾（小写比较）
# # === 放在本文件现有工具函数旁 ===
_UNIT_TOKENS = (
    r"(?:pcs|g|ml|bottle|carton|cup|head|clove|bag|bags|bunch|slice|can|"
    r"瓣|个|個|枚|颗|顆|袋|盒|罐|根|片|块|杯|串|条|朵|把|头|粒|瓶)"
)

# def _dedupe_qty_unit_tail(line: str) -> str:
#     """
#     若一行里出现 >=2 次 “<num> <unit>” 片段，保留第一次并裁掉后续内容。
#     例：
#       'yogurt 4 pcs 1.0 pcs'  -> 'yogurt 4 pcs'
#       'grapes 1 bag 1.0 pcs'  -> 'grapes 1 bag'
#     """
#     s = (line or "").strip()
#     if not s:
#         return s
#     pat = re.compile(rf"(\d+(?:\.\d+)?)\s*{_UNIT_TOKENS}\b", flags=re.IGNORECASE)
#     matches = list(pat.finditer(s))
#     if len(matches) >= 2:
#         end = matches[0].end()
#         return s[:end].strip()
#     return s

def _strip_fake_unit_from_name(name: str) -> str:
    """
    剪掉 name 末尾误拼的“<num> <unit>”，避免把单位/数量留在名称里。
    如：'oranges 6 pcs' -> 'oranges'
    """
    s = (name or "").strip()
    s = re.sub(rf"\s*\d+(?:\.\d+)?\s*{_UNIT_TOKENS}\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _align_to_allowed(name: str, allowed: list[str]) -> str:
    """
    名称轻量对齐到允许词表：
    - 完全匹配（不区分大小写）直接返回词表形式
    - 否则尝试“包含关系”宽松对齐（'cherry tomatoes' ~ 'tomato'）
    - 仅在一致性聚合（consensus）流程里用于“轻度归一名词”
    """
    if not name:
        return name
    nl = name.lower()
    amap = {a.lower(): a for a in allowed}
    if nl in amap:
        return amap[nl]
    for a in allowed:
        al = a.lower()
        if nl in al or al in nl:
            return a
    return name

def _fmt_items(items: List[InventoryItem]) -> str:
    """调试打印用：以统一的结构化格式输出项目行。"""
    return "\n".join(f"- {it.name} {it.quantity} {it.unit.value}" for it in items)

# === 单次 LLM 识别 ===
def _scan_with_llm_once(image_bytes: bytes,
                   model: str = "gpt-4o-mini",
                   language: str = "zh",
                   labels: Optional[List[str]] = None,
                   seed: Optional[int] = 1) -> List[InventoryItem]:
    """
    单次 LLM 视觉理解：
    1) 让模型输出逐行的“名称 数量 单位”文本（中/英提示）
    2) 行级预处理：去行首编号（1)、•）、必要时可启用尾巴去重
    3) 调用 _normalize_line_for_units：给非显式数量的行补成“1 个”，并统一一些“x2/2x”等写法
    4) parse_qty_unit：解析成 (name, qty, base_unit[g/ml/pcs])，失败则兜底为 pcs
    5) merge_inventory_items：同名同单位合并，最后输出结构化清单
    """
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
    except Exception as e:
        raise RuntimeError(
            "LLM 后端不可用。请先 `pip install langchain-openai` 并设置 OPENAI_API_KEY。"
        ) from e

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatOpenAI(model=model, temperature=0)

    if language == "zh":
        instr = (
            "你现在在看一张【冰箱内部】的照片。请列出你能清楚识别到的食材（生鲜或包装食品）。\n"
            "要求：\n"
            "1) 逐行输出，每行格式：名称 数量 单位（例如：鸡蛋 6 个 / 牛奶 1 瓶 / 西兰花 1 颗）。\n"
            "2) 不要编造；不确定就不要写。尽量使用 pcs（个/瓶/袋/盒/颗/瓣/把/束/头…），液体可用 ml，粉类可用 g。\n"
            "3) 只输出清单本身，不要解释；不要输出任何个人信息或二维码内容。\n"
        )
    else:
        instr = (
            "You are looking at a photo of the inside of a fridge.\n"
            "List food items you can clearly identify.\n"
            "1) One per line: name quantity unit (e.g., eggs 6 pcs / milk 1 bottle / broccoli 1 head)\n"
            "2) Do not guess. Prefer pcs for discrete items; ml for liquids; g for powders.\n"
            "3) Output list only. No explanations. No PII.\n"
        )

    content = [
        {"type": "text", "text": instr},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
    ]
    text = llm.invoke([HumanMessage(content=content)]).content or ""

    lines = [ln.strip(" -•\t").strip() for ln in text.splitlines() if ln.strip()]
    items: List[InventoryItem] = []
    for ln in lines:
        ln = _strip_leading_index(ln)
        
        # ln = _dedupe_qty_unit_tail(ln)
        
        normalized = (
            _normalize_line_for_units(ln) if any(ch.isdigit() for ch in ln)
            else _normalize_line_for_units(f"{ln} 1 个")
        )
        
        # print(normalized)
        
        name, qty, base = parse_qty_unit(normalized)
        try:
            u = Unit(base)
        except Exception:
            u = Unit.pcs
        
        if name and qty > 0:
            it = InventoryItem(name=name, quantity=float(qty), unit=u)
            # print(f"[LLM parse] name={it.name!r}, quantity={it.quantity}, unit={it.unit.value}")
            items.append(it)

    return merge_inventory_items(items)

# =============================================================================
#                         多次/切块的一致性聚合（可选）
# =============================================================================

def aggregate_consensus(runs: List[List[InventoryItem]],
                        debug: bool = False,
                        allowed: Optional[List[str]] = None) -> List[InventoryItem]:
    """
    将多次 LLM 结果做一致性聚合：
    - 先对每轮结果做轻度名称对齐（_align_to_allowed），再合并
    - 按 (name, unit) 聚合数量，取中位数作为最终数量（pcs 四舍五入整数；g/ml 保留一位）
    - debug=True 时打印每轮与最终聚合结果
    """
    from statistics import median
    from collections import defaultdict
    allowed = allowed or _build_allowed_names(None)
    cleaned_runs: List[List[InventoryItem]] = []

    for i, items in enumerate(runs, 1):
        fixed = []
        for it in items:
            nm = _align_to_allowed(_strip_fake_unit_from_name(it.name), allowed)
            fixed.append(InventoryItem(name=nm, quantity=float(it.quantity), unit=it.unit))
        fixed = merge_inventory_items(fixed)
        fixed.sort(key=lambda it: (it.name.lower(), it.unit.value))
        cleaned_runs.append(fixed)
        if debug:
            print(f"[LLM run #{i}]")
            print(_fmt_items(fixed))

    buckets: Dict[tuple, List[float]] = defaultdict(list)
    for items in cleaned_runs:
        for it in items:
            key = (_canon_name(it.name), it.unit.value)
            buckets[key].append(float(it.quantity))

    out: List[InventoryItem] = []
    for (name, unit), arr in buckets.items():
        if not arr: 
            continue
        q = median(arr)
        q = round(q) if unit == "pcs" else round(q, 1)
        out.append(InventoryItem(name=name, quantity=q, unit=Unit(unit)))

    out.sort(key=lambda it: (it.name.lower(), it.unit.value))
    if debug:
        print("[LLM consensus]")
        print(_fmt_items(out))
    return out

def scan_fridge_with_llm_consensus(image_bytes: bytes,
                                   n_runs: int = 3,
                                   model: str = "gpt-4o-mini",
                                   language: str = "zh",
                                   labels: Optional[List[str]] = None,
                                   seed: Optional[int] = 1,
                                   debug: bool = False,
                                   return_runs: bool = False):
    """
    同一张图多次调用 LLM，做一致性聚合（中位数）：
    - n_runs 次单次识别 -> aggregate_consensus -> 返回最终/或最终+历史
    - 注意：此模式增加调用次数，成本更高；只有需要更稳时才使用
    """
    allowed = _build_allowed_names(labels)
    runs = [
        _scan_with_llm_once(image_bytes, model=model, language=language, labels=allowed, seed=seed)
        for _ in range(max(1, n_runs))
    ]
    consensus = aggregate_consensus(runs, debug=debug, allowed=allowed)
    return (consensus, runs) if return_runs else consensus

# =============================================================================
#                               统一入口（LLM / OVD）
# =============================================================================

def _scan_with_llm(image_bytes: bytes,
                   model: str = "gpt-4o-mini",
                   labels: Optional[List[str]] = None,
                   language: str = "zh",
                   seed: Optional[int] = 1,
                   mode: str = "single",
                   debug: bool = False,
                   return_runs: bool = False):
    """
    LLM 后端统一入口：
    - mode='single'   ：单次解析（便宜、快、足够稳定）
    - mode='consensus'：多次解析 + 中位数聚合（更稳，成本更高）
    - debug=True      ：打印结构化清单
    - return_runs=True：返回 (最终输出, [每轮输出])
    """
    if mode == "single":
        items = _scan_with_llm_once(image_bytes, model=model, language=language, labels=labels, seed=seed)
        if debug:
            print("[LLM single]")
            print(_fmt_items(items))
        return (items, [items]) if return_runs else items
    elif mode == "consensus":
        return scan_fridge_with_llm_consensus(
            image_bytes, n_runs=3, model=model, language=language, labels=labels,
            seed=seed, debug=debug, return_runs=return_runs
        )
    else:
        raise ValueError("mode 仅支持 'single' | 'consensus' ")


def scan_fridge_from_image(image_bytes: bytes,
                           backend: str = "llm",
                           **kwargs):
    """
    顶层统一 API：
      - backend='ovd'：使用 OWL-ViT 零样本检测（无需联网；需本地加载模型）
          * 参数：labels（可选）、score（阈值，默认 0.25）、language（显示语言，默认 'zh'）
      - backend='llm'：使用视觉 LLM（默认 single）
          * 参数：model、language、labels（可选轻度对齐）、seed、mode('single'/'consensus')、
                 debug（打印）、return_runs（返回每轮结果）
    返回：List[InventoryItem] 或 (List[InventoryItem], List[List[InventoryItem]])（当 return_runs=True）
    """
    if backend == "ovd":
        return _scan_with_ovd(
            image_bytes,
            labels=kwargs.get("labels"),
            score=kwargs.get("score", 0.25),
            language=kwargs.get("language", "zh"), 
        )
    elif backend == "llm":
        return _scan_with_llm(
            image_bytes,
            model=kwargs.get("model", "gpt-4o-mini"),
            language=kwargs.get("language", "zh"),  
            labels=kwargs.get("labels"),
            seed=kwargs.get("seed", 1),
            mode=kwargs.get("mode", "single"),
            debug=kwargs.get("debug", False),
            return_runs=kwargs.get("return_runs", False),
        )
    else:
        raise ValueError("backend 仅支持 'ovd' | 'llm'")
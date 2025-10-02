# i18n/zh_lexicon.py
# -*- coding: utf-8 -*-
"""
双向中英词典（conditions / allergens / ingredients），基于 JSON 词表加载。
特性：
- 支持 zh→en / en→zh 双向映射
- 支持同义词/别名、简繁/全角归一
- 可加载多个 JSON 并自动合并/去重
- 支持拼音兜底与可选模糊匹配（rapidfuzz）
- 规范英文名 canonical_en 作为内部统一键；可附带 code, category 字段

JSON 文件格式示例（每类一个或多个 JSON 文件）：
{
  "entries": [
    {
      "canonical_en": "hypertension",
      "zh_aliases": ["高血压", "高血壓", "血压高"],
      "en_aliases": ["htn"],
      "code": "I10",
      "category": "cardio"
    }
  ]
}
"""

from __future__ import annotations
from typing import Dict, List, Optional
import os
import re
import glob
import json
from i18n.utils import normalize_zh

# 可选：模糊匹配库（未安装也可跑）
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None
    fuzz = None

# 可选：pypinyin 用于拼音兜底（未安装也可跑）
try:
    from pypinyin import lazy_pinyin
except Exception:
    def lazy_pinyin(s):  # 简易兜底：逐字符“拼音”，保证不报错
        return [c for c in (s or "")]


# -----------------------
# 规范化工具
# -----------------------
# def normalize_zh(s: str) -> str:
#     """
#     中文字符串规范化：
#     - 去首尾空格、转小写
#     - 全角转半角（含全角引号 → ASCII 引号）
#     - 删除常见标点（包含 ASCII " 和 '）
#     - 压缩多空格
#     - 若含中文字符，移除所有空格（OCR 常见把汉字分隔开）
#     """
#     s = (s or "").strip().lower()

#     # 全角→半角
#     def _dbc2sbc(ch: str) -> str:
#         code = ord(ch)
#         if code == 0x3000:     # 全角空格
#             return " "
#         if 0xFF01 <= code <= 0xFF5E:
#             return chr(code - 0xFEE0)
#         return ch

#     s = "".join(_dbc2sbc(ch) for ch in s)

#     # 删除常见标点（新增 ASCII 引号 " '）
#     s = re.sub(r'[，。、“”‘’"\'！（）()【】\[\]{}：:；;·\-—_、/\\]', " ", s)

#     # 压缩多空格
#     s = re.sub(r"\s+", " ", s).strip()

#     # 若包含中文字符，把空格全部去掉，避免 OCR/手输造成的“分词”
#     if re.search(r"[\u4e00-\u9fff]", s):
#         s = s.replace(" ", "")

#     return s


def normalize_en(s: str) -> str:
    """
    英文字符串规范化：去空格、转小写、压缩多空格。
    """
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def pinyin_key(s: str) -> str:
    """
    将中文词条转成拼音串，便于做拼音级别的等价匹配。
    未安装 pypinyin 时，使用逐字符兜底，保证稳定性。
    """
    return "".join(lazy_pinyin(s or ""))


# -----------------------
# 词条结构
# -----------------------
class LexEntry:
    """
    单个词条：规范英文名 + 中英文别名 + 可选 code / category
    """
    def __init__(
        self,
        canonical_en: str,
        zh_aliases: List[str] | None = None,
        en_aliases: List[str] | None = None,
        code: Optional[str] = None,
        category: Optional[str] = None,
    ):
        self.canonical_en = canonical_en
        self.zh_aliases = [z for z in (zh_aliases or []) if z]
        self.en_aliases = [e for e in (en_aliases or []) if e]
        self.code = code
        self.category = category


class Bidict:
    """
    双向/多别名词典：支持 zh→en / en→zh, 含同义词与兜底匹配。
    - self.zh2en: 规范化中文/中文别名 → 规范英文名
    - self.en2zh: 规范英文名/英文别名 → 规范中文主名
    """
    def __init__(self):
        self.entries: List[LexEntry] = []
        self.zh2en: Dict[str, str] = {}
        self.en2zh: Dict[str, str] = {}

    def _rebuild(self) -> None:
        self.zh2en.clear()
        self.en2zh.clear()
        for e in self.entries:
            en_key = normalize_en(e.canonical_en)
            # 反向映射用第一个中文别名作为默认展示名
            zh_main = normalize_zh(e.zh_aliases[0]) if e.zh_aliases else None
            if zh_main:
                self.en2zh[en_key] = zh_main
            # 中文别名 → 规范英文
            for zh in e.zh_aliases:
                self.zh2en[normalize_zh(zh)] = en_key
            # 英文别名 → 规范英文，同时给 en→zh 一个中文展示
            for al in e.en_aliases:
                en_al = normalize_en(al)
                if zh_main and en_al not in self.en2zh:
                    self.en2zh[en_al] = zh_main
                # 允许英文别名通过 zh2en_lookup 命中
                self.zh2en[en_al] = en_key

    def add_or_update(self, new: LexEntry) -> None:
        """
        合并或新增词条：
        - 若已存在相同 canonical_en，则合并别名/补充元信息；
        - 否则追加新词条。
        """
        nk = normalize_en(new.canonical_en)
        for e in self.entries:
            if normalize_en(e.canonical_en) == nk:
                e.zh_aliases = sorted(set(e.zh_aliases + new.zh_aliases))
                e.en_aliases = sorted(set(e.en_aliases + new.en_aliases))
                if not e.code and new.code:
                    e.code = new.code
                if not e.category and new.category:
                    e.category = new.category
                self._rebuild()
                return
        self.entries.append(new)
        self._rebuild()

    # ---------- 加载 JSON ----------
    def load_json(self, path: str) -> int:
        """
        从单个 JSON 文件加载词条，返回加载条数。
        JSON 格式：{"entries": [ {canonical_en, zh_aliases, en_aliases, code, category}, ... ]}
        """
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cnt = 0
        for row in payload.get("entries", []):
            cen = (row.get("canonical_en") or "").strip()
            if not cen:
                continue
            self.add_or_update(
                LexEntry(
                    canonical_en=cen,
                    zh_aliases=row.get("zh_aliases", []) or [],
                    en_aliases=row.get("en_aliases", []) or [],
                    code=row.get("code"),
                    category=row.get("category"),
                )
            )
            cnt += 1
        return cnt

    def load_json_dir(self, dir_path: str, pattern: str = "*.json") -> int:
        """
        批量加载目录下的多个 JSON 文件（例如 base.json + custom.json),返回累计条数。
        """
        n = 0
        for fp in glob.glob(os.path.join(dir_path, pattern)):
            n += self.load_json(fp)
        return n

    # ---------- 查找 ----------
    def zh2en_lookup(self, s: str, fuzzy: bool = True, pinyin_fallback: bool = True) -> str:
        """
        中文（或英文别名）→ 规范英文名。
        1) 直接命中
        2) 拼音兜底（pypinyin 可用时更准；否则逐字符兜底）
        3) 模糊匹配兜底（需 rapidfuzz）
        全部失败时，返回规范化后的小写 key，让上游决定如何处理。
        """
        key = normalize_zh(s)
        # 1) 直接命中
        if key in self.zh2en:
            return self.zh2en[key]

        # 2) 拼音兜底
        if pinyin_fallback and self.zh2en:
            pk = pinyin_key(key)
            for zh_alias, en in self.zh2en.items():
                if pinyin_key(zh_alias) == pk:
                    return en

        # 3) 模糊匹配兜底
        if fuzzy and process and self.zh2en:
            cand = process.extractOne(key, list(self.zh2en.keys()), scorer=fuzz.WRatio)
            if cand and cand[1] >= 88:
                return self.zh2en[cand[0]]

        # 4) 兜底：返回规范化英文（此处是 key 本身的小写），上游可再做 LLM 翻译或人工确认
        return key

    def en2zh_lookup(self, s: str, fuzzy: bool = False) -> str:
        """
        英文（或英文别名）→ 中文主名（若无中文别名则回传原文）。
        可选模糊匹配。
        """
        key = normalize_en(s)
        if key in self.en2zh:
            return self.en2zh[key]
        if fuzzy and process and self.en2zh:
            cand = process.extractOne(key, list(self.en2zh.keys()), scorer=fuzz.WRatio)
            if cand and cand[1] >= 88:
                return self.en2zh[cand[0]]
        return s


# -----------------------
# 三个大类词典实例
# -----------------------
CONDITIONS = Bidict()
ALLERGENS = Bidict()
INGREDIENTS = Bidict()


def init_lexicons_from_json(base_dir: str = "i18n") -> dict:
    """
    应用启动时调用一次，加载：
      base_dir/conditions.json
      base_dir/allergens.json
      base_dir/ingredients.json
    你也可以手动多次调用 load_json / load_json_dir 加载 base + custom 多份文件。
    """
    n1 = CONDITIONS.load_json(os.path.join(base_dir, "conditions.json"))
    n2 = ALLERGENS.load_json(os.path.join(base_dir, "allergens.json"))
    n3 = INGREDIENTS.load_json(os.path.join(base_dir, "ingredients.json"))
    return {"conditions": n1, "allergens": n2, "ingredients": n3}


# 统一 API（业务侧调用）
def map_condition_zh2en(s: str) -> str:
    return CONDITIONS.zh2en_lookup(s)


def map_allergy_zh2en(s: str) -> str:
    return ALLERGENS.zh2en_lookup(s)


def map_ing_zh2en(s: str) -> str:
    return INGREDIENTS.zh2en_lookup(s)


def map_ing_en2zh(s: str) -> str:
    return INGREDIENTS.en2zh_lookup(s)


__all__ = [
    "LexEntry",
    "Bidict",
    "CONDITIONS",
    "ALLERGENS",
    "INGREDIENTS",
    "init_lexicons_from_json",
    "map_condition_zh2en",
    "map_allergy_zh2en",
    "map_ing_zh2en",
    "map_ing_en2zh",
    "normalize_zh",
    "normalize_en",
    "pinyin_key",
]



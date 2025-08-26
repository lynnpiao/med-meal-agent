# i18n/profile_enum_map.py
## -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Literal

from models.schemas import Sex, ActivityLevel  # 直接用你的枚举，避免拼写错

def _normalize_zh(s: str) -> str:
    import re
    s = (s or "").strip().lower()
    # 全角转半角
    def _dbc2sbc(ch):
        code = ord(ch)
        if code == 0x3000:
            return " "
        if 0xFF01 <= code <= 0xFF5E:
            return chr(code - 0xFEE0)
        return ch
    s = "".join(_dbc2sbc(ch) for ch in s)
    # 去常见标点与多空格
    s = re.sub(r"[，。、“”‘’！（）()【】\[\]{}：:；;·\-—_、/\\]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# —— 性别映射 ——（含常见同义词）
_SEX_ZH2EN = {
    "男": Sex.male, "男性": Sex.male, "男生": Sex.male, "男的": Sex.male,
    "女": Sex.female, "女性": Sex.female, "女生": Sex.female, "女的": Sex.female,
    # 英文/缩写兜底
    "male": Sex.male, "m": Sex.male, "boy": Sex.male,
    "female": Sex.female, "f": Sex.female, "girl": Sex.female,
}

# —— 活动水平映射 ——（含同义词/语义接近词）
_ACTIVITY_ZH2EN = {
    # sedentary
    "久坐": ActivityLevel.sedentary, "静止": ActivityLevel.sedentary, "不运动": ActivityLevel.sedentary,
    "少运动": ActivityLevel.sedentary, "坐班": ActivityLevel.sedentary, "办公室": ActivityLevel.sedentary,
    "辦公室": ActivityLevel.sedentary, "低活动": ActivityLevel.sedentary, "低活動": ActivityLevel.sedentary,
    "sedentary": ActivityLevel.sedentary,
    # light
    "轻度": ActivityLevel.light, "輕度": ActivityLevel.light, "轻体力": ActivityLevel.light, "輕體力": ActivityLevel.light,
    "轻量": ActivityLevel.light, "日常走动": ActivityLevel.light, "日常走動": ActivityLevel.light,
    "light": ActivityLevel.light,
    # moderate
    "中等": ActivityLevel.moderate, "中度": ActivityLevel.moderate, "适中": ActivityLevel.moderate, "適中": ActivityLevel.moderate,
    "一般": ActivityLevel.moderate, "moderate": ActivityLevel.moderate,
    # active
    "活跃": ActivityLevel.active, "活躍": ActivityLevel.active, "经常运动": ActivityLevel.active, "經常運動": ActivityLevel.active,
    "较高": ActivityLevel.active, "較高": ActivityLevel.active, "重体力": ActivityLevel.active, "重體力": ActivityLevel.active,
    "active": ActivityLevel.active,
    # very_active
    "非常活跃": ActivityLevel.very_active, "非常活躍": ActivityLevel.very_active, "高强度": ActivityLevel.very_active, "高強度": ActivityLevel.very_active,
    "运动量大": ActivityLevel.very_active, "運動量大": ActivityLevel.very_active, "训练多": ActivityLevel.very_active, "訓練多": ActivityLevel.very_active,
    "very active": ActivityLevel.very_active, "very_active": ActivityLevel.very_active,
}

def map_sex_zh2en(s: str, default: Sex | None = None) -> Sex:
    key = _normalize_zh(s)
    if key in _SEX_ZH2EN:
        return _SEX_ZH2EN[key]
    # 兜底：若未识别，用默认；否则抛错更安全
    if default is not None:
        return default
    raise ValueError(f"无法识别的性别: {s!r}（建议输入：男/女）")

def map_activity_zh2en(s: str, default: ActivityLevel = ActivityLevel.light) -> ActivityLevel:
    key = _normalize_zh(s)
    if key in _ACTIVITY_ZH2EN:
        return _ACTIVITY_ZH2EN[key]
    # 兜底用 light，避免报错中断流程
    return default

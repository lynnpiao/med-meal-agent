# parsers/user_profile_zh.py
# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Dict, Any
from models.schemas import UserProfile
from i18n.profile_enum_map import map_sex_zh2en, map_activity_zh2en
from i18n.zh_lexicon import map_condition_zh2en, map_allergy_zh2en

def build_user_profile_from_zh(payload: Dict[str, Any]) -> UserProfile:
    sex = map_sex_zh2en(payload.get("sex", ""))
    activity = map_activity_zh2en(payload.get("activity_level", "轻度"))
    conditions_en = [map_condition_zh2en(x) for x in payload.get("conditions", [])]
    allergies_en = [map_allergy_zh2en(x) for x in payload.get("allergies", [])]

    return UserProfile(
        name=(payload.get("name") or None),
        age=int(payload["age"]),
        sex=sex,
        height_cm=float(payload["height_cm"]),
        weight_kg=float(payload["weight_kg"]),
        activity_level=activity,
        conditions=conditions_en,     # Pydantic 再做去重/小写
        allergies=allergies_en,       # 同上
        share_health_data=bool(payload.get("share_health_data", False)),
        health_report=payload.get("health_report"),  # 若已是结构化 dict，Pydantic 会校验
    )

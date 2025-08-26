# tests/test_models_userprofile_from_zh.py
# -*- coding: utf-8 -*-
import importlib
import sys
from pathlib import Path
import pytest

# 确保能 import 项目包
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.schemas import Sex, ActivityLevel, UserProfile
from parsers.user_profile_zh import build_user_profile_from_zh
from i18n.profile_enum_map import map_sex_zh2en, map_activity_zh2en

def test_build_user_profile_from_zh_basic(zh_lexicon_mod):
    """
    基本流程：中文输入 → 英文枚举与规范名；去重与小写由 Pydantic 校验器完成。
    """
    payload = {
        "name": "李四",
        "age": 32,
        "sex": "女",
        "height_cm": 165,
        "weight_kg": 60,
        "activity_level": "非常活跃",
        "conditions": ["高血压", "高血脂", "  高血脂  ", " type 2 diabetes "],
        "allergies": ["乳制品", "花生", "  花生 "],
        "share_health_data": True,
    }
    prof = build_user_profile_from_zh(payload)

    # 基本字段
    assert isinstance(prof, UserProfile)
    assert prof.name == "李四"
    assert prof.age == 32
    assert prof.sex == Sex.female
    assert prof.activity_level == ActivityLevel.very_active
    assert prof.height_cm == 165.0
    assert prof.weight_kg == 60.0
    assert prof.share_health_data is True

    # conditions / allergies 映射并去重（词库需包含相应条目）
    # 允许顺序不同，这里用集合断言
    assert set(prof.conditions) >= {"hypertension", "hyperlipidemia", "t2dm"}
    assert set(prof.allergies) == {"milk", "peanut"}

def test_build_user_profile_dedup_and_norm(zh_lexicon_mod):
    """
    Pydantic validators: 去重 + 去空格 + 小写
    """
    payload = {
        "age": 30, "sex": "男", "height_cm": 175, "weight_kg": 70,
        "activity_level": "中等",
        "conditions": [" Hypertension ", "t2dm", "T2DM", ""],
        "allergies": ["花生", "  花生  ", "  shellfish "],
    }
    prof = build_user_profile_from_zh(payload)
    assert prof.sex == Sex.male
    assert prof.activity_level == ActivityLevel.moderate
    assert prof.conditions == ["hypertension", "t2dm"]
    assert prof.allergies == ["peanut", "shellfish"]

def test_conditions_english_passthrough_normalization(zh_lexicon_mod):
    """
    英文直通：若已是英文（含奇怪大小写/空白），只做规范化（strip + lower）与去重。
    """
    payload = {
        "age": 40, "sex": "男", "height_cm": 180, "weight_kg": 82,
        "activity_level": "轻度",
        # 英文已有目标名，但大小写/空白杂乱
        "conditions": ["  HyPerLiPidEmia ", "T2DM", " hyperlipidemia "],
        "allergies": ["  PeAnUt  ", "peanut"],
    }
    prof = build_user_profile_from_zh(payload)
    # 去重 + 规范化
    assert prof.conditions == ["hyperlipidemia", "t2dm"]
    assert prof.allergies == ["peanut"]

def test_health_report_passthrough(zh_lexicon_mod):
    """
    若已提供结构化 HealthReport 字段，Pydantic 会自动校验并接收。
    """
    payload = {
        "age": 29, "sex": "女", "height_cm": 160, "weight_kg": 50,
        "activity_level": "轻度",
        "conditions": [], "allergies": [],
        "health_report": {
            "systolic_bp": 130, "diastolic_bp": 80,
            "hba1c_percent": 5.8
        }
    }
    prof = build_user_profile_from_zh(payload)
    assert prof.health_report is not None
    assert prof.health_report.systolic_bp == 130
    assert prof.health_report.hba1c_percent == 5.8

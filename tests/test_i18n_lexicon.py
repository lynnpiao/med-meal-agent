# tests/test_i18n_lexicon.py
# -*- coding: utf-8 -*-
import importlib.util as _iu
import pytest

HAS_PYPINYIN = _iu.find_spec("pypinyin") is not None
HAS_RAPIDFUZZ = _iu.find_spec("rapidfuzz") is not None

def test_conditions_zh2en(zh_lexicon_mod):
    zl = zh_lexicon_mod
    assert zl.map_condition_zh2en("高血压") == "hypertension"

def test_allergens_zh2en(zh_lexicon_mod):
    zl = zh_lexicon_mod
    # 根据你的 allergens.json 调整断言
    assert zl.map_allergy_zh2en("牛奶") == "milk"

def test_ingredients_roundtrip(zh_lexicon_mod):
    zl = zh_lexicon_mod
    assert zl.map_ing_zh2en("西兰花") == "broccoli"
    zh = zl.map_ing_en2zh("broccoli")
    assert isinstance(zh, str) and len(zh) > 0

def test_en_alias_hits_canonical(zh_lexicon_mod):
    zl = zh_lexicon_mod
    got = zl.map_ing_zh2en("garlic")  # 需要 ingredients.json 的 en_aliases 有 "garlic"
    assert got in ("garlic clove", "garlic")

def test_normalization(zh_lexicon_mod):
    zl = zh_lexicon_mod
    messy = "  ＂大　蒜 ＂  "
    got = zl.map_ing_zh2en(messy)
    assert got in ("garlic clove", "garlic")

@pytest.mark.skipif(not HAS_PYPINYIN, reason="需要 pypinyin 才能测试拼音兜底")
def test_pinyin_fallback_traditional(zh_lexicon_mod):
    zl = zh_lexicon_mod
    # 即便词库未显式列出“西蘭花”，拼音兜底应返回 broccoli
    assert zl.map_ing_zh2en("西蘭花") == "broccoli" 

@pytest.mark.skipif(not HAS_RAPIDFUZZ, reason="需要 rapidfuzz 才能测试模糊兜底")
def test_fuzzy_fallback(zh_lexicon_mod):
    zl = zh_lexicon_mod
    assert zl.map_ing_zh2en("西籣花") == "broccoli"  # 轻微错别字

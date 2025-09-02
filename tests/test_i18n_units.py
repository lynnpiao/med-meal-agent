# tests/test_i18n_units.py
# -*- coding: utf-8 -*-
import math

from i18n.units import (
    glucose_mgdl_from_mmol, chol_mgdl_from_mmol, tg_mgdl_from_mmol,
    creat_mgdl_from_umol, uric_mgdl_from_umol, vitd_ngml_from_nmol,
    parse_zh_qty_unit, parse_qty_unit
)

def _approx(a, b, tol=1e-2):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)

def test_lab_conversions():
    assert glucose_mgdl_from_mmol(5.5) == 5.5 * 18.0
    assert round(chol_mgdl_from_mmol(4.1), 2) == round(4.1 * 38.67, 2)
    assert round(tg_mgdl_from_mmol(1.7), 2) == round(1.7 * 88.57, 2)
    assert round(creat_mgdl_from_umol(88.4), 3) == 1.0
    assert round(uric_mgdl_from_umol(594.8), 2) == round(594.8 / 59.48, 2)
    assert round(vitd_ngml_from_nmol(50.0), 3) == round(50.0 / 2.496, 3)

def test_parse_zh_qty_unit():
    assert parse_zh_qty_unit("鸡胸肉 半斤") == ("鸡胸肉", 250.0, "g")
    assert parse_zh_qty_unit("牛奶 200 ml") == ("牛奶", 200.0, "ml")
    assert parse_zh_qty_unit("半斤 鸡胸肉") == ("鸡胸肉", 250.0, "g")
    assert parse_zh_qty_unit("200 ml 牛奶") == ("牛奶", 200.0, "ml")
    assert parse_zh_qty_unit("鸡蛋 2个") == ("鸡蛋", 2.0, "pcs")
    assert parse_zh_qty_unit("鸡胸肉 0.5 斤") == ("鸡胸肉", 250.0, "g")
    assert parse_zh_qty_unit("鸡胸肉 1斤") == ("鸡胸肉", 500.0, "g")
    assert parse_zh_qty_unit("花生  半两") == ("花生", 25.0, "g")  # 1 两=50g → 半两=25g

def test_parse_zh_qty_unit_pcs_tokens():
    assert parse_zh_qty_unit("土豆 3袋") == ("土豆", 3.0, "pcs")
    assert parse_zh_qty_unit("黄瓜 半个") == ("黄瓜", 0.5, "pcs")
    assert parse_zh_qty_unit("豆角 半袋") == ("豆角", 0.5, "pcs")
    assert parse_zh_qty_unit("西红柿 1瓶") == ("西红柿", 1.0, "pcs")
    assert parse_zh_qty_unit("3袋 土豆") == ("土豆", 3.0, "pcs")
    assert parse_zh_qty_unit("半个 黄瓜") == ("黄瓜", 0.5, "pcs")
    assert parse_zh_qty_unit("半袋 豆角") == ("豆角", 0.5, "pcs")
    assert parse_zh_qty_unit("1瓶 西红柿") == ("西红柿", 1.0, "pcs")
    
def test_parse_zh_qty_unit_en_units_if_enabled():
    # 以下断言依赖你已在 units.py 中加入英文单位映射（tbsp/tsp/lb 等）
    name, qty, unit = parse_zh_qty_unit("橄榄油 1 tbsp")
    assert name == "橄榄油" and unit == "ml" and _approx(qty, 15.0)

    name, qty, unit = parse_zh_qty_unit("酱油 2 tsp")
    assert name == "酱油" and unit == "ml" and _approx(qty, 10.0)

    name, qty, unit = parse_zh_qty_unit("牛肉 1 lb")
    assert name == "牛肉" and unit == "g" and _approx(qty, 453.592, tol=0.01)
    
def test_fraction_numbers_and_unicode():
    assert parse_qty_unit("油 1 1/2 tbsp")[1:] == (15.0*1.5, "ml")
    assert parse_qty_unit("盐 ½ tsp")[1:] == (2.5, "ml")

def test_fluid_ounce_variants():
    assert parse_qty_unit("牛奶 2 fl. oz")[1:] == (2*29.5735, "ml")
    assert parse_qty_unit("牛奶 2 fluid ounces")[1:] == (2*29.5735, "ml")

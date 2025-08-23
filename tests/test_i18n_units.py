# tests/test_i18n_units.py
# -*- coding: utf-8 -*-
from i18n.units import (
    glucose_mgdl_from_mmol, chol_mgdl_from_mmol, tg_mgdl_from_mmol,
    creat_mgdl_from_umol, uric_mgdl_from_umol, vitd_ngml_from_nmol,
    parse_zh_qty_unit
)

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
    assert parse_zh_qty_unit("鸡蛋 2个") == ("鸡蛋", 2.0, "pcs")
    assert parse_zh_qty_unit("鸡胸肉 0.5 斤") == ("鸡胸肉", 250.0, "g")
    assert parse_zh_qty_unit("鸡胸肉 1斤") == ("鸡胸肉", 500.0, "g")
    assert parse_zh_qty_unit("花生  半两") == ("花生", 25.0, "g")  # 1 两=50g → 半两=25g

    # 未知单位 → 默认 g（数量原样）
    assert parse_zh_qty_unit("土豆 3袋") == ("土豆", 3.0, "g")

# tests/test_parsers_health_report_zh_full.py
# -*- coding: utf-8 -*-
import sys
import re
import math
from pathlib import Path
import pytest
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import parsers.health_report_zh as hr
from parsers.health_report_zh import HRSource, HRExtract
from models.schemas import HealthReport

BASE = ROOT / "parsers" / "base_dir"

def _approx(a, b, rel=1e-3, abs_tol=1e-9):
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)

@pytest.fixture(autouse=True)
def ensure_re_in_module():
    import re
    if not hasattr(hr, "re"):
        setattr(hr, "re", re)

def test_text_file_full_extraction(monkeypatch):
    text_path = BASE / "test.txt"
    content = text_path.read_text(encoding="utf-8", errors="ignore")

    extract = HRExtract(
        systolic_bp_mmHg=135,
        diastolic_bp_mmHg=70,
        hba1c_percent=6.4,
        ldl_value=3.3, ldl_unit="mmol/L",
        alt_value=48, alt_unit="U/L",
        tsh_value=2.1, tsh_unit="mIU/L",
        hemoglobin_value=130, hemoglobin_unit="g/L",
        ferritin_value=60, ferritin_unit="μg/L",
        gdm=False,
        medications_text="二甲双胍，左甲状腺素。",
    )

    def fake_structured(text: str, model="gpt-4o-mini"):
        assert text.strip() == content.strip()
        return extract

    monkeypatch.setattr(hr, "_llm_structured_from_text", fake_structured)

    report: HealthReport = hr.extract_health_report(HRSource(data=content, kind="text"))
    
    assert report.systolic_bp == 135
    assert report.diastolic_bp == 70
    assert report.hba1c_percent == 6.4
    assert _approx(report.ldl_mg_dl, 3.3 * 38.67)
    assert report.alt_u_l == 48
    assert report.tsh_u_iu_ml == 2.1
    assert _approx(report.hemoglobin_g_dl, 13.0)
    assert report.ferritin_ng_ml == 60
    assert report.gdm is False
    assert report.medications == ["二甲双胍", "左甲状腺素"]

def _load_two_images():
    def _img(path_lower: str, path_upper: str):
        p = (BASE / path_lower)
        if not p.exists():
            p = (BASE / path_upper)
        return p
    img1_path = _img("test11.jpg", "TEST11.JPG")
    img2_path = _img("test22.jpg", "TEST22.JPG")
    return img1_path.read_bytes(), img2_path.read_bytes()

def _fake_ocr_texts(img1_bytes: bytes, img2_bytes: bytes):
    """
    生成一个 fake_ocr(b)：
    - img1：给 BP、HbA1c、ALT、用药与备注（部分字段）
    - img2：给全套实验室（含 BP 冲突值用于覆盖）、肾功能/脂类/糖代谢/甲功/维D/血象铁代谢/孕周/GDM/用药/备注
    """
    txt1 = (
        "收缩压 126 mmHg；舒张压 77 mmHg；"
        "糖化血红蛋白 5.9 %；ALT 42 U/L；"
        "用药：阿司匹林，维生素D；备注：来自第一张"
    )
    txt2 = (
        "收缩压 130 mmHg；舒张压 80 mmHg；"
        "糖化血红蛋白 6.2 %；空腹血葡萄糖 5.6 mmol/L；"
        "口服葡萄糖耐量试验2小时血糖 8.0 mmol/L；"
        "总胆固醇 5.2 mmol/L；甘油三酯 1.7 mmol/L；"
        "高密度脂蛋白胆固醇 1.1 mmol/L；低密度脂蛋白胆固醇 3.2 mmol/L；非HDL 4.1 mmol/L；"
        "肌酐 88.4 µmol/L；eGFR 92；尿酸 420 µmol/L；"
        "AST 28 U/L；ALT 41 U/L；"
        "TSH 2.2 mIU/L；维生素D 50 nmol/L；"
        "Hb 130 g/L；Ferritin 60 ug/L；"
        "孕周 22 周；GDM：是；"
        "用药：阿司匹林，叶酸；备注：来自第二张"
    )
    def fake_ocr(b: bytes, model="gpt-4o-mini") -> str:
        if b == img1_bytes:
            return txt1
        elif b == img2_bytes:
            return txt2
        return ""
    return fake_ocr

def _generic_structurer():
    """
    通用 fake_structured(text) ：
    - 用正则从 OCR 文本抽取字段与单位
    - 正确区分 数值(第1捕获组) 和 单位(第2捕获组)
    """
    rx = {
    # 血压（单位基本固定 mmHg，不单独捕单位也可以）
    "sbp": re.compile(r"收缩压\s*([0-9.]+)\s*mmhg", re.I),
    "dbp": re.compile(r"舒张压\s*([0-9.]+)\s*mmhg", re.I),

    # HbA1c（%）
    "hba1c": re.compile(r"(?:hba1c|糖化血红蛋白)\s*[:：=]?\s*([0-9.]+)\s*%", re.I),

    # 血糖：双捕获（值, 单位）
    "fpg": re.compile(r"(?:空腹血(?:葡萄)?糖)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),
    "ogtt2h": re.compile(r"(?:ogtt|口服葡萄糖耐量试验)\s*2(?:小时|h)?(?:\s*血糖)?\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),

    # 血脂：双捕获
    "tc":   re.compile(r"(?:总胆固醇|TC)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),
    "tg":   re.compile(r"(?:甘油三酯|TG)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),
    "hdl":  re.compile(r"(?:高密度脂蛋白胆固醇|HDL)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),
    "ldl":  re.compile(r"(?:低密度脂蛋白胆固醇|LDL)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),
    "nonhdl": re.compile(r"(?:非HDL|non-?HDL)\s*[:：=]?\s*([0-9.]+)\s*(mmol/L|mg/dL)", re.I),

    # 肌酐/尿酸：可继续用“值/单位”两条，或写成双捕获版二选一
    "creat_val":  re.compile(r"肌酐\s*[:：=]?\s*([0-9.]+)\s*(?:µ?mol/L|umol/L|mg/dL)", re.I),
    "creat_unit": re.compile(r"肌酐\s*[0-9.]+\s*((?:µ?mol/L|umol/L|mg/dL))", re.I),

    "ua_val":  re.compile(r"(?:尿酸|UA)\s*[:：=]?\s*([0-9.]+)\s*(?:µ?mol/L|umol/L|mg/dL)", re.I),
    "ua_unit": re.compile(r"(?:尿酸|UA)\s*[0-9.]+\s*((?:µ?mol/L|umol/L|mg/dL))", re.I),

    # eGFR（多为无单位数值；不同写法视为等价）
    "egfr": re.compile(r"eGFR\s*[:：=]?\s*([0-9.]+)", re.I),

    # 肝功能（单位 U/L、IU/L 等价，不必捕单位）
    "alt": re.compile(r"(?:ALT|丙氨酸转氨酶)\s*[:：=]?\s*([0-9.]+)\s*(?:U/L|IU/L)", re.I),
    "ast": re.compile(r"(?:AST|天冬氨酸转氨酶)\s*[:：=]?\s*([0-9.]+)\s*(?:U/L|IU/L)", re.I),

    # TSH：双捕获
    "tsh_val": re.compile(
        r"(?:TSH|促甲状腺激素)\s*[:：=]?\s*([0-9.]+)\s*"
        r"(?:"
        r"(?:[mµu]iu|mu)\s*/\s*(?:m?l|l)"  # mIU/mL、mIU/L、uIU/mL、µIU/L、…
        r"|mU\s*/\s*L"                    # mU/L
        r")",
        re.I
    ),

    # 维生素 D：双捕获
    "vitd_val":  re.compile(r"(?:维生素D|25\(?OH\)?D)\s*[:：=]?\s*([0-9.]+)\s*(?:nmol/L|ng/mL)", re.I),
    "vitd_unit": re.compile(r"(?:维生素D|25\(?OH\)?D)\s*[0-9.]+\s*((?:nmol/L|ng/mL))", re.I),

    # Hb / Ferritin：数值 + 单位
    "hb_val":  re.compile(r"(?:Hb|血红蛋白)\s*[:：=]?\s*([0-9.]+)\s*(?:g/dL|g/L)", re.I),
    "hb_unit": re.compile(r"(?:Hb|血红蛋白)\s*[0-9.]+\s*((?:g/dL|g/L))", re.I),

    "ferr_val":  re.compile(r"(?:Ferritin|铁蛋白)\s*[:：=]?\s*([0-9.]+)\s*(?:ng/mL|μg/L|ug/L)", re.I),
    "ferr_unit": re.compile(r"(?:Ferritin|铁蛋白)\s*[0-9.]+\s*((?:ng/mL|μg/L|ug/L))", re.I),

    # 孕周 / GDM / 用药 / 备注
    "gest_weeks": re.compile(r"(?:孕周|妊娠.*?周)\s*[:：=]?\s*([0-9.]+)", re.I),
    "gdm_yes":    re.compile(r"(?:GDM|妊娠期糖尿病)\s*[:：=]?\s*(是|阳性)", re.I),
    "gdm_no":     re.compile(r"(?:GDM|妊娠期糖尿病)\s*[:：=]?\s*(否|阴性)", re.I),
    "meds":       re.compile(r"(?:用药|药物)\s*[:：]\s*([^\n；。]*)", re.I),
    "notes":      re.compile(r"(?:备注)\s*[:：]\s*([^\n]+)", re.I),
    
    }

    def g1(name: str, text: str) -> Optional[str]:
        m = rx[name].search(text)
        return m.group(1) if m else None

    def g2(name: str, text: str) -> Optional[str]:
        m = rx[name].search(text)
        return m.group(2) if (m and m.lastindex and m.lastindex >= 2) else None

    def gf(name: str, text: str) -> Optional[float]:
        s = g1(name, text)
        return float(s) if s is not None else None

    def fake_structured(text: str, model="gpt-4o-mini"):
        out = hr.HRExtract(
            systolic_bp_mmHg=gf("sbp", text),
            diastolic_bp_mmHg=gf("dbp", text),
            hba1c_percent=gf("hba1c", text),
        )

        # 糖代谢
        v, u = gf("fpg", text), g2("fpg", text)
        if v is not None and u:
            out.fasting_glucose_value, out.fasting_glucose_unit = v, u
        v2, u2 = gf("ogtt2h", text), g2("ogtt2h", text)
        if v2 is not None and u2:
            out.ogtt_2h_glucose_value, out.ogtt_2h_glucose_unit = v2, u2

        # 血脂
        for key, field in [("tc", "total_cholesterol"), ("tg", "triglycerides"),
                           ("hdl", "hdl"), ("ldl", "ldl"), ("nonhdl", "non_hdl")]:
            val, unit = gf(key, text), g2(key, text)
            if val is not None and unit:
                setattr(out, f"{field}_value", val)
                setattr(out, f"{field}_unit", unit)

        # 肌酐/尿酸/eGFR/肝功
        cv, cu = gf("creat_val", text), g1("creat_unit", text)
        if cv is not None and cu:
            out.creatinine_value, out.creatinine_unit = cv, cu
        egfr = gf("egfr", text)
        if egfr is not None:
            out.egfr_value, out.egfr_unit = egfr, "mL/min/1.73m2"
        alt = gf("alt", text)
        if alt is not None:
            out.alt_value, out.alt_unit = alt, "U/L"
        ast = gf("ast", text)
        if ast is not None:
            out.ast_value, out.ast_unit = ast, "U/L"
        uav, uau = gf("ua_val", text), g1("ua_unit", text)
        if uav is not None and uau:
            out.uric_acid_value, out.uric_acid_unit = uav, uau

        # TSH / 维D
        tsh = gf("tsh_val", text)
        if tsh is not None:
            out.tsh_value, out.tsh_unit = tsh, "mIU/L"
        vdv, vdu = gf("vitd_val", text), g1("vitd_unit", text)
        if vdv is not None and vdu:
            out.vitamin_d_value, out.vitamin_d_unit = vdv, vdu

        # Hb / Ferritin
        hbv, hbu = gf("hb_val", text), g1("hb_unit", text)
        if hbv is not None and hbu:
            out.hemoglobin_value, out.hemoglobin_unit = hbv, hbu
        fv, fu = gf("ferr_val", text), g1("ferr_unit", text)
        if fv is not None and fu:
            out.ferritin_value, out.ferritin_unit = fv, fu

        # 孕周 / GDM
        gw = gf("gest_weeks", text)
        if gw is not None:
            out.gestational_weeks_value = gw
        if rx["gdm_yes"].search(text):
            out.gdm = True
        elif rx["gdm_no"].search(text):
            out.gdm = False

        # 用药 / 备注
        meds = g1("meds", text)
        if meds:
            out.medications_text = meds
        notes = g1("notes", text)
        if notes:
            out.notes = notes

        return out

    return fake_structured

def _merge_from_images(monkeypatch, order="12"):
    img1_bytes, img2_bytes = _load_two_images()
    monkeypatch.setattr(hr, "_llm_ocr_image_to_text", _fake_ocr_texts(img1_bytes, img2_bytes))
    monkeypatch.setattr(hr, "_llm_structured_from_text", _generic_structurer())

    sources = (
        [HRSource(data=img1_bytes, kind="image"), HRSource(data=img2_bytes, kind="image")]
        if order == "12"
        else [HRSource(data=img2_bytes, kind="image"), HRSource(data=img1_bytes, kind="image")]
    )
    if hasattr(hr, "extract_health_report_multi"):
        return hr.extract_health_report_multi(sources)
    else:
        r = [hr.extract_health_report(s) for s in sources]
        assert hasattr(hr, "merge_health_reports"), "请先实现 merge_health_reports 或 extract_health_report_multi"
        return hr.merge_health_reports(r)

def test_merge_two_images_full_coverage_later_wins(monkeypatch):
    """
    顺序 [test11, test22]：test22 中与 test11 重复的指标应覆盖（后者覆盖前者）。
    同时验证单位换算与合并用药/备注。
    """
    rep = _merge_from_images(monkeypatch, order="12")

    # 覆盖：BP 与 HbA1c、ALT 来自 test2
    assert rep.systolic_bp == 130
    assert rep.diastolic_bp == 80
    assert rep.hba1c_percent == 6.2
    assert rep.alt_u_l == 41
    assert rep.ast_u_l == 28

    # 糖代谢
    assert _approx(rep.fasting_glucose_mg_dl, 5.6 * 18.0)   # 100.8 mg/dL
    assert _approx(rep.ogtt_2h_glucose_mg_dl, 8.0 * 18.0)   # 144 mg/dL

    # 血脂（mmol/L → mg/dL）
    assert _approx(rep.total_cholesterol_mg_dl, 5.2 * 38.67)
    assert _approx(rep.triglycerides_mg_dl, 1.7 * 88.57)
    assert _approx(rep.hdl_mg_dl, 1.1 * 38.67)
    assert _approx(rep.ldl_mg_dl, 3.2 * 38.67)
    assert _approx(rep.non_hdl_mg_dl, 4.1 * 38.67)

    # 肾功能/尿酸
    assert _approx(rep.creatinine_mg_dl, 1.0)               # 88.4 µmol/L → 1.0 mg/dL
    assert rep.egfr_ml_min_1_73m2 == 92
    assert _approx(rep.uric_acid_mg_dl, 420 / 59.48)

    # TSH / 维D / Hb / Ferritin
    assert rep.tsh_u_iu_ml == 2.2
    assert _approx(rep.vitamin_d_25oh_ng_ml, 50 / 2.496)    # nmol/L → ng/mL
    assert _approx(rep.hemoglobin_g_dl, 13.0)               # 130 g/L → 13.0 g/dL
    assert rep.ferritin_ng_ml == 60                         # 60 ug/L == 60 ng/mL

    # 孕周 / GDM
    assert rep.gestational_weeks == 22
    assert rep.gdm is True

    # 合并用药（去重保序）与备注拼接
    assert rep.medications == ["阿司匹林", "维生素D", "叶酸"]
    # print(rep.notes)
    assert rep.notes == "来自第一张 | 来自第二张"

def test_merge_two_images_full_coverage_order_swap(monkeypatch):
    """
    顺序 [test22, test11]：后者覆盖前者，但 None 不覆盖已有值。
    - BP/HbA1c/ALT 最终来自 test11（覆盖）
    - AST 保留来自 test22（test11 无 AST，不覆盖）
    其余实验室项仍来自 test22。
    """
    rep = _merge_from_images(monkeypatch, order="21")

    # 覆盖后结果
    assert rep.systolic_bp == 126
    assert rep.diastolic_bp == 77
    assert rep.hba1c_percent == 5.9
    assert rep.alt_u_l == 42
    assert rep.ast_u_l == 28  # test1 没有 AST，不覆盖

    # 糖代谢/血脂/肾功能/尿酸——仍来自 test2（未被覆盖）
    assert _approx(rep.fasting_glucose_mg_dl, 5.6 * 18.0)
    assert _approx(rep.ogtt_2h_glucose_mg_dl, 8.0 * 18.0)
    assert _approx(rep.total_cholesterol_mg_dl, 5.2 * 38.67)
    assert _approx(rep.triglycerides_mg_dl, 1.7 * 88.57)
    assert _approx(rep.hdl_mg_dl, 1.1 * 38.67)
    assert _approx(rep.ldl_mg_dl, 3.2 * 38.67)
    assert _approx(rep.non_hdl_mg_dl, 4.1 * 38.67)
    assert _approx(rep.creatinine_mg_dl, 1.0)
    assert rep.egfr_ml_min_1_73m2 == 92
    assert _approx(rep.uric_acid_mg_dl, 420 / 59.48)

    # TSH / 维D / Hb / Ferritin
    assert rep.tsh_u_iu_ml == 2.2
    assert _approx(rep.vitamin_d_25oh_ng_ml, 50 / 2.496)
    assert _approx(rep.hemoglobin_g_dl, 13.0)
    assert rep.ferritin_ng_ml == 60

    # 孕周 / GDM
    assert rep.gestational_weeks == 22
    assert rep.gdm is True

    # 用药合并与备注
    assert rep.medications == ["阿司匹林", "叶酸", "维生素D"]
    assert rep.notes == "来自第二张 | 来自第一张"
    
    

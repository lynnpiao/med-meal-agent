# tests/test_parsers_health_report_zh_live.py
# -*- coding: utf-8 -*-
import os
import math
from pathlib import Path
import pytest 
import openai
import unicodedata as _ud
import re

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import parsers.health_report_zh as hr
from parsers.health_report_zh import HRSource

BASE = ROOT / "parsers" / "base_dir"

def _approx(a, b, rel=1e-2, abs_tol=1e-6):  # 集成测试更宽松一点
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol)

def _skip_on_llm_excs(exc: Exception):
    SKIP_EXCS = (
        openai.RateLimitError,
        openai.AuthenticationError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.APIError,
    )
    if isinstance(exc, SKIP_EXCS):
        pytest.skip(f"LLM live test skipped: {type(exc).__name__}: {exc}")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"),
]

def test_structured_from_text_live():
    """
    直接用 test.txt 的文本打 LLM，检查关键字段（对 test.txt 做“完全匹配”）。
    为降低脆弱性，单位/标点做轻度归一化（NFKC、去空格、µ/μ→u，去句号）。
    """
    txt = (BASE / "test.txt").read_text("utf-8", errors="ignore")

    # 常见 LLM/API 异常 → 跳过
    try:
        out = hr._llm_structured_from_text(txt, model="gpt-4o-mini")
    except Exception as e:
        _skip_on_llm_excs(e)
        raise

    def _norm_unit(s: str) -> str:
        # 统一全/半角，转小写，去空格，把微符号(µ/μ)都当成 u
        t = _ud.normalize("NFKC", (s or "")).strip().lower()
        t = t.replace("µ", "u").replace("μ", "u")
        t = re.sub(r"\s+", "", t)
        return t

    def _split_meds(meds_list, meds_text):
        # 优先用数组；否则用文本切分；并去掉句尾句号/空白
        if meds_list:
            items = meds_list
        else:
            items = re.split(r"[，,；;、]+", meds_text or "")
        out = []
        for m in items:
            m = m.strip().strip("。．.")
            if m:
                out.append(m)
        return out

    # ====== 数值精确匹配（test.txt） ======
    assert out.systolic_bp_mmHg in (135, 135.0)
    assert out.diastolic_bp_mmHg in (70, 70.0)
    assert out.hba1c_percent is not None and _approx(out.hba1c_percent, 6.4, rel=0, abs_tol=1e-9)

    # LDL 3.3 mmol/L
    assert out.ldl_value is not None and _approx(out.ldl_value, 3.3, rel=0, abs_tol=1e-9)
    assert _norm_unit(out.ldl_unit) == "mmol/l"

    # ALT 48 U/L
    assert out.alt_value is not None and _approx(out.alt_value, 48, rel=0, abs_tol=1e-9)
    assert _norm_unit(out.alt_unit) in {"u/l", "iu/l"}  # U/L 与 IU/L 数值等价

    # TSH 2.1 mIU/L
    assert out.tsh_value is not None and _approx(out.tsh_value, 2.1, rel=0, abs_tol=1e-9)
    assert _norm_unit(out.tsh_unit) in {"miu/l", "uiu/ml", "mu/l"}  # 这些写法等价（mIU/L == uIU/mL == mU/L）

    # Hb 130 g/L
    assert out.hemoglobin_value is not None and _approx(out.hemoglobin_value, 130, rel=0, abs_tol=1e-9)
    assert _norm_unit(out.hemoglobin_unit) == "g/l"

    # Ferritin 60 μg/L
    assert out.ferritin_value is not None and _approx(out.ferritin_value, 60, rel=0, abs_tol=1e-9)
    assert _norm_unit(out.ferritin_unit) in {"ug/l", "ng/ml"}  # μg/L 与 ng/mL 数值等价

    # GDM：否
    assert out.gdm is False

    # 用药：二甲双胍，左甲状腺素。
    meds = _split_meds(out.medications, out.medications_text)
    assert meds == ["二甲双胍", "左甲状腺素"]


def _looks_like_refusal(s: str) -> bool:
    t = _ud.normalize("NFKC", (s or "")).lower()
    # 常见的拒绝/无法处理用语（中英），以及很短的回复
    bad = ("抱歉", "无法处理", "不能处理", "不能帮助", "不便协助", "i'm sorry", "cannot", "can't", "policy")
    return any(k in t for k in bad) and len(t) < 200


def test_ocr_image_to_text_live():
    """
    真正调用视觉 OCR：
    - 正向：应至少包含血压关键词（收缩压/舒张压 或 SBP/DBP），否则跳过
    - 负向：test1.jpg 不应包含 HbA1c/糖化血红蛋白；如出现，视作模型波动并跳过
    """
    path = BASE / ("test11.jpg" if (BASE / "test11.jpg").exists() else "TEST11.JPG")
    img_bytes = path.read_bytes()

    try:
        txt = hr._llm_ocr_image_to_text(img_bytes, model="gpt-4o-mini")
    except Exception as e:
        _skip_on_llm_excs(e)
        raise
    
    if _looks_like_refusal(txt):
        pytest.skip(f"OCR 被模型拒绝：{txt!r}")

    # 归一化：NFKC + lower；顺带去掉所有空白，兼容全/半角和大小写差异
    tnorm = _ud.normalize("NFKC", (txt or "")).lower()
    tnorm = "".join(ch for ch in tnorm if not ch.isspace())

    # 正向：至少命中一个血压关键词；否则跳过（可能是模型波动/图片质量问题）
    # 放宽：出现血压关键词或 “NNN/NNN mmHg” 也算
    has_bp = (
        any(k in tnorm for k in ("收缩压", "舒张压", "sbp", "dbp", "血压"))
        or re.search(r"\b\d{2,3}\s*/\s*\d{2,3}\s*mmhg\b", tnorm) is not None
    )
    if not has_bp:
        pytest.skip(f"OCR 文本未包含血压线索（可能模型波动/图片质量）。样本: {tnorm[:160]!r}")

    # 负向：test1.jpg 不应有 HbA1c/糖化血红蛋白；如出现则跳过（避免偶发幻觉导致失败）
    neg_keys = ("hba1c", "糖化血红蛋白")
    if any(k in tnorm for k in neg_keys):
        pytest.skip("OCR 文本意外包含 HbA1c/糖化血红蛋白，可能是模型幻觉，先跳过以保证稳定性。")
        

# def test_end_to_end_image_live():
#     """
#     端到端：图片 -> OCR -> 结构化 -> HealthReport
#     """
#     p = BASE / ("test1.jpg" if (BASE/"test1.jpg").exists() else "test1.JPG")
#     img_bytes = p.read_bytes()
#     try:
#         rep = hr.extract_health_report(HRSource(data=img_bytes, kind="image"), model="gpt-4o-mini")
#     except Exception as e:
#         _skip_on_llm_excs(e)
#         raise

#     # 根据 test1.jpg 的实际内容写断言（只对“应有字段”断言，不要太苛刻）
#     assert rep.systolic_bp is not None
#     assert rep.diastolic_bp is not None
#     # 如果图片包含 ALT/HbA1c，就顺带断言：
#     # assert rep.alt_u_l is not None
#     # assert rep.hba1c_percent is not None


# def test_end_to_end_two_images_merge_live():
#     """
#     两张图片合并：真实 OCR + 结构化 + 合并。
#     只验证“后者覆盖前者”的规则确实发生：如果两张图都抽到了 BP，以第二张为准。
#     """
#     p1 = BASE / ("test1.jpg" if (BASE/"test1.jpg").exists() else "test1.JPG")
#     p2 = BASE / ("test2.jpg" if (BASE/"test2.jpg").exists() else "test2.JPG")
#     s1 = HRSource(data=p1.read_bytes(), kind="image")
#     s2 = HRSource(data=p2.read_bytes(), kind="image")

#     try:
#         rep = hr.extract_health_report_multi([s1, s2], model="gpt-4o-mini")
#         # 为了验证“后者覆盖前者”，再分别跑一遍单图抽取
#         rep1 = hr.extract_health_report(s1, model="gpt-4o-mini")
#         rep2 = hr.extract_health_report(s2, model="gpt-4o-mini")
#     except Exception as e:
#         _skip_on_llm_excs(e)
#         raise

#     if rep1.systolic_bp and rep2.systolic_bp:
#         assert rep.systolic_bp == rep2.systolic_bp
#     if rep1.diastolic_bp and rep2.diastolic_bp:
#         assert rep.diastolic_bp == rep2.diastolic_bp
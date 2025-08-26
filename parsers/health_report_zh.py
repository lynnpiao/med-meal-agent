# parsers/health_report_zh.py
# -*- coding: utf-8 -*-
"""
LLM 驱动的中文体检报告抽取（MVP）：
- 文本：直接丢给 LLM 结构化
- 图片：用 LLM 视觉 OCR 抽原文文本，再走结构化
- PDF/DOCX：先提取文本；若是扫描件可选转为图片逐页 OCR（见 TODO 钩子）

需要环境变量：
- OPENAI_API_KEY（langchain-openai 读取）

依赖：
- langchain-openai
- pypdf
- python-docx
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Literal, Union, List, Iterable
from dataclasses import dataclass
from PIL import Image, ImageDraw
import base64
import io
import os
import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser

from i18n.units import (
    glucose_mgdl_from_mmol, chol_mgdl_from_mmol, tg_mgdl_from_mmol,
    creat_mgdl_from_umol, uric_mgdl_from_umol, vitd_ngml_from_nmol
)
from models.schemas import HealthReport

# -----------------------------
# 结构化提取的临时 Schema（LLM 输出）
# -----------------------------
class HRExtract(BaseModel):
    # 血压
    systolic_bp_mmHg: Optional[float] = None
    diastolic_bp_mmHg: Optional[float] = None

    # 糖化
    hba1c_percent: Optional[float] = None

    # 血脂
    ldl_value: Optional[float] = None
    ldl_unit: Optional[str] = None               # mg/dL 或 mmol/L
    hdl_value: Optional[float] = None
    hdl_unit: Optional[str] = None
    triglycerides_value: Optional[float] = None
    triglycerides_unit: Optional[str] = None
    total_cholesterol_value: Optional[float] = None
    total_cholesterol_unit: Optional[str] = None # mg/dL 或 mmol/L
    non_hdl_value: Optional[float] = None
    non_hdl_unit: Optional[str] = None           # mg/dL 或 mmol/L

    # 血糖
    fasting_glucose_value: Optional[float] = None
    fasting_glucose_unit: Optional[str] = None   # mg/dL 或 mmol/L
    ogtt_2h_glucose_value: Optional[float] = None
    ogtt_2h_glucose_unit: Optional[str] = None

    # 肾功能
    creatinine_value: Optional[float] = None
    creatinine_unit: Optional[str] = None        # mg/dL 或 umol/L/µmol/L
    egfr_value: Optional[float] = None
    egfr_unit: Optional[str] = None              # mL/min/1.73m2（或类似写法）

    # 肝功能
    alt_value: Optional[float] = None
    alt_unit: Optional[str] = None               # U/L 或 IU/L
    ast_value: Optional[float] = None
    ast_unit: Optional[str] = None

    # 尿酸
    uric_acid_value: Optional[float] = None
    uric_acid_unit: Optional[str] = None         # mg/dL 或 umol/L/µmol/L

    # 甲状腺
    tsh_value: Optional[float] = None
    tsh_unit: Optional[str] = None               # uIU/mL, µIU/mL, mIU/L, mU/L（数值等价）

    # 维生素D
    vitamin_d_value: Optional[float] = None
    vitamin_d_unit: Optional[str] = None         # ng/mL 或 nmol/L

    # 血红蛋白/铁蛋白
    hemoglobin_value: Optional[float] = None
    hemoglobin_unit: Optional[str] = None        # g/dL 或 g/L（g/L→g/dL ÷10）
    ferritin_value: Optional[float] = None
    ferritin_unit: Optional[str] = None          # ng/mL 或 μg/L/ug/L（等价）

    # 产科
    gestational_weeks_value: Optional[float] = None
    gdm: Optional[bool] = None                   # 妊娠期糖尿病（是/否）

    # 其他
    medications: Optional[List[str]] = None      # 建议优先抽取为数组
    medications_text: Optional[str] = None       # 备选：逗号/顿号/分号分隔的字符串
    notes: Optional[str] = None


# -----------------------------
# 工具函数：单位规整与换算
# -----------------------------
def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _same(x: Optional[float]) -> Optional[float]:
    return x if x is not None else None

def _split_med_list(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # 用常见分隔符切分：中/英文逗号、分号、顿号、斜杠、竖线
    parts = re.split(r"[,\uFF0C;\uFF1B\u3001/|]+", s)
    out: List[str] = []
    for p in parts:
        # 去首尾空白
        p = p.strip()
        # 去掉两端常见标点（中英文句号、逗号、分号、顿号、空白点等）
        p = re.sub(r"^[\s\.\u3002，,;；、·•]+|[\s\.\u3002，,;；、·•]+$", "", p)
        if p:
            out.append(p)
    return out

def _to_mgdl_or_ngml(d: HRExtract) -> Dict[str, Any]:
    """
    把 HRExtract → schemas.HealthReport 所需字段与单位：
      - *_mg_dl / *_ng_ml / *_u_l / *_u_iu_ml / egfr_ml_min_1_73m2 等
      - 单位换算规则见注释
    """
    out: Dict[str, Any] = {}

    # —— 血压 ——
    out["systolic_bp"] = d.systolic_bp_mmHg
    out["diastolic_bp"] = d.diastolic_bp_mmHg

    # —— HbA1c ——
    out["hba1c_percent"] = d.hba1c_percent

    # —— 血糖 ——
    if d.fasting_glucose_value is not None:
        out["fasting_glucose_mg_dl"] = (
            glucose_mgdl_from_mmol(d.fasting_glucose_value)
            if "mmol" in _norm(d.fasting_glucose_unit)
            else d.fasting_glucose_value
        )
    if d.ogtt_2h_glucose_value is not None:
        out["ogtt_2h_glucose_mg_dl"] = (
            glucose_mgdl_from_mmol(d.ogtt_2h_glucose_value)
            if "mmol" in _norm(d.ogtt_2h_glucose_unit)
            else d.ogtt_2h_glucose_value
        )

    # —— 血脂 ——
    if d.ldl_value is not None:
        out["ldl_mg_dl"] = chol_mgdl_from_mmol(d.ldl_value) if "mmol" in _norm(d.ldl_unit) else d.ldl_value
    if d.hdl_value is not None:
        out["hdl_mg_dl"] = chol_mgdl_from_mmol(d.hdl_value) if "mmol" in _norm(d.hdl_unit) else d.hdl_value
    if d.triglycerides_value is not None:
        out["triglycerides_mg_dl"] = tg_mgdl_from_mmol(d.triglycerides_value) if "mmol" in _norm(d.triglycerides_unit) else d.triglycerides_value
    if d.total_cholesterol_value is not None:
        out["total_cholesterol_mg_dl"] = chol_mgdl_from_mmol(d.total_cholesterol_value) if "mmol" in _norm(d.total_cholesterol_unit) else d.total_cholesterol_value
    if d.non_hdl_value is not None:
        out["non_hdl_mg_dl"] = chol_mgdl_from_mmol(d.non_hdl_value) if "mmol" in _norm(d.non_hdl_unit) else d.non_hdl_value

    # —— 肾功能 ——
    if d.creatinine_value is not None:
        u = _norm(d.creatinine_unit)
        if "umol" in u or "µmol" in u:
            out["creatinine_mg_dl"] = creat_mgdl_from_umol(d.creatinine_value)
        else:
            out["creatinine_mg_dl"] = d.creatinine_value
    if d.egfr_value is not None:
        # eGFR 通常直接是 mL/min/1.73m²；有时写法不同（ml/min/1.73㎡ 等），数值等价，直接赋值
        out["egfr_ml_min_1_73m2"] = d.egfr_value

    # —— 肝功能 ——
    if d.alt_value is not None:
        # ALT 单位 U/L / IU/L 数值等价
        out["alt_u_l"] = d.alt_value
    if d.ast_value is not None:
        out["ast_u_l"] = d.ast_value

    # —— 尿酸 ——
    if d.uric_acid_value is not None:
        u = _norm(d.uric_acid_unit)
        if "umol" in u or "µmol" in u:
            out["uric_acid_mg_dl"] = uric_mgdl_from_umol(d.uric_acid_value)
        else:
            out["uric_acid_mg_dl"] = d.uric_acid_value

    # —— 甲状腺 TSH ——
    if d.tsh_value is not None:
        # 约定目标单位 uIU/mL；常见写法：mIU/L、µIU/mL、uIU/mL、mU/L
        # 它们在数值上等价（mIU/L == uIU/mL），因此直接赋值即可
        out["tsh_u_iu_ml"] = d.tsh_value

    # —— 维生素 D ——
    if d.vitamin_d_value is not None:
        out["vitamin_d_25oh_ng_ml"] = (
            vitd_ngml_from_nmol(d.vitamin_d_value)
            if "nmol" in _norm(d.vitamin_d_unit)
            else d.vitamin_d_value
        )

    # —— 血红蛋白/铁蛋白 ——
    if d.hemoglobin_value is not None:
        u = _norm(d.hemoglobin_unit)
        if "g/l" in u:
            out["hemoglobin_g_dl"] = d.hemoglobin_value / 10.0  # g/L → g/dL
        else:
            out["hemoglobin_g_dl"] = d.hemoglobin_value  # g/dL 直接用
    if d.ferritin_value is not None:
        u = _norm(d.ferritin_unit)
        # 目标单位 ng/mL；常见 μg/L、ug/L、mcg/L 与 ng/mL 数值等价
        if any(tok in u for tok in ["μg/l", "ug/l", "mcg/l"]):
            out["ferritin_ng_ml"] = d.ferritin_value
        else:
            out["ferritin_ng_ml"] = d.ferritin_value  # 已是 ng/mL

    # —— 产科 ——
    if d.gestational_weeks_value is not None:
        out["gestational_weeks"] = d.gestational_weeks_value
    out["gdm"] = d.gdm

    # —— 用药 ——
    meds: List[str] = []
    if isinstance(d.medications, list) and d.medications:
        meds.extend([m.strip() for m in d.medications if isinstance(m, str) and m.strip()])
    if d.medications_text:
        meds.extend(_split_med_list(d.medications_text))
    # 去重保持顺序
    seen = set(); meds_out: List[str] = []
    for m in meds:
        k = m.lower()
        if k not in seen:
            seen.add(k); meds_out.append(m)
    out["medications"] = meds_out

    # —— 备注 ——
    out["notes"] = d.notes
    return out

# -----------------------------
# LLM 结构化提取（文本→JSON）
# -----------------------------
def _llm_structured_from_text(text: str, model: str = "gpt-4o-mini") -> HRExtract:
    llm = ChatOpenAI(model=model, temperature=0)
    parser = PydanticOutputParser(pydantic_object=HRExtract)
    prompt = f"""
你是医疗体检报告的信息抽取助手。请从以下中文文本中抽取指标，
尽量保留原单位（如 mmol/L、mg/dL、µmol/L、ng/mL、g/L、U/L、mIU/L 等），没写的留空。
只输出 JSON（不要额外解释）。字段尽量按原文精确抄写单位与数值。

必须尝试抽取的字段（缺失可留空）：
- 收缩压/舒张压（mmHg）
- HbA1c（%）
- 血糖：空腹血糖（值+单位 mg/dL 或 mmol/L）、OGTT 2h（值+单位）
- 血脂：LDL/HDL/甘油三酯/总胆固醇/非HDL（各自的值+单位 mg/dL 或 mmol/L）
- 肾功能：肌酐（值+单位 mg/dL 或 µmol/L/umol/L）、eGFR（值+单位，mL/min/1.73m²）
- 肝功能：ALT、AST（值+单位 U/L 或 IU/L）
- 尿酸（值+单位 mg/dL 或 µmol/L/umol/L）
- TSH（值+单位：uIU/mL、µIU/mL、mIU/L 或 mU/L）
- 维生素D（25(OH)D，值+单位 ng/mL 或 nmol/L）
- 血红蛋白（值+单位 g/dL 或 g/L）
- 铁蛋白（值+单位 ng/mL 或 μg/L/ug/L）
- 孕周（周数，数字）
- 妊娠期糖尿病（gdm：是/否）
- 用药（若有，尽量抽取为列表；或以逗号/顿号/分号分隔的字符串）
- 备注（可选）

文本：
{text}

严格按以下 Pydantic 模式输出（仅 JSON）：
{parser.get_format_instructions()}
"""
    raw = llm.invoke(prompt).content
    return parser.parse(raw)


# -----------------------------
# 视觉 OCR（图片→纯文本）走 LLM
# -----------------------------
def preprocess_for_ocr(img_bytes: bytes) -> bytes:
    """
    对体检单进行“保护性预处理”：
    - 盖掉 Header（机构抬头/姓名/证件号）
    - 盖掉 Footer（电话/二维码/条码）
    - 额外盖掉右下角（常见二维码区域）
    - 盖掉中上方一条（有些报告标题下仍有 PII）
    目标：尽量降低被模型因 PII 拒绝的概率，同时不影响主要化验区块。
    """
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = im.size
    draw = ImageDraw.Draw(im)

    # Header：约 22% 高度
    draw.rectangle([0, 0, w, int(h * 0.22)], fill="white")

    # Footer：约 90%~100%
    draw.rectangle([0, int(h * 0.90), w, h], fill="white")

    # 右下角二维码/条形码常见区域
    draw.rectangle([int(w * 0.78), int(h * 0.78), w, h], fill="white")

    # 中上方再盖一条（视情况保守一点）
    draw.rectangle([0, int(h * 0.22), w, int(h * 0.27)], fill="white")

    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()

def _llm_ocr_image_to_text(image_bytes: bytes, model: str = "gpt-4o-mini", do_preprocess: bool = True) -> str:
    """
    用 LLM 的视觉能力从图片中“抄录”检测结果为纯文本。
    返回值直接再喂 _llm_structured_from_text。
    """
    # 先做图像预处理，尽量去掉可能触发拒答的 PII/二维码/机构抬头
    if do_preprocess:
        try:
            image_bytes = preprocess_for_ocr(image_bytes)
        except Exception:
            # 任何图像处理失败，直接回落到原图
            pass
        
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    llm = ChatOpenAI(model=model, temperature=0)
    # LangChain 支持用“多模态内容”作为 HumanMessage content
    # 我们要求模型输出“原文数值+单位的纯文本”，不需要解释。
    content = [
        {
            "type": "text",
            "text": (
                "这是一张中文体检/化验单图片。"
                "请仅抄录与以下医学指标相关的条目为纯文本：血压、血糖、HbA1c、"
                "血脂(LDL/HDL/甘油三酯/总胆固醇/非HDL)、肌酐、eGFR、尿酸、ALT、AST、"
                "25(OH)D、血红蛋白、铁蛋白、孕周、妊娠期糖尿病、用药、备注。"
                "【重要】忽略并不要输出任何个人身份或联系方式信息（如姓名、证件号、用户ID、电话号码、地址、二维码/条形码等）。"
                "直接逐条输出“项目 数值 单位”的原文文本，不要解释、不翻译。缺失的项目不必输出。"
            ),
        },
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}" }},
    ]
    raw = llm.invoke([HumanMessage(content=content)]).content
    return raw or ""


# -----------------------------
# 文档加载：PDF / DOCX
# -----------------------------
def _pdf_to_text(pdf_bytes: bytes) -> str:
    try:
        import pypdf
    except Exception as e:
        raise RuntimeError("需要依赖 pypdf 来读取 PDF 文本，请先安装。") from e
    reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
    texts: List[str] = []
    for page in reader.pages:
        texts.append(page.extract_text() or "")
    return "\n".join(texts).strip()

def _docx_to_text(docx_bytes: bytes) -> str:
    try:
        import docx
    except Exception as e:
        raise RuntimeError("需要依赖 python-docx 来读取 DOCX 文本，请先安装。") from e
    doc = docx.Document(io.BytesIO(docx_bytes))
    return "\n".join(p.text for p in doc.paragraphs).strip()


# -----------------------------
# 统一入口
# -----------------------------
SourceType = Literal["text", "image", "pdf", "docx", "auto"]

@dataclass
class HRSource:
    data: Union[str, bytes]              # 文本字符串 或 文件字节
    kind: SourceType = "auto"            # 自动判别或指定
    filename: Optional[str] = None       # 用于 auto 判别扩展名
    use_vision_ocr_for_pdf: bool = False # 若 PDF 疑似扫描件，可选用 LLM 视觉 OCR（需要额外依赖将 PDF 转图片）


def extract_health_report(source: HRSource, model: str = "gpt-4o-mini") -> HealthReport:
    """
    统一入口：
      - 文本（含 OCR 文本）→ 直接结构化
      - 图片 → LLM 视觉 OCR → 文本 → 结构化
      - PDF/DOCX → 先抽文本 → 结构化
      - PDF 扫描件：可选 use_vision_ocr_for_pdf=True（需要你自行在 TODO 中将 PDF 转页图）

    返回：models.schemas.HealthReport
    """
    # 判别来源类型
    kind = source.kind
    if kind == "auto":
        ext = (os.path.splitext(source.filename or "")[1] or "").lower()
        if isinstance(source.data, str):
            kind = "text"
        elif ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            kind = "image"
        elif ext == ".pdf":
            kind = "pdf"
        elif ext == ".docx":
            kind = "docx"
        else:
            # 未知时尝试按文本解析失败再按图片
            kind = "text" if isinstance(source.data, str) else "image"

    # 路由
    if kind == "text":
        text = source.data if isinstance(source.data, str) else source.data.decode("utf-8", errors="ignore")
        extracted = _llm_structured_from_text(text, model=model)
        unified = _to_mgdl_or_ngml(extracted)
        return HealthReport(**unified)

    elif kind == "image":
        if not isinstance(source.data, (bytes, bytearray)):
            raise ValueError("图片模式需要 bytes 数据")
        ocr_text = _llm_ocr_image_to_text(bytes(source.data), model=model)
        extracted = _llm_structured_from_text(ocr_text, model=model)
        unified = _to_mgdl_or_ngml(extracted)
        return HealthReport(**unified)

    elif kind == "pdf":
        if not isinstance(source.data, (bytes, bytearray)):
            raise ValueError("PDF 模式需要 bytes 数据")
        text = _pdf_to_text(bytes(source.data))
        if (not text or len(text) < 30) and source.use_vision_ocr_for_pdf:
            # TODO（可选）：将 PDF 转为图片逐页喂 _llm_ocr_image_to_text
            # 参考：pdf2image.convert_from_bytes(pdf_bytes) -> List[PIL.Image]
            # 然后把每页转 PNG bytes 做 _llm_ocr_image_to_text，最后拼接
            pass
        extracted = _llm_structured_from_text(text, model=model)
        unified = _to_mgdl_or_ngml(extracted)
        return HealthReport(**unified)

    elif kind == "docx":
        if not isinstance(source.data, (bytes, bytearray)):
            raise ValueError("DOCX 模式需要 bytes 数据")
        text = _docx_to_text(bytes(source.data))
        extracted = _llm_structured_from_text(text, model=model)
        unified = _to_mgdl_or_ngml(extracted)
        return HealthReport(**unified)

    else:
        raise ValueError(f"不支持的来源类型：{kind!r}")


# -----------------------------
# 兼容旧API（文本版）
# -----------------------------
def extract_health_report_from_zh(text: str, model: str = "gpt-4o-mini") -> HealthReport:
    """
    兼容旧函数：输入中文文本，返回 HealthReport。
    """
    extracted = _llm_structured_from_text(text, model=model)
    unified = _to_mgdl_or_ngml(extracted)
    return HealthReport(**unified)


def merge_health_reports(reports: Iterable[HealthReport]) -> HealthReport:
    """
    将多个 HealthReport 合并为一个：
    - 规则：后者覆盖前者（later wins），None 不覆盖
    - medications：合并去重，保序
    - notes：拼接（用 " | " 连接）
    """
    merged = HealthReport()  # 全部字段 Optional，空构造合法
    meds_acc: List[str] = []
    notes_acc: List[str] = []

    def _merge_one(rep: HealthReport):
        nonlocal meds_acc, notes_acc
        # 常规字段：后者覆盖前者（遍历“类”的字段定义，避免实例上的已弃用属性）
        for name in HealthReport.model_fields.keys():
            if name in {"medications", "notes"}:
                continue
            val = getattr(rep, name, None)
            if val is not None:
                setattr(merged, name, val)

        # medications：合并去重
        if rep.medications:
            seen = {m.lower(): m for m in meds_acc}
            for m in rep.medications:
                k = m.lower()
                if k not in seen:
                    meds_acc.append(m)
                    seen[k] = m

        # notes：拼接
        if rep.notes:
            notes_acc.append(rep.notes)

    for r in reports:
        _merge_one(r)

    if meds_acc:
        merged.medications = meds_acc
    if notes_acc:
        merged.notes = " | ".join(notes_acc)

    return merged

def extract_health_report_multi(sources: List[HRSource], model: str = "gpt-4o-mini") -> HealthReport:
    """
    依次解析多份来源（文本/图片/PDF/DOCX），并合并为一个 HealthReport。
    后者覆盖前者。
    """
    reps: List[HealthReport] = []
    for src in sources:
        reps.append(extract_health_report(src, model=model))
    return merge_health_reports(reps)
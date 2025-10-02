import re
from typing import Final

# 预编译：含/不含 ASCII 引号两套标点模式
_PUNCT_WITH_QUOTES: Final[re.Pattern] = re.compile(r'[，。、“”‘’"\'！（）()【】\[\]{}：:；;·\-—_、/\\]')
_PUNCT_NO_QUOTES:   Final[re.Pattern] = re.compile(r'[，。、“”‘’！（）()【】\[\]{}：:；;·\-—_、/\\]')
_CJK_ANY:           Final[re.Pattern] = re.compile(r'[\u4e00-\u9fff]')  # 简体常用汉字区

def _dbc2sbc_char(ch: str) -> str:
    """全角→半角（含全角空格）。"""
    code = ord(ch)
    if code == 0x3000:      # 全角空格
        return " "
    if 0xFF01 <= code <= 0xFF5E:
        return chr(code - 0xFEE0)
    return ch

def normalize_zh_unified(
    s: str | None,
    *,
    delete_quotes: bool = True,        # 是否删除 ASCII 引号 " 和 '
    strip_all_spaces_if_cjk: bool = True,  # 若含中文字符，是否移除所有空格（防 OCR 分词）
) -> str:
    """
    中文字符串规范化（统一版）：
    1) 去首尾空格、转小写
    2) 全角→半角（包含全角空格、全角符号）
    3) 删除常见中英文标点（可选是否包含 ASCII 引号）
    4) 压缩多空格为单空格
    5) 可选：若含中文字符，移除全部空格（适合 OCR/手输被分开的汉字）

    参数:
        delete_quotes: 删除 ASCII 引号 " 和 '（对应你第一版的做法）
        strip_all_spaces_if_cjk: 若检测到中文字符，删除全部空格（对应你第一版的做法）
    """
    s = (s or "").strip().lower()

    # 全角→半角
    s = "".join(_dbc2sbc_char(ch) for ch in s)

    # 删标点
    punct_re = _PUNCT_WITH_QUOTES if delete_quotes else _PUNCT_NO_QUOTES
    s = punct_re.sub(" ", s)

    # 压缩空格
    s = re.sub(r"\s+", " ", s).strip()

    # 含中文则去掉所有空格（可选）
    if strip_all_spaces_if_cjk and _CJK_ANY.search(s):
        s = s.replace(" ", "")

    return s

# ---------- 兼容别名：复刻两版原始行为 ----------
def normalize_zh(s: str) -> str:
    """等价于你第一版：删除引号 + 中文时去掉所有空格"""
    return normalize_zh_unified(s, delete_quotes=True, strip_all_spaces_if_cjk=True)

def _normalize_zh(s: str) -> str:
    """等价于你第二版：保留引号 + 不清除中文里的空格"""
    return normalize_zh_unified(s, delete_quotes=False, strip_all_spaces_if_cjk=False)

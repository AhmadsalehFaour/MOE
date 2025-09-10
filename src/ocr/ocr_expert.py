# -*- coding: utf-8 -*-
"""
OCR Expert
----------
- استخراج نص متعدد اللغات من الصور باستخدام Tesseract عبر pytesseract.
- كشف اللغة تلقائياً (langdetect بشكل افتراضي، وfastText LID-176 إن توفر).
- ترميم الكلمات منخفضة الثقة عبر LLaMA من خلال Ollama باستخدام سياق سابق/لاحق.
- إرجاع رموز OCR (tokens) مع الصناديق (bboxes) والثقة.

المتطلبات (تمت إضافتها في requirements/Dockerfile المقترح):
- pillow
- pytesseract
- langdetect
- requests
- (اختياري) fasttext + نموذج lid.176.bin

بيئة Ollama:
- OLLAMA_URL (افتراضي: http://127.0.0.1:11434)
- OLLAMA_MODEL (افتراضي: llama3)
- FASTTEXT_LID_MODEL (اختياري: مسار نموذج LID-176، افتراضي: /app/models/lid.176.bin)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any
import os
import re
import logging

import pytesseract
from pytesseract import Output

# كشف اللغة (خفيف)
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except Exception:
    _HAS_LANGDETECT = False
    detect = None  # type: ignore

# fastText (اختياري وأدق)
try:
    import fasttext  # type: ignore
    _HAS_FASTTEXT = True
except Exception:
    _HAS_FASTTEXT = False
    fasttext = None  # type: ignore

import requests

# إعداد سجلّات خفيفة
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ---------- بيانات ونتائج ----------

@dataclass
class OCRToken:
    text: str
    conf: float                       # [0..1]
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    block_num: Optional[int] = None
    par_num: Optional[int] = None
    line_num: Optional[int] = None
    word_num: Optional[int] = None

@dataclass
class OCRResult:
    language: str                     # ISO-639-1 إن أمكن (ar, en, ...)
    text: str                         # نص مُعاد تركيبه بأسطر
    tokens: List[OCRToken]

# ---------- أدوات مساعدة ----------

_TESS_TO_ISO = {
    "eng": "en",
    "ara": "ar",
    "fra": "fr",
    "deu": "de",
    "spa": "es",
    "ita": "it",
    "tur": "tr",
    "rus": "ru",
    "chi_sim": "zh",
    "chi_tra": "zh",
}

def tesseract_code_to_iso(code: str) -> str:
    code = (code or "").lower()
    return _TESS_TO_ISO.get(code, code[:2] if len(code) >= 2 else code)

def _normalize_conf(val: Any) -> float:
    """يحّول ثقة pytesseract إلى [0..1]."""
    try:
        f = float(val)
    except Exception:
        return 0.0
    if f < 0:
        return 0.0
    # pytesseract عادةً 0..100
    if f > 1.0:
        f = f / 100.0
    if f > 1.0:
        f = 1.0
    return f

def _build_tesseract_config(psm: Optional[int], oem: Optional[int]) -> str:
    cfg = []
    if psm is not None:
        cfg += ["--psm", str(psm)]
    if oem is not None:
        cfg += ["--oem", str(oem)]
    return " ".join(cfg)

# ---------- كشف اللغة ----------

_FT_MODEL = None  # fastText model (global singleton)

def _load_fasttext_if_available() -> None:
    """تحميل نموذج fastText إن أمكن وكان الملف موجوداً."""
    global _FT_MODEL
    if not _HAS_FASTTEXT:
        return
    if _FT_MODEL is not None:
        return
    model_path = os.getenv("FASTTEXT_LID_MODEL", "/app/models/lid.176.bin")
    if os.path.exists(model_path):
        try:
            logger.info("Loading fastText LID model from %s", model_path)
            _FT_MODEL = fasttext.load_model(model_path)  # type: ignore
        except Exception as e:
            logger.warning("Failed to load fastText model: %s", e)

def _detect_lang_fasttext(text: str) -> Optional[str]:
    if _FT_MODEL is None:
        return None
    t = text.strip().replace("\n", " ")
    if not t:
        return None
    try:
        labels, probs = _FT_MODEL.predict(t, k=1)  # type: ignore
        if not labels:
            return None
        lang = labels[0].replace("__label__", "")
        # أحياناً يرجع مثل 'ar', 'en', وهذا مناسب
        return lang
    except Exception:
        return None

def _detect_lang_langdetect(text: str) -> Optional[str]:
    if not _HAS_LANGDETECT or not detect:
        return None
    t = text.strip()
    if len(t) < 3:
        return None
    try:
        return detect(t)
    except Exception:
        return None

def vote_language(samples: List[str]) -> str:
    """
    تصويت بسيط بين fastText (إن وُجد) وlangdetect على عينات متعددة.
    """
    _load_fasttext_if_available()
    from collections import Counter

    votes: List[str] = []
    for s in samples:
        s = s.strip()
        if len(s) < 3:
            continue

        # fastText أولوية إن وُجد
        ft_lang = _detect_lang_fasttext(s)
        if ft_lang:
            votes.append(ft_lang)

        # langdetect للمساندة
        ld_lang = _detect_lang_langdetect(s)
        if ld_lang:
            votes.append(ld_lang)

    if not votes:
        # إن تعذر الكشف نرجّح العربية لو لغة تيسراكت العربية موجودة عادة
        return "ar"

    lang = Counter(votes).most_common(1)[0][0]
    # توحيد بعض الأكواد الطويلة إلى ISO-639-1 عند الإمكان
    return lang[:2] if len(lang) > 2 else lang

# ---------- ترميم الكلمات منخفضة الثقة عبر LLaMA/Ollama ----------

def llama_fill_unk(
    sentence_with_unk: str,
    ollama_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: int = 120,
) -> str:
    ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3")

    system = "You are a helpful assistant that restores corrupted words in text using context."
    prompt = (
        "Replace each <UNK> in the sentence with the most likely word(s) "
        "based on the left and right context, preserving the original language and style.\n\n"
        f"Sentence:\n{sentence_with_unk}\n\n"
        "Return only the corrected sentence."
    )

    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "system": system, "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning("Ollama call failed (%s). Returning original text.", e)
        return sentence_with_unk.replace("<UNK>", "").strip()

def _window_indices(centers: List[int], total: int, win: int) -> List[Tuple[int, int]]:
    """
    يحسب نوافذ سياقية حول مواضع الـ <UNK> لتقليل طول الطلب إلى LLM.
    """
    bounds: List[Tuple[int, int]] = []
    for c in centers:
        s = max(0, c - win)
        e = min(total, c + win + 1)
        bounds.append((s, e))
    # دمج النوافذ المتداخلة
    if not bounds:
        return []
    bounds.sort()
    merged = [bounds[0]]
    for b in bounds[1:]:
        last_s, last_e = merged[-1]
        if b[0] <= last_e:
            merged[-1] = (last_s, max(last_e, b[1]))
        else:
            merged.append(b)
    return merged

def repair_low_confidence_text(
    tokens: List[OCRToken],
    conf_threshold: float = 0.75,
    use_windows: bool = True,
    window_size: int = 8,
    ollama_url: Optional[str] = None,
    ollama_model: Optional[str] = None,
) -> str:
    """
    يضع <UNK> بدل الكلمات منخفضة الثقة ثم يطلب من LLaMA ترميمها.
    - use_windows: إن كان النص طويلاً، نرسل نوافذ سياق بدلاً من الجملة كاملة.
    """
    if not tokens:
        return ""

    words = [t.text for t in tokens]
    confs = [t.conf for t in tokens]
    # تأكيد التطبيع [0..1]
    confs = [_normalize_conf(c) for c in confs]

    low_idxs = [i for i, c in enumerate(confs) if words[i].strip() and c < conf_threshold]
    if not low_idxs:
        return " ".join(words)

    if not use_windows or len(words) < 64:
        # نهج بسيط: استبدل في النص كله
        seq = words[:]
        for i in low_idxs:
            seq[i] = "<UNK>"
        sentence = " ".join(seq)
        return llama_fill_unk(sentence, ollama_url, ollama_model)

    # نهج النوافذ: ترميم موضعي ثم دمج
    repaired = words[:]
    for s, e in _window_indices(low_idxs, len(words), window_size):
        window_seq = repaired[s:e]
        # استبدل فقط الـ low داخل النافذة
        local = []
        for i, w in enumerate(window_seq, start=s):
            if i in low_idxs:
                local.append("<UNK>")
            else:
                local.append(w)
        window_text = " ".join(local)
        fixed = llama_fill_unk(window_text, ollama_url, ollama_model)
        # دمج: إن فشل أو رجّع نص أقصر بكثير، نتجاهله
        if fixed and fixed.count("<UNK>") <= window_text.count("<UNK>"):
            # تقسيم بسيط للكلمات (قد يختلف العدد)
            fixed_words = fixed.split()
            # إذا الطول مطابق، نبدّل مباشرة
            if len(fixed_words) == len(local):
                repaired[s:e] = fixed_words
            else:
                # محاولة اندماج متسامحة: استبدل الـ <UNK> فقط بمخرجات من fixed
                # نعثر على المواضع بترتيبها
                fi = 0
                for j in range(s, e):
                    if repaired[j] == "<UNK>":
                        # تقدّم في fixed_words للبحث عن أول كلمة ليست <UNK>
                        while fi < len(fixed_words) and fixed_words[fi] == "<UNK>":
                            fi += 1
                        if fi < len(fixed_words):
                            repaired[j] = fixed_words[fi]
                            fi += 1
    return " ".join(repaired)

# ---------- خبير الـ OCR ----------

class OCRExpert:
    """
    خبير OCR باستخدام Tesseract عبر pytesseract.
    - languages: قائمة رموز تيسراكت (eng, ara, ...) تُستخدم عند عدم الكشف التلقائي.
    - auto_detect: إن كانت True سنمرّر اتحاد اللغات إلى تيسراكت ثم نكشف اللغة من النص.
    - psm/oem: خيارات تيسراكت (صفحة/محرك).
    - min_token_len: تجاهل الرموز الأقصر من هذا الطول أثناء إعادة البناء والكشف.
    """

    def __init__(
        self,
        languages: Optional[List[str]] = None,
        auto_detect: bool = True,
        psm: Optional[int] = None,
        oem: Optional[int] = None,
        min_token_len: int = 1,
    ) -> None:
        self.languages = languages or ["ara", "eng"]
        self.auto_detect = auto_detect
        self.psm = psm
        self.oem = oem
        self.min_token_len = min_token_len

    # ----- API -----

    def run_ocr(self, image: Any, lang_codes: Optional[List[str]] = None) -> OCRResult:
        """
        يجري OCR ويعيد النص والرموز. إذا auto_detect=True تمرر اتحاد اللغات إلى تيسراكت.
        """
        use_langs = lang_codes or self.languages
        if not isinstance(use_langs, list) or not use_langs:
            use_langs = self.languages

        tess_lang = "+".join(use_langs)
        config = _build_tesseract_config(self.psm, self.oem)

        data = pytesseract.image_to_data(image, output_type=Output.DICT, lang=tess_lang, config=config)

        tokens: List[OCRToken] = []
        n = len(data.get("text", []))
        for i in range(n):
            txt = data["text"][i]
            if not (txt and txt.strip()):
                continue
            # تجاهل الرموز القصيرة جداً
            if len(txt.strip()) < self.min_token_len:
                continue

            conf = _normalize_conf(data.get("conf", ["-1"])[i])
            x, y, w, h = (
                int(data["left"][i]),
                int(data["top"][i]),
                int(data["width"][i]),
                int(data["height"][i]),
            )
            token = OCRToken(
                text=txt.strip(),
                conf=conf,
                bbox=(x, y, x + w, y + h),
                block_num=int(data.get("block_num", [0])[i]) if "block_num" in data else None,
                par_num=int(data.get("par_num", [0])[i]) if "par_num" in data else None,
                line_num=int(data.get("line_num", [0])[i]) if "line_num" in data else None,
                word_num=int(data.get("word_num", [0])[i]) if "word_num" in data else None,
            )
            tokens.append(token)

        # إعادة بناء نص بأسطر وفق block/line قدر الإمكان
        full_text = self._reconstruct_text(data, tokens)

        # كشف اللغة
        if self.auto_detect:
            # عينات للكشف (أفضل 50 كلمة معقولة)
            samples = [t.text for t in tokens if len(t.text) >= 2][:50]
            lang = vote_language(samples) if samples else "ar"
        else:
            # اختيار ISO مكافئ لأول لغة ممررة
            lang = tesseract_code_to_iso(use_langs[0])

        return OCRResult(language=lang, text=full_text, tokens=tokens)

    # ----- Helpers -----

    def _reconstruct_text(self, raw: Dict[str, List[Any]], tokens: List[OCRToken]) -> str:
        """
        إعادة بناء النص بأسطر باستخدام block_num/par_num/line_num إن توفرت،
        وإلا نرجع دمجاً بسيطاً للكلمات.
        """
        if not tokens:
            return ""

        # إن كان لدينا line_num: نبني سطور
        if "line_num" in raw and "text" in raw:
            lines: Dict[Tuple[int, int, int], List[str]] = {}
            n = len(raw["text"])
            for i in range(n):
                txt = raw["text"][i]
                if not (txt and txt.strip()):
                    continue
                block = int(raw.get("block_num", [0])[i]) if "block_num" in raw else 0
                par = int(raw.get("par_num", [0])[i]) if "par_num" in raw else 0
                line = int(raw.get("line_num", [0])[i]) if "line_num" in raw else 0
                key = (block, par, line)
                lines.setdefault(key, []).append(txt.strip())
            # ترتيب المفاتيح
            ordered = sorted(lines.keys())
            rebuilt_lines = [" ".join(lines[k]) for k in ordered if lines.get(k)]
            return "\n".join([re.sub(r"\s+", " ", ln).strip() for ln in rebuilt_lines if ln.strip()])

        # fallback: دمج بسيط
        return " ".join([t.text for t in tokens])

# ---------- مُخرجات جاهزة للتسلسل ----------

def ocr_result_asdict(res: OCRResult) -> Dict[str, Any]:
    return {
        "language": res.language,
        "text": res.text,
        "tokens": [asdict(t) for t in res.tokens],
    }

__all__ = [
    "OCRToken",
    "OCRResult",
    "OCRExpert",
    "repair_low_confidence_text",
    "llama_fill_unk",
    "ocr_result_asdict",
]

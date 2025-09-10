# -*- coding: utf-8 -*-
"""
MoE Image Describer (updated)
-----------------------------

Features:
- OCR route: Tesseract (via pytesseract) + language detection +
  low-confidence repair with LLaMA (Ollama)
- Vision fallback: tries your project's VisionExpert/Gating/Ollama if
  available; otherwise, uses a generic LLM caption
- Language-aware captioning (ask LLM to write the caption in the requested
  language)

Environment:
- OLLAMA_URL (default: http://127.0.0.1:11434)
- OLLAMA_MODEL (default: llama3)

Depends on:
- requests
- src.ocr.ocr_expert (provided)
- (optionally) your project's vision/gate/llama/prompts modules if present
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import os
import requests

# ---- Optional imports (project-specific).
# We degrade gracefully if missing. ----
try:
    # Your original stack (if exists)
    # Expecting something like:
    #   class VisionExpert:  def __call__(self, img_pil, topk=5) -> Any
    #                         or .extract(img_pil, topk=5) -> Any
    #   class GatingNetwork: def decide(self, vis) -> Any
    #                        with .route/.rationale
    from src.moe.vision import VisionExpert  # type: ignore
except Exception:
    VisionExpert = None  # type: ignore

try:
    from src.moe.gate import GatingNetwork  # type: ignore
except Exception:
    GatingNetwork = None  # type: ignore

# If you have a wrapper around Ollama (nice, but optional)
try:
    from src.moe.llama import OllamaClient  # type: ignore
except Exception:
    OllamaClient = None  # type: ignore

# If you had system prompts/makers in your project
try:
    from src.moe.prompts import SYSTEM_PROMPT  # type: ignore
    from src.moe.prompts import make_caption_prompt  # type: ignore
except Exception:
    SYSTEM_PROMPT, make_caption_prompt = None, None  # type: ignore

# ---- OCR Expert (we provide this module) ----
from src.ocr.ocr_expert import OCRExpert, repair_low_confidence_text, OCRResult


# ============================================================================
# Utility: direct Ollama call (fallback if no OllamaClient wrapper exists)
# ============================================================================
def _ollama_generate(prompt: str,
                     system: Optional[str] = None,
                     model: Optional[str] = None,
                     url: Optional[str] = None,
                     timeout: int = 180) -> str:
    """
    Minimal Ollama call. Returns response string or raises for HTTP errors.
    """
    url = url or os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    model = model or os.getenv("OLLAMA_MODEL", "llama3")
    payload = {"model": model, "prompt": prompt, "stream": False}
    if system:
        payload["system"] = system
    resp = requests.post(f"{url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return (resp.json() or {}).get("response", "").strip()


# ============================================================================
# Helper: try to run vision & extract hints in a robust way
# ============================================================================
def _safe_run_vision(vision, img_pil, topk: int = 5):
    """
    Calls your VisionExpert in a tolerant manner. Supports two common styles:
    - vision(img_pil, topk=topk)
    - vision.extract(img_pil, topk=topk)

    Returns `vis` (opaque object from your project) or None on failure.
    """
    if vision is None:
        return None
    try:
        # common pattern 1: callable vision
        return vision(img_pil, topk=topk)
    except TypeError:
        # common pattern 2: explicit method
        try:
            return vision.extract(img_pil, topk=topk)  # type: ignore
        except Exception:
            return None
    except Exception:
        return None


def _vis_hints(vis) -> Tuple[List[str], List[str], Optional[float]]:
    """
    Extracts (labels, colors, complexity) from various possible shapes of
    `vis`.
    - If vis is a dict, looks for keys: "topk_labels", "colors", "complexity"
    - If object, tries attributes with those names
    - labels may be [(label, prob), ...] or [label, ...];
      we normalize to [str, ...]
    """
    labels: List[str] = []
    colors: List[str] = []
    complexity: Optional[float] = None

    if vis is None:
        return labels, colors, complexity

    # dict shape
    if isinstance(vis, dict):
        raw_labels = vis.get("topk_labels", [])
        colors = vis.get("colors", []) or []
        complexity = vis.get("complexity", None)
    else:
        # object shape
        raw_labels = getattr(vis, "topk_labels", []) or []
        colors = getattr(vis, "colors", []) or []
        complexity = getattr(vis, "complexity", None)

    # normalize labels
    out_labels: List[str] = []
    for item in raw_labels:
        if isinstance(item, (list, tuple)) and item:
            out_labels.append(str(item[0]))
        else:
            out_labels.append(str(item))
    labels = out_labels
    # normalize colors to str
    colors = [str(c) for c in colors if c]
    return labels, colors, complexity


# ============================================================================
# Main class
# ============================================================================
class MoEImageDescriber:
    def __init__(self,
                 device: str = "cpu",
                 ollama_url: str = "http://127.0.0.1:11434",
                 ollama_model: str = "llama3",
                 p_threshold: float = 0.55,
                 entropy_hi: float = 3.5,
                 # OCR
                 ocr_enabled: bool = True,
                 ocr_auto_lang: bool = True,
                 ocr_langs: Optional[List[str]] = None,
                 ocr_conf_threshold: float = 0.75) -> None:

        # ---- store params ----
        self.device = device
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        self.p_threshold = p_threshold
        self.entropy_hi = entropy_hi

        # ---- OCR ----
        self.ocr_enabled = ocr_enabled
        self.ocr_auto_lang = ocr_auto_lang
        self.ocr_langs = ocr_langs or ["ara", "eng"]
        self.ocr_conf_threshold = ocr_conf_threshold
        self.ocr = OCRExpert(languages=self.ocr_langs,
                             auto_detect=self.ocr_auto_lang)

        # ---- Try to wire your original components (if present) ----
        self.vision = None
        self.gate = None
        self.llama = None

        if VisionExpert is not None:
            try:
                self.vision = VisionExpert(device=device)  # type: ignore
            except Exception:
                self.vision = None

        if GatingNetwork is not None:
            try:
                self.gate = GatingNetwork(p_threshold=p_threshold,
                                          entropy_hi=entropy_hi) 
                # type: ignore
            except Exception:
                self.gate = None

        if OllamaClient is not None:
            try:
                self.llama = OllamaClient(ollama_url, ollama_model) 
                # type: ignore
            except Exception:
                self.llama = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def describe(self,
                 img_pil,
                 topk: int = 5,
                 max_tokens: int = 200,
                 language: str = "English") -> Dict[str, Any]:
        """
        Entry point. If OCR is enabled and returns non-empty text -> OCR route.
        Else -> Vision (or generic) caption route.
        """
        # ------------------- OCR route -------------------
        if self.ocr_enabled:
            ocr_res: OCRResult = self.ocr.run_ocr(
                img_pil,
                lang_codes=None if self.ocr_auto_lang else self.ocr_langs
            )
            if len((ocr_res.text or "").strip()) >= 3:
                repaired = repair_low_confidence_text(
                    ocr_res.tokens,
                    conf_threshold=self.ocr_conf_threshold,
                    ollama_url=self.ollama_url,
                    ollama_model=self.ollama_model,
                )
                explanation = self._explain_text_with_llm(
                    text=repaired,
                    source_lang=ocr_res.language,
                    target_lang=language,
                    max_tokens=max_tokens
                )
                return {
                    "route": "ocr",
                    "language": ocr_res.language,
                    "text_raw": ocr_res.text,
                    "text_repaired": repaired,
                    "explanation": explanation,
                    "tokens": [t.__dict__ for t in ocr_res.tokens],
                    # let UI show something in caption box
                    "caption": repaired,
                    "rationale": {
                        "note": (
                            "OCR path chosen; low-confidence tokens repaired "
                            "with LLaMA."
                        ),
                        "ui_language": language,
                    },
                }

        # ------------------- Vision (or generic) route -------------------
        return self._normal_caption_path(img_pil, topk, max_tokens, language)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _normal_caption_path(self,
                             img_pil,
                             topk: int,
                             max_tokens: int,
                             language: str) -> Dict[str, Any]:
        """
        Tries your original Vision + Gate + LLaMA/template pipeline.
        If unavailable, generates a generic caption using LLM.
        """
        vis = _safe_run_vision(self.vision, img_pil, topk=topk)
        labels, colors, complexity = _vis_hints(vis)

        # If your GatingNetwork exists, we attempt to use it to set a route
        # and rationale.
        decision = None
        if self.gate is not None and vis is not None:
            try:
                decision = self.gate.decide(vis)  # type: ignore
            except Exception:
                decision = None

        # prefer your LLaMA wrapper if present
        if self.llama is not None:
            try:
                # If your project has a prompt maker
                if (
                    make_caption_prompt is not None
                    and SYSTEM_PROMPT is not None
                    and labels
                ):
                    user_prompt = make_caption_prompt(
                        labels, colors, complexity
                    )  # type: ignore
                    user_prompt += f"\n\nWrite the caption in {language}."
                    caption = self.llama.generate(
                        SYSTEM_PROMPT, user_prompt, max_tokens=max_tokens
                    )  # type: ignore
                else:
                    # Simple prompt using hints (or generic if none)
                    caption = self._llm_caption_via_wrapper(
                        labels, colors, language, max_tokens
                    )
            except Exception:
                caption = self._llm_caption_direct(
                    labels, colors, language, max_tokens
                )
        else:
            # direct Ollama call
            caption = self._llm_caption_direct(
                labels, colors, language, max_tokens
            )

        result: Dict[str, Any] = {
            "route": "vision",
            "caption": caption,
            "labels": labels,
            "colors": colors,
            "complexity": complexity,
            "rationale": {
                "note": (
                    (
                        (
                            (
                                "Vision route (fallback). "
                                "If your project's gate/vision exist, "
                                "they were used where possible."
                            )
                        )
                    )
                )
            },
        }

        # If we had a gate decision, include it
        if decision is not None:
            try:
                result["route"] = getattr(decision, "route", "vision")
                result["rationale"] = getattr(decision, "rationale",
                                              result["rationale"])
            except Exception:
                pass

        return result

    def _llm_caption_via_wrapper(self,
                                 labels: List[str],
                                 colors: List[str],
                                 language: str,
                                 max_tokens: int) -> str:
        """
        Uses your OllamaClient wrapper if available.
        """
        assert self.llama is not None
        system = "You are a helpful captioning assistant."
        hints = ""
        if labels:
            hints += f"Objects: {', '.join(labels[:7])}.\n"
        if colors:
            hints += f"Prominent colors: {', '.join(colors[:5])}.\n"
        prompt = (
            f"Write a concise, natural image caption in {language}.\n"
            f"{hints}"
            "If hints are empty, write a generic but plausible caption "
            "without hallucinating specific brands or text."
        )
        return self.llama.generate(system, prompt, max_tokens=max_tokens)

    def _llm_caption_direct(self,
                            labels: List[str],
                            colors: List[str],
                            language: str,
                            max_tokens: int) -> str:
        """
        Calls Ollama directly (no wrapper).
        """
        system = "You are a helpful captioning assistant."
        hints = ""
        if labels:
            hints += f"Objects: {', '.join(labels[:7])}.\n"
        if colors:
            hints += f"Prominent colors: {', '.join(colors[:5])}.\n"
        prompt = (
            f"{system}\n\n"
            f"Write a concise, natural image caption in {language}.\n"
            f"{hints}"
            "If hints are empty, write a generic caption without inventing" +
            "specific text from the image."
        )
        try:
            return _ollama_generate(
                prompt=prompt,
                system=None,
                model=self.ollama_model,
                url=self.ollama_url,
                timeout=180
            )
        except Exception:
            # last-resort template
            base = "A photo"
            if labels:
                base += " of " + ", ".join(labels[:3])
            if colors:
                base += " with " + ", ".join(colors[:3]) + " tones"
            base += "."
            # Translate with a tiny follow-up, if language isn't English
            if language.lower().startswith(("en", "english")):
                return base
            try:
                trans_prompt = (
                    "You are a translator.\n"
                    f"Translate to {language} " +
                    "WITHOUT adding extra words:\n\n{base}"
                )
                return _ollama_generate(prompt=trans_prompt, system=None,
                                        model=self.ollama_model,
                                        url=self.ollama_url, timeout=60)
            except Exception:
                return base

    def _explain_text_with_llm(self, text: str, source_lang: str,
                               target_lang: str, max_tokens: int) -> str:
        """
        Summarize/explain OCR-extracted text in the *target* UI language.

        Args:
            text: The extracted (repaired) text from OCR.
            source_lang: Detected language of the source text (e.g., "en", "ar").
            target_lang: Desired output language for the user (e.g., "Arabic", "ar").
            max_tokens: Soft cap for response length (the helper may ignore if not supported).

        Returns:
            A summary + explanation in the requested target language, 
            or a short fallback.
        """
        tgt = (target_lang or "").strip()
        sys_msg = (
            "You are a multilingual assistant that summarizes and explains extracted text accurately. "
            "Do not invent facts. Preserve numbers and named entities exactly as written."
        )

        prompt = (
            f"The following text is in language: {source_lang}.\n\n"
            f"Return your entire answer STRICTLY in {tgt}.\n"
            f"Keep the whole answer concise (≈{max_tokens} tokens or fewer).\n\n"
            "Text:\n"
            "```text\n"
            f"{text}\n"
            "```\n\n"
            "Tasks:\n"
            f"- Provide a concise summary (2–3 sentences) in {tgt}.\n"
            f"- Explain any uncommon or domain-specific terms (if present) in {tgt}.\n"
            f"- If there are entities like dates, ID numbers, phone numbers, company names, list them in {tgt}.\n"
            f"- If translation is needed, translate faithfully to {tgt} without inventing details or changing numbers.\n"
            f"- If a section is not applicable, write 'None' in {tgt}.\n"
            f"- IMPORTANT: Answer strictly in {tgt}."
        )

        try:
            # Uses the helper defined earlier in this module.
            return _ollama_generate(
                prompt=prompt,
                system=sys_msg,
                model=self.ollama_model,
                url=self.ollama_url,
                timeout=180,
            )
        except Exception as e:
            # Best-effort graceful fallback in the target language.
            try:
                import logging
                logging.getLogger(__name__).exception("Explain-with-LLM failed")
            except Exception:
                pass

            if tgt.lower() in ("ar", "arabic", "العربية", "عربي"):
                return f"(تعذّر إنشاء الشرح) {text[:400]}"
            else:
                return f"(Summary unavailable) {text[:400]}"

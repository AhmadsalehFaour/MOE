# app.py
import os
import traceback
import streamlit as st
from PIL import Image
from src.moe.describer import MoEImageDescriber

# ===================== Page config (must be first) =====================
st.set_page_config(
    page_title="MoE Image Captioner + OCR",
    page_icon="🤖",
    layout="centered"
)

# ===================== Styles (RTL, monospace) =========================
st.markdown(
    """
<style>
.rtl {
  direction: rtl;
  text-align: right;
  line-height: 1.8;
  font-size: 1.05rem;
  font-family: "Noto Naskh Arabic", "Amiri", "Segoe UI", Tahoma, sans-serif;
}
.mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "
    Liberation Mono", "Courier New", monospace;
  font-size: 0.95rem;
  white-space: pre-wrap;
}
.small-dim {
  opacity: 0.8;
  font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===================== Helpers =====================
def render_text(text: str, language: str | None):
    """Render text with RTL if Arabic-like."""
    lang = (language or "").strip().lower()
    if lang in ("arabic", "ar", "العربية", "عربي"):
        st.markdown(f'<div class="rtl">{text}</div>', unsafe_allow_html=True)
    else:
        st.markdown(text)


def env_default(name: str, fallback: str) -> str:
    return os.getenv(name, fallback)


# Keep last result across reruns
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None

# ===================== Header =====================
st.title("🖼️ Mixture-of-Experts Image Captioner + 📝 OCR")
st.caption(
    "Vision (ResNet-50) + Gating + Ollama (e.g., llama3) + "
    "OCR (Tesseract)"
)

# ===================== Settings =====================
with st.expander("Settings", expanded=False):
    # LLM/Ollama
    ollama_url = st.text_input(
        "Ollama URL",
        env_default("OLLAMA_URL", "http://127.0.0.1:11434")
    )
    ollama_model = st.text_input(
        "Ollama Model", env_default("OLLAMA_MODEL", "llama3")
    )
    # Gate / Vision thresholds (kept for compatibility)
    threshold = st.slider("Gate: confidence threshold", 0.0, 1.0, 0.55, 0.01)
    entropy_hi = st.slider("Gate: entropy upper bound", 0.0, 6.0, 3.5, 0.1)
    # add 'cuda' in your own GPU image if needed
    device = st.selectbox("Device", ["cpu"], index=0)
    caption_lang = st.selectbox(
        "Caption language (vision route)",
        ["English", "Spanish", "French", "German", "Arabic"]
    )
with st.expander("OCR", expanded=True):
    ocr_enabled = st.checkbox("Enable OCR route", value=True)
    ocr_auto_lang = st.checkbox("Auto-detect OCR language", value=True)
    # Tesseract language packs (codes): eng, ara, fra, deu, spa ...
    ocr_langs = st.multiselect(
        "OCR languages (when not auto)",
        options=["ara", "eng", "fra", "deu", "spa"],
        default=["ara", "eng"]
    )
    ocr_conf_threshold = st.slider(
        "Low-confidence repair threshold", 0.0, 1.0, 0.75, 0.01
    )

# ===================== Upload =====================
uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    try:
        img = Image.open(uploaded).convert("RGB")
    except Exception as e:
        st.error(f"Failed to open image: {e}")
        img = None

    if img is not None:
        st.image(img, caption="Uploaded image", use_container_width=True)

        if st.button("Generate caption / OCR"):
            st.session_state.error = None
            try:
                moe = MoEImageDescriber(
                    device=device,
                    ollama_url=ollama_url,
                    ollama_model=ollama_model,
                    p_threshold=threshold,
                    entropy_hi=entropy_hi,
                    # OCR wiring
                    ocr_enabled=ocr_enabled,
                    ocr_auto_lang=ocr_auto_lang,
                    ocr_langs=ocr_langs,
                    ocr_conf_threshold=ocr_conf_threshold,
                )
                with st.spinner("Thinking..."):
                    result = moe.describe(
                        img,
                        topk=5,
                        max_tokens=200,
                        language=caption_lang
                    )
                    st.session_state.result = result
            except Exception as e:
                st.session_state.error = f"{type(e).__name__}: {e}"
                st.session_state.result = None
                st.error("Processing failed. See details below.")

# ===================== Output =====================
res = st.session_state.result
err = st.session_state.error

if err:
    with st.expander("Error details", expanded=False):
        st.markdown(f"<div class='mono'>{err}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='mono small-dim'>{traceback.format_exc()}</div>",
            unsafe_allow_html=True
        )

if res:
    route = str(res.get("route", ""))
    st.markdown(f"**Selected route:** `{route}`")

    # ---------- OCR route ----------
    if route == "ocr":
        st.subheader("Detected OCR language")
        st.code(res.get("language", ""), language="text")

        st.subheader("Extracted Text (repaired)")
        render_text(res.get("text_repaired", ""), res.get("language", "en"))

        with st.expander("Raw OCR (preview)"):
            # Show first 80 tokens to avoid huge output
            tokens = res.get("tokens", [])
            preview = tokens[:80] if isinstance(tokens, list) else tokens
            st.json({"len": len(tokens) if isinstance(tokens, list) else 0,
                     "preview": preview}, expanded=False)

        st.subheader("Explanation")
        # ← استخدم لغة الواجهة
        render_text(res.get("explanation", ""), caption_lang)
        st.divider()

    # ---------- Vision route (or shared caption) ----------
    st.subheader("Caption (Vision Route or Shared)")
    render_text(res.get("caption", ""), caption_lang)

    if res.get("rationale"):
        with st.expander("Rationale"):
            st.json(res.get("rationale"), expanded=False)

# ===================== Footer =====================
st.caption(
    "Tip: Use Docker build args to add more Tesseract languages "
    "or enable fastText LID."
)
st.markdown("---")

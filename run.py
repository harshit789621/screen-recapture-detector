import time
import io
import pandas as pd
from PIL import Image
import streamlit as st
from predict import predict
st.set_page_config(
    page_title="Screen Recapture Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown("""
<style>
/* ── Page background ── */
[data-testid="stAppViewContainer"] {
    background: #0d0f14;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* ── Verdict card ── */
.verdict-card {
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
    margin-top: 12px;
}
.verdict-real {
    background: #0d2b1e;
    border: 1.5px solid #1a7a4a;
}
.verdict-fake {
    background: #2b0d0d;
    border: 1.5px solid #7a1a1a;
}
.verdict-label {
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.verdict-real .verdict-label  { color: #4ade80; }
.verdict-fake .verdict-label  { color: #f87171; }
.verdict-score {
    font-size: 52px;
    font-weight: 800;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
}
.verdict-real .verdict-score  { color: #4ade80; }
.verdict-fake .verdict-score  { color: #f87171; }
.verdict-desc {
    font-size: 13px;
    color: #8b8f9e;
    margin-top: 8px;
}

/* ── Score bar ── */
.bar-wrap {
    background: #1c1f2b;
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    margin: 14px 0 6px;
}
.bar-fill-real {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #1a7a4a, #4ade80);
    transition: width 0.4s ease;
}
.bar-fill-fake {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #7a1a1a, #f87171);
    transition: width 0.4s ease;
}
.bar-labels {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: #555a6b;
}

/* ── Bulk table ── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* ── Metrics row ── */
[data-testid="stMetric"] {
    background: #151821;
    border: 1px solid #252836;
    border-radius: 10px;
    padding: 16px 20px;
}

/* ── Tab styling ── */
[data-testid="stTabs"] button {
    font-weight: 600;
    letter-spacing: 0.04em;
}

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

THRESHOLD = 0.50   # scores above this are flagged as screen / fake

def score_to_html(score: float) -> str:
    """Render the colour-coded verdict card + bar."""
    pct    = int(score * 100)
    is_fake = score >= THRESHOLD

    card_cls = "verdict-fake" if is_fake else "verdict-real"
    bar_cls  = "bar-fill-fake" if is_fake else "bar-fill-real"
    label    = "⚠️  SCREEN / RECAPTURE" if is_fake else "✅  REAL PHOTO"
    desc     = (
        "This image shows the characteristics of a re-photographed screen."
        if is_fake else
        "This image shows the characteristics of a genuine real-world photograph."
    )

    return f"""
    <div class="verdict-card {card_cls}">
        <div class="verdict-label">{label}</div>
        <div class="verdict-score">{score:.4f}</div>
        <div class="bar-wrap">
            <div class="{bar_cls}" style="width:{pct}%"></div>
        </div>
        <div class="bar-labels"><span>Real (0)</span><span>Screen (1)</span></div>
        <div class="verdict-desc">{desc}</div>
    </div>
    """
def run_predict(uploaded_file) -> tuple[float, float]:
    """Call predict() and return (score, latency_ms)."""
    # Reset pointer — st.image() may have consumed it
    uploaded_file.seek(0)
    t0    = time.perf_counter()
    score = predict(uploaded_file)
    ms    = (time.perf_counter() - t0) * 1000
    return score, ms
st.markdown("""
<div style="padding: 8px 0 24px">
    <span style="font-size:11px;font-weight:700;letter-spacing:.18em;
                 text-transform:uppercase;color:#555a6b">
    </span>
    <h1 style="margin:6px 0 4px;font-size:2rem;font-weight:800;color:#e8eaf0">
        🔍 Screen Recapture Detector
    </h1>
    <p style="margin:0;color:#8b8f9e;font-size:14px">
        Detects whether an image is a genuine photograph or a
        photo-of-a-screen (recapture fraud).&nbsp;
        Score 0 = real · Score 1 = screen.
    </p>
</div>
""", unsafe_allow_html=True)

tab_single, tab_bulk = st.tabs(["📷  Single Image", "📁  Bulk Upload"])
with tab_single:
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload one image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="single",
        help="Supports JPG, PNG, BMP, WEBP",
    )

    if uploaded is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col_img, col_gap, col_result = st.columns([5, 0.4, 4])

        with col_img:
            st.image(uploaded, caption=uploaded.name, use_container_width=True)

        with col_result:
            with st.spinner("Analysing…"):
                score, ms = run_predict(uploaded)

            # Verdict card
            st.markdown(score_to_html(score), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            m1, m2 = st.columns(2)
            m1.metric("Fraud Score",  f"{score:.4f}")
            m2.metric("Latency",      f"{ms:.0f} ms")

            st.markdown(
                f"<p style='font-size:12px;color:#555a6b;margin-top:10px'>"
                f"Decision threshold: {THRESHOLD} — adjust in run.py if needed.</p>",
                unsafe_allow_html=True,
            )

with tab_bulk:
    st.markdown("<br>", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        accept_multiple_files=True,
        key="bulk",
        help="Select any number of files — they are processed one by one.",
    )

    if uploaded_files:
        st.markdown(
            f"<p style='color:#8b8f9e;font-size:13px'>"
            f"{len(uploaded_files)} file(s) selected. Processing…</p>",
            unsafe_allow_html=True,
        )

        progress_bar = st.progress(0, text="Starting…")
        results      = []

        for idx, f in enumerate(uploaded_files):
            progress_bar.progress(
                (idx) / len(uploaded_files),
                text=f"Analysing {f.name}  ({idx + 1}/{len(uploaded_files)})",
            )
            try:
                score, ms = run_predict(f)
                verdict   = "SCREEN" if score >= THRESHOLD else "REAL"
            except Exception as e:
                score, ms, verdict = None, None, f"ERROR: {e}"

            results.append({
                "File":      f.name,
                "Score":     round(score, 4) if score is not None else "—",
                "Verdict":   verdict,
                "Latency ms": round(ms, 0) if ms is not None else "—",
            })

        progress_bar.progress(1.0, text="Done ✓")
        st.markdown("<br>", unsafe_allow_html=True)
        total    = len(results)
        n_screen = sum(1 for r in results if r["Verdict"] == "SCREEN")
        n_real   = sum(1 for r in results if r["Verdict"] == "REAL")
        n_err    = sum(1 for r in results if str(r["Verdict"]).startswith("ERROR"))
        avg_lat  = (
            sum(r["Latency ms"] for r in results if isinstance(r["Latency ms"], float))
            / max(total - n_err, 1)
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total images",      total)
        c2.metric("✅ Real",            n_real)
        c3.metric("⚠️ Screen / Fake",  n_screen)
        c4.metric("Avg latency",       f"{avg_lat:.0f} ms")
        st.markdown("<br>", unsafe_allow_html=True)
        df = pd.DataFrame(results)
        def _colour_verdict(val):
            if val == "REAL":
                return "background-color:#0d2b1e; color:#4ade80; font-weight:600"
            elif val == "SCREEN":
                return "background-color:#2b0d0d; color:#f87171; font-weight:600"
            return "color:#f59e0b"

        st.dataframe(
            df.style.map(_colour_verdict, subset=["Verdict"]),
            use_container_width=True,
            hide_index=True,
        )
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download results as CSV",
            data=csv,
            file_name="recapture_results.csv",
            mime="text/csv",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            "<p style='font-size:13px;font-weight:700;color:#8b8f9e;"
            "letter-spacing:.06em;text-transform:uppercase'>Image previews</p>",
            unsafe_allow_html=True,
        )

        COLS = 5
        rows = [uploaded_files[i:i+COLS] for i in range(0, len(uploaded_files), COLS)]
        res_lookup = {r["File"]: r for r in results}

        for row_files in rows:
            cols = st.columns(COLS)
            for col, f in zip(cols, row_files):
                with col:
                    f.seek(0)
                    img = Image.open(f)
                    col.image(img, use_container_width=True)
                    r = res_lookup.get(f.name, {})
                    verdict = r.get("Verdict", "?")
                    score_v = r.get("Score",   "?")
                    colour  = "#4ade80" if verdict == "REAL" else (
                              "#f87171" if verdict == "SCREEN" else "#f59e0b")
                    col.markdown(
                        f"<p style='text-align:center;font-size:11px;"
                        f"color:{colour};font-weight:700;margin-top:4px'>"
                        f"{verdict} · {score_v}</p>",
                        unsafe_allow_html=True,
                    )
st.markdown("<br><hr style='border-color:#252836'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#555a6b;font-size:12px'>"
    "FFT · LBP · Color · Sharpness  ·  SVM classifier  ·  "
    "Held-out accuracy: 93.1%  ·  ~400 ms / image (CPU)"
    "</p>",
    unsafe_allow_html=True,
)
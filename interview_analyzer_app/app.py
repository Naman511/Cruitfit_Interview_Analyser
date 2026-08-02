"""
Interview Audio Analyzer — Streamlit App
=========================================

Upload an interview recording, optionally supply API keys for one or more
free LLM providers (Hugging Face, Groq, Google Gemini), and get:

  - a full transcript with interviewer/interviewee split
  - per-answer LLM quality scores (consensus across whichever providers you configure)
  - an AI-generated-text risk signal for each answer, powered by the local,
    keyless ai-text-detector package (replacing GPTZero — no key needed)
  - speaking-time / response-latency analysis
  - an overall hiring recommendation with supporting rationale

Run with:  streamlit run app.py
"""

import os
import tempfile
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyzer import InterviewAnalyzer

st.set_page_config(page_title="Interview Audio Analyzer", page_icon="🎙️", layout="wide")

# --------------------------------------------------------------------------- #
# Sidebar — API keys & options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🎙️ Interview Analyzer")
    st.caption("Multi-LLM scoring + local AI-detection")

    st.subheader("1. LLM API keys (optional)")
    st.caption("Add at least one for real LLM scoring. None supplied → a basic heuristic score is used instead.")
    hf_key = st.text_input("Hugging Face API key", type="password", help="Free at huggingface.co/settings/tokens")
    groq_key = st.text_input("Groq API key", type="password", help="Free at console.groq.com")
    google_key = st.text_input("Google Gemini API key", type="password", help="Free at aistudio.google.com/app/apikey")

    st.subheader("2. AI-detection")
    st.success("Uses the local **ai-text-detector** package (no API key needed) instead of GPTZero.", icon="✅")
    st.caption("If a Gemini key is provided above, it's also used as a second AI-risk opinion.")

    st.subheader("3. Options")
    model_size = st.selectbox("Whisper model size", ["tiny", "base", "small", "medium", "large"], index=1,
                               help="Larger = more accurate, slower.")
    position_context = st.text_input("Position context (optional)", placeholder="e.g. Data Scientist")

    st.subheader("4. ffmpeg (only if you get a 'not found' error)")
    ffmpeg_dir = st.text_input(
        "ffmpeg folder (optional)",
        placeholder=r"e.g. C:\ffmpeg\bin",
        help="Only needed if the app can't find ffmpeg even though it works in your terminal "
             "(common right after a winget/choco install, before a reboot). Run `where ffmpeg` "
             "in a terminal where it works and paste the folder here.",
    )

# --------------------------------------------------------------------------- #
# Main — upload & run
# --------------------------------------------------------------------------- #
st.header("Upload interview recording")
audio_file = st.file_uploader("Audio file (wav, mp3, m4a, etc.)", type=["wav", "mp3", "m4a", "flac", "ogg"])

run_col, status_col = st.columns([1, 3])
run_clicked = run_col.button("▶️ Run analysis", type="primary", disabled=audio_file is None)

if "analyzer" not in st.session_state:
    st.session_state.analyzer = None
if "recommendation" not in st.session_state:
    st.session_state.recommendation = None

if run_clicked and audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.name)[1]) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(msg, frac=None):
        status_text.info(msg)
        if frac is not None:
            progress_bar.progress(min(max(frac, 0.0), 1.0))

    try:
        analyzer = InterviewAnalyzer(tmp_path, progress_cb=progress_cb)
        analyzer.set_api_keys(huggingface=hf_key, groq=groq_key, google=google_key)
        recommendation = analyzer.run(model_size=model_size, position_context=position_context, ffmpeg_dir=ffmpeg_dir)
        st.session_state.analyzer = analyzer
        st.session_state.recommendation = recommendation
        status_text.success("Analysis complete!")
    except RuntimeError as e:
        status_text.error(f"Setup issue: {e}")
    except Exception as e:
        status_text.error(f"Analysis failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
analyzer = st.session_state.analyzer
rec = st.session_state.recommendation

if analyzer is not None and rec:
    st.divider()
    st.header("🎯 Hiring Recommendation")

    rec_colors = {
        "STRONGLY RECOMMEND": "#1a7f37", "RECOMMEND": "#2da44e",
        "CONSIDER WITH RESERVATIONS": "#bf8700", "DO NOT RECOMMEND": "#cf222e",
        "REJECT — HIGH AI-ASSISTANCE RISK": "#8250df",
    }
    color = rec_colors.get(rec["recommendation"], "#57606a")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall score", f"{rec['final_score']:.1f}/100")
    c2.metric("Avg. answer score", f"{rec['avg_response_score']:.1f}/10")
    c3.metric("AI-risk (avg. probability)", f"{rec['avg_ai_probability']*100:.0f}%")
    c4.metric("High-risk answers", f"{rec['high_ai_risk_count']}/{rec['total_qa_pairs']}")

    st.markdown(
        f"<div style='padding:14px;border-radius:8px;background:{color}22;border:1px solid {color};'>"
        f"<span style='font-size:1.3em;font-weight:700;color:{color};'>{rec['recommendation']}</span></div>",
        unsafe_allow_html=True,
    )

    if rec.get("consensus_providers_used"):
        st.caption(f"LLM providers used: {', '.join(rec['consensus_providers_used'])}")
    else:
        st.caption("No LLM API keys configured — scores are from the basic heuristic fallback only.")

    col_s, col_c = st.columns(2)
    with col_s:
        st.subheader("✅ Strengths")
        for s in rec["strengths"]:
            st.markdown(f"- {s}")
    with col_c:
        st.subheader("⚠️ Concerns")
        for c in rec["concerns"]:
            st.markdown(f"- {c}")

    # ---- Timing ---- #
    st.divider()
    st.header("⏱️ Speaking-time & response latency")
    timing = analyzer.timing_analysis
    t1, t2 = st.columns([1, 2])
    with t1:
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Interviewer", "Interviewee"],
            values=[timing["interviewer_time"], timing["interviewee_time"]],
            hole=0.45, marker=dict(colors=["#89b4fa", "#f38ba8"]),
        )])
        fig_pie.update_layout(title="Speaking time split", margin=dict(t=40, b=0, l=0, r=0), height=320)
        st.plotly_chart(fig_pie, use_container_width=True)
    with t2:
        st.metric("Avg. response latency", f"{timing['avg_response_latency']:.2f}s")
        st.metric("Interviewee speaking %", f"{timing['interviewee_percentage']:.1f}%")
        if timing["response_latencies"]:
            fig_lat = px.histogram(x=timing["response_latencies"], nbins=15,
                                    labels={"x": "Response latency (s)"}, title="Response latency distribution")
            fig_lat.update_layout(height=300, margin=dict(t=40, b=0, l=0, r=0))
            st.plotly_chart(fig_lat, use_container_width=True)

    # ---- Per-question analysis ---- #
    st.divider()
    st.header("❓ Per-question analysis")
    df = analyzer.results_dataframe()

    # Surface any provider call failures instead of letting them fail silently
    # into the heuristic fallback.
    all_errors = []
    for _, row in df.iterrows() if not df.empty else []:
        for err in row.get("llm_errors", []) or []:
            all_errors.append(f"Q{row['qa_index']+1}: {err}")
    if all_errors:
        with st.expander(f"⚠️ {len(all_errors)} API call issue(s) — click to see why a provider was skipped", expanded=False):
            for e in all_errors:
                st.code(e, language=None)

    if not df.empty:
        # Build a long-form (one row per provider per question) frame so the
        # score distribution can be broken out and colored by which LLM
        # actually produced each score, instead of one anonymous blob.
        provider_rows = []
        for _, row in df.iterrows():
            pscores = row.get("provider_scores", {}) or {}
            if pscores:
                for provider, score in pscores.items():
                    provider_rows.append({"qa_index": row["qa_index"], "provider": provider, "score": score})
            else:
                provider_rows.append({"qa_index": row["qa_index"], "provider": "heuristic", "score": row["llm_score"]})
        provider_df = pd.DataFrame(provider_rows)

        cc1, cc2 = st.columns(2)
        with cc1:
            fig_score = px.histogram(
                provider_df, x="score", color="provider", nbins=10, barmode="overlay", opacity=0.7,
                title="LLM score distribution (by provider)", labels={"score": "Score (1-10)", "provider": "LLM"},
            )
            fig_score.update_layout(height=320, legend_title_text="LLM")
            st.plotly_chart(fig_score, use_container_width=True)
        with cc2:
            fig_ai = px.histogram(df, x="ai_probability", nbins=10, title="AI-risk probability distribution",
                                   labels={"ai_probability": "AI probability"})
            fig_ai.update_layout(height=320)
            st.plotly_chart(fig_ai, use_container_width=True)

        fig_scatter = px.scatter(df, x="ai_probability", y="llm_score", trendline="ols" if len(df) > 1 else None,
                                  labels={"ai_probability": "AI probability", "llm_score": "LLM score"},
                                  title="Answer score vs. AI-risk")
        fig_scatter.update_layout(height=350)
        st.plotly_chart(fig_scatter, use_container_width=True)

        risk_badge = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
        for _, row in df.iterrows():
            with st.expander(f"Q{row['qa_index']+1}: {row['question'][:90]}"):
                st.markdown(f"**Answer:** {row['answer']}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("LLM score", f"{row['llm_score']:.1f}/10")
                m2.metric("Confidence", f"{row['llm_confidence']:.2f}")
                m3.metric("AI risk", f"{risk_badge.get(row['ai_risk_level'],'')} {row['ai_risk_level']}")
                m4.metric("Latency", f"{row['response_latency']:.1f}s")
                if row.get("llm_highlight"):
                    st.info(f"💡 **What stood out:** {row['llm_highlight']}")
                st.markdown(f"**Feedback:** {row['llm_feedback']}")
                if row.get("consensus_providers"):
                    st.caption(f"Scored by: {', '.join(row['consensus_providers'])}")
                else:
                    st.caption("Scored by: heuristic fallback (no LLM key produced a usable response for this answer)")
                if row.get("ai_detection_services"):
                    st.caption(f"AI-risk checked by: {', '.join(row['ai_detection_services'])}")
                if row.get("llm_errors"):
                    with st.expander("Show API errors for this question"):
                        for e in row["llm_errors"]:
                            st.code(e, language=None)

        st.divider()
        st.header("📥 Export")
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download full Q&A analysis (CSV)", csv, file_name="interview_qa_analysis.csv", mime="text/csv")
    else:
        st.info("No Q&A pairs were detected — check that the recording has clear back-and-forth turns.")

elif audio_file is None:
    st.info("Upload an interview recording to get started. Add API keys in the sidebar for full LLM scoring — "
            "AI-detection works out of the box with no key required.")
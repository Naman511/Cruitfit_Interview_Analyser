# Interview Audio Analyzer (Streamlit)

Upload an interview recording and get automated transcription, Q&A extraction,
multi-LLM answer scoring, AI-generated-text risk signals, and an overall
hiring recommendation — all in one page.

## What changed vs. the original Colab script

- **AI detection**: GPTZero (paid/API-key) has been replaced with the local,
  keyless [`ai-text-detector`](https://github.com/lynote-ai/ai-text-detector)
  package. It runs entirely on your machine — no API key, no network call —
  and returns an explainable `score` (0–100), `verdict`, and top signals. A
  lightweight pattern-based heuristic is still kept as a second, always-on
  opinion, and Gemini can optionally add a third if you supply a key.
- **UI**: the Colab `input()` prompts and `files.upload()` calls are replaced
  with a Streamlit sidebar (API key fields) and file uploader.
- **No FastAPI layer**: everything runs in-process inside the Streamlit app
  (upload → analyze → render), so there's no separate backend to stand up.
  If you later want to expose this analysis as an API (e.g. for a mobile
  client or batch jobs), wrap `analyzer.InterviewAnalyzer` in a small FastAPI
  service — the class has no Streamlit dependency, so it drops in cleanly.

## Setup

```bash
cd interview_analyzer_app
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Whisper + torch are heavy installs (a few GB) and the first run downloads the
selected Whisper model — expect a few minutes on first launch.

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Using the app

1. **(Optional) Add LLM API keys** in the sidebar — Hugging Face, Groq, and/or
   Google Gemini. All three have free tiers. You can add just one, several, or
   none (a basic word-count heuristic is used as a fallback if none are set).
2. **AI detection needs no key** — it's on by default via the local detector.
3. **Upload** your interview recording (wav/mp3/m4a/flac/ogg).
4. Pick a **Whisper model size** (bigger = more accurate, slower) and an
   optional **position context** string to steer the LLM scoring.
5. Click **Run analysis** and watch the progress messages — transcription is
   usually the slowest step.
6. Review the hiring recommendation, timing charts, per-question breakdown,
   and download the full results as CSV.

## Notes & limitations

- Speaker separation uses simple audio-feature clustering (pitch, MFCCs,
  spectral centroid, etc.) rather than a trained diarization model — it works
  best on two-speaker recordings with reasonably distinct voices.
- The AI-detection score is a **risk estimate, not proof** — per the
  ai-text-detector project itself, it's meant for triage and explanation, not
  disciplinary or high-stakes decisions on its own.
- Free-tier LLM APIs are rate-limited; long interviews with many Q&A pairs
  may take a while or hit rate limits mid-run.

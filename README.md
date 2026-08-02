# Cruitfit Interview Analyzer

An AI-powered tool for analyzing job interview recordings using multiple Large Language Models (LLMs) and advanced audio processing techniques.

<img width="1918" height="817" alt="image" src="https://github.com/user-attachments/assets/225d67aa-1c01-41f9-ad25-acd3d5e2a157" />

<img width="1917" height="832" alt="image" src="https://github.com/user-attachments/assets/16615208-eaef-467e-9727-5e914de76e6d" />

<img width="1917" height="806" alt="image" src="https://github.com/user-attachments/assets/e86f0fa8-efb6-4e75-aae5-f73e075ce830" />

<img width="1565" height="787" alt="image" src="https://github.com/user-attachments/assets/3e0e414f-625c-4f2d-ba16-a820b4d437f9" />


## Important Disclaimer

**This tool is for research and educational purposes only.** It should not be used as the sole basis for hiring decisions. Automated interview analysis systems can introduce bias, may violate employment laws, and raise significant privacy concerns. Always involve human judgment in hiring processes and ensure compliance with applicable laws and regulations.

## Features

- **Multi-LLM Evaluation**: Consensus scoring across whichever free AI providers you configure (Hugging Face, Groq, Google Gemini) — with a per-question "what stood out" highlight, not just a number
- **Speaker Identification**: Automatically distinguishes between interviewer and interviewee
- **AI Detection**: Flags potentially AI-generated responses using the local, keyless [`ai-text-detector`](https://github.com/lynote-ai/ai-text-detector) package — no API key or network call required
- **Timing Analysis**: Analyzes response latencies and speaking-time distribution
- **Comprehensive Scoring**: Evaluates responses on quality, authenticity, timing, and consistency
- **Visual Analytics**: Interactive charts, including a score distribution broken out by which LLM produced each score
- **Detailed Reports**: Per-question breakdown with scores, feedback, and CSV export
- **Built-in diagnostics**: If a provider API call fails (bad key, decommissioned model, rate limit), the reason is shown in the app instead of silently falling back

## Quick Start

### Try it Online

**[Run in Google Colab](https://colab.research.google.com/drive/1Q2YVX7iejgnotLQ6f8a-4LlgE3iR_fR6?usp=sharing)**

Click the link above to test the original notebook version directly in your browser without any setup required.

### Run the Streamlit App Locally (Recommended)

```bash
git clone https://github.com/Naman511/Cruitfit_Interview_Analyser
cd Cruitfit_Interview_Analyser
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Whisper + torch are heavy installs (a few GB), and the first run downloads the selected Whisper model — expect a few minutes on first launch. Whisper also requires the `ffmpeg` command-line tool to be installed separately and on your PATH (`winget install ffmpeg` / `brew install ffmpeg` / `apt install ffmpeg`).

### Using the App

1. **(Optional) Add LLM API keys** in the sidebar — Hugging Face, Groq, and/or Google Gemini. All three have free tiers. Add just one, several, or none (a basic word-count heuristic is used as a fallback if none are set).
2. **AI detection needs no key** — it's on by default via the local detector.
3. **Upload** your interview recording (wav/mp3/m4a/flac/ogg).
4. Pick a **Whisper model size** (bigger = more accurate, slower) and an optional **position context** string to steer the LLM scoring.
5. Click **Run analysis** and watch the progress messages — transcription is usually the slowest step.
6. Review the hiring recommendation, timing charts, per-question breakdown (including the "what stood out" highlight for each answer), and download the full results as CSV.

## API Keys Setup

Cruitfit works with free tiers of multiple AI services. Provide as many or as few as you like:

- **Hugging Face** (Free): [Get API Key](https://huggingface.co/settings/tokens)
- **Groq** (Free): [Get API Key](https://console.groq.com/)
- **Google Gemini** (Free): [Get API Key](https://aistudio.google.com/app/apikey)
- **AI detection**: no key needed — runs locally via `ai-text-detector`

## Requirements

```
streamlit
plotly
pandas
numpy
requests
scikit-learn
librosa
openai-whisper
torch
torchaudio
git+https://github.com/lynote-ai/ai-text-detector.git
```

(`ffmpeg` is also required as a separate system-level install — see above.)

## How It Works

1. **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
2. **Speaker Diarization**: Identifies different speakers using audio feature clustering (pitch, MFCCs, spectral centroid, etc.)
3. **Role Classification**: Determines interviewer vs. interviewee based on question patterns
4. **Q&A Extraction**: Pairs questions with corresponding answers
5. **Multi-LLM Evaluation**: Scores responses using whichever configured AI models are available, for reliability
6. **AI Detection**: Flags potentially artificial responses via the local ai-text-detector, with a pattern-based heuristic and (optionally) Gemini as additional opinions
7. **Comprehensive Analysis**: Combines all factors into a final recommendation

## Output

Cruitfit provides:

- **Hiring Recommendation**: STRONGLY RECOMMEND / RECOMMEND / CONSIDER WITH RESERVATIONS / DO NOT RECOMMEND / REJECT, with detailed reasoning
- **Numerical Score**: 0–100 overall performance score
- **Detailed Breakdown**: Individual Q&A analysis with scores, a per-answer highlight, and feedback
- **Visual Charts**: Speaking time, per-LLM score distributions, AI-detection results
- **CSV Exports**: All data available for further analysis

## Limitations and Risks

- **Bias Risk**: AI models may introduce unconscious bias into evaluations
- **Privacy Concerns**: Audio and transcript text are sent to whichever third-party LLM providers you configure (nothing is sent anywhere for AI detection, which runs locally)
- **False Positives**: AI detection is a risk estimate, not proof, and may incorrectly flag human responses — treat it as a triage signal, not a verdict
- **Audio Quality**: Poor recordings may affect transcription and speaker-separation accuracy
- **Language Support**: Optimized for English interviews
- **Legal Compliance**: May not comply with employment laws in all jurisdictions
- **Model churn**: Free-tier LLM providers frequently deprecate model IDs; if a provider stops responding, check the in-app diagnostics panel for the exact error

## Best Practices

- Use as a supplementary tool, not a primary decision-maker
- Always involve human reviewers in final decisions
- Ensure candidates consent to AI-powered analysis
- Regularly audit for bias and fairness
- Test thoroughly with diverse candidate pools
- Comply with local employment and privacy laws

## Contributing

Contributions are welcome! Please ensure your changes:

- Include appropriate tests
- Follow ethical AI practices
- Consider bias and fairness implications
- Update documentation as needed

## License

This project is licensed under the MIT License — see the LICENSE file for details.

## Ethics Statement

Cruitfit is designed to assist, not replace, human judgment in hiring processes. We strongly encourage:

- Transparent communication with candidates about AI usage
- Regular bias testing and mitigation
- Compliance with employment laws and regulations
- Ethical use that promotes fairness and diversity

## Acknowledgments

- OpenAI Whisper for speech recognition
- [`ai-text-detector`](https://github.com/lynote-ai/ai-text-detector) for local, keyless AI-detection signals
- Hugging Face, Groq, and Google for free-tier evaluation APIs
- The open-source community for essential libraries

## Support

- Open an issue for bugs or feature requests
- Check the [Colab notebook](https://colab.research.google.com/drive/1Q2YVX7iejgnotLQ6f8a-4LlgE3iR_fR6?usp=sharing) for a quick live example
- Review the code documentation (`analyzer.py`, `app.py`) for technical details

---

**Remember**: This tool should enhance human decision-making, not replace it. Always prioritize fairness, transparency, and legal compliance in your hiring processes.

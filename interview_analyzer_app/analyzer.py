"""
Core engine for the Interview Audio Analyzer.

Adapted from a Colab notebook into a reusable, importable module for the
Streamlit app in app.py. Key change from the original script: GPTZero has
been replaced with the local, keyless `ai-text-detector` package
(https://github.com/lynote-ai/ai-text-detector) for AI-generated-text risk
signals, so no AI-detection API key is required at all.
"""

import os
import re
import time
import warnings
from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Optional heavy deps are imported lazily inside functions/methods that need
# them, so the module can be imported (e.g. for a quick UI render) even if
# whisper/librosa/torch aren't installed yet.
# ---------------------------------------------------------------------------


class MultiLLMEvaluator:
    """Scores interview answers via user-supplied LLM API keys, and estimates
    AI-generated-text risk using the local ai-text-detector package (with a
    lightweight pattern-based signal kept as a second, always-on opinion)."""

    def __init__(self):
        self.api_keys: Dict[str, str] = {}

    def set_api_keys(self, huggingface: str = "", groq: str = "", google: str = ""):
        self.api_keys = {}
        if huggingface:
            self.api_keys["huggingface"] = huggingface.strip()
        if groq:
            self.api_keys["groq"] = groq.strip()
        if google:
            self.api_keys["google"] = google.strip()

    # ------------------------------------------------------------------ #
    # Answer quality scoring
    # ------------------------------------------------------------------ #
    def score_answer_with_llm(self, question: str, answer: str, context: str = "") -> Dict:
        prompt = f"""You are an expert interview evaluator. Analyze this interview response and provide a detailed evaluation.

INTERVIEW QUESTION: {question}

CANDIDATE'S ANSWER: {answer}

JOB CONTEXT: {context}

Respond in this EXACT format:
SCORE: [number from 1-10]
HIGHLIGHT: [1-2 sentences on the single most notable, specific thing about THIS answer — good or bad, concrete, not generic]
REASONING: [2-3 sentences explaining your score]
STRENGTHS: [bullet points of strengths]
WEAKNESSES: [bullet points of areas for improvement]
RECOMMENDATION: [brief hiring recommendation]"""

        results = []
        errors = []
        if "huggingface" in self.api_keys:
            r, err = self._evaluate_with_huggingface(prompt)
            if r:
                results.append(r)
            if err:
                errors.append(err)
        if "groq" in self.api_keys:
            r, err = self._evaluate_with_groq(prompt)
            if r:
                results.append(r)
            if err:
                errors.append(err)
        if "google" in self.api_keys:
            r, err = self._evaluate_with_google(prompt)
            if r:
                results.append(r)
            if err:
                errors.append(err)

        if results:
            out = self._aggregate_llm_results(results)
        else:
            out = self._basic_heuristic_evaluation(answer)
        out["errors"] = errors
        return out

    def _evaluate_with_huggingface(self, prompt: str) -> "tuple[Optional[Dict], Optional[str]]":
        # HF retired the old per-model "api-inference.huggingface.co" serverless
        # endpoint for most chat models in favor of an OpenAI-compatible router.
        models = [
            "meta-llama/Llama-3.3-70B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct-1M",
        ]
        headers = {"Authorization": f"Bearer {self.api_keys['huggingface']}", "Content-Type": "application/json"}
        last_err = None
        for model in models:
            try:
                resp = requests.post(
                    "https://router.huggingface.co/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 400,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return self._parse_llm_response(text, "huggingface"), None
                last_err = f"Hugging Face ({model}): HTTP {resp.status_code} — {resp.text[:200]}"
            except Exception as e:
                last_err = f"Hugging Face ({model}): {e}"
        return None, last_err

    def _evaluate_with_groq(self, prompt: str) -> "tuple[Optional[Dict], Optional[str]]":
        # llama3-8b-8192 / llama3-70b-8192 / mixtral-8x7b-32768 were all
        # decommissioned by Groq in 2025; these are the current replacements.
        models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
        headers = {"Authorization": f"Bearer {self.api_keys['groq']}", "Content-Type": "application/json"}
        last_err = None
        for model in models:
            try:
                payload = {
                    "messages": [{"role": "user", "content": prompt}],
                    "model": model,
                    "temperature": 0.3,
                    "max_tokens": 400,
                }
                resp = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers, json=payload, timeout=30,
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return self._parse_llm_response(text, "groq"), None
                last_err = f"Groq ({model}): HTTP {resp.status_code} — {resp.text[:200]}"
            except Exception as e:
                last_err = f"Groq ({model}): {e}"
        return None, last_err

    def _evaluate_with_google(self, prompt: str) -> "tuple[Optional[Dict], Optional[str]]":
        # gemini-pro / gemini-1.5-flash have been shut down; 2.5-flash is the
        # current stable model as of mid-2026 (2.5-flash-lite as a fallback).
        last_err = None
        for model_name in ["gemini-2.5-flash", "gemini-2.5-flash-lite"]:
            try:
                url = (
                    f"https://generativelanguage.googleapis.com/v1beta/models/"
                    f"{model_name}:generateContent?key={self.api_keys['google']}"
                )
                payload = {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.3, "maxOutputTokens": 400},
                }
                resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=30)
                if resp.status_code == 200:
                    result = resp.json()
                    if result.get("candidates"):
                        text = result["candidates"][0]["content"]["parts"][0]["text"]
                        return self._parse_llm_response(text, "google"), None
                    last_err = f"Google ({model_name}): no candidates in response — {str(result)[:200]}"
                else:
                    last_err = f"Google ({model_name}): HTTP {resp.status_code} — {resp.text[:200]}"
            except Exception as e:
                last_err = f"Google ({model_name}): {e}"
        return None, last_err

    @staticmethod
    def _parse_llm_response(text: str, provider: str) -> Dict:
        try:
            score_match = re.search(r"SCORE[:\s]*(\d{1,2})", text, re.IGNORECASE)
            score = int(score_match.group(1)) if score_match else 5
            score = max(1, min(10, score))

            def grab(label, stop):
                m = re.search(rf"{label}[:\s]*(.*?)(?={stop}|$)", text, re.IGNORECASE | re.DOTALL)
                return m.group(1).strip() if m else "Not provided"

            highlight = grab("HIGHLIGHT", "REASONING")
            reasoning = grab("REASONING", "STRENGTHS")
            strengths = grab("STRENGTHS", "WEAKNESSES")
            weaknesses = grab("WEAKNESSES", "RECOMMENDATION")
            recommendation_match = re.search(r"RECOMMENDATION[:\s]*(.*)", text, re.IGNORECASE | re.DOTALL)
            recommendation = recommendation_match.group(1).strip() if recommendation_match else "Not provided"

            return {
                "score": score, "highlight": highlight, "reasoning": reasoning, "strengths": strengths,
                "weaknesses": weaknesses, "recommendation": recommendation,
                "provider": provider, "raw_response": text,
            }
        except Exception:
            return {
                "score": 5, "highlight": "Not available", "reasoning": "Parse error", "strengths": "N/A",
                "weaknesses": "N/A", "recommendation": "N/A", "provider": provider, "raw_response": text,
            }

    @staticmethod
    def _aggregate_llm_results(results: List[Dict]) -> Dict:
        scores = [r["score"] for r in results]
        avg_score = float(np.mean(scores))
        score_std = float(np.std(scores))
        confidence = max(0.5, 1.0 - (score_std / 3.0))

        return {
            "score": round(avg_score, 1),
            "highlight": " | ".join(f"[{r['provider']}] {r['highlight']}" for r in results),
            "feedback": "\n".join(f"[{r['provider']}] {r['reasoning']}" for r in results),
            "strengths": "\n".join(f"[{r['provider']}] {r['strengths']}" for r in results),
            "weaknesses": "\n".join(f"[{r['provider']}] {r['weaknesses']}" for r in results),
            "recommendations": "\n".join(f"[{r['provider']}] {r['recommendation']}" for r in results),
            "confidence": confidence,
            "consensus_providers": [r["provider"] for r in results],
            "provider_scores": {r["provider"]: r["score"] for r in results},
            "score_variance": score_std,
            "evaluation_method": "multi-llm-consensus",
        }

    @staticmethod
    def _basic_heuristic_evaluation(answer: str) -> Dict:
        word_count = len(answer.split())
        if word_count < 10:
            score, feedback = 3, "Response is too brief"
        elif word_count < 50:
            score, feedback = 5, "Response length is adequate"
        elif word_count < 200:
            score, feedback = 7, "Good response length with detail"
        else:
            score, feedback = 6, "Response may be too lengthy"
        return {
            "score": score, "highlight": "No LLM key active — this is a word-count heuristic, not a quality read.",
            "feedback": feedback, "strengths": "Basic evaluation only",
            "weaknesses": "No API keys supplied — add one for deeper analysis",
            "recommendations": "Add a Groq/HF/Gemini key for richer scoring",
            "confidence": 0.3, "consensus_providers": [], "provider_scores": {"heuristic": score},
            "evaluation_method": "basic_heuristic",
        }

    # ------------------------------------------------------------------ #
    # AI-generated-text risk detection
    # ------------------------------------------------------------------ #
    def detect_ai_generated_content(self, text: str) -> Dict:
        results = []

        local_result = self._detect_with_local_detector(text)
        if local_result:
            results.append(local_result)

        if "google" in self.api_keys:
            google_result = self._detect_with_google_ai(text)
            if google_result:
                results.append(google_result)

        results.append(self._pattern_based_ai_detection(text))
        return self._aggregate_ai_detection_results(results)

    @staticmethod
    def _detect_with_local_detector(text: str) -> Optional[Dict]:
        """Uses the local, keyless ai-text-detector package
        (https://github.com/lynote-ai/ai-text-detector) instead of GPTZero."""
        try:
            from aidetect import analyze_text
            result = analyze_text(text)
            return {
                "ai_probability": max(0.0, min(1.0, result.score / 100.0)),
                "service": "ai-text-detector",
                "details": {
                    "verdict": getattr(result, "verdict", "unknown"),
                    "confidence": getattr(result, "confidence", "unknown"),
                    "signals": [
                        {"name": s.name, "note": s.note} for s in result.strongest_signals()
                    ] if hasattr(result, "strongest_signals") else [],
                },
            }
        except ImportError:
            return None
        except Exception:
            return None

    def _detect_with_google_ai(self, text: str) -> Optional[Dict]:
        prompt = f"""Analyze the following text and determine if it appears to be AI-generated or human-written. Look for patterns like overly formal or structured language, generic responses without personal experience, perfect grammar and flow, repetitive phrasing, and lack of natural speech patterns.

Text to analyze: "{text}"

Respond with:
AI_PROBABILITY: [number from 0.0 to 1.0]
REASONING: [brief explanation]"""
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"gemini-1.5-flash:generateContent?key={self.api_keys['google']}"
            )
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1, "maxOutputTokens": 200},
            }
            resp = requests.post(url, headers={"Content-Type": "application/json"}, json=payload, timeout=25)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("candidates"):
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    m = re.search(r"AI_PROBABILITY[:\s]*([\d.]+)", response_text)
                    ai_probability = float(m.group(1)) if m else 0.5
                    reasoning_m = re.search(r"REASONING[:\s]*(.*)", response_text, re.DOTALL)
                    reasoning = reasoning_m.group(1).strip() if reasoning_m else "Not provided"
                    return {
                        "ai_probability": min(1.0, max(0.0, ai_probability)),
                        "service": "google_gemini",
                        "details": {"reasoning": reasoning},
                    }
        except Exception:
            pass
        return None

    @staticmethod
    def _pattern_based_ai_detection(text: str) -> Dict:
        indicators = {"repetitive_phrases": 0, "formal_language": 0, "perfect_grammar": 0, "generic_responses": 0}
        words = text.lower().split()
        word_counts: Dict[str, int] = {}
        for w in words:
            if len(w) > 3:
                word_counts[w] = word_counts.get(w, 0) + 1
        max_rep = max(word_counts.values()) if word_counts else 1
        if max_rep > len(words) * 0.1:
            indicators["repetitive_phrases"] = 0.3

        formal_words = ["furthermore", "consequently", "nevertheless", "however", "therefore", "moreover"]
        if sum(1 for w in formal_words if w in text.lower()) > 2:
            indicators["formal_language"] = 0.4

        sentences = [s.strip() for s in text.split(".") if s.strip()]
        if len(sentences) > 2:
            avg_len = np.mean([len(s.split()) for s in sentences])
            if avg_len > 20:
                indicators["perfect_grammar"] = 0.2

        generic_phrases = ["as an ai", "i don't have personal", "i cannot", "i'm unable to", "based on my training"]
        if sum(1 for p in generic_phrases if p in text.lower()) > 0:
            indicators["generic_responses"] = 0.6

        ai_probability = min(1.0, sum(indicators.values()))
        return {"ai_probability": ai_probability, "service": "pattern_analysis", "details": indicators}

    @staticmethod
    def _aggregate_ai_detection_results(results: List[Dict]) -> Dict:
        if not results:
            return {"ai_probability": 0.5, "risk_level": "Medium", "confidence": 0.1, "services": [], "individual_results": []}
        probs = [r["ai_probability"] for r in results]
        avg_prob = float(np.mean(probs))
        prob_std = float(np.std(probs)) if len(probs) > 1 else 0.0
        confidence = max(0.3, 1.0 - prob_std)
        risk_level = "High" if avg_prob > 0.7 else "Medium" if avg_prob > 0.4 else "Low"
        return {
            "ai_probability": avg_prob, "risk_level": risk_level, "confidence": confidence,
            "services": [r["service"] for r in results], "individual_results": results,
            "agreement_score": 1.0 - prob_std,
        }


class InterviewAnalyzer:
    """Full pipeline: transcription -> speaker ID -> Q&A extraction -> timing
    -> multi-LLM scoring + AI-risk detection -> hiring recommendation."""

    def __init__(self, audio_file_path: str, progress_cb=None):
        self.audio_file_path = audio_file_path
        self.whisper_model = None
        self.segments: List[Dict] = []
        self.speakers: Dict[str, str] = {}
        self.qa_pairs: List[Dict] = []
        self.timing_analysis: Dict = {}
        self.evaluation_results: List[Dict] = []
        self.hiring_recommendation: Dict = {}
        self.llm_evaluator = MultiLLMEvaluator()
        self._progress_cb = progress_cb or (lambda msg, frac=None: None)

    def _progress(self, msg: str, frac: Optional[float] = None):
        self._progress_cb(msg, frac)

    def set_api_keys(self, huggingface="", groq="", google=""):
        self.llm_evaluator.set_api_keys(huggingface, groq, google)

    def load_whisper_model(self, model_size="base", ffmpeg_dir: str = ""):
        import whisper
        self._check_ffmpeg(ffmpeg_dir)
        self._progress(f"Loading Whisper model ({model_size})...", 0.05)
        self.whisper_model = whisper.load_model(model_size)

    @staticmethod
    def _check_ffmpeg(ffmpeg_dir: str = ""):
        """Whisper (and librosa's audio loading) shell out to the ffmpeg
        executable. If it isn't found, subprocess calls fail with a cryptic
        '[WinError 2] The system cannot find the file specified' on Windows
        (or 'No such file or directory' on macOS/Linux).

        A common gotcha: `winget install ffmpeg` updates the user PATH env
        var, but an already-running process (or a terminal that was opened
        before the PATH refresh propagated) won't see it until a full
        logoff/reboot. So beyond shutil.which, we also: (1) honor an
        explicit directory the user points us to, and (2) probe common
        Windows install locations and, if found, prepend them to this
        process's PATH so it works without a reboot."""
        import os
        import shutil

        if ffmpeg_dir:
            candidate = os.path.join(ffmpeg_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if os.path.isfile(candidate):
                os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            else:
                raise RuntimeError(f"No ffmpeg executable found at: {candidate}")

        if shutil.which("ffmpeg") is not None:
            return

        # Probe typical install locations that winget/choco/manual installs use,
        # in case PATH hasn't refreshed for this process yet.
        probe_dirs = []
        if os.name == "nt":
            local_appdata = os.environ.get("LOCALAPPDATA", "")
            probe_dirs += [
                os.path.join(local_appdata, "Microsoft", "WinGet", "Links"),
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                r"C:\ProgramData\chocolatey\bin",
            ]
            # WinGet nests the real exe under a version-specific subfolder; do a shallow scan.
            winget_pkgs = os.path.join(local_appdata, "Microsoft", "WinGet", "Packages")
            if os.path.isdir(winget_pkgs):
                for root, dirs, files in os.walk(winget_pkgs):
                    if "ffmpeg.exe" in files:
                        probe_dirs.append(root)

        for d in probe_dirs:
            candidate = os.path.join(d, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if os.path.isfile(candidate):
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                if shutil.which("ffmpeg") is not None:
                    return

        raise RuntimeError(
            "ffmpeg was not found on this app's PATH, even though it may work in your terminal.\n\n"
            "This usually means ffmpeg was installed (e.g. via winget) but the PATH update hasn't "
            "propagated to this process yet — that needs a full sign-out/sign-in or reboot to fix "
            "at the OS level.\n\n"
            "Fastest fix without rebooting: find your ffmpeg.exe (run `where ffmpeg` in the terminal "
            "where it DOES work), then paste the folder path into the 'ffmpeg folder (optional)' "
            "field in the sidebar and rerun."
        )

    def transcribe_audio(self):
        self._progress("Transcribing audio...", 0.15)
        result = self.whisper_model.transcribe(self.audio_file_path, word_timestamps=True, verbose=False)
        segments = []
        for seg in result["segments"]:
            segments.append({
                "start": seg["start"], "end": seg["end"],
                "text": seg["text"].strip(), "words": seg.get("words", []),
            })
        self.segments = segments
        self._progress(f"Transcription complete — {len(segments)} segments", 0.35)
        return segments

    def _extract_audio_features(self, start, end, sr=16000):
        import librosa
        y, _ = librosa.load(self.audio_file_path, offset=start, duration=end - start, sr=sr)
        if len(y) < sr * 0.1:
            return None
        features = []
        pitches, _ = librosa.piptrack(y=y, sr=sr, threshold=0.1)
        features.append(np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfcc, axis=1))
        features.append(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        features.append(np.mean(librosa.feature.zero_crossing_rate(y)))
        features.append(np.mean(librosa.feature.rms(y=y)))
        return np.array(features)

    def identify_speakers(self):
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        self._progress("Identifying speakers...", 0.40)
        features_list, valid_idx = [], []
        for i, seg in enumerate(self.segments):
            feats = self._extract_audio_features(seg["start"], seg["end"])
            if feats is not None and not np.any(np.isnan(feats)):
                features_list.append(feats)
                valid_idx.append(i)

        if len(features_list) < 2:
            for seg in self.segments:
                seg["speaker"] = "Speaker_0"
            return

        normalized = StandardScaler().fit_transform(features_list)
        labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(normalized)
        for i, idx in enumerate(valid_idx):
            self.segments[idx]["speaker"] = f"Speaker_{labels[i]}"
        for i, seg in enumerate(self.segments):
            if "speaker" not in seg:
                prev = next((self.segments[j]["speaker"] for j in range(i - 1, -1, -1) if "speaker" in self.segments[j]), "Speaker_0")
                seg["speaker"] = prev

    def determine_interviewer_interviewee(self):
        self._progress("Determining interviewer vs. interviewee...", 0.45)
        stats = {"Speaker_0": {"questions": 0, "segments": 0}, "Speaker_1": {"questions": 0, "segments": 0}}
        patterns = [r"\?", r"\bwhat\b", r"\bhow\b", r"\bwhy\b", r"\bwhen\b", r"\bwhere\b", r"\bwho\b",
                    r"\bcan you\b", r"\bwould you\b", r"\bcould you\b", r"\btell me\b", r"\bexplain\b"]
        for seg in self.segments:
            sp = seg.get("speaker", "Speaker_0")
            stats.setdefault(sp, {"questions": 0, "segments": 0})
            stats[sp]["segments"] += 1
            if any(re.search(p, seg["text"].lower()) for p in patterns):
                stats[sp]["questions"] += 1

        ratios = {sp: v["questions"] / max(v["segments"], 1) for sp, v in stats.items()}
        interviewer = max(ratios, key=ratios.get) if ratios else "Speaker_0"
        interviewee = next((sp for sp in stats if sp != interviewer), "Speaker_1")
        self.speakers = {"interviewer": interviewer, "interviewee": interviewee}
        for seg in self.segments:
            seg["role"] = "Interviewer" if seg.get("speaker") == interviewer else "Interviewee"

    def extract_qa_pairs(self):
        self._progress("Extracting Q&A pairs...", 0.50)
        qa_pairs, current_q, current_a = [], None, []

        def flush():
            if current_q and current_a:
                qa_pairs.append({
                    "question": current_q["text"], "answer": " ".join(p["text"] for p in current_a),
                    "question_start": current_q["start"], "question_end": current_q["end"],
                    "answer_start": current_a[0]["start"], "answer_end": current_a[-1]["end"],
                })

        for seg in self.segments:
            text = seg["text"].strip()
            if not text:
                continue
            if seg["role"] == "Interviewer":
                flush()
                current_q = {"text": text, "start": seg["start"], "end": seg["end"]}
                current_a = []
            elif seg["role"] == "Interviewee" and current_q:
                current_a.append({"text": text, "start": seg["start"], "end": seg["end"]})
        flush()

        self.qa_pairs = qa_pairs
        return qa_pairs

    def analyze_timing(self):
        self._progress("Analyzing speaking-time distribution...", 0.55)
        interviewer_time = sum(s["end"] - s["start"] for s in self.segments if s["role"] == "Interviewer")
        interviewee_time = sum(s["end"] - s["start"] for s in self.segments if s["role"] == "Interviewee")
        total = interviewer_time + interviewee_time or 1
        latencies = [max(0, qa["answer_start"] - qa["question_end"]) for qa in self.qa_pairs]
        for qa, lat in zip(self.qa_pairs, latencies):
            qa["response_latency"] = lat

        self.timing_analysis = {
            "interviewer_time": interviewer_time, "interviewee_time": interviewee_time,
            "total_speaking_time": total,
            "interviewer_percentage": interviewer_time / total * 100,
            "interviewee_percentage": interviewee_time / total * 100,
            "response_latencies": latencies,
            "avg_response_latency": float(np.mean(latencies)) if latencies else 0,
            "max_response_latency": float(np.max(latencies)) if latencies else 0,
            "min_response_latency": float(np.min(latencies)) if latencies else 0,
        }
        return self.timing_analysis

    def evaluate_responses(self, position_context: str = ""):
        self._progress("Scoring responses with your configured LLM(s)...", 0.60)
        self.evaluation_results = []
        n = max(len(self.qa_pairs), 1)
        for i, qa in enumerate(self.qa_pairs):
            self._progress(f"Evaluating Q&A pair {i + 1}/{len(self.qa_pairs)}...", 0.60 + 0.30 * (i + 1) / n)
            llm_eval = self.llm_evaluator.score_answer_with_llm(qa["question"], qa["answer"], position_context)
            ai_detection = self.llm_evaluator.detect_ai_generated_content(qa["answer"])
            latency = max(0, qa.get("answer_start", 0) - qa.get("question_end", 0))

            evaluation = {
                "qa_index": i, "question": qa["question"], "answer": qa["answer"],
                "llm_score": llm_eval["score"], "llm_highlight": llm_eval.get("highlight", ""),
                "llm_feedback": llm_eval["feedback"],
                "llm_strengths": llm_eval.get("strengths", ""), "llm_weaknesses": llm_eval.get("weaknesses", ""),
                "llm_confidence": llm_eval.get("confidence", 0.5),
                "consensus_providers": llm_eval.get("consensus_providers", []),
                "provider_scores": llm_eval.get("provider_scores", {}),
                "llm_errors": llm_eval.get("errors", []),
                "ai_probability": ai_detection["ai_probability"], "ai_risk_level": ai_detection["risk_level"],
                "ai_confidence": ai_detection["confidence"], "ai_detection_services": ai_detection["services"],
                "response_latency": latency, "answer_length": len(qa["answer"].split()),
            }
            self.evaluation_results.append(evaluation)
            qa.update(evaluation)
        return self.evaluation_results

    def generate_hiring_recommendation(self):
        self._progress("Generating hiring recommendation...", 0.95)
        if not self.evaluation_results:
            return {}

        avg_score = float(np.mean([r["llm_score"] for r in self.evaluation_results]))
        avg_ai_prob = float(np.mean([r["ai_probability"] for r in self.evaluation_results]))
        avg_latency = float(np.mean([r["response_latency"] for r in self.evaluation_results]))
        avg_conf = float(np.mean([r["llm_confidence"] for r in self.evaluation_results]))
        high_risk = sum(1 for r in self.evaluation_results if r["ai_risk_level"] == "High")
        score_std = float(np.std([r["llm_score"] for r in self.evaluation_results]))

        score_component = (avg_score / 10) * 35
        authenticity_component = (1 - avg_ai_prob) * 25
        if 1 <= avg_latency <= 3:
            time_component = 15
        elif avg_latency < 1:
            time_component = 12
        else:
            time_component = max(0, 15 - (avg_latency - 3) * 1.5)
        consistency_component = max(0, 15 - score_std * 1.5)
        confidence_component = avg_conf * 10
        final_score = score_component + authenticity_component + time_component + consistency_component + confidence_component

        if final_score >= 85 and high_risk == 0:
            recommendation = "STRONGLY RECOMMEND"
        elif final_score >= 70 and high_risk <= 1:
            recommendation = "RECOMMEND"
        elif final_score >= 55 and high_risk <= 2:
            recommendation = "CONSIDER WITH RESERVATIONS"
        elif high_risk > len(self.evaluation_results) * 0.5:
            recommendation = "REJECT — HIGH AI-ASSISTANCE RISK"
        else:
            recommendation = "DO NOT RECOMMEND"

        strengths, concerns = [], []
        if avg_score >= 7:
            strengths.append(f"Strong average response quality ({avg_score:.1f}/10)")
        elif avg_score < 5:
            concerns.append(f"Below-average response quality ({avg_score:.1f}/10)")
        if avg_ai_prob < 0.3:
            strengths.append("Low AI-assistance risk — responses read as authentic")
        elif high_risk > 0:
            concerns.append(f"High AI-assistance risk flagged in {high_risk} response(s)")
        if avg_conf >= 0.7:
            strengths.append("High confidence across evaluations")
        elif avg_conf < 0.4:
            concerns.append("Low evaluation confidence — recommend manual review")
        if 1 <= avg_latency <= 3:
            strengths.append("Response timing suggests good preparation")
        elif avg_latency > 5:
            concerns.append("Slow response times may indicate under-preparation")
        if score_std < 1.5:
            strengths.append("Consistent performance across questions")
        elif score_std > 2.5:
            concerns.append("Inconsistent performance across questions")

        interviewee_pct = self.timing_analysis.get("interviewee_percentage", 0)
        if 60 <= interviewee_pct <= 80:
            strengths.append("Good balance of speaking time")
        elif interviewee_pct < 50:
            concerns.append("Candidate spoke relatively little — may indicate low engagement")
        elif interviewee_pct > 85:
            concerns.append("Candidate dominated the conversation — may indicate weak listening")

        self.hiring_recommendation = {
            "recommendation": recommendation, "final_score": final_score,
            "avg_response_score": avg_score, "avg_ai_probability": avg_ai_prob,
            "avg_response_time": avg_latency, "avg_confidence": avg_conf,
            "high_ai_risk_count": high_risk, "strengths": strengths, "concerns": concerns,
            "total_qa_pairs": len(self.evaluation_results),
            "interviewee_speaking_percentage": interviewee_pct,
            "consensus_providers_used": sorted(set(
                p for r in self.evaluation_results for p in r.get("consensus_providers", [])
            )),
        }
        self._progress("Done.", 1.0)
        return self.hiring_recommendation

    def run(self, model_size="base", position_context="", ffmpeg_dir=""):
        self.load_whisper_model(model_size, ffmpeg_dir=ffmpeg_dir)
        self.transcribe_audio()
        self.identify_speakers()
        self.determine_interviewer_interviewee()
        self.extract_qa_pairs()
        self.analyze_timing()
        self.evaluate_responses(position_context)
        return self.generate_hiring_recommendation()

    def results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.evaluation_results)
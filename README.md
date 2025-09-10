# Cruitfit Interview Analyzer

An AI-powered tool for analyzing job interview recordings using multiple Large Language Models (LLMs) and advanced audio processing techniques.

## Important Disclaimer

**This tool is for research and educational purposes only.** It should not be used as the sole basis for hiring decisions. Automated interview analysis systems can introduce bias, may violate employment laws, and raise significant privacy concerns. Always involve human judgment in hiring processes and ensure compliance with applicable laws and regulations.

## Features

- **Multi-LLM Evaluation**: Uses consensus scoring from multiple AI providers (Hugging Face, Groq, Google Gemini)
- **Speaker Identification**: Automatically distinguishes between interviewer and interviewee
- **AI Detection**: Identifies potentially AI-generated responses
- **Timing Analysis**: Analyzes response latencies and speaking time distribution
- **Comprehensive Scoring**: Evaluates responses on quality, authenticity, timing, and consistency
- **Visual Analytics**: Generates detailed charts and performance metrics
- **Detailed Reports**: Produces comprehensive analysis with actionable insights

## Quick Start

### Try it Online (Recommended)

**[Run in Google Colab](https://colab.research.google.com/drive/1Q2YVX7iejgnotLQ6f8a-4LlgE3iR_fR6?usp=sharing)**

Click the link above to test Cruitfit Interview Analyzer directly in your browser without any setup required.

### Local Installation

```bash
git clone https://github.com/Naman511/Cruitfit_Interview_Analyser/edit/main/README.md
cd cruitfit-interview-analyzer
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from cruitfit_analyzer import analyze_interview_enhanced

# Analyze an interview recording
analyzer = analyze_interview_enhanced(
    audio_file_path="interview.mp3",
    model_size="base",
    position_context="Software Engineer"
)

# View results
print(f"Recommendation: {analyzer.hiring_recommendation['recommendation']}")
print(f"Score: {analyzer.hiring_recommendation['final_score']:.1f}/100")
```

### Google Colab Usage

1. Open the [Colab notebook](https://colab.research.google.com/drive/1Q2YVX7iejgnotLQ6f8a-4LlgE3iR_fR6?usp=sharing)
2. Upload your audio file when prompted
3. Provide API keys for enhanced features (optional)
4. Run all cells to get comprehensive analysis

## API Keys Setup

Cruitfit works with free tiers of multiple AI services. You can provide API keys for enhanced features:

- **Hugging Face** (Free): [Get API Key](https://huggingface.co/settings/tokens)
- **Groq** (Free): [Get API Key](https://console.groq.com/)
- **Google Gemini** (Free): [Get API Key](https://aistudio.google.com/app/apikey)
- **GPTZero** (Free tier): [Get API Key](https://gptzero.me/)

## Requirements

```
openai-whisper
librosa
scikit-learn
pandas
numpy
matplotlib
requests
aiohttp
transformers
sentence-transformers
google-generativeai
pyannote.audio
torch
torchaudio
pydub
```

## How It Works

1. **Audio Transcription**: Uses OpenAI Whisper for accurate speech-to-text conversion
2. **Speaker Diarization**: Identifies different speakers using audio feature clustering
3. **Role Classification**: Determines interviewer vs interviewee based on question patterns
4. **Q&A Extraction**: Pairs questions with corresponding answers
5. **Multi-LLM Evaluation**: Scores responses using multiple AI models for reliability
6. **AI Detection**: Identifies potentially artificial responses
7. **Comprehensive Analysis**: Combines all factors into final recommendations

## Output

Cruitfit provides:

- **Hiring Recommendation**: RECOMMEND, CONSIDER, or REJECT with detailed reasoning
- **Numerical Score**: 0-100 overall performance score
- **Detailed Breakdown**: Individual Q&A analysis with scores and feedback
- **Visual Charts**: Speaking time, score distributions, AI detection results
- **CSV Exports**: All data available for further analysis

## Limitations and Risks

- **Bias Risk**: AI models may introduce unconscious bias into evaluations
- **Privacy Concerns**: Audio data is processed by third-party AI services
- **False Positives**: AI detection may incorrectly flag human responses
- **Audio Quality**: Poor recordings may affect accuracy
- **Language Support**: Optimized for English interviews
- **Legal Compliance**: May not comply with employment laws in all jurisdictions

## Best Practices

- Use as a supplementary tool, not primary decision-maker
- Always involve human reviewers in final decisions
- Ensure candidates consent to AI-powered analysis
- Regularly audit for bias and fairness
- Test thoroughly with diverse candidate pools
- Comply with local employment and privacy laws

## Contributing

Contributions are welcome! Please read our contributing guidelines and ensure your changes:

- Include appropriate tests
- Follow ethical AI practices
- Consider bias and fairness implications
- Update documentation as needed

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Ethics Statement

Cruitfit is designed to assist, not replace, human judgment in hiring processes. We strongly encourage:

- Transparent communication with candidates about AI usage
- Regular bias testing and mitigation
- Compliance with employment laws and regulations
- Ethical use that promotes fairness and diversity

## Acknowledgments

- OpenAI Whisper for speech recognition
- Hugging Face for transformer models
- Google, Groq, and other AI providers for evaluation services
- The open-source community for essential libraries

## Support

- Open an issue for bugs or feature requests
- Check the [Colab notebook](https://colab.research.google.com/drive/1Q2YVX7iejgnotLQ6f8a-4LlgE3iR_fR6?usp=sharing) for live examples
- Review the code documentation for technical details

---

**Remember**: This tool should enhance human decision-making, not replace it. Always prioritize fairness, transparency, and legal compliance in your hiring processes.

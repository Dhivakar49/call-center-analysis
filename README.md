# Call Center Compliance API

## Description
A FastAPI-based REST API that accepts MP3 call recordings via Base64 encoding and performs multi-stage AI analysis. The pipeline transcribes Tamil/Hindi call center audio using Groq Whisper, romanizes it to Tanglish/Hinglish, and analyzes SOP compliance, payment intent, and business metrics using Groq LLaMA. Transcripts are indexed in ChromaDB for semantic search.

## Tech Stack
- Language/Framework: Python, FastAPI
- Key libraries: groq, noisereduce, soundfile, chromadb, ffmpeg-python
- LLM/AI models used: Groq Whisper large-v3 (STT), Groq LLaMA 3.3 70B Versatile (NLP)

## Setup Instructions
1. Clone the repository
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Set environment variables
```bash
cp .env.example .env
# Add your GROQ_API_KEY in .env
```
4. Run the application
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## Approach
- Base64 MP3 is decoded and enhanced (8kHz→16kHz upsampling + noise reduction via noisereduce)
- Audio is split into 30s chunks and transcribed using Groq Whisper large-v3
- Raw transcript is sent to Groq LLaMA 3.3 70B for phonetic romanization (Tanglish/Hinglish) and NLP analysis
- SOP compliance is measured against: Greeting → Identification → Problem Statement → Solution → Closing
- Payment preference and rejection reason are strictly classified from allowed enums
- Every transcript is indexed in ChromaDB for semantic search
- All classification values are validated before returning to prevent scoring failures

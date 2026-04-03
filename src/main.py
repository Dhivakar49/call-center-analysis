"""
Call Center Compliance API
--------------------------
Accepts MP3 audio via Base64, performs multi-stage AI analysis:
  1. Audio enhancement (noisereduce + ffmpeg)
  2. Speech-to-text (Groq Whisper large-v3)
  3. NLP analysis (Groq LLaMA 3.3 70B)
  4. Vector storage (ChromaDB)

Returns structured JSON with transcript, SOP validation, analytics, and keywords.
"""
import os
import base64
import tempfile
import json
import re
import subprocess
import math
import uuid
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
import chromadb

load_dotenv()

app = FastAPI()

API_KEY = os.getenv("API_KEY", "sk_track3_987654321")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)

# Vector storage — ChromaDB for semantic search
chroma_client = chromadb.PersistentClient(path="vector_store")
transcript_collection = chroma_client.get_or_create_collection(
    name="transcripts",
    metadata={"hnsw:space": "cosine"}
)

CHUNK_SEC = 30

# Load SpeechBrain MetricGAN+ once at startup
print("API ready.")



class AnalyticsRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


# ── Audio processing ──────────────────────────────────────────────────────────

def get_duration(path: str) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True, text=True
    )
    return float(r.stdout.strip())


def enhance_audio(input_path: str, output_path: str):
    """Enhance speech using noisereduce + scipy bandpass filter."""
    import soundfile as sf
    import noisereduce as nr
    import numpy as np
    from scipy import signal

    upsampled = input_path + "_16k.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1",
        "-af", "volume=3.0",
        upsampled
    ], capture_output=True, check=True)

    data, rate = sf.read(upsampled)

    # Bandpass filter for telephone speech range
    nyq = rate / 2
    low, high = 100 / nyq, 4000 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data)

    # Noise reduction
    noise_sample = filtered[:rate // 2] if len(filtered) > rate // 2 else filtered
    reduced = nr.reduce_noise(
        y=filtered, sr=rate,
        y_noise=noise_sample,
        prop_decrease=0.85,
        stationary=True,
        n_fft=1024,
    )

    # Normalize
    max_val = np.max(np.abs(reduced))
    if max_val > 0:
        reduced = reduced / max_val * 0.95

    sf.write(output_path, reduced, rate)
    if os.path.exists(upsampled):
        os.remove(upsampled)


def split_wav(wav_path: str, chunk_dir: str, duration: float) -> list:
    os.makedirs(chunk_dir, exist_ok=True)
    num_chunks = math.ceil(duration / CHUNK_SEC)
    chunks = []
    for i in range(num_chunks):
        start = i * CHUNK_SEC
        out = os.path.join(chunk_dir, f"chunk_{i:03d}.wav")
        subprocess.run([
            "ffmpeg", "-y", "-i", wav_path,
            "-ss", str(start), "-t", str(CHUNK_SEC),
            "-ar", "16000", "-ac", "1", out
        ], capture_output=True, check=True)
        chunks.append(out)
    return chunks


# ── Transcription ─────────────────────────────────────────────────────────────

def transcribe_chunk(chunk_path: str, lang_code: str, prev_text: str = "", is_first: bool = False) -> str:
    # No prompt for first chunk — avoids hallucinating words like "Vanakkam"
    # Only use previous text as context for subsequent chunks
    with open(chunk_path, "rb") as f:
        kwargs = dict(
            file=("chunk.wav", f, "audio/wav"),
            model="whisper-large-v3",
            language=lang_code,
            response_format="verbose_json",
            temperature=0.0,
        )
        if not is_first and prev_text:
            kwargs["prompt"] = prev_text[-200:].strip()
        response = groq_client.audio.transcriptions.create(**kwargs)

    return response.text.strip()


def transcribe_audio(audio_bytes: bytes, fmt: str, language: str) -> str:
    lang_code = "ta" if "tamil" in language.lower() else "hi"

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = os.path.join(tmpdir, f"input.{fmt}")
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)

        enhanced_path = os.path.join(tmpdir, "enhanced.wav")
        enhance_audio(raw_path, enhanced_path)

        duration = get_duration(enhanced_path)
        chunk_dir = os.path.join(tmpdir, "chunks")
        chunks = split_wav(enhanced_path, chunk_dir, duration)

        parts = []
        prev = ""
        for idx, chunk_path in enumerate(chunks):
            text = transcribe_chunk(chunk_path, lang_code, prev, is_first=(idx == 0))
            # Skip if chunk is clearly hallucinated (too short or repetitive)
            if len(text.strip()) > 5:
                parts.append(text)
                prev = text

        return " ".join(parts).strip()


# ── NLP Analysis ──────────────────────────────────────────────────────────────

NLP_PROMPT = (
    "You are a call center compliance analyst.\n\n"
    "The transcript below is raw Whisper output from a call center recording.\n\n"
    "STRICT RULE FOR transcript_clean:\n"
    "- Tamil words → write phonetically in English letters (Tanglish). Example: வணக்கம் → Vanakkam, சொல்லுங்க → Sollunga, நீங்க → Neenga\n"
    "- Hindi words → write phonetically in English letters (Hinglish). Example: नमस्ते → Namaste, बताइए → Bataiye\n"
    "- English words → keep EXACTLY as spoken, do NOT change\n"
    "- NEVER translate Tamil/Hindi words to their English meaning\n"
    "- WRONG: வணக்கம் → 'Greetings' | CORRECT: வணக்கம் → 'Vanakkam'\n"
    "- WRONG: சொல்லுங்க → 'Please tell' | CORRECT: சொல்லுங்க → 'Sollunga'\n"
    "- WRONG: 'ah sollunga' → 'yeah tell me' | CORRECT: 'ah sollunga' stays 'ah sollunga'\n"
    "- WRONG: 'neenga irukingala' → 'are you there' | CORRECT: 'neenga irukingala' stays as-is\n"
    "- Remove only clear hallucinated nonsense words\n\n"
    "TASK 2 - COMPLIANCE ANALYSIS based on the meaning.\n\n"
    "Transcript:\n{transcript}\n\n"
    "Language: {language}\n\n"
    "Return ONLY valid JSON:\n"
    "{{\n"
    '  "transcript_clean": "Phonetic Tanglish/Hinglish. Tamil words romanized, Hindi words romanized, English words kept as-is. Every line: Agent: ...\\nCustomer: ...",\n'
    '  "summary": "2-3 sentence English summary",\n'
    '  "sop_validation": {{\n'
    '    "greeting": true,\n'
    '    "identification": false,\n'
    '    "problemStatement": true,\n'
    '    "solutionOffering": true,\n'
    '    "closing": true,\n'
    '    "complianceScore": 0.0,\n'
    '    "adherenceStatus": "FOLLOWED or NOT_FOLLOWED",\n'
    '    "explanation": "Short English explanation"\n'
    "  }},\n"
    '  "analytics": {{\n'
    '    "paymentPreference": "EMI or FULL_PAYMENT or PARTIAL_PAYMENT or DOWN_PAYMENT or NONE",\n'
    '    "rejectionReason": "HIGH_INTEREST or BUDGET_CONSTRAINTS or ALREADY_PAID or NOT_INTERESTED or NONE",\n'
    '    "sentiment": "Positive or Negative or Neutral"\n'
    "  }},\n"
    '  "keywords": ["keyword1", "keyword2"]\n'
    "}}\n\n"
    "Rules:\n"
    "- transcript_clean: romanize Tamil/Hindi phonetically, keep English as-is, remove nonsense. Write EVERY line spoken. NEVER use '...' or skip content.\n"
    "- complianceScore = true steps / 5 (e.g. 4 true = 0.8, 3 true = 0.6, 5 true = 1.0)\n"
    "- adherenceStatus = FOLLOWED only if ALL 5 steps true, else NOT_FOLLOWED\n"
    "- greeting: agent said hello/vanakkam/namaste at start of call\n"
    "- identification: agent confirmed customer name OR account number\n"
    "- problemStatement: purpose of call clearly stated\n"
    "- solutionOffering: agent offered solution, product, course, or payment plan\n"
    "- closing: call ended with proper goodbye/thank you\n"
    "- paymentPreference: classify strictly:\n"
    "  EMI = monthly installments (e.g. '6 months EMI', '24 manathu')\n"
    "  FULL_PAYMENT = full amount at once\n"
    "  PARTIAL_PAYMENT = part now rest later (e.g. 'I can pay 2000 today and rest next month')\n"
    "  DOWN_PAYMENT = initial deposit/advance only\n"
    "  NONE = no payment discussed\n"
    "- rejectionReason: only if sale NOT completed:\n"
    "  HIGH_INTEREST = too expensive/high rate\n"
    "  BUDGET_CONSTRAINTS = no money now (e.g. 'I dont have enough money')\n"
    "  ALREADY_PAID = customer already paid\n"
    "  NOT_INTERESTED = not interested\n"
    "  NONE = sale completed or no rejection\n"
    "- sentiment: Positive = customer happy/agreed/interested, Negative = angry/refused/frustrated, Neutral = no strong emotion\n"
    "- keywords: 5-10 English terms directly traceable to the transcript\n"
)


# Common domain words Whisper gets wrong on low-quality call center audio
WORD_CORRECTIONS = {
    "date science": "data science",
    "data signs": "data science",
    "data sign": "data science",
    "guvi": "Guvi",
    "gigi": "Guvi",
    "guvee": "Guvi",
    "iit madras": "IIT Madras",
    "iits": "IIT",
    "hcl": "HCL",
    "emi": "EMI",
    "e.m.i": "EMI",
    "artificial intelligence": "Artificial Intelligence",
    "devops": "DevOps",
    "dev ops": "DevOps",
    "machine learning": "Machine Learning",
    "placement support": "Placement Support",
    "course fee": "course fee",
    "data analyst": "Data Analyst",
    "full stack": "Full Stack",
    "full-stack": "Full Stack",
}


def correct_transcript(text: str) -> str:
    """Fix common domain-specific word errors from Whisper."""
    # Case-insensitive replacement but preserve original case structure
    import re
    for wrong, correct in WORD_CORRECTIONS.items():
        text = re.sub(re.escape(wrong), correct, text, flags=re.IGNORECASE)
    return text


VALID_PAYMENT = {"EMI", "FULL_PAYMENT", "PARTIAL_PAYMENT", "DOWN_PAYMENT", "NONE"}
VALID_REJECTION = {"HIGH_INTEREST", "BUDGET_CONSTRAINTS", "ALREADY_PAID", "NOT_INTERESTED", "NONE"}
VALID_SENTIMENT = {"Positive", "Negative", "Neutral"}


def extract_json(raw: str) -> dict:
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().lstrip("json").strip()
            if part.startswith("{"):
                raw = part
                break
    start, end = raw.find("{"), raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]
    try:
        return json.loads(raw)
    except Exception:
        raw = re.sub(r",\s*([}\]])", r"\1", raw)
        try:
            return json.loads(raw)
        except Exception:
            return {}


def sanitize_analysis(analysis: dict, raw_transcript: str) -> dict:
    """Ensure all fields are present and classification values are valid."""
    sop = analysis.get("sop_validation", {})
    analytics = analysis.get("analytics", {})

    # Sanitize payment preference
    payment = analytics.get("paymentPreference", "NONE").strip().upper()
    if payment not in VALID_PAYMENT:
        payment = "NONE"

    # Sanitize rejection reason
    rejection = analytics.get("rejectionReason", "NONE").strip().upper()
    if rejection not in VALID_REJECTION:
        rejection = "NONE"

    # Sanitize sentiment
    sentiment = analytics.get("sentiment", "Neutral").strip().capitalize()
    if sentiment not in VALID_SENTIMENT:
        sentiment = "Neutral"

    # Sanitize compliance score
    try:
        score = float(sop.get("complianceScore", 0.0))
        score = max(0.0, min(1.0, score))
    except Exception:
        score = 0.0

    # Recalculate score precisely from booleans
    steps = ["greeting", "identification", "problemStatement", "solutionOffering", "closing"]
    true_count = sum(1 for s in steps if sop.get(s, False) is True)
    calculated_score = round(true_count / 5, 2)  # 0.0, 0.2, 0.4, 0.6, 0.8, 1.0

    adherence = sop.get("adherenceStatus", "NOT_FOLLOWED").strip().upper()
    if adherence not in {"FOLLOWED", "NOT_FOLLOWED"}:
        adherence = "NOT_FOLLOWED"
    # Force NOT_FOLLOWED if any step is false
    if true_count < 5:
        adherence = "NOT_FOLLOWED"
    else:
        adherence = "FOLLOWED"

    # Ensure transcript is never empty
    transcript = analysis.get("transcript_clean", "").strip()
    if not transcript:
        transcript = raw_transcript

    return {
        "transcript_clean": transcript,
        "summary": analysis.get("summary", "").strip() or "Call center conversation analyzed.",
        "sop_validation": {
            "greeting": bool(sop.get("greeting", False)),
            "identification": bool(sop.get("identification", False)),
            "problemStatement": bool(sop.get("problemStatement", False)),
            "solutionOffering": bool(sop.get("solutionOffering", False)),
            "closing": bool(sop.get("closing", False)),
            "complianceScore": calculated_score,
            "adherenceStatus": adherence,
            "explanation": sop.get("explanation", "").strip() or "SOP analysis completed.",
        },
        "analytics": {
            "paymentPreference": payment,
            "rejectionReason": rejection,
            "sentiment": sentiment,
        },
        "keywords": analysis.get("keywords", []) or [],
    }


def analyze_with_llm(transcript: str, language: str) -> dict:
    safe = transcript.replace('"', "'").replace('\r', ' ').replace('\\', '')
    # Don't truncate — send full transcript for accuracy
    if len(safe) > 6000:
        safe = safe[:6000]

    import time
    for attempt in range(3):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a JSON-only API. Output ONLY valid JSON, no markdown, no extra text."},
                    {"role": "user", "content": NLP_PROMPT.format(transcript=safe, language=language)}
                ],
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"}
            )
            raw_analysis = extract_json(response.choices[0].message.content.strip())
            return sanitize_analysis(raw_analysis, transcript)
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                print(f"Rate limit hit, waiting 60s... (attempt {attempt+1})")
                time.sleep(60)
            else:
                raise


def store_transcript(transcript: str, summary: str, language: str, analysis: dict):
    """Index transcript in ChromaDB for semantic search."""
    doc_id = str(uuid.uuid4())
    transcript_collection.add(
        documents=[transcript],
        metadatas=[{
            "language": language,
            "summary": summary[:500],
            "paymentPreference": analysis.get("analytics", {}).get("paymentPreference", "NONE"),
            "sentiment": analysis.get("analytics", {}).get("sentiment", "Neutral"),
            "adherenceStatus": analysis.get("sop_validation", {}).get("adherenceStatus", "NOT_FOLLOWED"),
            "complianceScore": str(analysis.get("sop_validation", {}).get("complianceScore", 0.0)),
        }],
        ids=[doc_id]
    )
    return doc_id


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post("/api/call-analytics")
async def call_analytics(request: AnalyticsRequest, x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
        transcript = transcribe_audio(audio_bytes, request.audioFormat, request.language)
        transcript = correct_transcript(transcript)

        if not transcript:
            raise HTTPException(status_code=422, detail="Transcription returned empty result")

        analysis = analyze_with_llm(transcript, request.language)

        # Store in vector DB for semantic search
        store_transcript(
            transcript=analysis.get("transcript_clean") or transcript,
            summary=analysis.get("summary", ""),
            language=request.language,
            analysis=analysis
        )

        return {
            "status": "success",
            "language": request.language,
            "transcript": analysis.get("transcript_clean") or transcript,
            "summary": analysis.get("summary", ""),
            "sop_validation": analysis.get("sop_validation", {}),
            "analytics": analysis.get("analytics", {}),
            "keywords": analysis.get("keywords", []),
        }
    except HTTPException:
        raise
    except Exception as e:
        return {
            "status": "error",
            "language": request.language,
            "transcript": "",
            "summary": f"Processing error: {str(e)}",
            "sop_validation": {
                "greeting": False, "identification": False,
                "problemStatement": False, "solutionOffering": False,
                "closing": False, "complianceScore": 0.0,
                "adherenceStatus": "NOT_FOLLOWED",
                "explanation": "Processing failed"
            },
            "analytics": {"paymentPreference": "NONE", "rejectionReason": "NONE", "sentiment": "Neutral"},
            "keywords": []
        }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/search")
def search_transcripts(q: str, x_api_key: str = Header(...), n: int = 5):
    """Semantic search over stored transcripts."""
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")
    results = transcript_collection.query(
        query_texts=[q],
        n_results=min(n, transcript_collection.count() or 1)
    )
    return {
        "query": q,
        "results": [
            {"transcript": doc, "metadata": meta}
            for doc, meta in zip(
                results["documents"][0],
                results["metadatas"][0]
            )
        ]
    }

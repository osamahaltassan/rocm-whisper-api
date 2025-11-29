import whisper
import torch
import os
import traceback
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse, PlainTextResponse

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
model = None

# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------

def format_timestamp_srt(seconds: float) -> str:
    """Convert seconds to SRT format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Convert seconds to VTT format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: list) -> str:
    """Generate SRT subtitle format from Whisper segments."""
    srt_lines = []
    for i, segment in enumerate(segments, start=1):
        start = format_timestamp_srt(segment["start"])
        end = format_timestamp_srt(segment["end"])
        text = segment["text"].strip()
        srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(srt_lines)


def generate_vtt(segments: list) -> str:
    """Generate WebVTT subtitle format from Whisper segments."""
    vtt_lines = ["WEBVTT\n"]
    for segment in segments:
        start = format_timestamp_vtt(segment["start"])
        end = format_timestamp_vtt(segment["end"])
        text = segment["text"].strip()
        vtt_lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(vtt_lines)


def build_verbose_response(result: dict, duration: float) -> dict:
    """Build OpenAI verbose_json response format."""
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "id": seg.get("id", 0),
            "seek": seg.get("seek", 0),
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
            "tokens": seg.get("tokens", []),
            "temperature": seg.get("temperature", 0.0),
            "avg_logprob": seg.get("avg_logprob", 0.0),
            "compression_ratio": seg.get("compression_ratio", 0.0),
            "no_speech_prob": seg.get("no_speech_prob", 0.0),
        })
    
    return {
        "task": "transcribe",
        "language": result.get("language", ""),
        "duration": round(duration, 2),
        "text": result.get("text", "").strip(),
        "segments": segments,
    }


async def process_audio(
    file: UploadFile,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
) -> dict:
    """Process audio file and return Whisper result."""
    if not model:
        raise HTTPException(
            status_code=503,
            detail="Whisper model is not available. Check server logs."
        )
    
    contents = await file.read()
    temp_path = None
    
    try:
        suffix = os.path.splitext(file.filename or ".wav")[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            temp_path = tmp.name
        
        print(f"Transcribing: {file.filename}")
        
        transcribe_opts = {
            "fp16": False,
            "task": "transcribe",
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.6,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
        }
        if language:
            transcribe_opts["language"] = language
        if prompt:
            transcribe_opts["initial_prompt"] = prompt
        
        result = model.transcribe(temp_path, **transcribe_opts)
        
        # Calculate duration from segments
        duration = 0.0
        if result.get("segments"):
            duration = result["segments"][-1]["end"]
        
        print("Transcription successful.")
        return {"result": result, "duration": duration}
    
    except Exception as e:
        print(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------------------------------
# FastAPI Application
# --------------------------------------------------------------------------

app = FastAPI(
    title="Whisper API",
    description="OpenAI-compatible Whisper API running on ROCm",
    version="2.0.0",
)


@app.on_event("startup")
def load_whisper_model():
    """Load Whisper model on startup."""
    global model
    try:
        print("=" * 50)
        print(f"PyTorch version: {torch.__version__}")
        if hasattr(torch.version, 'hip'):
            print(f"ROCm version: {torch.version.hip}")
        print(f"GPU available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Device: {torch.cuda.get_device_name(0)}")
        print(f"Loading model: {MODEL_NAME}")
        print("=" * 50)

        # Load to CPU first, then move to GPU if available
        print("Loading model to CPU...")
        cpu_model = whisper.load_model(MODEL_NAME, device="cpu")
        
        if DEVICE == "cuda":
            print("Moving model to GPU...")
            model = cpu_model.to(DEVICE)
        else:
            model = cpu_model
        
        print(f"Model ready on: {DEVICE}")

    except Exception as e:
        print("=" * 50)
        print("FAILED TO LOAD MODEL")
        traceback.print_exc()
        print("=" * 50)
        model = None


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "running",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),  # Accepted but ignored (uses env var)
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0.0),
):
    """
    OpenAI-compatible transcription endpoint.
    
    Supported response_format: json, text, verbose_json, vtt
    """
    data = await process_audio(file, language, prompt, temperature or 0.0)
    result = data["result"]
    duration = data["duration"]
    
    if response_format == "text":
        return PlainTextResponse(content=result.get("text", "").strip())
    
    elif response_format == "verbose_json":
        return JSONResponse(content=build_verbose_response(result, duration))
    
    elif response_format == "vtt":
        vtt_content = generate_vtt(result.get("segments", []))
        return PlainTextResponse(content=vtt_content, media_type="text/vtt")
    
    else:  # json (default)
        return JSONResponse(content={"text": result.get("text", "").strip()})


@app.post("/v1/audio/transcriptions/srt")
async def transcribe_srt(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    temperature: Optional[float] = Form(0.0),
):
    """
    Transcription endpoint that returns SRT subtitle format.
    """
    data = await process_audio(file, language, prompt, temperature or 0.0)
    result = data["result"]
    
    srt_content = generate_srt(result.get("segments", []))
    return PlainTextResponse(content=srt_content, media_type="text/plain")
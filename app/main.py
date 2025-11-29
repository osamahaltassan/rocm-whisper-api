import whisper
import torch
import os
import traceback
import tempfile
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse

# --------------------------------------------------------------------------
# 1. 모델 및 장치 설정
# --------------------------------------------------------------------------

# ROCm 지원이 가능한지 확인하고 장치를 설정합니다.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 사용할 Whisper 모델을 선택합니다.
MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
model = None

def generate_srt(segments):
    """Generate SRT subtitle format from Whisper segments."""
    srt_output = []
    for i, segment in enumerate(segments, start=1):
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        text = segment["text"].strip()
        
        srt_output.append(f"{i}")
        srt_output.append(f"{start_time} --> {end_time}")
        srt_output.append(text)
        srt_output.append("")  # Blank line between entries
    
    return "\n".join(srt_output)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

# --------------------------------------------------------------------------
# 2. FastAPI 애플리케이션 초기화 및 모델 로드
# --------------------------------------------------------------------------
app = FastAPI(
    title="ROCm Whisper API",
    description="An API to transcribe audio files using OpenAI's Whisper on ROCm.",
    version="1.1.0" # 기능 추가로 버전 업데이트
)

@app.on_event("startup")
def load_whisper_model():
    """
    FastAPI 앱이 시작될 때 Whisper 모델을 로드합니다.
    """
    global model
    try:
        print("="*50)
        print(f"PyTorch version: {torch.__version__}")
        if hasattr(torch.version, 'hip'):
            print(f"Torch is built with ROCm: {torch.version.hip}")
        print(f"Is ROCm (GPU) available? -> {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Attempting to load Whisper model: '{MODEL_NAME}'")
        print("="*50)

        # 1단계: 모델을 CPU에 먼저 로드합니다.
        print(f"Step 1: Loading model '{MODEL_NAME}' onto CPU...")
        cpu_model = whisper.load_model(MODEL_NAME, device="cpu")
        print("Step 1: Model loaded on CPU successfully.")

        # 2단계: GPU가 사용 가능하면 모델을 GPU로 이동시킵니다.
        if DEVICE == "cuda":
            print(f"Step 2: Moving model to GPU ({DEVICE})...")
            model = cpu_model.to(DEVICE)
            print("Step 2: Model moved to GPU successfully.")
        else:
            model = cpu_model
        
        print(f"\n✅ Whisper model '{MODEL_NAME}' is ready on device: {DEVICE}.\n")

    except Exception as e:
        print("="*50)
        print("❌ FAILED TO LOAD WHISPER MODEL ❌")
        traceback.print_exc()
        print("="*50)
        model = None


@app.get("/", summary="Health Check", description="API 서버의 상태를 확인합니다.")
def read_root():
    status = "running"
    model_status = "loaded" if model else "failed_to_load"
    return {"status": status, "model_status": model_status, "model_name": MODEL_NAME}

@app.post("/transcribe", summary="Transcribe Audio File", description="오디오 파일을 텍스트로 변환합니다.")
async def transcribe_audio(
    file: UploadFile = File(...),
    response_format: Optional[str] = Form("text")
):
    if not model:
        raise HTTPException(status_code=503, detail="Whisper model is not available. Check server logs for details.")
    
    contents = await file.read()

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_audio_file:
            temp_audio_file.write(contents)
            temp_path = temp_audio_file.name
        
        print(f"Transcribing file: {file.filename} (format: {response_format})")
        
        # Force task="transcribe" to disable translation
        result = model.transcribe(temp_path, fp16=False, task="transcribe")

        duration = 0
        if result.get("segments"):
            last_segment = result["segments"][-1]
            duration = last_segment["end"]

        print("Transcription successful.")

        # Handle different response formats
        if response_format == "srt":
            srt_content = generate_srt(result["segments"])
            return JSONResponse(content={
                "filename": file.filename,
                "duration_seconds": round(duration, 2),
                "language": result["language"],
                "srt": srt_content
            })
        else:  # Default to text
            return JSONResponse(content={
                "filename": file.filename,
                "duration_seconds": round(duration, 2),
                "language": result["language"],
                "text": result["text"]
            })
    except Exception as e:
        print(f"❌ An error occurred during transcription: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if 'temp_path' in locals() and os.path.exists(temp_path):
            os.remove(temp_path)


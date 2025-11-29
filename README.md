# ROCm Whisper API

OpenAI-compatible Whisper API running on AMD GPUs via ROCm.

Based on: https://github.com/jjajjara/rocm-whisper-api

## Quick Start

```bash
docker run -d \
  --name rocm-whisper-api \
  --device /dev/kfd --device /dev/dri \
  -p 8080:8080 \
  -e HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  -e WHISPER_MODEL=base \
  -v whisper-cache:/root/.cache/whisper \
  ghcr.io/osamahaltassan/rocm-whisper-api:1.1
```

Or with docker-compose:

```yaml
services:
  rocm-whisper-api:
    image: ghcr.io/osamahaltassan/rocm-whisper-api:1.1
    container_name: rocm-whisper-api
    restart: unless-stopped
    ports:
      - "8080:8080"
    environment:
      - HSA_OVERRIDE_GFX_VERSION=11.0.0
      - WHISPER_MODEL=base
    devices:
      - "/dev/kfd:/dev/kfd"
      - "/dev/dri:/dev/dri"
    volumes:
      - whisper-cache:/root/.cache/whisper

volumes:
  whisper-cache:
```

## Model Cache

Mount `/root/.cache/whisper` to persist downloaded models across container restarts. Without this volume, Whisper will re-download the model every time the container starts.

| Model  | Size    | VRAM  |
| ------ | ------- | ----- |
| tiny   | ~75 MB  |       |
| base   | ~145 MB |       |
| small  | ~465 MB |       |
| medium | ~1.5 GB |       |
| large  | ~3 GB   | 7178M |

## Endpoints

|Endpoint|Method|Description|
|---|---|---|
|`/`|GET|Health check|
|`/v1/audio/transcriptions`|POST|OpenAI-compatible transcription|
|`/v1/audio/transcriptions/srt`|POST|SRT subtitle output|

## API Reference

### POST /v1/audio/transcriptions

OpenAI-compatible endpoint. Accepts multipart form data.

**Parameters:**

|Parameter|Type|Required|Description|
|---|---|---|---|
|`file`|file|Yes|Audio file (mp3, wav, m4a, webm, etc.)|
|`model`|string|No|Accepted but ignored (uses `WHISPER_MODEL` env var)|
|`language`|string|No|ISO-639-1 code (e.g., `en`, `ko`, `ja`)|
|`prompt`|string|No|Text to guide transcription style|
|`response_format`|string|No|`json` (default), `text`, `verbose_json`, `vtt`|
|`temperature`|float|No|Sampling temperature (0-1)|

**Response Formats:**

```bash
# json (default)
curl -X POST -F "file=@audio.mp3" http://localhost:8080/v1/audio/transcriptions
# {"text": "Transcribed text..."}

# text
curl -X POST -F "file=@audio.mp3" -F "response_format=text" http://localhost:8080/v1/audio/transcriptions
# Transcribed text...

# verbose_json
curl -X POST -F "file=@audio.mp3" -F "response_format=verbose_json" http://localhost:8080/v1/audio/transcriptions
# {"task": "transcribe", "language": "en", "duration": 12.5, "text": "...", "segments": [...]}

# vtt
curl -X POST -F "file=@audio.mp3" -F "response_format=vtt" http://localhost:8080/v1/audio/transcriptions
# WEBVTT
# 00:00:00.000 --> 00:00:03.500
# First segment...
```

### POST /v1/audio/transcriptions/srt

Dedicated SRT subtitle endpoint.

**Parameters:**

|Parameter|Type|Required|Description|
|---|---|---|---|
|`file`|file|Yes|Audio file|
|`language`|string|No|ISO-639-1 code|
|`prompt`|string|No|Text to guide transcription|
|`temperature`|float|No|Sampling temperature (0-1)|

**Example:**

```bash
curl -X POST -F "file=@audio.mp3" http://localhost:8080/v1/audio/transcriptions/srt -o output.srt
```

## Configuration

|Variable|Default|Description|
|---|---|---|
|`WHISPER_MODEL`|`base`|Model size: `tiny`, `base`, `small`, `medium`, `large`|
|`HSA_OVERRIDE_GFX_VERSION`|`11.0.0`|ROCm GPU override (for RDNA3)|

## Integration

### Open WebUI

Set Speech-to-Text URL to:

```
http://rocm-whisper-api:8080/v1
```

### Python (openai client)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="not-needed")

with open("audio.mp3", "rb") as f:
    transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
print(transcript.text)
```

### Python (requests)

```python
import requests

with open("audio.mp3", "rb") as f:
    response = requests.post(
        "http://localhost:8080/v1/audio/transcriptions",
        files={"file": f},
    )
print(response.json()["text"])
```
## 🧪 How to Test

The API supports two output formats: **text** (default) and **SRT** subtitles. Translation is disabled - the API only transcribes in the original language.

### 1. Text Output (Default)

**cURL:**
```bash
curl -X POST -F "file=@test.m4a" http://localhost:8080/transcribe
```

**Python:**
```bash
python3 client_example.py test.m4a
# or explicitly
python3 client_example.py test.m4a --format text
```

**Response:**
```json
{
  "filename": "test.m4a",
  "duration_seconds": 12.5,
  "language": "en",
  "text": "This is the transcribed text..."
}
```

---

### 2. SRT Subtitle Output

**cURL:**
```bash
curl -X POST \
  -F "file=@test.m4a" \
  -F "response_format=srt" \
  http://localhost:8080/transcribe
```

**Python:**
```bash
python3 client_example.py test.m4a --format srt
```

**Response:**
```json
{
  "filename": "test.m4a",
  "duration_seconds": 12.5,
  "language": "en",
  "srt": "1\n00:00:00,000 --> 00:00:03,500\nThis is the first segment...\n\n2\n00:00:03,500 --> 00:00:07,200\nAnd this is the second...\n\n"
}
```

**Save SRT to file:**
```bash
curl -X POST \
  -F "file=@test.m4a" \
  -F "response_format=srt" \
  http://localhost:8080/transcribe | jq -r '.srt' > output.srt
```

---

### 📋 Output Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| `text` | Plain text transcription | General transcription, note-taking |
| `srt` | SubRip subtitle format | Video subtitles, accessibility |

**Note:** Translation is permanently disabled. The API only transcribes audio in its original language. To get English output from non-English audio, use a separate translation service.
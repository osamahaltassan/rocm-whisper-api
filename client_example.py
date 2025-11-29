#!/usr/bin/env python3
"""
Client example for OpenAI-compatible Whisper API.
Supports both standard transcription and SRT output.
"""

import requests
import argparse
import os
import sys

API_BASE = os.getenv("WHISPER_API_URL", "http://localhost:8080")


def transcribe(
    file_path: str,
    response_format: str = "json",
    language: str = None,
    prompt: str = None,
):
    """Transcribe audio using /v1/audio/transcriptions endpoint."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    url = f"{API_BASE}/v1/audio/transcriptions"
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {"response_format": response_format}
        
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        print(f"POST {url}")
        print(f"File: {file_path}, Format: {response_format}")
        
        response = requests.post(url, files=files, data=data, timeout=300)

    if response.status_code == 200:
        if response_format in ("text", "vtt"):
            print(response.text)
        else:
            print(response.json())
    else:
        print(f"Error {response.status_code}: {response.text}")
        sys.exit(1)


def transcribe_srt(file_path: str, language: str = None, prompt: str = None):
    """Get SRT output using /v1/audio/transcriptions/srt endpoint."""
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    url = f"{API_BASE}/v1/audio/transcriptions/srt"
    
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f)}
        data = {}
        
        if language:
            data["language"] = language
        if prompt:
            data["prompt"] = prompt

        print(f"POST {url}")
        print(f"File: {file_path}")
        
        response = requests.post(url, files=files, data=data, timeout=300)

    if response.status_code == 200:
        print(response.text)
    else:
        print(f"Error {response.status_code}: {response.text}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper API Client")
    parser.add_argument("file", help="Audio file path")
    parser.add_argument(
        "--format",
        choices=["json", "text", "verbose_json", "vtt", "srt"],
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument("--language", help="Language code (e.g., en, ko, ja)")
    parser.add_argument("--prompt", help="Prompt to guide transcription")
    parser.add_argument("--output", "-o", help="Save output to file")

    args = parser.parse_args()

    # Capture output
    if args.format == "srt":
        transcribe_srt(args.file, args.language, args.prompt)
    else:
        transcribe(args.file, args.format, args.language, args.prompt)
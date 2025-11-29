import requests
import os
import argparse
import mimetypes

API_URL = "http://localhost:8080/transcribe"

def get_mime_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type:
        return mime_type
    
    ext = os.path.splitext(file_path)[1].lower()
    mime_map = {
        '.m4a': 'audio/mp4',
        '.mp3': 'audio/mpeg',
        '.wav': 'audio/wav',
        '.webm': 'audio/webm',
        '.ogg': 'audio/ogg'
    }
    return mime_map.get(ext, 'application/octet-stream')

def transcribe_audio(file_path, response_format='text'):
    if not os.path.exists(file_path):
        print(f"Error: File not found '{file_path}'")
        return

    mime_type = get_mime_type(file_path)
    print(f"File: '{os.path.basename(file_path)}', MIME: '{mime_type}', Format: {response_format}")

    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, mime_type)}
        data = {'response_format': response_format}
        
        print(f"Sending to {API_URL}...")
        
        try:
            response = requests.post(API_URL, files=files, data=data, timeout=300)

            if response.status_code == 200:
                result = response.json()
                print("\n--- Transcription Success ---")
                print(f"Filename: {result.get('filename')}")
                print(f"Duration: {result.get('duration_seconds')}s")
                print(f"Language: {result.get('language')}")
                
                if response_format == 'srt':
                    print(f"\n--- SRT Output ---")
                    print(result.get('srt'))
                else:
                    print(f"\n--- Text Output ---")
                    print(result.get('text'))
            else:
                print(f"\n--- Error ---")
                print(f"Status: {response.status_code}")
                print(f"Response: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"\n--- Request Error ---")
            print(f"Failed to connect: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper API")
    parser.add_argument("filepath", help="Path to audio file")
    parser.add_argument("--format", choices=['text', 'srt'], default='text', 
                       help="Output format: text or srt (default: text)")
    
    args = parser.parse_args()
    transcribe_audio(args.filepath, args.format)
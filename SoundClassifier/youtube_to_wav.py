# Download and trim YouTube videos, and extract audio as wav files.
import os
import yt_dlp
import subprocess
from pathlib import Path
from urllib.parse import urlparse
import re

dataset_path = Path.home() / "Documents/soundClassifier/audioSnippets"

# Define categories
categories = {
    "LABEL": [  # Replace LABEL
        {"url": "URL", "start": "00:00:09", "duration": "00:00:05"},  # Replace URL
        {"url": "URL", "start": "00:00:04", "duration": "00:00:03"}
    ],
      "LABEL": [
        {"url": "URL", "start": "00:00:09", "duration": "00:00:05"},
        {"url": "URL", "start": "00:00:04", "duration": "00:00:03"}
    ],
      "LABEL": [
        {"url": "URL", "start": "00:00:09", "duration": "00:00:05"},
        {"url": "URL", "start": "00:00:04", "duration": "00:00:03"}
    ]
}

# Convert short YouTube URL to full length version
def convert_shorts_url(url):
    parsed = urlparse(url)
    if "shorts" in parsed.path:
        video_id = parsed.path.split("/")[2]
        return f"https://www.youtube.com/watch?v={video_id}"
    return url

# Download and trim URL and extract as wav file
def download_snippet(url, start, duration, category, index, save_path=dataset_path):
    url = convert_shorts_url(url)
    category_path = save_path / category
    category_path.mkdir(parents=True, exist_ok=True)
    
    # Output filename = category + index
    out_audio = category_path / f"{category}{index}.wav"

    temp_video = category_path / "temp_video.mp4"

    # Download video
    ydl_opts = {"format": "best", "outtmpl": str(temp_video)}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    # Trim and extract audio
    subprocess.run([
        "ffmpeg",
        "-i", str(temp_video),
        "-ss", start,
        "-t", duration,
        "-q:a", "0",
        "-map", "a",
        str(out_audio)
    ])

    temp_video.unlink()  # delete temp video after extraction
    print(f"Saved snippet: {out_audio}")

for category, snippets in categories.items():
    for idx, snippet in enumerate(snippets, start=1):
        download_snippet(snippet["url"], snippet["start"], snippet["duration"], category, idx)

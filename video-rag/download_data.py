"""
Fetches the sample video this recipe indexes: NASA's "Explains Cold Zone
Above Tropics" -- a real, public-domain (CC0), 5:12 video with genuine
narration synced to a whiteboard diagram, not stock footage.
"""

import os
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_VIDEO_URL = "https://images-assets.nasa.gov/video/ARC-20140529-AAV3515-NASA-Explains-Cold-Zone-Above-Tropics/ARC-20140529-AAV3515-NASA-Explains-Cold-Zone-Above-Tropics~mobile.mp4"
DEFAULT_VIDEO_PATH = DATA_DIR / "video.mp4"


def download_sample_video(url: str = DEFAULT_VIDEO_URL, dest: Path = DEFAULT_VIDEO_PATH) -> Path:
    """Download the sample video if it isn't already cached locally."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        print(f"Using cached video: {dest}")
        return dest

    print(f"Downloading sample video from {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    print(f"Saved to {dest} ({dest.stat().st_size / 1024 / 1024:.1f} MB)")
    return dest


if __name__ == "__main__":
    video_url = os.environ.get("SAMPLE_VIDEO_URL", DEFAULT_VIDEO_URL)
    download_sample_video(video_url)

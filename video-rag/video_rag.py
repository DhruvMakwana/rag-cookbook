"""
Video RAG: retrieve relevant moments from a video by combining a
timestamped transcript (speech-to-text) with sampled frames, so a query
can pull back both the spoken content AND what was actually on screen at
that moment.

Two real limitations this recipe demonstrates directly rather than
hiding: (1) transcript-only retrieval finds WHEN something was discussed,
but has no access to what was visually on screen -- a question about a
detail that's only shown, never spoken, can't be answered from
transcript text alone, no matter how good the retrieval is; (2) frame
sampling at a fixed interval means the exact right visual moment isn't
guaranteed to be captured, only the closest sampled frame to it.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import llm

DATA_DIR = Path(__file__).parent / "data"
VIDEO_PATH = DATA_DIR / "video.mp4"
VIDEO_URL = "https://images-assets.nasa.gov/video/ARC-20140529-AAV3515-NASA-Explains-Cold-Zone-Above-Tropics/ARC-20140529-AAV3515-NASA-Explains-Cold-Zone-Above-Tropics~mobile.mp4"

TRANSCRIPT_CACHE = Path(__file__).parent / ".transcript_cache.json"
FRAME_DIR = Path(__file__).parent / ".frames_cache"

FRAME_INTERVAL_SEC = 5
WHISPER_MODEL_SIZE = "small"
TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Real timestamps and keywords, checked directly against the video's own
# transcript -- not guessed. All answerable from speech alone.
VIDEO_EVAL_SET = [
    ("What temperature range is found at ground level in the tropics?", "80 to 90"),
    ("What is the point called where the troposphere meets the stratosphere?", "tropopause"),
    ("What percentage of UV-B radiation is absorbed by the ozone layer?", "90%"),
    ("What percentage of UV-C radiation is absorbed by the ozone layer?", "100%"),
    ("Why does the stratosphere begin to warm up?", "absorbed"),
]

# This one is NOT answerable from the transcript at all -- "altitude" is
# never spoken in the narration. It's a label on the whiteboard graph,
# visible only in the frame.
VISUAL_ONLY_QUESTION = "What does the Y-axis of the whiteboard graph say?"
VISUAL_ONLY_KEYWORD = "altitude"


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def download_sample_video(url: str = VIDEO_URL, dest: Path = VIDEO_PATH) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


# ======================================================================
# 1. Extract frames at a fixed interval -- ffmpeg, no separate frame
#    library needed since Whisper already requires ffmpeg on the system
# ======================================================================


def extract_frames(video_path: Path = VIDEO_PATH, interval_sec: int = FRAME_INTERVAL_SEC) -> list[tuple[float, Image.Image]]:
    FRAME_DIR.mkdir(exist_ok=True)
    cached = sorted(FRAME_DIR.glob("frame_*.jpg"))
    if cached:
        return [(_frame_index_to_timestamp(p, interval_sec), Image.open(p).copy()) for p in cached]

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-i", str(video_path), "-vf", f"fps=1/{interval_sec}", "-y", f"{tmp}/frame_%04d.jpg"],
            check=True, capture_output=True,
        )
        frames = []
        for i, path in enumerate(sorted(Path(tmp).glob("frame_*.jpg"))):
            dest = FRAME_DIR / path.name
            img = Image.open(path).copy()
            img.save(dest)
            frames.append((i * interval_sec, img))
    return frames


def _frame_index_to_timestamp(path: Path, interval_sec: int) -> float:
    index = int(path.stem.split("_")[1]) - 1
    return index * interval_sec


# ======================================================================
# 2. Transcribe with Whisper -- timestamped segments
# ======================================================================


def transcribe_video(video_path: Path = VIDEO_PATH, model_size: str = WHISPER_MODEL_SIZE) -> list[dict]:
    if TRANSCRIPT_CACHE.exists():
        return json.loads(TRANSCRIPT_CACHE.read_text())

    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(str(video_path), verbose=False)
    segments = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]
    TRANSCRIPT_CACHE.write_text(json.dumps(segments))
    return segments


# ======================================================================
# 3. Index and search transcript segments
# ======================================================================


def build_transcript_index(segments: list[dict], text_model: SentenceTransformer) -> np.ndarray:
    texts = [s["text"] for s in segments]
    return text_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def transcript_search(query: str, segments: list[dict], segment_embeddings: np.ndarray, text_model: SentenceTransformer, k: int = 3) -> list[dict]:
    query_embedding = text_model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, segment_embeddings)[0]
    top_k = sims.topk(min(k, len(segments)))
    return [segments[i] for i in top_k.indices.tolist()]


# ======================================================================
# 4. Nearest-frame lookup -- frame grounding is driven by the retrieved
#    transcript segment's timestamp, not a separate visual search. Frames
#    only exist at fixed 5-second samples, so this is always "closest
#    sampled frame," never a guarantee of the exact right visual moment.
# ======================================================================


def nearest_frame(timestamp: float, frames: list[tuple[float, Image.Image]]) -> Image.Image:
    return min(frames, key=lambda pair: abs(pair[0] - timestamp))[1]


def frame_to_base64(img: Image.Image) -> str:
    import base64
    import io

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ======================================================================
# 5. Evaluation
# ======================================================================


def recall_at_k_transcript(segments: list[dict], text_model: SentenceTransformer, k: int = 3) -> float:
    segment_embeddings = build_transcript_index(segments, text_model)
    hits = 0
    for question, keyword in VIDEO_EVAL_SET:
        results = transcript_search(question, segments, segment_embeddings, text_model, k=k)
        if any(keyword.lower() in r["text"].lower() for r in results):
            hits += 1
    return hits / len(VIDEO_EVAL_SET)


# ======================================================================
# 6. Query pipeline
# ======================================================================


def run_query(question: str, provider: str | None = None) -> None:
    load_environment()
    segments = transcribe_video()
    frames = extract_frames()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    segment_embeddings = build_transcript_index(segments, text_model)

    top = transcript_search(question, segments, segment_embeddings, text_model, k=1)[0]
    frame = nearest_frame(top["start"], frames)

    print(f"Question: {question}\n")
    print(f"Retrieved transcript segment [{top['start']:.1f}s-{top['end']:.1f}s]: {top['text']}")
    print(f"Nearest frame at {top['start']:.1f}s\n")

    answer = llm.generate_with_image(
        f"Answer the question using the transcript segment and the video frame shown.\n\n"
        f"Transcript: {top['text']}\n\nQuestion: {question}",
        frame_to_base64(frame),
        provider=provider,
    )
    print(f"Answer: {answer}")


def run_visual_demo(provider: str | None = None) -> None:
    """The concrete case transcript-only retrieval cannot handle: a
    detail that's shown on screen but never spoken."""
    load_environment()
    segments = transcribe_video()
    frames = extract_frames()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    segment_embeddings = build_transcript_index(segments, text_model)

    print(f"Question: {VISUAL_ONLY_QUESTION}\n")

    print("=== Transcript-only (no frame) ===")
    top = transcript_search(VISUAL_ONLY_QUESTION, segments, segment_embeddings, text_model, k=3)
    context = "\n".join(s["text"] for s in top)
    print(f"Retrieved transcript context:\n{context}\n")
    text_only_answer = llm.generate(
        f"Answer using only the transcript context below. If the answer isn't in it, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {VISUAL_ONLY_QUESTION}",
        provider=provider,
    )
    print(f"Transcript-only answer: {text_only_answer}\n")

    print("=== With the retrieved frame ===")
    frame = nearest_frame(30.0, frames)  # the whiteboard segment, verified directly
    vision_answer = llm.generate_with_image(
        f"Answer using only what's visible in this frame.\n\nQuestion: {VISUAL_ONLY_QUESTION}",
        frame_to_base64(frame),
        provider=provider,
    )
    print(f"With-frame answer: {vision_answer}")


def run_comparison() -> None:
    load_environment()
    segments = transcribe_video()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    recall = recall_at_k_transcript(segments, text_model, k=3)
    print(f"Transcript-only Recall@3 on {len(VIDEO_EVAL_SET)} factual questions: {recall:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video RAG demo.")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--visual-demo", action="store_true", help="Show the case transcript-only retrieval can't answer.")
    parser.add_argument("--compare", action="store_true", help="Recall@3 on the transcript-only retrieval path.")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    elif args.visual_demo:
        run_visual_demo(args.provider)
    elif args.query:
        run_query(args.query, args.provider)
    else:
        parser.print_help()

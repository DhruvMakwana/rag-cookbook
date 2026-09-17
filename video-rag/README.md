# Video RAG

Retrieval over a real video: a timestamped transcript (speech-to-text) plus sampled frames, so a query pulls back both the spoken content and what was actually on screen at that moment. Uses a real, public-domain NASA video with genuine narration synced to a whiteboard diagram — not stock footage with a music track and no speech. Concept write-up: [Video RAG](https://dhruvmakwana.github.io/rag-deep-dive/video-rag/).

Needs an Anthropic (or OpenAI) API key for generation, and the `ffmpeg` system binary. Transcription and embedding are fully local.

## The sample video

NASA's ["NASA Explains Cold Zone Above Tropics"](https://images.nasa.gov) (public domain, CC0) — 5:12, ~12MB. NASA scientist Paul Newman explains atmospheric layers (troposphere, tropopause, stratosphere) and ozone/UV absorption while drawing a live graph on a whiteboard. Verified directly before choosing it: the video mixes animated title cards, animated illustrations, hand-lettered word overlays, and genuine live-action whiteboard footage — and the whiteboard's axis labels ("ALTITUDE", "TEMPERATURE") are never spoken in the narration, which is exactly the kind of visual-only detail this recipe needs to demonstrate a real limit of transcript-only retrieval.

## Pipeline

1. **Frame extraction** — `ffmpeg`, one frame every 5 seconds. Fixed-interval sampling, not scene detection: simple, and adequate for a 5-minute video.
2. **Transcription** — `openai-whisper` (`small` model), timestamped segments. Whisper shells out to the system `ffmpeg` binary internally, so it's a real prerequisite even though the transcription code never calls it directly.
3. **Retrieval** — transcript segments are embedded with a dedicated text embedder and searched by cosine similarity, same pattern as the naive text RAG used elsewhere in this repo, just with a timestamp attached to each unit instead of a page or chunk id.
4. **Frame grounding** — once a transcript segment is retrieved, the sampled frame closest to that segment's start time is pulled and passed to a vision-capable LLM alongside the transcript text.

## Measured: Recall@3 on transcript-only retrieval — 3/5

```text
PASS  What temperature range is found at ground level in the tropics?
PASS  What is the point called where the troposphere meets the stratosphere?
PASS  What percentage of UV-B radiation is absorbed by the ozone layer?
FAIL  What percentage of UV-C radiation is absorbed by the ozone layer?
FAIL  Why does the stratosphere begin to warm up?
```

Both misses have a distinct, checked cause:

- **UV-C question**: the exact right segment (*"There's another class called UVC... 100% of the UVC"*) exists in the transcript, but wasn't in the top-3 — embedding similarity favored two adjacent, near-duplicate segments about UV-B's 90% figure and generic "absorbed by the ozone layer" text instead. Two segments discussing very similar content (percentages, absorption, ozone) crowded out the one with the actually-different number.
- **Stratosphere-warming question**: retrieval correctly found the segment containing *"the reason the stratosphere begins to warm up is because"* — but Whisper's own segmentation cut the sentence there, and the actual reason (*"...radiation from the Sun is being absorbed in the ozone layer"*) landed in the very next segment, which wasn't retrieved. This is the temporal-transcript version of a chunk-boundary problem: the cause and its explanation were split across two retrieval units.

## The case transcript-only retrieval can't solve at all

Asked *"What does the Y-axis of the whiteboard graph say?"* (the answer, "ALTITUDE," is never spoken anywhere in the video):

```text
Transcript-only answer: The transcript doesn't specify what the Y-axis of
the whiteboard graph says.

With-frame answer: Based on the image, the Y-axis of the whiteboard graph
appears to say "ALTITUDE" (written vertically along the left side of the
graph).
```

The transcript-only path correctly declines rather than guessing — a real, honest negative result, not a hallucination. Passing the actual frame image, retrieved by timestamp proximity to the closest relevant transcript segment, is what makes this answerable at all.

## What this recipe does NOT do

No separate visual (CLIP-based) frame search — frame grounding here is driven entirely by proximity to the retrieved transcript segment's timestamp, not by embedding the query against frame images directly. A fuller pipeline could add that (embed frames with a CLIP-family model — see [Multimodal RAG](../multimodal-rag/) and [Vision RAG](../vision-rag/) in this repo for exactly that mechanism, already covered there) to catch cases where the relevant frame doesn't line up with any well-matching transcript segment at all. Not implemented here to keep this recipe focused on the timestamp-linking mechanism specific to video.

## Install

```bash
brew install ffmpeg   # or your platform's equivalent -- a real system dependency
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

## Run

```bash
python video_rag.py --compare                                              # Recall@3 above
python video_rag.py --visual-demo                                          # the whiteboard Y-axis case above
python video_rag.py --query "What percentage of UV-B radiation is absorbed by the ozone layer?"
```

First run transcribes the video (~1 minute on an 8GB M3 Mac, CPU-only) and extracts frames, caching both locally (`.transcript_cache.json`, `.frames_cache/`) so later runs skip straight to retrieval.

## Files

| File | Role |
|---|---|
| `video_rag.py` | Frame extraction, transcription, retrieval, frame grounding, CLI — the file you actually run |
| `video_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation, including image-grounded generation — Anthropic / OpenAI |
| `download_data.py` | Fetches the sample NASA video |

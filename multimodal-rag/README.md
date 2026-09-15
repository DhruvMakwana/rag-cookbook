# Multimodal RAG

Retrieval across text **and** the sample paper's own real embedded figures — the architecture diagram, the scaled dot-product attention diagram, the multi-head attention diagram, pulled directly out of the PDF, not stock images. Two indexing approaches, measured honestly against each other: a **unified embedding space** (everything, one space) vs. **separate indexes + RRF fusion** (a dedicated text embedder for text, a CLIP-family model only for images). Concept write-up: [Multimodal RAG](https://dhruvmakwana.github.io/rag-deep-dive/multimodal-rag/).

Needs an LLM key for the final generation step only — both retrieval approaches are fully local.

## Which embedding model — measured, not assumed

Base CLIP is genuinely old (2021). Before building around it, 2 more recent candidates were tested head-to-head, in a real, isolated venv, not just cited from a leaderboard:

| Model | Text Recall@3 | Image queries correct (of 4) | Notes |
|---|---|---|---|
| `clip-ViT-B-32` (base CLIP) | 0.375 | 2/4 | The obvious default — turned out not to be the best one |
| `google/siglip-base-patch16-224` | 0.25 | 1/4 | Often cited as CLIP's successor — scored *worse* on both axes here |
| **`jinaai/jina-clip-v2`** | **0.75** | 2/4 | **Winner** — same image accuracy as base CLIP, double the text quality |

`jina-clip-v2` is what this recipe actually uses. Getting it running took a real fix, not a shrug: its custom model code imports `clip_loss` from `transformers.models.clip.modeling_clip`, which newer `transformers` versions removed — pin `transformers==4.46.3` (already in `requirements.txt`) and it loads and runs correctly. It also needs `trust_remote_code=True` (it ships custom model code from Jina AI's own HuggingFace repo) and its own `AutoModel.encode_text` / `encode_image` API — its `sentence-transformers` integration didn't correctly route image inputs in testing.

## If you have more memory to spare

`jina-clip-v2` was picked for this recipe because it's the best-measured candidate that still fits an 8GB machine. With more RAM/VRAM available, worth trying instead (check the [MTEB multimodal leaderboard](https://huggingface.co/spaces/mteb/leaderboard) for current standings before committing):

| Model | Access | Notes |
|---|---|---|
| `Qwen3-VL-Embedding-8B` | Local, ~16GB+ VRAM recommended | Top of the MTEB multimodal leaderboard at the time of writing; too large to load alongside the rest of this stack on 8GB |
| Voyage Multimodal 3.5 | API only | No local memory cost regardless of machine specs — swap `MultimodalEncoder` for API calls |
| Gemini Embedding 2 (multimodal) | API only | Same trade-off as Voyage — no local footprint, but adds network latency and per-call cost |

Swapping in a local model just means pointing `CLIP_MODEL_NAME` at it and updating `MultimodalEncoder` if its API surface differs from `AutoModel.encode_text`/`encode_image` (as `jina-clip-v2` did vs. base CLIP's `SentenceTransformer.encode`). An API-based model needs `MultimodalEncoder.encode` rewritten to call the provider's endpoint instead of a local `transformers` model.

## The real trade-off, measured

| | Recall@3 on the same 8 factual questions used across this repo |
|---|---|
| Dedicated text embedder (`all-MiniLM-L6-v2`) | **1.000** |
| `jina-clip-v2`'s text encoder, same corpus | **0.750** |

Putting text into a shared multimodal space so it can be compared against images still costs **25 points of Recall@3** on plain factual questions, even with the best-measured model — a real, quantified, reproducible cost, smaller than base CLIP's 62-point cost, but not zero. The unified approach genuinely enables one thing separate indexes can't (a single similarity ranking across both modalities), but it pays for that with meaningfully worse plain-text retrieval.

**Separate indexes + RRF fusion is the better default here**, verified two ways:
- An image-appropriate question ("How are multiple attention heads concatenated together?") correctly retrieves both the relevant text explanation and the matching Multi-Head Attention diagram
- A plain factual question ("What optimizer did they use?") correctly returns only text — no image gets pulled in just because it happened to be the least-irrelevant of 3

That second case only works because of a real bug caught during testing: RRF has no concept of absolute relevance, only rank — with just 3 images in the pool, whichever one scored marginally highest still became "rank 1" and got a real fusion boost ahead of relevant text. Fixed with a minimum similarity threshold (`CLIP_RELEVANCE_THRESHOLD`) gating whether a modality contributes to fusion at all. A second, deeper property was found chasing an image-only demo and left honestly unresolved: RRF produces an *exact* tie at rank 0 between modalities on every query, a structural fact about rank-based fusion that no tiebreaker actually fixes — see the code comments on `reciprocal_rank_fusion` for the full story.

**Not every image query is correctly answered even by the best model measured** — asked about scaled dot-product attention specifically, `jina-clip-v2` retrieves the Multi-Head Attention diagram instead (both are visually and conceptually similar attention diagrams). Reported as-is, not hidden — 2 of 4 test queries were wrong, and that's part of the real, measured picture.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

## Run

```bash
python multimodal_rag.py --compare-text-quality
python multimodal_rag.py --mode fused --query "How are multiple attention heads concatenated together?"
python multimodal_rag.py --mode unified --query "How are multiple attention heads concatenated together?"
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`):

```bash
python multimodal_rag.py --mode fused --query "..." --provider openai
```

## Files

| File | Role |
|---|---|
| `multimodal_rag.py` | Image extraction + both indexing approaches + CLI — the file you actually run |
| `multimodal_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

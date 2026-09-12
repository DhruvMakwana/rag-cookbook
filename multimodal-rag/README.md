# Multimodal RAG

Retrieval across text **and** the sample paper's own real embedded figures — the architecture diagram, the scaled dot-product attention diagram, the multi-head attention diagram, pulled directly out of the PDF, not stock images. Two indexing approaches, measured honestly against each other: a **unified CLIP embedding space** (everything, one space) vs. **separate indexes + RRF fusion** (a dedicated text embedder for text, CLIP only for images). Concept write-up: [Multimodal RAG](https://dhruvmakwana.github.io/rag-deep-dive/multimodal-rag/).

Needs an LLM key for the final generation step only — both retrieval approaches are fully local.

## The real trade-off, measured

| | Recall@3 on the same 8 factual questions used across this repo |
|---|---|
| Dedicated text embedder (`all-MiniLM-L6-v2`) | **1.000** |
| CLIP's text encoder, same corpus | **0.375** |

Putting text into CLIP's shared space so it can be compared against images costs **62 points of Recall@3** on plain factual questions — a real, quantified, reproducible cost, not a hypothetical caveat. The unified approach genuinely enables one thing separate indexes can't (a single similarity ranking across both modalities), but it pays for that with meaningfully worse plain-text retrieval.

**Separate indexes + RRF fusion is the better default here**, verified two ways:
- An image-appropriate question ("How is scaled dot-product attention computed?") correctly retrieves both the relevant text explanation and the matching diagram
- A plain factual question ("What optimizer did they use?") correctly returns only text — no image gets pulled in just because it happened to be the least-irrelevant of 3

That second case only works because of a real bug caught during testing: RRF has no concept of absolute relevance, only rank — with just 3 images in the pool, whichever one scored marginally highest (measured: all three scored ~0.22-0.23, genuine noise, for an unrelated question) still became "rank 1" and got a real fusion boost ahead of relevant text. Fixed with a minimum similarity threshold (`CLIP_RELEVANCE_THRESHOLD`) gating whether a modality contributes to fusion at all.

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
python multimodal_rag.py --mode fused --query "How is scaled dot-product attention computed?"
python multimodal_rag.py --mode unified --query "How is scaled dot-product attention computed?"
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

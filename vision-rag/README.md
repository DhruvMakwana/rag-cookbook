# Vision RAG

Retrieval with no text extraction anywhere in the pipeline. Each PDF page is rendered as a full image, embedded with a ColPali-family late-interaction model, and — when a page is selected — the raw page image is what the LLM reads at generation time, not OCR'd text or a caption. Compared directly against the naive chunk-and-embed text RAG used elsewhere in this repo, on the same sample paper. Concept write-up: [Vision RAG](https://dhruvmakwana.github.io/rag-deep-dive/vision-rag/).

Needs a vision-capable LLM key (Anthropic or OpenAI — see below) for generation; retrieval itself is fully local.

## How this differs from Multimodal RAG

The [Multimodal RAG](../multimodal-rag/) recipe in this repo extracts individual figures out of a PDF and retrieves text chunks + those figures, either in one CLIP space or as separate indexes fused with RRF — text is still extracted and chunked either way. This recipe never extracts text at all: a whole page is the retrieval unit, embedded directly as pixels, and the page image itself is what the LLM sees when answering. That matters most on documents where meaning lives in layout — tables, multi-column figures, charts — which text extraction can silently scramble.

## The model — and a real checkpoint bug it ships with

[`vidore/ColSmolVLM-256M-base`](https://huggingface.co/vidore/ColSmolVLM-256M-base) — the lightest model in the ColPali/ColVision family with a usable [ViDoRe](https://huggingface.co/blog/manu/colpali) benchmark score (~80 NDCG@5, vs. ~89 for the 2-3B leaders like ColQwen2.5). It's the one ColVision checkpoint plausibly runnable on an 8GB machine without a dedicated GPU — ColQwen2/2.5 (2-3B params) realistically need one.

This specific checkpoint's `adapter_config.json` sets `base_model_name_or_path` to `./models/ColSmolVLM-256M-Base`, a local training path that doesn't exist outside the run that produced it. `transformers` auto-detects the adapter config on load and tries to resolve that path as a Hugging Face repo id, which fails with an `HFValidationError`. The full merged weights are already in the repo's `model.safetensors`, so `load_vision_model()` in `vision_rag.py` downloads the snapshot, patches a local copy of `adapter_config.json` to point at the real base model (`HuggingFaceTB/SmolVLM-256M-Base`, from the model card), and loads from that local copy instead.

## How retrieval works: late interaction (MaxSim), not a single vector

Each page is embedded as a grid of patch vectors, not pooled into one summary vector. Concretely, for this model: a page image is split into a 4×4 grid of 512×512 tiles plus one whole-page thumbnail tile (17 tiles total); each tile is patchified into a 32×32 grid of non-overlapping 16×16px patches (1,024 raw patches/tile), then a 4×4 pixel-shuffle merge reduces that to 8×8 = 64 tokens/tile. 17 × 64 = **1,088 image tokens per page** (verified directly — not a fixed constant of the model, it follows from the page's pixel dimensions). A query is embedded the same way, one vector per token. Scoring sums, for every query token, the single best-matching patch anywhere on the page:

```
score(query, page) = Σ (for each query token) max (over all page patches) dot_product
```

This is why a page can win on the strength of ONE region — one table cell, one line of a chart legend — even if the rest of the page is unrelated to the query. It's also why the index is heavier than a single-vector embedding: 1,088 vectors per page here, vs. one per chunk for the naive baseline.

## Measured: Recall@3, naive text RAG vs. Vision RAG

On the same 8 factual questions used across this repo, retrieving over the same "Attention Is All You Need" PDF:

| | Recall@3 |
|---|---|
| Naive text RAG (chunk-level) | **1.000** |
| Vision RAG (page-level) | **0.875** |

Vision RAG misses one question here: *"How many attention heads did they use?"* retrieves the paper's attention-visualization figure pages (which literally show per-head attention diagrams) ahead of the page with the actual answer (`h = 8`, in Table 3) — the query's wording matches those figures visually/semantically more strongly than it matches a small table cell. Rephrasing with the table named directly (*"...per Table 3?"*) retrieves the right page. Page-level retrieval is also a coarser unit than a precise 500-character chunk, so on a mostly plain-text paper like this one, naive chunking's precision has room to win outright — consistent with published benchmarks showing Vision RAG's real advantage is concentrated on layout/table-heavy documents, not universal.

## Where Vision RAG actually wins: table-layout integrity

Asked *"What BLEU score did the base model get on the newstest2013 development set, per Table 3?"* (ground truth: 25.8, a Table 3 cell):

- **Naive text RAG** retrieves a passage that mentions Table 3 and BLEU (*"...establishing a new state-of-the-art BLEU score of 28.4. The configuration of this model is listed in the bottom line of Table 3..."*) without containing Table 3's actual data — 28.4 is the *big* model's headline test-set score from a different table entirely. The LLM, to its credit, doesn't guess: it correctly says the base-model dev-set score isn't in the given context.
- **Vision RAG** retrieves the actual Table 3 page image and reads **25.8** directly off it — correct.

Run it yourself: `python vision_rag.py --table-demo`.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Only Anthropic and OpenAI are supported for `--mode vision` — generation needs a vision-capable model that can read the retrieved page image directly. Ollama works for `--mode naive` only.

## Run

```bash
python vision_rag.py --compare              # Recall@3: naive text RAG vs. Vision RAG
python vision_rag.py --table-demo            # the table-layout failure mode, both approaches, real answers
python vision_rag.py --mode vision --query "How many attention heads did the base model use, per Table 3?"
python vision_rag.py --mode naive  --query "What optimizer did they use for training?"
```

Override the LLM provider per-call:

```bash
python vision_rag.py --mode vision --query "..." --provider openai
```

## Files

| File | Role |
|---|---|
| `vision_rag.py` | Page rendering, vision + naive retrieval, both indexing approaches, CLI — the file you actually run |
| `vision_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama, including image-grounded generation |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

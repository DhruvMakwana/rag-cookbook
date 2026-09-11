# Graph RAG

Build a knowledge graph from the corpus — entities and relationships extracted per chunk, deduplicated, clustered into communities, each community summarized — then answer queries two ways: **local search** (entity-link the query, traverse the graph, generate from the traversed facts + source text) and **global search** (map-reduce over community summaries, for broad synthesis questions a single top-k retrieval can't answer). Concept write-up: [Graph RAG](https://dhruvmakwana.github.io/rag-deep-dive/graph-rag/).

Uses `networkx` for the graph itself — in-process, no graph database server, same "no extra infra" philosophy as this repo's FAISS-flat-index naive RAG. Needs an LLM key for extraction, summarization, and generation.

## What building the graph actually costs

On the sample paper (92 chunks): **92 extraction calls** (one per chunk) → 605 entities, 819 relationships → Louvain community detection (a real graph algorithm, not an LLM call) → **70 communities** → **70 summarization calls**. That's genuinely expensive relative to naive RAG's cost (embed once, no LLM calls at all) — this is graph RAG's most-cited real weakness, measured directly here rather than just asserted. The graph is cached to disk after building so this only runs once per corpus.

## Verified with real, honest comparisons

**Local search — a genuine win, verified against naive retrieval directly:**

| | Naive @ k=1 | Naive @ k=3 | Naive @ k=5 | Local search |
|---|---|---|---|---|
| *"What connects label smoothing to the BLEU score improvements?"* | ❌ | ❌ | ✅ | ✅ (found via entity-linking + 1-hop traversal, no k needed) |

**Global search — a genuine win on a broad synthesis question:** asked *"What are the main components of the Transformer architecture and how do they work together?"*, naive top-5 retrieval clustered around one narrative thread (parallelization, encoder/decoder stacks) and never surfaced positional encoding, residual connections, or the decoder's causal masking at all. Global search, drawing from 10 of 70 relevant communities, covered all of it. Same underlying lesson [RAPTOR](https://dhruvmakwana.github.io/rag-deep-dive/retrieval-methods/#raptor) already demonstrated elsewhere on this site — flat top-k can't synthesize across a whole corpus — but clustering on **graph community structure** instead of **embedding similarity**, a genuinely different basis for the same idea.

**A real, honest limitation found during testing:** entity deduplication uses embedding similarity with an 0.88 cosine-similarity threshold — verified directly that "Encoder" and "the Encoder stack" (genuinely the same entity) score only **0.833**, just under the threshold, and don't get merged. Not hidden — this is a real imperfection of similarity-based deduplication, not a hypothetical one.

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
python graph_rag.py --build                    # build + cache the graph (slow, one-time — ~92 LLM calls)
python graph_rag.py --mode local --query "What connects label smoothing to the BLEU score improvements?"
python graph_rag.py --mode global --query "What are the main components of the Transformer architecture?"
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`) instead of changing `.env`:

```bash
python graph_rag.py --mode local --query "..." --provider openai
```

## A note on cost

Global search costs **one LLM call per community, every query** (70 calls on this sample corpus, plus 1 reduce call) — genuinely expensive. [LazyGraphRAG](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/) (Microsoft Research) is a real, more recent alternative that skips upfront community summarization and does relevance filtering lazily at query time instead — not implemented here, but worth knowing as the answer to "this is too expensive."

## Files

| File | Role |
|---|---|
| `graph_rag.py` | Extraction + dedup + graph build + communities + local/global search + CLI — the file you actually run |
| `graph_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

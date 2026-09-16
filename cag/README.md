# CAG (Cache-Augmented Generation)

Skip retrieval entirely: load the whole knowledge base into the LLM's context and reuse a cached prefill across repeated queries, instead of retrieving relevant chunks per query. Measured directly against the same 8 factual questions used across this repo, on the same sample paper, loaded whole. Concept write-up: [CAG](https://dhruvmakwana.github.io/rag-deep-dive/cag/).

Needs an Anthropic API key. Retrieval is not used at all — the entire point of this technique.

## Scope: what this recipe actually measures, and why

The original CAG paper ("Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks", [arXiv:2412.15605](https://arxiv.org/abs/2412.15605)) precomputes and reuses a model's raw KV-cache directly — its own [reference implementation](https://github.com/hhhuang/CAG) manipulates `past_key_values` on a self-hosted, open-weight model (`Llama-3.1-8B-Instruct` by default). That's only possible because you control the model's internals. Hosted APIs (Claude, OpenAI) never expose `past_key_values` for you to save and reload, and an 8B+ model isn't practical to self-host on an 8GB machine with no dedicated GPU (16GB+ RAM even quantized).

This recipe measures the closest real equivalent available via a hosted API: Anthropic's `cache_control` prompt caching, applied to the entire sample document as the system prompt. It's a genuine substitution for the paper's actual mechanism, not the literal thing — reuse of a server-side cache keyed on an exact prefix match, rather than a client-controlled KV-cache. OpenAI's caching is automatic with no equivalent explicit control, so it isn't part of the measured comparison here.

## What's actually being compared

- **Fresh** — the entire document (39,524 characters, ~15,125 tokens) sent as the system prompt on every call, no caching. This is plain long-context stuffing, not CAG — it's the baseline CAG's caching layer is compared against.
- **Cached** — the same system prompt, annotated with a `cache_control` breakpoint (5-minute TTL). The first call against the document pays a real premium to write the cache; every later call within the TTL reads from it instead of reprocessing the whole document.

## Measured: cost and latency, real API calls, 8 questions

Two independent full runs against the same document and questions:

| | Fresh (total) | Cached (total) | Change |
|---|---|---|---|
| Cost, run 1 | $0.2536 | $0.0399 | **-84.3%** |
| Cost, run 2 | $0.2544 | $0.0370 | **-85.5%** |
| Latency, run 1 | 27.96s | 31.63s | **+13.1%** (slower) |
| Latency, run 2 | 28.30s | 26.93s | **-4.8%** (faster) |

**Cost reduction is dramatic and consistent** — cache reads are billed at 0.1x the base input price, vs. 1.0x for fresh and 1.25x for the initial cache write, so reprocessing a ~15K-token document on every call is genuinely expensive compared to reading it from cache.

**Latency reduction was NOT reliably observed at this document size.** Across two runs the delta swung from +13% (cached slower) to -5% (cached faster) — noise-level, not a measured win. Anthropic's own published claim is "up to 85% latency reduction," which is real at the scale where prefill compute dominates total request time — a document with hundreds of thousands of cached tokens, say. At ~15K tokens, network round-trip and output generation dominate wall-clock time far more than the prefill compute caching actually saves, so the latency benefit doesn't show up here. Cost savings and latency savings are not the same claim, and this recipe's own numbers show them diverging.

**Answer accuracy is identical whichever way it's measured (6-7 of 8 both ways)** — expected, since caching doesn't change what content the model reads, only how the unchanged prefix is billed and computed. The one recurring "FAIL" on this repo's standard eval set is a keyword-matching artifact: the model answers `d_model = 512` (correct) where the eval keyword is `dmodel = 512` (no underscore) — not a real quality difference between fresh and cached.

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
python cag.py --compare                                          # the full fresh-vs-cached comparison above
python cag.py --mode cached --query "How many attention heads did they use?"
python cag.py --mode fresh  --query "What optimizer did they use?"
```

## Files

| File | Role |
|---|---|
| `cag.py` | Document loading, fresh vs. cached generation, CLI — the file you actually run |
| `cag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Anthropic-only generation with raw usage/timing exposed — real numbers, not estimates |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

# Adaptive RAG

An LLM router classifies each query before retrieval, and the pipeline applies only the pre-retrieval technique that query's category actually needs — instead of running every technique on every query (expensive, and often counterproductive on queries that don't have the failure mode a technique targets, as measured in `query-transformation/`). Ties together `query-transformation/` and `post-retrieval/` into one end-to-end pipeline: **router → matching pre-retrieval fix (or none) → wide retrieval → cross-encoder rerank**. Concept write-up: [Build Adaptive RAG From Scratch](https://dhruvmakwana.github.io/rag-deep-dive/tutorials/advanced-rag-from-scratch/).

Needs an LLM key — the router alone is one LLM call per query, and most pre-retrieval techniques need one too.

## What the 5-query comparison actually shows

Router categories map to techniques already measured individually: `CLEAR` → no-op, `VAGUE` → rewrite, `COMPOUND` → decompose, `BROAD` → multi-query/RRF, `NEEDS_CONTEXT` → step-back. Each demo query reuses a scenario already keyword-verified in `query-transformation/`.

| Question | Category | Naive | In candidate pool | In final output |
|---|---|---|---|---|
| What optimizer did they use for training? | CLEAR | ✅ | ✅ | ✅ |
| why dont they just use RNNs like everyone else did before | VAGUE | ❌ | ✅ | ❌ |
| What optimizer did they use, and how many attention heads did they use? | COMPOUND | ❌ | ✅ | ✅ |
| What lets this model connect words that are far apart... | BROAD | ❌ | ✅ | ❌ |
| What value did they use for Pdrop during training? | NEEDS_CONTEXT | ✅ | ✅ | ✅ |

**The "candidate pool" column is the whole point of this recipe.** Two of five queries (VAGUE, BROAD) get correctly routed, correctly fixed by pre-retrieval — the target fact genuinely enters the candidate pool, which naive retrieval never achieves — and then **lost again by cross-encoder reranking**, because the reranker is just another model reading text, and it disagrees with what's actually the most precise answer. Stacking techniques isn't automatically monotonic: a later stage can undo an earlier stage's fix. This surfaced two real bugs during development, both described in code comments in `adaptive_rag.py`:

1. Reranking a VAGUE query's candidates against the *original* colloquial wording (instead of the rewritten query) reintroduces the exact vocabulary-mismatch problem rewriting just fixed.
2. Reranking a decomposed COMPOUND question's merged candidate pool against one single query collapses it back to one topic, undoing decomposition's guarantee that each sub-question gets independent representation.

Both are fixed in the code (COMPOUND now reranks per sub-question with guaranteed slots). VAGUE and BROAD's final-stage loss is **not** fixed — it's a real, reproducible finding (stable across repeated runs), left as-is and reported honestly rather than tuned away.

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
python adaptive_rag.py --query "What optimizer did they use for training?"
python adaptive_rag.py --compare              # the 5-query table above
python adaptive_rag.py --query "..." --compress   # add contextual compression as a final step
```

## A note on reproducibility

The router and most pre-retrieval techniques make live, unseeded LLM calls, so exact wording can vary run to run. The CLEAR/VAGUE/COMPOUND/BROAD/NEEDS_CONTEXT pattern in the comparison table was verified stable across repeated runs, but individual chunk rankings near a decision boundary can shift slightly.

## Files

| File | Role |
|---|---|
| `adaptive_rag.py` | Router + pipeline + CLI — the file you actually run |
| `adaptive_rag_docs.py` | **Documentation only** — self-contained per-section version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

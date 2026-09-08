# Query Transformation (Pre-Retrieval)

Six pre-retrieval query transformation techniques, each measured with a real before/after — naive retrieval on the raw query vs. retrieval after the technique runs, against the same sample document and (where applicable) the same 8-question Recall@k eval set used in `chunking-strategies/` and `embedding-model-selection/`. Concept write-up: [Advanced RAG — Pre-retrieval](https://dhruvmakwana.github.io/rag-deep-dive/advanced-rag-pre-retrieval/).

All 6 need an LLM key — query transformation is inherently LLM-driven.

| Technique | Measured result |
|---|---|
| `rewrite` | **Regressed** Recall@3 (0.88 → 0.62) on already-clear questions — rewriting adds verbosity with nothing to fix when the query wasn't broken |
| `multi_query` | **Regressed** slightly (0.88 → 0.75) — RRF rewards consistency across paraphrases, which can dilute an already-optimal single match |
| `hyde` | **Tied** on aggregate (0.88 = 0.88), but fixed one real naive miss while introducing a different one — and isn't perfectly reproducible run-to-run (the hypothetical answer is regenerated fresh each call) |
| `step_back` | **Tied** — this single, well-organized paper doesn't separate general context from specific facts enough to need it |
| `decompose` | **Clean win** on a genuinely compound question — naive retrieval misses one of two facts even at k=8; decomposition reliably finds both at k=6 |
| `ircot` | **Depends entirely on whether the underlying fact is findable at all** — succeeds cleanly when both facts are individually locatable; fails (honestly, without hallucinating) when a fact is buried in a way no query phrasing can surface |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```
Fill in a key for whichever `LLM_PROVIDER` you set — every technique here needs one.

## Run

```bash
python query_transformation.py --technique rewrite
python query_transformation.py --technique multi_query
python query_transformation.py --technique hyde
python query_transformation.py --technique step_back
python query_transformation.py --technique decompose
python query_transformation.py --technique ircot
```

## A note on reproducibility

Every technique here makes at least one live LLM call with no fixed seed, so **exact wording (and occasionally the exact Recall@k number) will vary slightly run to run** — this is real, not a bug to "fix." Multiple runs during development showed the overall pattern (which techniques help, tie, or hurt on this corpus) staying consistent, even when specific numbers shifted by one hit out of eight.

## Files

| File | Role |
|---|---|
| `query_transformation.py` | All 6 techniques + the CLI — the file you actually run |
| `query_transformation_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

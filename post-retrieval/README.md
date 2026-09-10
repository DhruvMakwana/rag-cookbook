# Post-retrieval

Three post-retrieval techniques, each measured with a real before/after — naive bi-encoder retrieval vs. retrieval after the technique runs, on a scenario matched to that technique's actual "when to use" case. Concept write-up: [Advanced RAG — Post-retrieval](https://dhruvmakwana.github.io/rag-deep-dive/advanced-rag-post-retrieval/).

`cross_encoder` needs no LLM key — it's fully local. `compression` and `listwise` both need one.

| Technique | Scenario | Measured result |
|---|---|---|
| `cross_encoder` | A question whose answer sits in a candidate pool but ranks outside naive's top-3 | **Fixes a real miss** — naive top-3 misses the target chunk (mid-pack rank), cross-encoder reranking over the same 15-candidate pool correctly promotes it to top-3 |
| `compression` | A single-fact question retrieved with a generous top-5 budget | **~94% size reduction** — 5 chunks / ~2000 chars compressed down to ~130 chars of actually-relevant text, target fact survives compression |
| `listwise` | Same scenario as `cross_encoder`, for direct comparison | **Also fixes the same miss** — but at roughly 230x the latency (one full LLM round-trip vs. one local forward pass), since it sends the whole candidate list to the LLM in one call instead of scoring each candidate independently |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Fill in a key for whichever `LLM_PROVIDER` you set. Not needed for `cross_encoder` alone.

## Run

```bash
python post_retrieval.py --technique cross_encoder
python post_retrieval.py --technique compression
python post_retrieval.py --technique listwise
```

## A note on reproducibility

`compression` and `listwise` both make live, unseeded LLM calls, so exact wording (and occasionally the exact True/False outcome) can vary slightly run to run. `cross_encoder` is fully deterministic — same model, same input, same output every time.

## Files

| File | Role |
|---|---|
| `post_retrieval.py` | All 3 techniques + the CLI — the file you actually run |
| `post_retrieval_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

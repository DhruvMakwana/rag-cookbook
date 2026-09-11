# Retrieval Methods

Six retrieval methods, each run for real against the same sample document (or a small metadata-tagged corpus for the self-query demo). Concept write-up: [Retrieval Methods](https://dhruvmakwana.github.io/rag-deep-dive/retrieval-methods/).

| Method | Needs a key? | What it does |
|---|---|---|
| `mmr` | No | Diversifies top-k results — real, tested `langchain_core` MMR utility |
| `hierarchical` | No | Two-stage retrieval: find the best section first, then rank chunks within it |
| `self_query` | Yes | LLM splits a natural-language query into a semantic part + structured metadata filters |
| `hypothetical_questions` | Yes | Generates plausible questions per chunk at index time; retrieval matches question-to-question |
| `raptor` | Yes | Clusters chunks, summarizes each cluster with an LLM — real clustering (`scikit-learn`), real summaries |
| `adaptive` | Yes | Simplified FLARE: LLM self-reports confidence in its own draft answer, only retrieves if confidence is low |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure (only needed for `self_query`, `hypothetical_questions`, `raptor`, `adaptive`)

```bash
cp .env.example .env
```
Fill in a key for whichever `LLM_PROVIDER` you set. `mmr` and `hierarchical` run fully locally with no key.

## Run

```bash
python retrieval_methods.py --method mmr
python retrieval_methods.py --method hierarchical
python retrieval_methods.py --method self_query
python retrieval_methods.py --method hypothetical_questions
python retrieval_methods.py --method raptor
python retrieval_methods.py --method adaptive --threshold 4   # compare with --threshold 3
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`) instead of changing `.env`:

```bash
python retrieval_methods.py --method mmr --provider openai
```

## A real finding worth knowing before you trust `adaptive`

`--method adaptive --threshold 3` reproduces a real failure: asked "what's the exact FLOPS count for training the big Transformer model," the model self-reports confidence 3, the (too-permissive) threshold lets it through, and it confidently states **3.3×10¹⁸ FLOPs** — which is actually the paper's figure for the *base* model, not the *big* one (the real answer is 2.3×10¹⁹, a full order of magnitude higher). `--threshold 4` catches this and correctly falls back to retrieval instead of shipping the wrong number — though even then, plain top-1 dense retrieval alone doesn't find the actual table row either (a numeric table row embeds poorly with a general-purpose dense model), so it honestly reports "not found in context" rather than hallucinating. See the blog page for the full story.

## Files

| File | Role |
|---|---|
| `retrieval_methods.py` | All 6 methods + the CLI — the file you actually run |
| `retrieval_methods_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

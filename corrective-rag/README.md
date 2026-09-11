# Corrective RAG (CRAG)

A lightweight evaluator grades the whole retrieved candidate set in one batched call, and the verdict — `Correct`, `Incorrect`, or `Ambiguous` — decides what happens next: not one confidence gate, but three genuinely different corrective actions. Concept write-up: [Corrective RAG (CRAG)](https://dhruvmakwana.github.io/rag-deep-dive/corrective-rag/).

| Verdict | Action |
|---|---|
| `Correct` | Split retrieved chunks into sentence-level "knowledge strips," re-score each one, keep only the strips that individually hold up |
| `Incorrect` | Discard retrieved chunks entirely, rewrite the query into search-engine keywords, fall back to a real web search |
| `Ambiguous` | Both — refined internal strips AND web search results, combined before generation |

Needs an LLM key (evaluator, query rewriting, refinement scoring, and generation all use one) and a **Tavily API key** for the web-search fallback — free tier, no credit card: [tavily.com](https://tavily.com).

## Verified with all three verdicts, on real data

| Question | Scores | Verdict | Source |
|---|---|---|---|
| "What optimizer did they use for training?" | `[1, 5, 2, 1, 1]` | Correct | internal (refined) |
| "What is the current stock price of Nvidia?" | `[1, 1, 1, 1, 1]` | Incorrect | web (real Tavily call — returned actual current NVDA price data) |
| "What is the significance of this paper compared to more recent transformer variants?" | `[2, 1, 1, 3, 1]` | Ambiguous | internal (refined) + web |

All three ran against a live Tavily search, not a mock — the "Incorrect" case above genuinely fetched current stock data the sample paper obviously doesn't contain, and the model correctly cited it as coming from that external source.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Fill in an LLM key and `TAVILY_API_KEY`. Only the `Correct` verdict path works without Tavily — `Incorrect` and `Ambiguous` both need it.

## Run

```bash
python corrective_rag.py --query "What optimizer did they use for training?"
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`) instead of changing `.env`:

```bash
python corrective_rag.py --query "..." --provider openai
```

## Files

| File | Role |
|---|---|
| `corrective_rag.py` | Evaluator + refinement + web search + generation + CLI — the file you actually run |
| `corrective_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

# RAGAS Deep Dive

Runs a real naive RAG pipeline over the sample paper, then scores its real retrieved contexts and real generated answers with RAGAS's four core metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall — to see whether the four-metric diagnostic table actually holds up against a real system, not a curated example. Concept write-up: [RAGAS Deep Dive](https://dhruvmakwana.github.io/rag-deep-dive/ragas/).

Needs an Anthropic API key — used both for the naive RAG pipeline's own generation and as RAGAS's judge LLM. Retrieval is fully local.

## Scope: ragas ships two parallel evaluation APIs right now

Current `ragas` (0.4.x) has a legacy `evaluate()` + `EvaluationDataset` + named-metric-class API (what every existing tutorial shows) and a newer per-sample `ragas.metrics.collections` + `.ascore()` API the current docs actually lead with. The legacy path still works — every import just fires a `DeprecationWarning` pointing at the new one — but it's the only one that produces the four-metrics-together diagnostic table this recipe is about; the modern collections classes can't be passed to `evaluate()` at all (different base class). This recipe uses the legacy-but-functional path.

## Setup requirements for this ragas release

1. **`langchain-community==0.3.31` needs pinning.** `ragas/llms/base.py` imports `langchain_community.chat_models.vertexai`, a module later `langchain-community` releases removed (moved to a separate `langchain-google-vertexai` package), and `ragas` carries no upper pin on it.
2. **Embeddings go through `LangchainEmbeddingsWrapper`.** `ragas`'s own `HuggingfaceEmbeddings` class fails pydantic validation on direct instantiation in this release — wrap a real `langchain-huggingface` embeddings object with `LangchainEmbeddingsWrapper` instead, the same pattern used for the judge LLM via `LangchainLLMWrapper`.
3. **The judge LLM wrapper needs `bypass_temperature=True`.** `LangchainLLMWrapper` sets a `temperature` value on the underlying model before every call by default. Current Claude models no longer accept that parameter (adaptive thinking replaces sampling controls), so `bypass_temperature=True` — the wrapper's own documented flag for this case — is required.

## Measured: does the diagnostic table hold up on a real system?

Aggregate scores across the standard 8-question eval set used throughout this repo:

```text
faithfulness:                          0.8845
answer_relevancy:                      0.8071
llm_context_precision_with_reference:  0.6250
context_recall:                        0.8750
```

Per-question, one result stands out — and it's the most useful one in the whole run:

```text
Q: What BLEU score did they get on English-to-German translation?
A: "...the Transformer model achieved a BLEU score of 28.4..." (factually correct)

faithfulness: 0.167   context_precision: 0.0   context_recall: 0.0
```

The answer is factually correct — 28.4 is the real EN-DE BLEU score. But none of the top-3 retrieved chunks actually contain it: one chunk is the results table cut off right before the Transformer's own rows appear, and the other two both discuss the *English-to-French* numbers (41.0, 41.8) from a different part of the paper. Retrieval genuinely missed the right table row. The model answered correctly anyway — from its own training data on this well-known paper, not from what was actually retrieved. RAGAS caught exactly this: a factually right answer with near-zero grounding in the retrieved context, which a simple "is the final answer correct" check would have missed entirely.

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
python ragas_deep_dive.py --compare
```

First run takes ~1 minute total (naive RAG generation for 8 questions, then the RAGAS scoring pass itself, which runs in well under a minute).

## Files

| File | Role |
|---|---|
| `ragas_deep_dive.py` | Naive RAG pipeline, RAGAS evaluation, CLI — the file you actually run |
| `ragas_deep_dive_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation for the naive baseline — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the sample "Attention Is All You Need" PDF used across this repo |

# Evaluation Metrics

Retrieval-quality metrics (Precision@k, Recall@k, MRR, NDCG, Hit Rate) computed against real, verifiable ground truth, and generation-quality metrics (BLEU, ROUGE-L, METEOR, embedding-based semantic similarity) computed against real locally-generated answers. Concept write-up: [Evaluation Metrics](https://dhruvmakwana.github.io/rag-deep-dive/evaluation-metrics/).

No paid API needed — generation runs through a local [Ollama](https://ollama.com) model, retrieval embeddings run locally via `sentence-transformers`.

## Ground truth, built from the data itself

The standard 8-question eval set used across this repo includes a `keyword` field — a substring that only appears on the page(s) that actually answer that question. This recipe uses that field to build real relevance sets (which chunks are actually relevant to a question) by checking which chunks contain the keyword, rather than hand-picking page numbers.

## Setup

Needs Ollama running locally with a model pulled:

```bash
ollama serve &
ollama pull qwen3:4b
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

METEOR needs NLTK's `wordnet`/`punkt` data downloaded once:

```bash
python3 -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('omw-1.4')"
```

If that download fails with an SSL certificate error (common on a fresh macOS Python.org install), set `SSL_CERT_FILE` to the `certifi` bundle first: `export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())")`.

## Run

```bash
python evaluation_metrics.py --part retrieval  # Precision@k, Recall@k, MRR, NDCG, Hit Rate
python evaluation_metrics.py --part generation # BLEU, ROUGE-L, METEOR, semantic similarity (generates 8 real answers via Ollama -- a few minutes)
python evaluation_metrics.py --part all        # both
```

## Scope note

This recipe covers retrieval metrics and the classic/embedding-based generation metrics. The LLM-judged groundedness/faithfulness diagnostic table (Faithfulness, Answer Relevancy, Context Precision, Context Recall) is a separate, more involved measurement already covered with real, tested code in [RAGAS Deep Dive](../ragas-deep-dive/) — this page doesn't duplicate it.

## Files

| File | Role |
|---|---|
| `evaluation_metrics.py` | Corpus + retrieval metrics + generation metrics -- the file you actually run |
| `evaluation_metrics_docs.py` | **Documentation only** -- self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `download_data.py` | Fetches the sample "Attention Is All You Need" PDF used across this repo |

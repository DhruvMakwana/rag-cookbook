# Production Considerations

Real reliability patterns (retry with backoff, a circuit breaker) against deliberately flaky/failing calls, and real per-stage latency, token-usage, and semantic-cache instrumentation across a naive RAG pipeline. Concept write-up: [Production Considerations](https://dhruvmakwana.github.io/rag-deep-dive/production-considerations/).

No paid API needed — generation runs through a local Ollama model.

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

## Run

```bash
python production_considerations.py --part retry            # retry-with-backoff against a call that fails twice then succeeds
python production_considerations.py --part circuit-breaker   # circuit breaker against a call that fails persistently
python production_considerations.py --part latency           # per-stage latency breakdown (embed/retrieve/generate)
python production_considerations.py --part cost              # per-stage token usage across several questions
python production_considerations.py --part cache             # semantic cache hit/miss against paraphrased questions
python production_considerations.py --part all                # everything
```

## Files

| File | Role |
|---|---|
| `production_considerations.py` | Corpus + all five demos -- the file you actually run |
| `production_considerations_docs.py` | **Documentation only** -- self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `download_data.py` | Fetches the sample "Attention Is All You Need" PDF used across this repo |

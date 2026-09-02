# Naive RAG — from scratch

A minimal, dependency-light naive RAG pipeline: PDF → chunk → embed → FAISS index → retrieve → generate. No LangChain/LlamaIndex — every step is plain Python so the mechanics stay visible. Concept write-up: [Naive RAG](https://dhruvmakwana.github.io/rag-deep-dive/naive-rag/) · Full walkthrough: [Build Naive RAG From Scratch](https://dhruvmakwana.github.io/rag-deep-dive/tutorials/naive-rag-from-scratch/).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

`sentence-transformers` pulls in PyTorch — this step downloads a few hundred MB and can take a couple of minutes.

## Configure

```bash
cp .env.example .env
```

Edit `.env` and set `LLM_PROVIDER` to one of `anthropic`, `openai`, or `ollama`, then fill in the matching API key. `.env` is git-ignored — it will never be committed. **Only install/keep the SDK for the provider you're actually using** if you want a lighter environment.

- **Anthropic / OpenAI:** paste your key into `.env`. That's it.
- **Ollama (free, fully local, no key):** install [Ollama](https://ollama.com), run `ollama pull llama3.1` once, then set `LLM_PROVIDER=ollama`.

## Run

```bash
python rag_naive.py "What is multi-head attention?"
```

First run downloads the sample PDF ("Attention Is All You Need", arXiv) into `data/`, chunks and embeds it, and caches a FAISS index there — later runs reuse the cache. Force a rebuild with `--rebuild`, or pull more/fewer chunks with `--k 6`.

```bash
python rag_naive.py "What optimizer did they use?" --provider openai --k 6
```

## Using your own PDF

```bash
export PDF_PATH=/path/to/your.pdf   # or set PDF_PATH in .env
python rag_naive.py "your question" --rebuild
```

`--rebuild` is required the first time you point at a different PDF, since the index/chunks are cached under `data/` from whatever was indexed last.

## Files

| File | Role |
|---|---|
| `download_data.py` | Fetches the default sample PDF (cached, not committed) |
| `llm.py` | Pluggable generation step — Anthropic / OpenAI / Ollama |
| `rag_naive.py` | The pipeline itself: load → chunk → embed → index → retrieve → prompt → generate |
| `.env.example` | Template for your own `.env` (never commit the real one) |

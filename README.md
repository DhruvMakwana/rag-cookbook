# RAG Cookbook

Runnable, from-scratch implementations of the techniques covered in [RAG, Deep Dive](https://dhruvmakwana.github.io/rag-deep-dive/) — one folder per technique, no LangChain/LlamaIndex abstraction layer, so every step of each pipeline stays visible and readable.

## Recipes

| Recipe | What it builds | Status |
|---|---|---|
| [`naive-rag/`](naive-rag/) | PDF → chunk → embed → FAISS → retrieve → generate, pluggable across Anthropic/OpenAI/Ollama | ✅ |

More recipes land alongside new pages on the blog — hybrid search + RRF, a prompted Self-RAG loop, CRAG with web-search fallback, and others are planned next.

## Using a recipe

Each folder is self-contained: its own `requirements.txt`, its own `README.md` with install/configure/run instructions, its own `.env.example`. Start with that folder's README.

```bash
git clone git@github.com:DhruvMakwana/rag-cookbook.git
cd rag-cookbook/naive-rag
# follow that folder's README from here
```

## A note on API keys

Every recipe reads credentials from a local `.env` file that is git-ignored in every recipe folder — never committed, never logged, never printed by any script here. Copy the provided `.env.example`, fill in only the key(s) you need, and you're set.

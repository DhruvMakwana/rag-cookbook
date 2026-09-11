# Agentic RAG

Instead of a hand-rolled router deciding which retrieval strategy to use (`query-transformation/`, `adaptive-rag/`), or fixed pre-programmed branching (`corrective-rag/`, `self-rag/`), an LLM agent picks its own tools at runtime via **Anthropic's native tool-calling API** — a genuinely different mechanism from every other recipe's prompted `<tag>` classification, and the answer to the routing gap flagged on the [Graph RAG](https://dhruvmakwana.github.io/rag-deep-dive/graph-rag/) page (choosing local vs. global search there was a manual CLI flag, not automatic). Concept write-up: [Agentic RAG](https://dhruvmakwana.github.io/rag-deep-dive/agentic-rag/).

Four tools, each a real technique already built and verified elsewhere in this repo — `naive_vector_search`, `graph_local_search`, `graph_global_search`, `web_search` (real Tavily) — and the agent decides which to call, how many times, and when it has enough. A real ReAct-style Thought → Action → Observation loop, made of actual API `tool_use` blocks, not parsed tags.

**This recipe's agent loop is Anthropic-only** — tool-calling request/response shapes differ enough across providers (unlike plain text generation) that there's no honest single abstraction here, so it doesn't use this repo's usual pluggable `llm.py`.

## Verified: the agent chose correctly on all 4 question types, unprompted

| Question | Tool chosen | Correct? |
|---|---|---|
| "What optimizer did they use for training?" (simple fact) | `naive_vector_search` | ✅ |
| "What are the main components of the Transformer architecture and how do they work together?" (broad synthesis) | `graph_global_search` | ✅ |
| "What connects label smoothing to the BLEU score improvements?" (specific relational) | `graph_local_search` | ✅ |
| "What is the current stock price of Nvidia?" (outside the corpus) | `web_search` (real Tavily call) | ✅ |

No routing logic was written for any of this — the tool *descriptions* alone were enough for the agent to pick correctly every time, across a genuinely diverse set of question types.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Fill in `ANTHROPIC_API_KEY` and `TAVILY_API_KEY`.

## Run

```bash
python agentic_rag.py --build   # build + cache the knowledge graph (slow, one-time — ~92 LLM calls)
python agentic_rag.py --query "What optimizer did they use for training?"
```

## Files

| File | Role |
|---|---|
| `agentic_rag.py` | Graph construction + the 4 tools + the agent loop + CLI — the file you actually run |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

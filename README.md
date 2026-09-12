# RAG Cookbook

Runnable, from-scratch implementations of the techniques covered in [RAG, Deep Dive](https://dhruvmakwana.github.io/rag-deep-dive/) — one folder per technique, no LangChain/LlamaIndex abstraction layer, so every step of each pipeline stays visible and readable.

## Recipes

| Recipe | What it builds | Status |
|---|---|---|
| [`naive-rag/`](naive-rag/) | PDF → chunk → embed → FAISS → retrieve → generate, pluggable across Anthropic/OpenAI/Ollama | ✅ |
| [`chunking-strategies/`](chunking-strategies/) | 10 chunking strategies from scratch — fixed-size through late chunking, proposition chunking, and adaptive strategy selection | ✅ |
| [`embedding-model-selection/`](embedding-model-selection/) | 5 demos measuring Recall@k on a real eval set — mismatched models, Matryoshka truncation, quantization, instruction prefixes, BGE-M3's dense+sparse+multi-vector | ✅ |
| [`retrieval-methods/`](retrieval-methods/) | 6 retrieval methods — MMR diversification, self-query metadata filtering, hypothetical questions, hierarchical retrieval, RAPTOR, adaptive retrieval (simplified FLARE) | ✅ |
| [`query-transformation/`](query-transformation/) | 6 pre-retrieval techniques — query rewriting, multi-query/RAG-Fusion, HyDE, step-back prompting, decomposition, IRCoT — each measured before/after, several with genuinely surprising (including negative) results | ✅ |
| [`post-retrieval/`](post-retrieval/) | 3 post-retrieval techniques — cross-encoder reranking, contextual compression, LLM-based listwise reranking (RankGPT-style) — measured before/after, plus a real latency comparison between cross-encoder and listwise reranking | ✅ |
| [`adaptive-rag/`](adaptive-rag/) | An LLM router classifies each query and applies only the matching pre-retrieval technique, then cross-encoder reranks — ties `query-transformation/` and `post-retrieval/` together into one pipeline, with an honest finding that reranking can undo an earlier stage's fix on 2 of 5 demo queries | ✅ |
| [`self-rag/`](self-rag/) | A prompted Self-RAG loop — grade relevance, generate, grade groundedness (via a local NLI model, not an LLM call), grade usefulness, retry or fall back honestly. Includes a genuine finding that Claude Sonnet 5 is remarkably resistant to hallucinating in this setup | ✅ |
| [`corrective-rag/`](corrective-rag/) | A batched retrieval evaluator grades the whole candidate set at once (Correct / Incorrect / Ambiguous), each verdict triggering a genuinely different corrective action — knowledge refinement, a real Tavily web-search fallback, or both. Verified against all 3 verdicts with real, live data | ✅ |
| [`graph-rag/`](graph-rag/) | Extract entities/relationships per chunk into a knowledge graph (networkx), detect communities (Louvain), summarize each — then local search (entity-link + traverse) and global search (map-reduce over community summaries). Both verified as genuine wins over naive retrieval, with real extraction-cost numbers reported honestly | ✅ |
| [`agentic-rag/`](agentic-rag/) | An LLM agent picks its own retrieval tool at runtime via Anthropic's native tool-calling API (not a hand-rolled router) — naive search, graph local search, graph global search, or a real Tavily web search fallback. Verified correct on all 4 question types, unprompted | ✅ |
| [`modular-rag/`](modular-rag/) | The same routing decision built 3 ways — keyword heuristics, embedding-based semantic routing, an LLM call — measured for real accuracy and latency on both a design set and a genuinely held-out set. Heuristic accuracy collapses from 1.00 to 0.40 on unseen phrasing; the LLM router costs ~30-1500x the latency but is the only one both accurate and consistent | ✅ |

More recipes land alongside new pages on the blog — hybrid search + RRF, and others are planned next.

## Using a recipe

Each folder is self-contained: its own `requirements.txt`, its own `README.md` with install/configure/run instructions, its own `.env.example`. Start with that folder's README.

```bash
git clone git@github.com:DhruvMakwana/rag-cookbook.git
cd rag-cookbook/naive-rag
# follow that folder's README from here
```

## A note on API keys

Every recipe reads credentials from a local `.env` file that is git-ignored in every recipe folder — never committed, never logged, never printed by any script here. Copy the provided `.env.example`, fill in only the key(s) you need, and you're set.

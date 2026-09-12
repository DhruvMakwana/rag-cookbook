# Modular RAG

Adaptive RAG, Self-RAG, Corrective RAG, Agentic RAG are all specific instances of one bigger idea: RAG as swappable modules, not a fixed pipeline. Every router built elsewhere in this repo (`adaptive-rag/`, implicitly `graph-rag/`, `agentic-rag/`) makes its routing decision with an LLM call. This recipe asks the one question none of those pages tested: **does routing actually need an LLM call at all?** Concept write-up: [Modular RAG](https://dhruvmakwana.github.io/rag-deep-dive/modular-rag/).

The same 5-category routing decision `adaptive-rag/` makes, built 3 different ways, measured for real accuracy and latency — not just asserted:

1. **`heuristic_router`** — keyword/pattern rules, no model call, ~0 cost
2. **`embedding_router`** — cosine similarity to example utterances per category, one local embedding call, no LLM
3. **`llm_router`** — the exact router from `adaptive-rag/`, reused here as the accuracy baseline

Needs an LLM key only for `llm_router`; the other two are fully local.

## Measured: a real accuracy/latency/generalization trade-off, not a hypothetical one

Two separate test sets: a **design set** (the 9 queries `adaptive-rag/`'s router was originally validated on) and a **held-out set** (5 queries written *without* looking at `heuristic_router`'s actual keyword rules — the fair test of generalization, not just fitting the phrasing a router was tuned against).

| Router | Design-set accuracy | Held-out accuracy | Avg latency |
|---|---|---|---|
| `heuristic` | 1.00 | **0.40** | ~0ms |
| `embedding` | 0.67 | 0.80 | ~50ms |
| `llm` | 1.00 | 1.00 | ~1500ms |

**The honest story here is the heuristic router's collapse, not its headline number.** 100% on the set it was tuned against looks great — and is almost meaningless, since the rules were written while looking at those exact 9 queries. Tested on 5 genuinely new phrasings it had never seen, accuracy drops to 40%: real, measured brittleness, not a hypothetical caveat. The embedding router shows the opposite, more trustworthy pattern — accuracy is *similar* across both sets (0.67 → 0.80), because it was never fit to either one; its confusions are consistent, not held-out-specific.

The LLM router is the only one that's both accurate everywhere **and** consistent — at roughly 30-1500x the latency of the other two, and a real per-call cost the others don't have at all.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```

Only needed for `llm_router`.

## Run

```bash
python modular_rag.py --compare
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`):

```bash
python modular_rag.py --compare --provider openai
```

## Files

| File | Role |
|---|---|
| `modular_rag.py` | All 3 routers + the CLI — the file you actually run |
| `modular_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |

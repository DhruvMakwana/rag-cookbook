# Self-RAG (prompted)

A prompted Self-RAG loop — no fine-tuning, just a sequence of grading calls standing in for the original paper's trained reflection tokens (the dominant way Self-RAG is actually implemented in practice): retrieve → grade relevance per chunk → generate from only the relevant ones → grade groundedness → grade usefulness → return, regenerate once, or fall back to an honest "insufficient support" response. Concept write-up: [Self-RAG](https://dhruvmakwana.github.io/rag-deep-dive/self-rag/).

Needs an LLM key — relevance grading, generation, and usefulness grading are all LLM calls. Groundedness grading is fully local (no key needed for that specific check).

## The one genuinely new piece: NLI-based groundedness

Most write-ups (including the original notes this page is based on) describe the groundedness/faithfulness check as another LLM call: "is this answer supported by the context?" This recipe uses a local **NLI (Natural Language Inference) cross-encoder** instead — `cross-encoder/nli-deberta-v3-base`, premise = retrieved context, hypothesis = the generated answer, checking for entailment. Verified directly:

| Test | Result |
|---|---|
| Answer matches context ("Adam optimizer" claim vs. context stating exactly that) | `entailment` — grounded |
| Answer contradicts context ("SGD" claim vs. context stating "Adam") | `contradiction` — not grounded |
| No relevant context retrieved at all | `no_context` — not grounded |

This is meaningfully cheaper than an LLM-judge call for this specific check (local inference, ~200-500ms, vs. a full LLM round-trip) and, per recent benchmarking, comparably accurate for QA-over-retrieved-context faithfulness specifically. **Known limitation, stated plainly:** this checks the whole answer as one hypothesis against the whole context as one premise — a more precise version would decompose the answer into individual claims and check each one, since this version can miss a mostly-grounded answer with one ungrounded clause buried in it.

## What actually happened testing this

Claude Sonnet 5 turned out to be remarkably resistant to hallucinating in this setup — across several deliberate attempts (context that omits the requested fact, no explicit "say so if missing" instruction, topically-related-but-non-answering context), it consistently declined rather than filled in a plausible-sounding answer from its own pretraining knowledge. That's a genuinely useful, honest finding on its own — it changes when a groundedness-checking stage actually earns its keep with a frontier model like this one, versus older/smaller/open models more prone to filling gaps.

Given that, the groundedness and usefulness checks are verified two ways: **controlled unit tests** (a deliberately wrong or hedged answer, checked directly against the grading functions — confirms the checks correctly fire when something IS wrong) and **one genuine, unforced end-to-end failure**: asking a question about a fact already known (from `query-transformation/`'s IRCoT testing) to embed poorly and be unfindable by any query phrasing. Relevance grading correctly discarded all 5 retrieved chunks as irrelevant, generation honestly declined instead of guessing, both checks correctly flagged the result, and the pipeline retried once before falling back to its honest final message — a real trace, not manufactured.

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
python self_rag.py --query "What optimizer did they use for training?"
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`) instead of changing `.env` — has no effect on the groundedness check, which is fully local:

```bash
python self_rag.py --query "..." --provider openai
```

## Files

| File | Role |
|---|---|
| `self_rag.py` | All 4 grading stages + the loop + CLI — the file you actually run |
| `self_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

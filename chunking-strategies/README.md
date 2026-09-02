# Chunking Strategies

Ten chunking strategies implemented from scratch against the same sample document, so you can actually see how differently each one splits identical text. Concept write-up: [Chunking Strategies](https://dhruvmakwana.github.io/rag-deep-dive/chunking/).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` + `transformers` pull in PyTorch — expect a few hundred MB and a couple of minutes, one time only. `late_chunking` additionally downloads `answerdotai/ModernBERT-base` (~570MB) on first run.

## Configure (only needed for `proposition` and `agentic`)

```bash
cp .env.example .env
```
Fill in a key for whichever `LLM_PROVIDER` you set (`anthropic` | `openai` | `ollama`). Every other strategy runs fully locally with no key.

## Run

```bash
python chunking_strategies.py --strategy recursive
python chunking_strategies.py --strategy semantic --threshold 0.5
python chunking_strategies.py --strategy all          # runs all 10, one after another
```

| `--strategy` | Needs | What it demonstrates |
|---|---|---|
| `fixed_size` | nothing | Naive baseline: cut every N characters |
| `recursive` | nothing | Paragraph → sentence → word fallback splitting |
| `structure_aware` | nothing | Splits a Markdown sample on real headers (the PDF has none, so this uses its own sample text) |
| `semantic` | local embedding model | Cuts where adjacent-sentence similarity drops |
| `sentence_window` | local embedding model* | Precise anchor + surrounding context window |
| `small_to_big` | nothing | Small child chunks linked to a larger parent |
| `late_chunking` | local long-context model | Embeds the whole doc first, pools per chunk after — see the blog page for why this is structurally different from every other method here |
| `proposition` | LLM key | Atomic, self-contained factual statements (Dense X Retrieval) |
| `agentic` | LLM key | LLM decides boundaries directly |
| `adaptive` | local embedding model | Runs 3 strategies, scores coherence, picks the winner |

\* `sentence_window` doesn't strictly need embeddings to run, but the CLI loads the model anyway since it's grouped with the other embedding-based demos.

## Files

| File | Role |
|---|---|
| `chunking_strategies.py` | All 10 strategies + the CLI |
| `llm.py` | Pluggable generation step — Anthropic / OpenAI / Ollama (used by `proposition`, `agentic`) |
| `download_data.py` | Fetches the same sample PDF used in `naive-rag/` |

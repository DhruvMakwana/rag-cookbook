# Embedding Model Selection

Five demos, each measuring **Recall@k on a small, fact-verified eval set** (8 questions with keywords confirmed against the actual "Attention Is All You Need" text) rather than just printing vectors. Concept write-up: [Embedding Model Selection](https://dhruvmakwana.github.io/rag-deep-dive/embedding-models/).

No LLM API key needed anywhere — every model here runs fully locally.

| Demo | What it measures | Model(s) |
|---|---|---|
| `mismatched` | Query with a different model than you indexed with | `all-MiniLM-L6-v2` + `multi-qa-MiniLM-L6-cos-v1` |
| `matryoshka` | Recall@k at 768/512/256/128 truncated dimensions | `nomic-embed-text-v1.5` |
| `quantization` | Recall@k at float32/int8/binary precision | `nomic-embed-text-v1.5` |
| `instruction` | Recall@k with vs. without the model's real prefix convention | `nomic-embed-text-v1.5` |
| `bge_m3` | Dense + sparse + ColBERT-style multi-vector from one model | `BAAI/bge-m3` |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run — one demo at a time on memory-constrained machines

```bash
python embedding_models.py --demo mismatched
python embedding_models.py --demo matryoshka
python embedding_models.py --demo quantization
python embedding_models.py --demo instruction
python embedding_models.py --demo bge_m3
```

Measured peak memory: ~1.2GB for `nomic-embed-text-v1.5`, ~1GB for `BAAI/bge-m3`, well under 1GB for the two small models in `mismatched`. Each `python` invocation fully releases memory on exit, so running demos one at a time (rather than `--demo all`) is the safe default on an 8GB machine or one already under memory pressure from other apps.

```bash
python embedding_models.py --demo all   # fine on a machine with headroom to spare
```

## A genuinely negative result, reported honestly

The `instruction` demo measured **no difference** between prefixed and unprefixed queries on this specific 8-question eval set — even though `nomic-embed-text-v1.5`'s own documentation recommends the prefixes, based on evaluation at real benchmark scale. An eval set this small doesn't have the statistical power to reliably show every real effect; see the blog page for what that means for how much to trust any single small benchmark, including this one.

## Files

| File | Role |
|---|---|
| `embedding_models.py` | All 5 demos + the CLI — the file you actually run |
| `embedding_models_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `download_data.py` | Fetches the same sample PDF used in `naive-rag/` and `chunking-strategies/` |

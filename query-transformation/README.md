# Query Transformation (Pre-Retrieval)

Six pre-retrieval query transformation techniques, each measured with a real before/after — naive retrieval on the raw query vs. retrieval after the technique runs. Each technique gets its own scenario built to match its actual documented "when to use" case (a vague query for rewriting, a broad multi-faceted question for multi-query, a numeric-fact question for HyDE, and so on) rather than one generic question set tested against all six — a mismatched test that can't show most of these techniques doing what they're actually for. Concept write-up: [Advanced RAG — Pre-retrieval](https://dhruvmakwana.github.io/rag-deep-dive/advanced-rag-pre-retrieval/).

The sample PDF's raw-dump front matter (permission notice + 7 authors' names/affiliations/emails) is trimmed before chunking — it's single-newline-separated with no real paragraph structure, and none of the demo questions are about authorship, so keeping it in only added noisy chunks competing for retrieval slots. After this cleanup, naive Recall@3 on the original generic 8-question eval set alone improved from 0.88 to 1.00 — a reminder that chunking/data hygiene often moves the needle more than query transformation does.

All 6 need an LLM key — query transformation is inherently LLM-driven.

| Technique | Scenario | Measured result |
|---|---|---|
| `rewrite` | Vague, colloquial query ("why dont they just use RNNs like everyone else did before") | **Fixes a real miss** — naive never finds the "precludes parallelization" passage at k=1-3; rewriting to explicit technical phrasing finds it reliably |
| `multi_query` | Broad, multi-faceted question ("what makes this model better at handling long-range dependencies?") | **Fixes a real miss, most of the time** — naive never surfaces the "maximum path length" passage at k=5; RAG-Fusion across 3 paraphrasings does in ~3 of 4 runs (this technique makes a fresh LLM call each run, so it isn't perfectly reproducible) |
| `hyde` | Numeric-fact question ("what BLEU score did they get?") | **Unreliable** — ties naive most runs (already close to perfect post-cleanup), but can occasionally *hurt*: a safety-tuned LLM sometimes refuses to hallucinate a specific number, and embedding that refusal text retrieves worse than the real question would have |
| `step_back` | Specific hyperparameter question with real general context available ("what value did they use for Pdrop?") | **Tied, genuinely** — naive already finds specific facts directly in this single, well-organized paper; there isn't enough separation between "general" and "specific" content here for step-back to add anything, even in a scenario built to fit its use case |
| `decompose` | Genuinely compound question ("what optimizer did they use, and how many attention heads?") | **Clean win** — naive retrieval misses one of the two facts even at k=6; decomposition into sub-questions reliably finds both |
| `ircot` | Multi-hop question, tested against both a findable and an unfindable underlying fact | **Depends entirely on whether the fact is findable at all** — succeeds cleanly when both facts are individually locatable; fails (honestly, without hallucinating) when a fact is buried in a way no query phrasing can surface |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
```
Fill in a key for whichever `LLM_PROVIDER` you set — every technique here needs one.

## Run

```bash
python query_transformation.py --technique rewrite
python query_transformation.py --technique multi_query
python query_transformation.py --technique hyde
python query_transformation.py --technique step_back
python query_transformation.py --technique decompose
python query_transformation.py --technique ircot
```

Override the LLM provider per-call with `--provider` (`anthropic`, `openai`, or `ollama`) instead of changing `.env`:

```bash
python query_transformation.py --technique rewrite --provider openai
```

## A note on reproducibility

Every technique here makes at least one live LLM call with no fixed seed, so **exact wording (and occasionally the exact Recall@k number) will vary slightly run to run** — this is real, not a bug to "fix." Multiple runs during development showed the overall pattern (which techniques help, tie, or hurt on this corpus) staying consistent, even when specific numbers shifted by one hit out of eight.

## Files

| File | Role |
|---|---|
| `query_transformation.py` | All 6 techniques + the CLI — the file you actually run |
| `query_transformation_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the same sample PDF used in the other recipes |

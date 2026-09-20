# RAGAS Deep Dive

Runs a real naive RAG pipeline over the sample paper, then scores its real retrieved contexts and real generated answers with RAGAS's four core metrics — Faithfulness, Answer Relevancy, Context Precision, Context Recall — to see whether the four-metric diagnostic table actually holds up against a real system, not a curated example. Concept write-up: [RAGAS Deep Dive](https://dhruvmakwana.github.io/rag-deep-dive/ragas/).

Needs an Anthropic API key — used both for the naive RAG pipeline's own generation and as RAGAS's judge LLM. Retrieval is fully local.

## Scope: ragas ships two parallel evaluation APIs, this recipe uses both

Current `ragas` (0.4.x) has a legacy `evaluate()` + `EvaluationDataset` + named-metric-class API (what every existing tutorial shows) and a newer per-sample `ragas.metrics.collections` + `.ascore()` API the current docs actually lead with. The legacy path still works — every import just fires a `DeprecationWarning` pointing at the new one — but it's the only one that produces the four-metrics-together diagnostic table (`--compare`'s first section uses it). This recipe also uses the modern collections API for four additional metrics (`--compare`'s second section) and for synthetic eval-set generation (`--testset`), since those aren't available through the legacy path.

## Setup requirements for this ragas release

1. **`langchain-community==0.3.31` needs pinning.** `ragas/llms/base.py` imports `langchain_community.chat_models.vertexai`, a module later `langchain-community` releases removed (moved to a separate `langchain-google-vertexai` package), and `ragas` carries no upper pin on it.
2. **Embeddings go through `LangchainEmbeddingsWrapper`.** `ragas`'s own `HuggingfaceEmbeddings` class fails pydantic validation on direct instantiation in this release — wrap a real `langchain-huggingface` embeddings object with `LangchainEmbeddingsWrapper` instead, the same pattern used for the judge LLM via `LangchainLLMWrapper`.
3. **The judge LLM wrapper needs `bypass_temperature=True`.** `LangchainLLMWrapper` sets a `temperature` value on the underlying model before every call by default. Current Claude models no longer accept that parameter (adaptive thinking replaces sampling controls), so `bypass_temperature=True` — the wrapper's own documented flag for this case — is required.
4. **The modern collections API needs a different fix for the same underlying problem.** It builds its LLM via `llm_factory` + the `instructor` library rather than LangChain, so `bypass_temperature` doesn't apply. Current Claude models reject both `temperature` and `top_p`, and this API has no equivalent constructor flag — both need removing directly from the constructed LLM's own `model_args` dict (`make_modern_judge_llm()` in `ragas_deep_dive.py` does this). Separately, `instructor>=1.17` is required against Anthropic models specifically — older versions fail to parse a response when the model returns a `ThinkingBlock` before its text content.
5. **`TestsetGenerator` needs the optional `rapidfuzz` package** for its internal relationship-builder transform — not a core `ragas` dependency, so it isn't pulled in automatically.

## Measured: does the diagnostic table hold up on a real system?

Aggregate scores across the standard 8-question eval set used throughout this repo, core four metrics:

```text
faithfulness:                          0.9062
answer_relevancy:                      0.8596
llm_context_precision_with_reference:  0.6250
context_recall:                        0.8750
```

Per-question, one result stands out — and it's the most useful one in the whole run:

```text
Q: What BLEU score did they get on English-to-German translation?
A: "...the Transformer model achieved a BLEU score of 28.4..." (factually correct)

faithfulness: 0.25   context_precision: 0.0   context_recall: 0.0
```

The generated number is correct — 28.4 is the real EN-DE BLEU score. But `context_recall` of 0.0 means the retrieved context didn't actually support that reference claim, and `context_precision` of 0.0 confirms none of the top-3 retrieved chunks were judged relevant to it either — retrieval missed the right table row, and the model answered correctly anyway from its own training data on this well-known paper, not from what was actually retrieved. RAGAS caught exactly this: a factually right answer with near-zero grounding in the retrieved context, which a simple "is the final answer correct" check would have missed entirely. This same pattern reproduced across multiple separate runs of this pipeline, not just once.

Aggregate scores for the four additional metrics from the modern collections API, same 8 questions:

```text
answer_correctness:   0.7238
semantic_similarity:  0.7899
factual_correctness:  0.4600
noise_sensitivity:    0.5089
```

`factual_correctness` scored 0.0 on two answers a human would call correct — including "The model was trained on 8 NVIDIA P100 GPUs" against a reference of "The models were trained on NVIDIA P100 GPUs" (0.877 semantic similarity, same fact, different phrasing). `FactualCorrectness`'s claim-decomposition-plus-NLI check is measurably stricter than embedding similarity on scope/phrasing differences a human reader would consider equivalent — worth knowing before treating a low `factual_correctness` score as proof of a factual error, rather than a possible phrasing mismatch.

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
python ragas_deep_dive.py --compare    # naive RAG + all 8 metrics (core 4 + extended 4)
python ragas_deep_dive.py --testset    # generate a synthetic eval set from the sample document
```

`--compare` takes a couple of minutes total (naive RAG generation for 8 questions, then two separate scoring passes). `--testset` takes under a minute for 6 synthetic questions.

Real, measured `--testset` output includes a case worth knowing about: the multi-hop synthesizer is skipped entirely (`No relationships match the provided condition. Cannot form clusters.`) even against the full ~40K-character source document — a single technical paper doesn't have enough thematic/entity overlap *between different sections* for the relationship-builder to find valid multi-hop clusters on. All generated questions come from the single-hop synthesizers instead. Separately, `TestsetGenerator` deliberately varies persona and query style — one generated question in a real run read `"wat did noam shazer propse in teh transformr paper"`, a genuine typo-laden phrasing variant, not a formatting error — which is a real argument for synthetic generation producing a more realistic eval set than hand-written clean questions.

## Files

| File | Role |
|---|---|
| `ragas_deep_dive.py` | Naive RAG pipeline, RAGAS evaluation, CLI — the file you actually run |
| `ragas_deep_dive_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation for the naive baseline — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the sample "Attention Is All You Need" PDF used across this repo |

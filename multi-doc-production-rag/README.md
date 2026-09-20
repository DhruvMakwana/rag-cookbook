# Multi-Document Production RAG

A capstone that ties together what the rest of this repo covers separately: a genuinely multi-document corpus (three real papers, not one toy PDF), a chunking-strategy comparison, a dense-vs-hybrid retrieval comparison, indexing the winning setup into a real vector database (Qdrant) with metadata filtering, and a final RAGAS evaluation of the assembled pipeline. Concept write-up: [Multi-Document Production RAG](https://dhruvmakwana.github.io/rag-deep-dive/tutorials/multi-doc-production-rag/).

No paid API required for the retrieval/chunking parts. The final RAGAS evaluation step needs an LLM — defaults to a local Ollama model, or set `LLM_PROVIDER=anthropic`/`openai` in `.env` for a hosted model.

## The corpus

Three real papers, closely related enough that genuine cross-document questions make sense: "Attention Is All You Need," "BERT," and "GPT-3" (~230 pages combined, 700-1000+ chunks depending on strategy — real scale, not a single 15-page toy PDF).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

If using the default Ollama provider:

```bash
ollama serve &
ollama pull qwen3:4b
```

## Run

```bash
python multi_doc_rag.py --part chunking    # compare fixed-size / recursive / sentence-window chunking
python multi_doc_rag.py --part retrieval   # compare dense-only vs. hybrid (BM25+RRF) retrieval
python multi_doc_rag.py --part qdrant      # index into Qdrant, demo metadata filtering across documents
python multi_doc_rag.py --part ragas       # naive RAG + RAGAS evaluation of the full pipeline
python multi_doc_rag.py --part all         # everything
```

## What's actually measured here

**Chunking comparison** (Recall@3 / Hit Rate@3 against 9 real, keyword-verified cross-document questions): sentence-window chunking clearly outperformed both fixed-size and recursive chunking on this corpus — 0.667 hit rate vs. 0.222 for fixed-size. Complete sentences as chunk boundaries mattered more than chunk-size tuning here.

**Dense vs. hybrid retrieval**: dense-only outperformed hybrid (BM25 + RRF) on this eval set — 0.667 vs. 0.444 hit rate. This is a real, checked-not-a-bug result: none of the eval questions have exact-match value (rare terms, codes, IDs) that BM25 is actually good at, so BM25's ranking for these natural-language questions is closer to noise, and RRF has no mechanism to down-weight a ranker that's contributing noise for a specific query — it trusts both rankers equally by construction. Hybrid search isn't unconditionally better; it depends on whether the corpus and query style actually have exact-match value to contribute.

**Qdrant metadata filtering**: "How many layers does the model have?" is genuinely ambiguous across this corpus (the Transformer, BERT, and GPT-3 all have a real, different layer count) — unfiltered retrieval's top-3 mixes the Transformer and GPT-3 papers and misses BERT entirely, even though BERT's own "L=12" answer is a real, correct one. Filtering to a specific document when that context is known is a real fix, not a hypothetical one.

**RAGAS evaluation**: of 9 real questions, 3 showed genuine retrieval misses (context_recall = 0.0). One of those three — "What are BERT's two pretraining tasks called?" — is a real instance of the same pattern documented in [RAGAS Deep Dive](../ragas-deep-dive/): the naive RAG pipeline answered the question correctly (Masked LM and Next Sentence Prediction are both real, correct facts), but scored faithfulness = 0.0, since the model answered from its own training knowledge rather than what was actually retrieved for that specific question.

## Files

| File | Role |
|---|---|
| `multi_doc_rag.py` | Corpus loading, three chunkers, retrieval/chunking comparisons, Qdrant indexing, RAGAS eval -- the file you actually run |
| `multi_doc_rag_docs.py` | **Documentation only** -- self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation for the naive RAG step -- Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the three sample papers used in this tutorial |

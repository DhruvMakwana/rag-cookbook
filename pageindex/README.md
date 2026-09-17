# PageIndex

Vectorless, reasoning-based retrieval: no embeddings, no vector database. A document's actual structure — headers, sections, table of contents — becomes a tree, and an LLM navigates that tree to find the right section, the way a person flips to the right chapter instead of scanning for similar-looking text. Compared directly against this repo's usual naive text RAG baseline. Concept write-up: [PageIndex](https://dhruvmakwana.github.io/rag-deep-dive/pageindex/).

Needs an Anthropic API key. Uses the real [`pageindex`](https://github.com/VectifyAI/PageIndex) package (VectifyAI), in local mode — documents are indexed on your machine using your own key, not sent to PageIndex's cloud.

## A different sample document than the rest of this repo, on purpose

Every other recipe here uses the "Attention Is All You Need" paper for consistency. This one doesn't: that paper has a shallow structure (~7 top-level sections, at most one level of sub-numbering), which wouldn't exercise what PageIndex is actually built for — navigating a document with real depth. This recipe uses NIST's **"Artificial Intelligence Risk Management Framework (AI RMF 1.0)"** instead: a real, free, 48-page regulatory document with genuine 4-level structure (Part → Section → Subsection → numbered category, e.g. `AI RMF Core → Govern → GOVERN 1.4`).

## How it works

1. **Tree construction** — a layout parser extracts the document's actual structure (headings, table of contents) directly, without an LLM, where possible. An LLM is used for two smaller things: confirming a detected heading really starts where the parser thinks it does, and writing each section's summary.
2. **Retrieval + generation, together** — given a question, the chat model is handed the tree (titles + summaries, not full text), reasons about which section(s) look relevant, pulls the actual text for just those sections, and answers — looping back for more if the first ones weren't enough. This is an agentic tool-calling loop under the hood, not a single classification call.

Both roles run on Anthropic models here, via LiteLLM's `anthropic/` model prefix (the library's own examples default to OpenAI): `anthropic/claude-haiku-4-5` for indexing (cheap, since a basic model is sufficient for structure/summaries), `anthropic/claude-sonnet-5` for the reasoning/chat step.

## Measured: PageIndex 5/5, naive text RAG 3/5

A 5-question eval set specific to this document (keywords checked directly against the real page text, not guessed):

```text
                                                              Naive   PageIndex (pages cited)
What is Risk Tolerance adapted from?                          PASS    PASS  [12]
How is GOVERN described relative to other functions?          PASS    PASS  [25,26,27]
What does MEASURE 2.7 evaluate?                                FAIL    PASS  [35]
Per Appendix B, are AI risks comprehensively addressed?        FAIL    PASS  [43,44]
Is it easy to tell if an opaque AI system is fair/secure?      PASS    PASS  [21]
```

Every PageIndex page citation was checked against the document's real tree structure and is accurate — it navigated to the actual section containing each answer, not just a plausible-sounding one.

**The two naive-RAG misses share a real, specific cause**, checked directly: for "What does MEASURE 2.7 evaluate?", the top-3 retrieved chunks were a nearby MAP-function table and generic TEVV process text — semantically close to the query's vocabulary ("measure", "evaluate"), but not the actual MEASURE 2.7 entry. Same pattern for the Appendix B question: the retrieved chunks were generic risk-management prose from elsewhere in the document, not Appendix B's specific text. In both cases, embedding similarity pulled toward text that *sounds* like the query rather than the document's own structural answer — exactly the failure mode this technique is built to avoid.

**What this comparison doesn't measure**: PageIndex's per-query cost. Retrieval and generation are one agentic call here, and the library doesn't expose a clean per-call token/cost breakdown the way a raw API call does (see the [CAG recipe](../cag/) for that kind of accounting) — each PageIndex query here took 6-12 seconds against naive RAG's <0.03 seconds, since it's running an actual reasoning loop over the tree rather than one cosine-similarity lookup. That latency and cost trade-off is real and should factor into when you reach for this.

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
python pageindex_rag.py --compare                                        # the comparison above
python pageindex_rag.py --mode pageindex --query "What does GOVERN 1.4 address?"
python pageindex_rag.py --mode naive     --query "What does GOVERN 1.4 address?"
```

First run indexes the document (~1 minute for 48 pages, a few cents) and caches the `doc_id` locally in `.pageindex_doc_id` — later runs reuse it instead of re-indexing.

## Files

| File | Role |
|---|---|
| `pageindex_rag.py` | Indexing, PageIndex search, naive text RAG baseline, CLI — the file you actually run |
| `pageindex_rag_docs.py` | **Documentation only** — self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `llm.py` | Pluggable generation for the naive baseline — Anthropic / OpenAI / Ollama |
| `download_data.py` | Fetches the NIST AI RMF PDF used by this recipe |

# Chunking Strategies

Ten chunking strategies, each backed by a real, widely-used, tested implementation — not hand-rolled from scratch — run against the same sample document so you can see how differently each one splits identical text. Concept write-up: [Chunking Strategies](https://dhruvmakwana.github.io/rag-deep-dive/chunking/).

| Strategy | Library | Class |
|---|---|---|
| `fixed_size` | [langchain-text-splitters](https://github.com/langchain-ai/langchain) | `CharacterTextSplitter` |
| `recursive` | langchain-text-splitters | `RecursiveCharacterTextSplitter` |
| `structure_aware` | langchain-text-splitters | `MarkdownHeaderTextSplitter` |
| `semantic` | [chonkie](https://pypi.org/project/chonkie) | `SemanticChunker` |
| `sentence_window` | [llama-index-core](https://github.com/run-llama/llama_index) | `SentenceWindowNodeParser` |
| `small_to_big` | llama-index-core | `HierarchicalNodeParser` |
| `late_chunking` | chonkie | `LateChunker` |
| `proposition` | [transformers](https://huggingface.co/chentong00/propositionizer-wiki-flan-t5-large) | the official Dense X Retrieval model (Chen et al., 2023) |
| `agentic` | chonkie | `SlumberChunker` (LLM-guided) |
| `adaptive` | — | orchestrates the 3 chunkers above; no library owns this meta-technique |

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pulls in PyTorch (via sentence-transformers/transformers) — a few hundred MB, a couple of minutes, one time only. `late_chunking` additionally downloads `nomic-ai/modernbert-embed-base`, and `proposition` downloads the ~780M-parameter propositionizer model, both on first run.

## Configure (only needed for `agentic`)

```bash
cp .env.example .env
```
Fill in a key for whichever `LLM_PROVIDER` you set (`anthropic` | `openai` | `ollama`). Every other strategy runs fully locally with no key — including `proposition`, which uses a locally-run model, not an API call.

## Run

```bash
python chunking_strategies.py --strategy recursive
python chunking_strategies.py --strategy semantic --threshold 0.5
python chunking_strategies.py --strategy all          # runs all 10, one after another
```

Each run prints full chunks (not truncated) so overlap and boundary behavior between consecutive chunks is directly visible, not just asserted.

## Files

| File | Role |
|---|---|
| `chunking_strategies.py` | All 10 strategies (each wrapping a real library call) + the CLI — the file you actually run |
| `chunking_strategies_docs.py` | **Documentation only** — a self-contained version of each function (own imports, own constants inlined) that the blog embeds section-by-section, so a copy-pasted block never assumes context from elsewhere in the file. Not imported by anything, not run as a script. If you change an algorithm above, mirror the change here — the two are not kept in sync automatically. |
| `llm.py` | Genie adapters for `agentic` — Anthropic/Ollama adapters implementing Chonkie's `BaseGenie` interface, plus Chonkie's own `OpenAIGenie` |
| `download_data.py` | Fetches the same sample PDF used in `naive-rag/` |

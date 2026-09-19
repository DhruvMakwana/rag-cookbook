# Vector Databases

The same chunk corpus (with real page-number metadata) indexed and queried against six vector databases -- FAISS, ChromaDB, Qdrant, Milvus Lite, pgvector, and Weaviate -- covering how each one actually handles HNSW parameters and metadata filtering, not just "here's how you add a vector." Concept write-up: [Vector Databases](https://dhruvmakwana.github.io/rag-deep-dive/vector-databases/).

No API key needed -- embeddings run locally via `sentence-transformers`. Pinecone is covered in the blog page as a reference (cloud-only, no self-host path, not run here).

## Corpus

The standard "Attention Is All You Need" PDF used across this repo, chunked with each chunk tagged by the real PDF page number it came from. Every database's metadata-filter demo filters on that page number.

## Setup per database

| Database | Needs |
|---|---|
| FAISS | Nothing extra -- runs in-process |
| ChromaDB | Nothing extra -- runs in-process |
| Qdrant | Nothing extra -- embedded mode, no server |
| Milvus Lite | Nothing extra -- embedded, pure-Python, no server |
| pgvector | A running Postgres with the extension: `docker run -d -e POSTGRES_PASSWORD=postgres -p 5433:5432 pgvector/pgvector:pg17` |
| Weaviate | A running server: `docker run -d -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true -e DEFAULT_VECTORIZER_MODULE=none cr.weaviate.io/semitechnologies/weaviate:1.39.5` |

Two setup notes worth knowing about upfront:

- On macOS, `faiss` needs to be imported before `torch` (pulled in by `sentence-transformers`) in the same process -- the reverse order loads two conflicting OpenMP runtimes and segfaults on the first FAISS call. `vector_databases.py` imports `faiss` first for this reason.
- `pymilvus` needs the `milvus_lite` extra (`pip install "pymilvus[milvus_lite]"`) for local, serverless use -- a plain `pip install pymilvus` doesn't pull it in.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python vector_databases.py --db all       # every database
python vector_databases.py --db qdrant    # just one
```

## Files

| File | Role |
|---|---|
| `vector_databases.py` | Shared corpus + one demo function per database -- the file you actually run |
| `vector_databases_docs.py` | **Documentation only** -- self-contained per-function version for the blog. Not run as a script, not kept in sync automatically. |
| `download_data.py` | Fetches the sample "Attention Is All You Need" PDF used across this repo |

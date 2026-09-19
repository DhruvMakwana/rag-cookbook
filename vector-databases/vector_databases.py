"""
Vector databases: the same real chunk corpus (with real page-number
metadata) indexed and queried against FAISS, ChromaDB, Qdrant, pgvector,
Weaviate, and Milvus Lite -- covering how each one actually handles HNSW
tuning, metadata filtering, and hybrid search, not just "here's how you
add a vector."
"""

import argparse
from pathlib import Path

# faiss must be imported before torch (pulled in by sentence-transformers)
# on macOS -- the reverse order loads two conflicting OpenMP runtimes and
# segfaults on the first FAISS call.
import faiss  # noqa: F401
import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

QUERIES = [
    "How many attention heads does the base model use?",
    "What optimizer and learning rate schedule were used for training?",
    "What BLEU score did the model achieve on English-to-German translation?",
]


def download_sample_pdf(url: str = DEFAULT_PDF_URL, dest: Path = DEFAULT_PDF_PATH) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


# ======================================================================
# Shared corpus -- chunks carry real page-number metadata so every
# database's filtered-search demo below filters on something genuine.
# ======================================================================


def build_corpus() -> list[dict]:
    """Returns a list of {"id", "text", "page"} dicts -- one per chunk,
    tagged with the real PDF page number it came from."""
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))

    chunks = []
    chunk_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "page": page_num})
            chunk_id += 1
    return chunks


def embed_corpus(chunks: list[dict], model: SentenceTransformer) -> np.ndarray:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
    return embeddings.astype("float32")


# ======================================================================
# FAISS -- no native metadata, so an IndexIDMap2 + external {id: page}
# dict is the standard pattern; filtering happens via IDSelectorBatch
# passed as `sel=` in SearchParameters at query time.
# ======================================================================


def demo_faiss(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import faiss

    dim = embeddings.shape[1]
    page_by_id = {c["id"]: c["page"] for c in chunks}
    ids = np.array([c["id"] for c in chunks], dtype="int64")

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.hnsw.efSearch = 64
    index = faiss.IndexIDMap2(hnsw)
    index.add_with_ids(embeddings, ids)

    query = QUERIES[1]  # optimizer / learning rate schedule -- appears late in the paper
    query_vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")

    print(f"\n--- FAISS: unfiltered top-3 for {query!r} ---")
    scores, result_ids = index.search(query_vec, 3)
    for score, cid in zip(scores[0], result_ids[0]):
        print(f"  page {page_by_id[cid]:>2}  score={score:.3f}")

    # Metadata filter: restrict the search to pages 7-9, where training
    # details live in this paper, using a real page-number boundary.
    allowed_ids = np.array([c["id"] for c in chunks if 7 <= c["page"] <= 9], dtype="int64")
    selector = faiss.IDSelectorBatch(allowed_ids)
    params = faiss.SearchParametersHNSW(sel=selector)

    print(f"--- FAISS: filtered to pages 7-9, top-3 for {query!r} ---")
    scores, result_ids = index.search(query_vec, 3, params=params)
    for score, cid in zip(scores[0], result_ids[0]):
        if cid == -1:
            continue
        print(f"  page {page_by_id[cid]:>2}  score={score:.3f}")


# ======================================================================
# ChromaDB -- metadata is native (a `where` filter alongside the vector
# query), and HNSW tuning lives in a structured `configuration={"hnsw": ...}`
# dict passed at collection-creation time.
# ======================================================================


def demo_chromadb(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import shutil

    import chromadb

    chroma_dir = Path(__file__).parent / "data" / "chroma_db"
    shutil.rmtree(chroma_dir, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    collection = client.create_collection(
        name="attention_paper",
        configuration={"hnsw": {"space": "cosine", "ef_construction": 200, "ef_search": 64, "max_neighbors": 32}},
    )
    collection.add(
        ids=[str(c["id"]) for c in chunks],
        embeddings=embeddings.tolist(),
        metadatas=[{"page": c["page"]} for c in chunks],
        documents=[c["text"] for c in chunks],
    )

    query = QUERIES[2]  # BLEU score question -- results live on page 8 in this paper
    query_vec = model.encode([query], normalize_embeddings=True).tolist()

    print(f"\n--- ChromaDB: unfiltered top-3 for {query!r} ---")
    result = collection.query(query_embeddings=query_vec, n_results=3)
    for page, dist in zip(result["metadatas"][0], result["distances"][0]):
        print(f"  page {page['page']:>2}  distance={dist:.3f}")

    # Native metadata filter: only pages 8-9, combined with the vector
    # query in a single call via `where`.
    print(f"--- ChromaDB: where page in [8, 9], top-3 for {query!r} ---")
    result = collection.query(query_embeddings=query_vec, n_results=3, where={"page": {"$in": [8, 9]}})
    for page, dist in zip(result["metadatas"][0], result["distances"][0]):
        print(f"  page {page['page']:>2}  distance={dist:.3f}")


# ======================================================================
# Qdrant -- `.search()` was removed from the client entirely (not just
# deprecated); `query_points()` is the only method now. Embedded mode
# (no server) via `QdrantClient(path=...)` is enough for this demo.
# ======================================================================


def demo_qdrant(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import shutil

    from qdrant_client import QdrantClient, models

    qdrant_dir = Path(__file__).parent / "data" / "qdrant_db"
    shutil.rmtree(qdrant_dir, ignore_errors=True)
    client = QdrantClient(path=str(qdrant_dir))

    dim = embeddings.shape[1]
    client.create_collection(
        collection_name="attention_paper",
        vectors_config=models.VectorParams(
            size=dim,
            distance=models.Distance.COSINE,
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200),
        ),
    )
    client.upsert(
        collection_name="attention_paper",
        points=[
            models.PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload={"page": c["page"], "text": c["text"]})
            for i, c in enumerate(chunks)
        ],
    )

    query = QUERIES[0]  # attention-heads question -- answered on page 3 in this paper
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    print(f"\n--- Qdrant: unfiltered top-3 for {query!r} ---")
    result = client.query_points(collection_name="attention_paper", query=query_vec, limit=3)
    for point in result.points:
        print(f"  page {point.payload['page']:>2}  score={point.score:.3f}")

    # Native metadata filter: exclude the appendix pages, expressed as a
    # Range condition on the real page-number payload field.
    print(f"--- Qdrant: filtered to page <= 10, top-3 for {query!r} ---")
    result = client.query_points(
        collection_name="attention_paper",
        query=query_vec,
        limit=3,
        query_filter=models.Filter(must=[models.FieldCondition(key="page", range=models.Range(lte=10))]),
    )
    for point in result.points:
        print(f"  page {point.payload['page']:>2}  score={point.score:.3f}")


# ======================================================================
# Milvus Lite -- a local, embedded, pure-Python `MilvusClient("./file.db")`
# server. Filtering uses a string expression syntax rather than a
# structured filter object.
# ======================================================================


def demo_milvus(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import shutil

    from pymilvus import DataType, MilvusClient

    # Despite the ".db" suffix suggesting a single file, MilvusClient
    # creates a directory here (WAL + storage) -- shutil.rmtree, not
    # Path.unlink, is what actually clears it between runs.
    milvus_path = Path(__file__).parent / "data" / "milvus_lite.db"
    shutil.rmtree(milvus_path, ignore_errors=True)
    client = MilvusClient(str(milvus_path))

    dim = embeddings.shape[1]
    schema = client.create_schema(auto_id=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("page", DataType.INT64)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=dim)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="HNSW", metric_type="COSINE", params={"M": 32, "efConstruction": 200})

    client.create_collection(collection_name="attention_paper", schema=schema, index_params=index_params)
    client.insert(
        collection_name="attention_paper",
        data=[{"id": c["id"], "page": c["page"], "vector": embeddings[i].tolist()} for i, c in enumerate(chunks)],
    )

    query = QUERIES[1]
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    print(f"\n--- Milvus Lite: unfiltered top-3 for {query!r} ---")
    result = client.search(collection_name="attention_paper", data=[query_vec], limit=3, output_fields=["page"], search_params={"params": {"ef": 64}})
    for hit in result[0]:
        print(f"  page {hit['entity']['page']:>2}  score={hit['distance']:.3f}")

    # Filter expression syntax -- a real string expression, not a
    # structured filter object, evaluated against the schema field.
    print(f"--- Milvus Lite: filter 'page >= 7 and page <= 9', top-3 for {query!r} ---")
    result = client.search(
        collection_name="attention_paper", data=[query_vec], limit=3, output_fields=["page"],
        filter="page >= 7 and page <= 9", search_params={"params": {"ef": 64}},
    )
    for hit in result[0]:
        print(f"  page {hit['entity']['page']:>2}  score={hit['distance']:.3f}")


# ======================================================================
# pgvector -- vectors and metadata live as ordinary Postgres columns, so
# filtering is just SQL `WHERE` combined with `ORDER BY ... <=> ... LIMIT`.
# Needs a running Postgres with the pgvector extension, e.g.:
#   docker run -d -e POSTGRES_PASSWORD=postgres -p 5433:5432 pgvector/pgvector:pg17
# ======================================================================

PG_DSN = "dbname=postgres user=postgres password=postgres host=localhost port=5433"


def demo_pgvector(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import psycopg
    from pgvector.psycopg import register_vector

    dim = embeddings.shape[1]
    with psycopg.connect(PG_DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)
        conn.execute("DROP TABLE IF EXISTS attention_paper")
        conn.execute(f"CREATE TABLE attention_paper (id BIGINT PRIMARY KEY, page INT, embedding VECTOR({dim}))")

        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO attention_paper (id, page, embedding) VALUES (%s, %s, %s)",
                [(c["id"], c["page"], embeddings[i]) for i, c in enumerate(chunks)],
            )

        conn.execute("CREATE INDEX ON attention_paper USING hnsw (embedding vector_cosine_ops) WITH (m = 32, ef_construction = 200)")
        conn.execute("SET hnsw.ef_search = 64")

        query = QUERIES[2]
        query_vec = model.encode(query, normalize_embeddings=True)

        print(f"\n--- pgvector: unfiltered top-3 for {query!r} ---")
        rows = conn.execute(
            "SELECT page, embedding <=> %s AS distance FROM attention_paper ORDER BY embedding <=> %s LIMIT 3",
            (query_vec, query_vec),
        ).fetchall()
        for page, distance in rows:
            print(f"  page {page:>2}  distance={distance:.3f}")

        # Plain SQL WHERE combined with the vector ORDER BY -- no
        # separate filter API, since page is just another column.
        print(f"--- pgvector: WHERE page BETWEEN 8 AND 9, top-3 for {query!r} ---")
        rows = conn.execute(
            "SELECT page, embedding <=> %s AS distance FROM attention_paper WHERE page BETWEEN 8 AND 9 ORDER BY embedding <=> %s LIMIT 3",
            (query_vec, query_vec),
        ).fetchall()
        for page, distance in rows:
            print(f"  page {page:>2}  distance={distance:.3f}")


# ======================================================================
# Weaviate -- vectors are supplied directly (`self_provided`, skipping
# its built-in vectorizer modules since embeddings already exist), and
# HNSW config nests inside `Configure.Vectors.self_provided(...)` as of
# client >= 4.16, not a bare top-level kwarg. Needs a running server:
#   docker run -d -p 8080:8080 -p 50051:50051 -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
#     -e DEFAULT_VECTORIZER_MODULE=none cr.weaviate.io/semitechnologies/weaviate:1.39.5
# ======================================================================


def demo_weaviate(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> None:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.query import Filter

    client = weaviate.connect_to_local()
    try:
        client.collections.delete("AttentionPaper")
        collection = client.collections.create(
            name="AttentionPaper",
            properties=[Property(name="page", data_type=DataType.INT)],
            vector_config=Configure.Vectors.self_provided(
                vector_index_config=Configure.VectorIndex.hnsw(ef_construction=200, max_connections=32),
            ),
        )

        with collection.batch.dynamic() as batch:
            for i, c in enumerate(chunks):
                batch.add_object(properties={"page": c["page"]}, vector=embeddings[i].tolist(), uuid=weaviate.util.generate_uuid5(c["id"]))

        query = QUERIES[0]
        query_vec = model.encode(query, normalize_embeddings=True).tolist()

        print(f"\n--- Weaviate: unfiltered top-3 for {query!r} ---")
        result = collection.query.near_vector(near_vector=query_vec, limit=3, return_metadata=["distance"])
        for obj in result.objects:
            print(f"  page {obj.properties['page']:>2}  distance={obj.metadata.distance:.3f}")

        print(f"--- Weaviate: filtered to page <= 10, top-3 for {query!r} ---")
        result = collection.query.near_vector(
            near_vector=query_vec, limit=3, return_metadata=["distance"],
            filters=Filter.by_property("page").less_or_equal(10),
        )
        for obj in result.objects:
            print(f"  page {obj.properties['page']:>2}  distance={obj.metadata.distance:.3f}")
    finally:
        client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vector database demos.")
    parser.add_argument("--db", choices=["faiss", "chroma", "qdrant", "pgvector", "weaviate", "milvus", "all"], default="all")
    args = parser.parse_args()

    chunks = build_corpus()
    print(f"Corpus: {len(chunks)} chunks across {len({c['page'] for c in chunks})} pages")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    embeddings = embed_corpus(chunks, model)
    print(f"Embeddings: {embeddings.shape}")

    if args.db in ("faiss", "all"):
        demo_faiss(chunks, embeddings, model)
    if args.db in ("chroma", "all"):
        demo_chromadb(chunks, embeddings, model)
    if args.db in ("qdrant", "all"):
        demo_qdrant(chunks, embeddings, model)
    if args.db in ("milvus", "all"):
        demo_milvus(chunks, embeddings, model)
    if args.db in ("pgvector", "all"):
        demo_pgvector(chunks, embeddings, model)
    if args.db in ("weaviate", "all"):
        demo_weaviate(chunks, embeddings, model)

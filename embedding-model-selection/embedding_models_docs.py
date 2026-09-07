"""
Documentation-only companion to embedding_models.py.

Not run top-to-bottom, not imported by anything. Every function below is
fully self-contained — its own imports, its own eval set and helper
inlined — so copy-pasting ANY single function into a fresh .py works on
its own. See embedding_models.py for why this split exists (same
rationale as chunking_strategies_docs.py in the chunking-strategies/
recipe): the real file stays DRY for maintenance, this one stays
standalone for reading.

If you change something in embedding_models.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:mismatched]
import numpy as np
from sentence_transformers import SentenceTransformer

# 8 question/keyword pairs, each keyword confirmed present verbatim in the
# actual "Attention Is All You Need" text before being used as ground truth.
EVAL_SET = [
    {"question": "How many attention heads did they use?", "keyword": "h = 8"},
    {"question": "What is the model's embedding dimension?", "keyword": "dmodel = 512"},
    {"question": "How many layers are in the encoder?", "keyword": "N = 6"},
    {"question": "What optimizer was used for training?", "keyword": "Adam"},
    {"question": "What BLEU score did they get on English-to-German translation?", "keyword": "28.4"},
    {"question": "What GPUs was the model trained on?", "keyword": "P100"},
    {"question": "How long did the base model train for?", "keyword": "12 hours"},
    {"question": "What dropout rate did they use?", "keyword": "Pdrop = 0.1"},
]


def recall_at_k(
    chunk_embeddings: np.ndarray,
    query_embeddings: np.ndarray,
    chunks: list[str],
    keywords: list[str],
    k: int = 3,
) -> float:
    """For each eval question, checks whether its keyword shows up in ANY
    of the top-k retrieved chunks — a simple, chunking-agnostic proxy for
    "did retrieval actually find the answer." `keywords[i]` must
    correspond to `query_embeddings[i]` — the caller owns that pairing,
    not this function. Returns the fraction of questions that pass."""
    if len(keywords) != query_embeddings.shape[0]:
        raise ValueError(
            f"{len(keywords)} keywords but {query_embeddings.shape[0]} query embeddings — must match 1:1"
        )
    query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
    chunk_norm = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-8)
    similarities = query_norm @ chunk_norm.T

    hits = 0
    for i, keyword in enumerate(keywords):
        top_k_indices = np.argsort(-similarities[i])[:k]
        retrieved_text = " ".join(chunks[idx] for idx in top_k_indices)
        if keyword in retrieved_text:
            hits += 1
    return hits / len(keywords)


def demo_mismatched_models(chunks: list[str], k: int = 3) -> None:
    """Index with one model, query with a DIFFERENT model — this is the
    one rule the embedding-models page calls non-negotiable. Vectors from
    different models live in different, incomparable spaces. The two
    models here are deliberately the same dimension (384) so the failure
    is SILENT — a differently-sized model would crash the matrix multiply
    outright, which is the easy case to catch; same-dimension-but-wrong-
    space is the dangerous one, since nothing errors, it just quietly
    returns nonsense rankings."""
    questions = [item["question"] for item in EVAL_SET]
    keywords = [item["keyword"] for item in EVAL_SET]

    model_a = SentenceTransformer("all-MiniLM-L6-v2")
    model_b = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

    chunk_vecs_a = model_a.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_a = model_a.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    same_model_recall = recall_at_k(chunk_vecs_a, query_vecs_a, chunks, keywords, k=k)

    query_vecs_b = model_b.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    mismatched_recall = recall_at_k(chunk_vecs_a, query_vecs_b, chunks, keywords, k=k)

    print(f"Same model for index + query: Recall@{k} = {same_model_recall:.2f}")
    print(f"Mismatched query model:       Recall@{k} = {mismatched_recall:.2f}")
# --8<-- [end:mismatched]


# --8<-- [start:matryoshka]
import numpy as np
from sentence_transformers import SentenceTransformer


def demo_matryoshka_truncation(chunks: list[str], questions: list[str], keywords: list[str], k: int = 3) -> None:
    """Embed the same chunks and queries at several truncated dimensions
    using ONE Matryoshka-trained model (nomic-embed-text-v1.5, trained to
    support 768/512/256/128-dim slices) — no retraining, just a smaller
    `truncate_dim` per call — and measure how much (if any) Recall@k is
    actually lost at each size. `recall_at_k` is defined the same way as
    in the `mismatched` snippet above; `keywords[i]` must correspond to
    `questions[i]`."""
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    for dim in (768, 512, 256, 128):
        chunk_vecs = model.encode(chunks, truncate_dim=dim, convert_to_numpy=True, show_progress_bar=False)
        query_vecs = model.encode(questions, truncate_dim=dim, convert_to_numpy=True, show_progress_bar=False)
        recall = recall_at_k(chunk_vecs, query_vecs, chunks, keywords, k=k)
        storage_kb = chunk_vecs.nbytes / 1024
        print(f"dim={dim:>4}  Recall@{k} = {recall:.2f}  storage for {len(chunks)} chunks = {storage_kb:.0f} KB")
# --8<-- [end:matryoshka]


# --8<-- [start:quantization]
import numpy as np
from sentence_transformers import SentenceTransformer


def demo_quantization(chunks: list[str], questions: list[str], keywords: list[str], k: int = 3) -> None:
    """Same model, same dimension — only the stored NUMBER FORMAT changes.
    float32 is 4 bytes/dim, int8 is 1 byte/dim (4x smaller), binary packs
    8 dims per byte (32x smaller than float32). Measures whether Recall@k
    survives each step down. `recall_at_k` is defined the same way as in
    the `mismatched` snippet above; `keywords[i]` must correspond to
    `questions[i]`."""
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    for precision in ("float32", "int8", "binary"):
        chunk_vecs = model.encode(chunks, precision=precision, convert_to_numpy=True, show_progress_bar=False)
        query_vecs = model.encode(questions, precision=precision, convert_to_numpy=True, show_progress_bar=False)
        # cosine similarity needs float math even for quantized vectors
        recall = recall_at_k(chunk_vecs.astype(np.float32), query_vecs.astype(np.float32), chunks, keywords, k=k)
        storage_kb = chunk_vecs.nbytes / 1024
        print(f"{precision:>8}  Recall@{k} = {recall:.2f}  storage for {len(chunks)} chunks = {storage_kb:.0f} KB")
# --8<-- [end:quantization]


# --8<-- [start:instruction]
from sentence_transformers import SentenceTransformer


def demo_instruction_prefix(chunks: list[str], questions: list[str], keywords: list[str], k: int = 3) -> None:
    """Instruction-tuned embedding models expect a short task prefix at
    inference time — for nomic-embed-text-v1.5 specifically, a DIFFERENT
    prefix for each side: "search_query: " on questions, "search_document: "
    on passages, not the same prefix on both. Compares Recall@k with vs.
    without those prefixes on the same model. `recall_at_k` is defined
    the same way as in the `mismatched` snippet above; `keywords[i]` must
    correspond to `questions[i]`."""
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)

    chunk_vecs_plain = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_plain = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    recall_plain = recall_at_k(chunk_vecs_plain, query_vecs_plain, chunks, keywords, k=k)

    prefixed_chunks = ["search_document: " + c for c in chunks]
    prefixed_questions = ["search_query: " + q for q in questions]
    chunk_vecs_prefixed = model.encode(prefixed_chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_prefixed = model.encode(prefixed_questions, convert_to_numpy=True, show_progress_bar=False)
    recall_prefixed = recall_at_k(chunk_vecs_prefixed, query_vecs_prefixed, chunks, keywords, k=k)

    print(f"No prefixes (plain text both sides):      Recall@{k} = {recall_plain:.2f}")
    print(f"Correct prefixes (search_query/document): Recall@{k} = {recall_prefixed:.2f}")
# --8<-- [end:instruction]


# --8<-- [start:bge_m3]
from FlagEmbedding import BGEM3FlagModel


def demo_bge_m3_multi_functionality(text: str) -> None:
    """One model, one encode() call, three different retrieval
    representations: a dense vector (semantic similarity), sparse
    lexical weights (keyword-style, BM25-like matching with learned
    weights instead of statistics), and ColBERT-style per-token vectors
    (late-interaction / multi-vector retrieval)."""
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False)
    output = model.encode([text], return_dense=True, return_sparse=True, return_colbert_vecs=True)

    dense = output["dense_vecs"][0]
    sparse = output["lexical_weights"][0]
    colbert = output["colbert_vecs"][0]

    decoded_sparse = {model.tokenizer.decode([int(token_id)]): weight for token_id, weight in sparse.items()}
    top_terms = sorted(decoded_sparse.items(), key=lambda kv: -kv[1])[:5]

    print(f"Dense vector:    shape={dense.shape}")
    print(f"Sparse weights:  top terms = {top_terms}")
    print(f"ColBERT vectors: shape={colbert.shape} (one vector per token)")
# --8<-- [end:bge_m3]

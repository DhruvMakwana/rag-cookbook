"""
Embedding model selection, demonstrated with measured Recall@k, not just
printed vectors. Every demo runs a small, fact-verified eval set (8
questions with keywords confirmed against the actual paper text) through
real retrieval so the effect of each technique is a measured number, not
an assertion.

No LLM API key needed anywhere in this recipe — every model here runs
fully locally.

On a memory-constrained machine, run demos one at a time rather than
`--demo all` — each `python` invocation fully releases its memory on
exit, but `--demo all` loads several models sequentially in one process
without guaranteed intermediate cleanup. Peak memory measured: ~1.2GB for
nomic-embed-text-v1.5, ~1GB for BGE-M3, well under 1GB for the two small
baseline models in the mismatched-model demo.

Run:
    python embedding_models.py --demo mismatched
    python embedding_models.py --demo matryoshka
    python embedding_models.py --demo quantization
    python embedding_models.py --demo instruction
    python embedding_models.py --demo bge_m3
    python embedding_models.py --demo all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

BASELINE_MODEL = "all-MiniLM-L6-v2"
MISMATCHED_MODEL = "multi-qa-MiniLM-L6-cos-v1"
MATRYOSHKA_MODEL = "nomic-ai/nomic-embed-text-v1.5"  # 137M params — deliberately small
HYBRID_MODEL = "BAAI/bge-m3"

# nomic-embed-text-v1.5's actual trained convention — not an invented prompt.
NOMIC_QUERY_PREFIX = "search_query: "
NOMIC_DOCUMENT_PREFIX = "search_document: "

# 8 question/keyword pairs, each keyword confirmed present verbatim in the
# actual "Attention Is All You Need" text before being used here.
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


# ======================================================================
# Setup / data loading / eval helpers
# ======================================================================


def load_sample_text() -> str:
    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


def cosine_sim_matrix(query_vecs: np.ndarray, chunk_vecs: np.ndarray) -> np.ndarray:
    query_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-8)
    chunk_norm = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-8)
    return query_norm @ chunk_norm.T


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
    similarities = cosine_sim_matrix(query_embeddings, chunk_embeddings)
    hits = 0
    for i, keyword in enumerate(keywords):
        top_k_indices = np.argsort(-similarities[i])[:k]
        retrieved_text = " ".join(chunks[idx] for idx in top_k_indices)
        if keyword in retrieved_text:
            hits += 1
    return hits / len(keywords)


# ======================================================================
# 1. The consistency rule — mismatched index/query embedding models
# ======================================================================
# --8<-- [start:mismatched]
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

    model_a = SentenceTransformer(BASELINE_MODEL)
    model_b = SentenceTransformer(MISMATCHED_MODEL)

    chunk_vecs_a = model_a.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_a = model_a.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    same_model_recall = recall_at_k(chunk_vecs_a, query_vecs_a, chunks, keywords, k=k)

    query_vecs_b = model_b.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    mismatched_recall = recall_at_k(chunk_vecs_a, query_vecs_b, chunks, keywords, k=k)

    print(f"Same model for index + query   ({BASELINE_MODEL}): Recall@{k} = {same_model_recall:.2f}")
    print(f"Mismatched query model         ({MISMATCHED_MODEL} query vs. {BASELINE_MODEL} index): Recall@{k} = {mismatched_recall:.2f}")
# --8<-- [end:mismatched]


# ======================================================================
# 2. Matryoshka truncation
# ======================================================================
# --8<-- [start:matryoshka]
def demo_matryoshka_truncation(chunks: list[str], k: int = 3) -> None:
    """Embed the same chunks and queries at several truncated dimensions
    using ONE Matryoshka-trained model — no retraining, just a smaller
    `truncate_dim` per call — and measure how much (if any) Recall@k is
    actually lost at each size."""
    questions = [item["question"] for item in EVAL_SET]
    keywords = [item["keyword"] for item in EVAL_SET]
    model = SentenceTransformer(MATRYOSHKA_MODEL, trust_remote_code=True)

    for dim in (768, 512, 256, 128):
        chunk_vecs = model.encode(chunks, truncate_dim=dim, convert_to_numpy=True, show_progress_bar=False)
        query_vecs = model.encode(questions, truncate_dim=dim, convert_to_numpy=True, show_progress_bar=False)
        recall = recall_at_k(chunk_vecs, query_vecs, chunks, keywords, k=k)
        storage_kb = (chunk_vecs.nbytes) / 1024
        print(f"dim={dim:>4}  Recall@{k} = {recall:.2f}  storage for {len(chunks)} chunks = {storage_kb:.0f} KB")
# --8<-- [end:matryoshka]


# ======================================================================
# 3. Quantization
# ======================================================================
# --8<-- [start:quantization]
def demo_quantization(chunks: list[str], k: int = 3) -> None:
    """Same model, same dimension — only the stored NUMBER FORMAT changes.
    float32 is 4 bytes/dim, int8 is 1 byte/dim (4x smaller), binary packs
    8 dims per byte (32x smaller than float32). Measures whether Recall@k
    survives each step down."""
    questions = [item["question"] for item in EVAL_SET]
    keywords = [item["keyword"] for item in EVAL_SET]
    model = SentenceTransformer(MATRYOSHKA_MODEL, trust_remote_code=True)

    for precision in ("float32", "int8", "binary"):
        chunk_vecs = model.encode(chunks, precision=precision, convert_to_numpy=True, show_progress_bar=False)
        query_vecs = model.encode(questions, precision=precision, convert_to_numpy=True, show_progress_bar=False)
        # cosine similarity needs float math even for quantized vectors
        recall = recall_at_k(chunk_vecs.astype(np.float32), query_vecs.astype(np.float32), chunks, keywords, k=k)
        storage_kb = chunk_vecs.nbytes / 1024
        print(f"{precision:>8}  Recall@{k} = {recall:.2f}  storage for {len(chunks)} chunks = {storage_kb:.0f} KB")
# --8<-- [end:quantization]


# ======================================================================
# 4. Instruction-tuned embeddings
# ======================================================================
# --8<-- [start:instruction]
def demo_instruction_prefix(chunks: list[str], k: int = 3) -> None:
    """Instruction-tuned embedding models expect a short task prefix at
    inference time — for nomic-embed-text-v1.5 specifically, a DIFFERENT
    prefix for each side: "search_query: " on questions, "search_document: "
    on passages, not the same prefix on both. Compares Recall@k with vs.
    without those prefixes on the same model."""
    questions = [item["question"] for item in EVAL_SET]
    keywords = [item["keyword"] for item in EVAL_SET]
    model = SentenceTransformer(MATRYOSHKA_MODEL, trust_remote_code=True)

    chunk_vecs_plain = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_plain = model.encode(questions, convert_to_numpy=True, show_progress_bar=False)
    recall_plain = recall_at_k(chunk_vecs_plain, query_vecs_plain, chunks, keywords, k=k)

    prefixed_chunks = [NOMIC_DOCUMENT_PREFIX + c for c in chunks]
    prefixed_questions = [NOMIC_QUERY_PREFIX + q for q in questions]
    chunk_vecs_prefixed = model.encode(prefixed_chunks, convert_to_numpy=True, show_progress_bar=False)
    query_vecs_prefixed = model.encode(prefixed_questions, convert_to_numpy=True, show_progress_bar=False)
    recall_prefixed = recall_at_k(chunk_vecs_prefixed, query_vecs_prefixed, chunks, keywords, k=k)

    print(f"No prefixes (plain text both sides):        Recall@{k} = {recall_plain:.2f}")
    print(f"Correct prefixes (search_query/document):   Recall@{k} = {recall_prefixed:.2f}")
# --8<-- [end:instruction]


# ======================================================================
# 5. BGE-M3 — dense + sparse + multi-vector from one model
# ======================================================================
# --8<-- [start:bge_m3]
def demo_bge_m3_multi_functionality(text: str) -> None:
    """One model, one encode() call, three different retrieval
    representations: a dense vector (semantic similarity), sparse
    lexical weights (keyword-style, BM25-like matching with learned
    weights instead of statistics), and ColBERT-style per-token vectors
    (late-interaction / multi-vector retrieval)."""
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(HYBRID_MODEL, use_fp16=False)
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


# ======================================================================
# CLI
# ======================================================================

DEMOS = ["mismatched", "matryoshka", "quantization", "instruction", "bge_m3"]


def run(demo: str) -> None:
    if demo == "bge_m3":
        text = load_sample_text()[:1500]
        demo_bge_m3_multi_functionality(text)
        return

    chunks = chunk_sample_text()

    if demo == "mismatched":
        demo_mismatched_models(chunks)
    elif demo == "matryoshka":
        demo_matryoshka_truncation(chunks)
    elif demo == "quantization":
        demo_quantization(chunks)
    elif demo == "instruction":
        demo_instruction_prefix(chunks)
    else:
        raise ValueError(f"Unknown demo: {demo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embedding model selection demos.")
    parser.add_argument("--demo", choices=DEMOS + ["all"], default="mismatched")
    args = parser.parse_args()

    if args.demo == "all":
        for d in DEMOS:
            print(f"\n{'=' * 60}\n{d}\n{'=' * 60}")
            run(d)
    else:
        run(args.demo)

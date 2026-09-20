"""
A capstone tutorial tying together what the rest of this repo covers
separately: a genuinely multi-document corpus (three real papers, not
one toy PDF), a chunking-strategy comparison, a dense-vs-hybrid
retrieval comparison, indexing the winner into a real vector database
(Qdrant) with metadata filtering, and a final RAGAS evaluation of the
assembled pipeline -- all real, all measured.
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from download_data import DATA_DIR, download_papers

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Real, verified cross-document eval set -- three questions per paper,
# each keyword confirmed present in that paper's real extracted text.
EVAL_SET = [
    {"question": "How many attention heads does the base Transformer model use?", "keyword": "h = 8", "doc": "attention_is_all_you_need.pdf", "reference": "The base Transformer model uses 8 attention heads (h = 8)."},
    {"question": "What optimizer was used to train the Transformer?", "keyword": "Adam", "doc": "attention_is_all_you_need.pdf", "reference": "The Transformer was trained with the Adam optimizer."},
    {"question": "What BLEU score did the Transformer achieve on English-to-German translation?", "keyword": "28.4", "doc": "attention_is_all_you_need.pdf", "reference": "The big Transformer model achieved a BLEU score of 28.4 on English-to-German translation."},
    {"question": "How many layers does BERT-base have?", "keyword": "L=12", "doc": "bert.pdf", "reference": "BERT-base has 12 layers (L=12)."},
    {"question": "What are BERT's two pretraining tasks called?", "keyword": "Masked LM", "doc": "bert.pdf", "reference": "BERT is pretrained on Masked LM (masked language modeling) and Next Sentence Prediction."},
    {"question": "What tokenization approach does BERT use?", "keyword": "WordPiece", "doc": "bert.pdf", "reference": "BERT uses WordPiece embeddings with a 30,000 token vocabulary."},
    {"question": "How many parameters does the full-size GPT-3 model have?", "keyword": "175 billion", "doc": "gpt3.pdf", "reference": "The full-size GPT-3 model has 175 billion parameters."},
    {"question": "What term does the GPT-3 paper use for adapting to a task without gradient updates?", "keyword": "in-context learning", "doc": "gpt3.pdf", "reference": "The paper calls this \"in-context learning.\""},
    {"question": "What is GPT-3's context window size in tokens?", "keyword": "2048", "doc": "gpt3.pdf", "reference": "GPT-3's context window is 2048 tokens."},
]


# ======================================================================
# Multi-document corpus with three chunking strategies -- every chunk
# keeps its real source document name and page number as metadata.
# ======================================================================


def load_documents() -> dict[str, str]:
    """Returns {filename: full_text}, keeping page boundaries as \\f
    (form feed) markers so page number can still be recovered."""
    from pypdf import PdfReader

    paths = download_papers()
    docs = {}
    for filename, path in paths.items():
        reader = PdfReader(str(path))
        docs[filename] = "\f".join(page.extract_text() or "" for page in reader.pages)
    return docs


def _pages(docs: dict[str, str]) -> list[tuple[str, int, str]]:
    """Yields (doc_name, page_number, page_text) -- chunking per page
    (rather than guessing offsets in a merged string) makes page
    attribution exact by construction instead of something to get
    subtly wrong later."""
    out = []
    for doc_name, text in docs.items():
        for page_num, page_text in enumerate(text.split("\f"), start=1):
            out.append((doc_name, page_num, page_text))
    return out


def chunk_fixed_size(docs: dict[str, str], size: int = 400, overlap: int = 50) -> list[dict]:
    chunks = []
    chunk_id = 0
    for doc_name, page_num, page_text in _pages(docs):
        for start in range(0, len(page_text), size - overlap):
            piece = page_text[start:start + size].strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks


def chunk_recursive(docs: dict[str, str], size: int = 400, overlap: int = 50) -> list[dict]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, separators=["\n\n", "\n", ". ", " ", ""])
    chunks = []
    chunk_id = 0
    for doc_name, page_num, page_text in _pages(docs):
        for piece in splitter.split_text(page_text):
            if len(piece.strip()) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece.strip(), "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks


def chunk_sentence_window(docs: dict[str, str], sentences_per_chunk: int = 5, overlap_sentences: int = 1) -> list[dict]:
    """A simple sentence-aware strategy: group N sentences per chunk
    with a 1-sentence overlap, instead of splitting mid-sentence at a
    fixed character count -- a cheap proxy for "semantic" chunking
    without needing an embedding-based splitter."""
    chunks = []
    chunk_id = 0
    step = max(1, sentences_per_chunk - overlap_sentences)
    for doc_name, page_num, page_text in _pages(docs):
        sentences = re.split(r"(?<=[.!?])\s+", page_text)
        for start in range(0, len(sentences), step):
            piece = " ".join(sentences[start:start + sentences_per_chunk]).strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "doc": doc_name, "page": page_num})
            chunk_id += 1
    return chunks


CHUNKERS = {"fixed": chunk_fixed_size, "recursive": chunk_recursive, "sentence": chunk_sentence_window}


# ======================================================================
# Part 1: chunking-strategy comparison -- ground truth built from
# EVAL_SET's real keywords (which chunk actually contains the answer),
# same pattern used elsewhere in this repo, now against a genuinely
# multi-document corpus where a chunk's wrong-document attribution is
# also a real failure mode to catch.
# ======================================================================


def build_relevance_set(chunks: list[dict], keyword: str, doc: str) -> set[int]:
    return {c["id"] for c in chunks if keyword.lower() in c["text"].lower() and c["doc"] == doc}


def dense_retrieve(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer, query: str, k: int) -> list[int]:
    query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    sims = embeddings @ query_vec
    top_k = np.argsort(-sims)[:k]
    return [chunks[i]["id"] for i in top_k]


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    return sum(1 for cid in retrieved[:k] if cid in relevant) / len(relevant)


def hit_rate_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    return 1.0 if any(cid in relevant for cid in retrieved[:k]) else 0.0


def evaluate_chunking_strategy(strategy: str, model: SentenceTransformer, k: int = 3) -> dict:
    docs = load_documents()
    chunks = CHUNKERS[strategy](docs)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    recalls, hits = [], []
    for item in EVAL_SET:
        relevant = build_relevance_set(chunks, item["keyword"], item["doc"])
        retrieved = dense_retrieve(chunks, embeddings, model, item["question"], k=max(k, 5))
        recalls.append(recall_at_k(retrieved, relevant, k))
        hits.append(hit_rate_at_k(retrieved, relevant, k))
    return {"strategy": strategy, "num_chunks": len(chunks), "recall_at_k": float(np.mean(recalls)), "hit_rate_at_k": float(np.mean(hits))}


def run_chunking_comparison(model: SentenceTransformer, k: int = 3) -> None:
    print(f"\n--- Chunking strategy comparison (Recall@{k}, Hit Rate@{k}, {len(EVAL_SET)} cross-document questions) ---")
    for strategy in CHUNKERS:
        result = evaluate_chunking_strategy(strategy, model, k=k)
        print(f"  {result['strategy']:<10} chunks={result['num_chunks']:>5}  recall@{k}={result['recall_at_k']:.3f}  hit_rate@{k}={result['hit_rate_at_k']:.3f}")


# ======================================================================
# Part 2: dense-only vs. hybrid (BM25 + dense + RRF) retrieval, on the
# winning chunking strategy from Part 1.
# ======================================================================


def hybrid_rrf_fusion(bm25_ranks: list[int], dense_ranks: list[int], k_const: int = 60) -> list[int]:
    scores: dict[int, float] = {}
    for rank, cid in enumerate(bm25_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
    for rank, cid in enumerate(dense_ranks, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k_const + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def run_retrieval_comparison(model: SentenceTransformer, chunking_strategy: str = "sentence", k: int = 3) -> None:
    from rank_bm25 import BM25Okapi

    docs = load_documents()
    chunks = CHUNKERS[chunking_strategy](docs)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    bm25 = BM25Okapi([t.lower().split() for t in texts])

    dense_recalls, dense_hits, hybrid_recalls, hybrid_hits = [], [], [], []
    for item in EVAL_SET:
        relevant = build_relevance_set(chunks, item["keyword"], item["doc"])

        dense_ranked = dense_retrieve(chunks, embeddings, model, item["question"], k=len(chunks))
        dense_recalls.append(recall_at_k(dense_ranked, relevant, k))
        dense_hits.append(hit_rate_at_k(dense_ranked, relevant, k))

        bm25_scores = bm25.get_scores(item["question"].lower().split())
        bm25_ranked_idx = np.argsort(-bm25_scores)
        bm25_ranked = [chunks[i]["id"] for i in bm25_ranked_idx]

        fused = hybrid_rrf_fusion(bm25_ranked, dense_ranked)
        hybrid_recalls.append(recall_at_k(fused, relevant, k))
        hybrid_hits.append(hit_rate_at_k(fused, relevant, k))

    print(f"\n--- Dense-only vs. hybrid (BM25+RRF) retrieval, {chunking_strategy} chunking, Recall@{k}/Hit Rate@{k} ---")
    print(f"  dense-only  recall@{k}={np.mean(dense_recalls):.3f}  hit_rate@{k}={np.mean(dense_hits):.3f}")
    print(f"  hybrid+RRF  recall@{k}={np.mean(hybrid_recalls):.3f}  hit_rate@{k}={np.mean(hybrid_hits):.3f}")


# ======================================================================
# Part 3: index the winning setup (sentence-window chunking, dense
# retrieval) into a real vector database -- Qdrant, embedded, with
# metadata filtering by source document.
# ======================================================================


def run_qdrant_indexing(model: SentenceTransformer, chunking_strategy: str = "sentence") -> None:
    import shutil

    from qdrant_client import QdrantClient, models

    docs = load_documents()
    chunks = CHUNKERS[chunking_strategy](docs)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    qdrant_dir = Path(__file__).parent / "data" / "qdrant_db"
    shutil.rmtree(qdrant_dir, ignore_errors=True)
    client = QdrantClient(path=str(qdrant_dir))

    dim = embeddings.shape[1]
    client.create_collection(
        collection_name="multi_doc_corpus",
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE, hnsw_config=models.HnswConfigDiff(m=32, ef_construct=200)),
    )
    client.upsert(
        collection_name="multi_doc_corpus",
        points=[
            models.PointStruct(id=c["id"], vector=embeddings[i].tolist(), payload={"doc": c["doc"], "page": c["page"], "text": c["text"]})
            for i, c in enumerate(chunks)
        ],
    )

    query = "How many layers does the model have?"
    query_vec = model.encode(query, normalize_embeddings=True).tolist()

    print(f"\n--- Qdrant: unfiltered top-3 for {query!r} (indexed {len(chunks)} chunks across {len(docs)} docs) ---")
    result = client.query_points(collection_name="multi_doc_corpus", query=query_vec, limit=3)
    for point in result.points:
        print(f"  {point.payload['doc']:<30} page {point.payload['page']:>2}  score={point.score:.3f}")

    print(f"--- Qdrant: filtered to doc='bert.pdf' only, top-3 for {query!r} ---")
    result = client.query_points(
        collection_name="multi_doc_corpus", query=query_vec, limit=3,
        query_filter=models.Filter(must=[models.FieldCondition(key="doc", match=models.MatchValue(value="bert.pdf"))]),
    )
    for point in result.points:
        print(f"  {point.payload['doc']:<30} page {point.payload['page']:>2}  score={point.score:.3f}")


# ======================================================================
# Part 4: RAGAS evaluation of the assembled pipeline -- winning
# chunking strategy, dense retrieval, real generation.
# ======================================================================


def run_naive_rag(model: SentenceTransformer, chunking_strategy: str = "sentence", provider: str | None = None) -> list[dict]:
    import llm

    docs = load_documents()
    chunks = CHUNKERS[chunking_strategy](docs)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    samples = []
    for item in EVAL_SET:
        retrieved_ids = dense_retrieve(chunks, embeddings, model, item["question"], k=3)
        by_id = {c["id"]: c for c in chunks}
        contexts = [by_id[cid]["text"] for cid in retrieved_ids]
        context_block = "\n---\n".join(contexts)
        answer = llm.generate(f"Answer the question using the context below, in one short sentence.\n\nContext:\n{context_block}\n\nQuestion: {item['question']}", provider=provider)
        samples.append({"question": item["question"], "answer": answer.strip(), "retrieved_contexts": contexts, "reference": item["reference"]})
    return samples


def run_ragas_eval(samples: list[dict]):
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall, ResponseRelevancy

    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-5"), bypass_temperature=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=f"sentence-transformers/{TEXT_EMBEDDING_MODEL}"))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(user_input=s["question"], response=s["answer"], retrieved_contexts=s["retrieved_contexts"], reference=s["reference"])
        for s in samples
    ])
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]
    return evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=4, timeout=180))


def run_full_pipeline_eval(model: SentenceTransformer, provider: str | None = None) -> None:
    print(f"\n--- Full pipeline: naive RAG over {len(EVAL_SET)} multi-document questions ---")
    samples = run_naive_rag(model, provider=provider)
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:150]}")
    print("\n--- RAGAS scores ---")
    result = run_ragas_eval(samples)
    df = result.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]].to_string())
    print("\nAggregate:", result)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")

    parser = argparse.ArgumentParser(description="Multi-document production RAG capstone tutorial.")
    parser.add_argument("--part", choices=["chunking", "retrieval", "qdrant", "ragas", "all"], default="all")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()

    docs = load_documents()
    print(f"Loaded {len(docs)} documents: {list(docs.keys())}")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

    if args.part in ("chunking", "all"):
        run_chunking_comparison(model)
    if args.part in ("retrieval", "all"):
        run_retrieval_comparison(model)
    if args.part in ("qdrant", "all"):
        run_qdrant_indexing(model)
    if args.part in ("ragas", "all"):
        run_full_pipeline_eval(model, provider=args.provider)

"""
Evaluation metrics: retrieval-quality metrics (Precision@k, Recall@k, MRR,
NDCG, Hit Rate) computed against real ground-truth relevance, and
generation-quality metrics (BLEU, ROUGE-L, METEOR, embedding-based
semantic similarity) computed against real locally-generated answers --
no paid API needed, generation runs through a local Ollama model.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"

# Same standard eval set used throughout this repo. `keyword` is a
# substring that only appears on the page(s) that actually answer the
# question -- used below to build real, verifiable ground-truth
# relevance sets, not hand-picked page numbers.
EVAL_SET = [
    {"question": "How many attention heads did they use?", "reference": "The base model used 8 attention heads (h = 8).", "keyword": "h = 8"},
    {"question": "What is the model's embedding dimension?", "reference": "The base model's embedding dimension is 512 (dmodel = 512).", "keyword": "dmodel"},
    {"question": "How many layers are in the encoder?", "reference": "The encoder has 6 layers (N = 6).", "keyword": "N = 6"},
    {"question": "What optimizer was used for training?", "reference": "They used the Adam optimizer.", "keyword": "Adam"},
    {"question": "What BLEU score did they get on English-to-German translation?", "reference": "The big model achieved a BLEU score of 28.4 on English-to-German translation.", "keyword": "28.4"},
    {"question": "What GPUs was the model trained on?", "reference": "The models were trained on NVIDIA P100 GPUs.", "keyword": "P100"},
    {"question": "How long did the base model train for?", "reference": "The base model trained for 12 hours.", "keyword": "12 hours"},
    {"question": "What dropout rate did they use?", "reference": "They used a dropout rate of Pdrop = 0.1.", "keyword": "Pdrop"},
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


def build_corpus() -> list[dict]:
    """Same chunking pattern used across this repo's other recipes --
    each chunk keeps the real page number it came from."""
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))

    chunks = []
    chunk_id = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            piece = text[start:start + CHUNK_SIZE].strip()
            if len(piece) < 40:
                continue
            chunks.append({"id": chunk_id, "text": piece, "page": page_num})
            chunk_id += 1
    return chunks


def ollama_generate(prompt: str, model: str = OLLAMA_MODEL) -> str:
    import requests

    response = requests.post(OLLAMA_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=180)
    response.raise_for_status()
    return response.json()["response"]


def run_naive_rag(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer) -> list[dict]:
    """Real naive RAG over EVAL_SET, generated locally via Ollama -- no
    paid API involved, so classic-metric behavior can be shown on real
    (not scripted) model output."""
    samples = []
    for item in EVAL_SET:
        retrieved_ids = dense_retrieve(chunks, embeddings, model, item["question"], k=3)
        by_id = {c["id"]: c for c in chunks}
        context_block = "\n---\n".join(by_id[cid]["text"] for cid in retrieved_ids)
        answer = ollama_generate(f"Answer the question using only the context below, in one short sentence.\n\nContext:\n{context_block}\n\nQuestion: {item['question']}")
        samples.append({"question": item["question"], "answer": answer.strip(), "reference": item["reference"]})
    return samples


# ======================================================================
# Retrieval metrics -- Precision@k, Recall@k, MRR, NDCG, Hit Rate.
# Ground truth is built from `EVAL_SET`'s `keyword` field: the chunks
# that actually contain the keyword are the real relevant set for that
# question, not hand-picked page numbers.
# ======================================================================


def build_relevance_sets(chunks: list[dict]) -> list[set[int]]:
    """Returns, for each EVAL_SET question, the set of chunk ids whose
    text actually contains that question's answer keyword."""
    relevance_sets = []
    for item in EVAL_SET:
        relevant = {c["id"] for c in chunks if item["keyword"].lower() in c["text"].lower()}
        relevance_sets.append(relevant)
    return relevance_sets


def dense_retrieve(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer, query: str, k: int) -> list[int]:
    query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    sims = embeddings @ query_vec
    top_k = np.argsort(-sims)[:k]
    return [chunks[i]["id"] for i in top_k]


def precision_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = retrieved[:k]
    return sum(1 for cid in top_k if cid in relevant) / k


def recall_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for cid in top_k if cid in relevant) / len(relevant)


def reciprocal_rank(retrieved: list[int], relevant: set[int]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    """Binary relevance NDCG@k: gain=1 for a relevant hit, discounted
    by log2(rank + 1); normalized by the ideal ordering (all relevant
    hits packed at the top)."""
    top_k = retrieved[:k]
    dcg = sum(1.0 / math.log2(rank + 1) for rank, cid in enumerate(top_k, start=1) if cid in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(retrieved: list[int], relevant: set[int], k: int) -> float:
    top_k = retrieved[:k]
    return 1.0 if any(cid in relevant for cid in top_k) else 0.0


def run_retrieval_metrics(chunks: list[dict], embeddings: np.ndarray, model: SentenceTransformer, k: int = 3) -> None:
    relevance_sets = build_relevance_sets(chunks)

    precisions, recalls, rrs, ndcgs, hits = [], [], [], [], []
    for item, relevant in zip(EVAL_SET, relevance_sets):
        retrieved = dense_retrieve(chunks, embeddings, model, item["question"], k=max(k, 5))
        precisions.append(precision_at_k(retrieved, relevant, k))
        recalls.append(recall_at_k(retrieved, relevant, k))
        rrs.append(reciprocal_rank(retrieved, relevant))
        ndcgs.append(ndcg_at_k(retrieved, relevant, k))
        hits.append(hit_rate_at_k(retrieved, relevant, k))
        print(f"  Q: {item['question'][:60]:<60} relevant_chunks={len(relevant):>2}  P@{k}={precisions[-1]:.2f}  R@{k}={recalls[-1]:.2f}  RR={rrs[-1]:.2f}")

    print(f"\nAggregate over {len(EVAL_SET)} questions:")
    print(f"  Precision@{k}: {np.mean(precisions):.3f}")
    print(f"  Recall@{k}:    {np.mean(recalls):.3f}")
    print(f"  MRR:           {np.mean(rrs):.3f}")
    print(f"  NDCG@{k}:      {np.mean(ndcgs):.3f}")
    print(f"  Hit Rate@{k}:  {np.mean(hits):.3f}")


# ======================================================================
# Generation metrics -- BLEU, ROUGE-L, METEOR (classic lexical-overlap
# metrics) and embedding-based semantic similarity, scored against real
# locally-generated answers.
# ======================================================================


def run_generation_metrics(samples: list[dict], model: SentenceTransformer) -> None:
    import sacrebleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    bleus, rouges, meteors, sims = [], [], [], []
    for s in samples:
        reference, answer = s["reference"], s["answer"]

        bleu = sacrebleu.sentence_bleu(answer, [reference]).score
        rouge_l = scorer.score(reference, answer)["rougeL"].fmeasure
        meteor = meteor_score([reference.split()], answer.split())
        ref_vec = model.encode(reference, normalize_embeddings=True)
        ans_vec = model.encode(answer, normalize_embeddings=True)
        semantic_sim = float(ref_vec @ ans_vec)

        bleus.append(bleu)
        rouges.append(rouge_l)
        meteors.append(meteor)
        sims.append(semantic_sim)

        print(f"\nQ: {s['question']}")
        print(f"  Reference: {reference}")
        print(f"  Answer:    {answer[:150]}")
        print(f"  BLEU={bleu:.1f}  ROUGE-L={rouge_l:.3f}  METEOR={meteor:.3f}  SemanticSim={semantic_sim:.3f}")

    print(f"\nAggregate over {len(samples)} questions:")
    print(f"  BLEU:        {np.mean(bleus):.1f}")
    print(f"  ROUGE-L:     {np.mean(rouges):.3f}")
    print(f"  METEOR:      {np.mean(meteors):.3f}")
    print(f"  SemanticSim: {np.mean(sims):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation metrics demo.")
    parser.add_argument("--part", choices=["retrieval", "generation", "all"], default="all")
    args = parser.parse_args()

    chunks = build_corpus()
    print(f"Corpus: {len(chunks)} chunks across {len({c['page'] for c in chunks})} pages")
    model = SentenceTransformer(TEXT_EMBEDDING_MODEL)

    if args.part in ("retrieval", "all"):
        texts = [c["text"] for c in chunks]
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        print("\n=== Retrieval metrics ===")
        run_retrieval_metrics(chunks, embeddings, model, k=3)

    if args.part in ("generation", "all"):
        if args.part == "generation":
            texts = [c["text"] for c in chunks]
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True)
        print("\n=== Generation metrics ===")
        samples = run_naive_rag(chunks, embeddings, model)
        run_generation_metrics(samples, model)

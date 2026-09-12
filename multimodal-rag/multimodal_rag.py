"""
Multimodal RAG against real data: the sample paper's own 3 embedded
figures (the architecture diagram, scaled dot-product attention,
multi-head attention) plus its text chunks, retrieved two ways —

1. unified: everything (text + images) embedded into ONE shared CLIP
   space, searched with a single similarity ranking
2. separate + fusion: text goes through a dedicated text embedder,
   images through CLIP, each ranked independently, then combined with
   RRF (same fusion used for multi-query/RAG-Fusion elsewhere in this
   repo)

Measures the real trade-off honestly: does putting everything in one
shared space actually cost you anything?

Run:
    python multimodal_rag.py --mode unified --query "..."
    python multimodal_rag.py --mode fused --query "..."
    python multimodal_rag.py --compare-text-quality
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from sentence_transformers import SentenceTransformer

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLIP_MODEL_NAME = "clip-ViT-B-32"

# Same 8 question/keyword pairs used across every other recipe in this
# repo — reused here, not re-invented, to measure text-retrieval
# quality under each indexing approach.
TEXT_EVAL_SET = [
    ("How many attention heads did they use?", "h = 8"),
    ("What is the model's embedding dimension?", "dmodel = 512"),
    ("How many layers are in the encoder?", "N = 6"),
    ("What optimizer was used for training?", "Adam"),
    ("What BLEU score did they get on English-to-German translation?", "28.4"),
    ("What GPUs was the model trained on?", "P100"),
    ("How long did the base model train for?", "12 hours"),
    ("What dropout rate did they use?", "Pdrop = 0.1"),
]


# ======================================================================
# Setup / data loading
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    abstract_start = full_text.find("Abstract")
    return full_text[abstract_start:] if abstract_start != -1 else full_text


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


# ======================================================================
# 1. Extract the paper's own real embedded figures — not stock images
# ======================================================================

IMAGE_LABELS = {
    "page2_Im1": "Figure 1: the overall Transformer encoder-decoder architecture diagram",
    "page3_Im2": "Scaled Dot-Product Attention diagram (Q, K, V, MatMul, Scale, Mask, SoftMax)",
    "page3_Im3": "Multi-Head Attention diagram (multiple parallel attention heads, Linear, Concat)",
}


def extract_images_from_pdf() -> dict[str, Image.Image]:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    reader = PdfReader(str(pdf_path))

    images = {}
    for page_idx, page in enumerate(reader.pages):
        for img in page.images:
            key = f"page{page_idx}_{Path(img.name).stem}"
            images[key] = img.image
    return images


# ======================================================================
# 2. Unified embedding space — everything in CLIP
# ======================================================================


def build_unified_index(chunks: list[str], images: dict[str, Image.Image], clip_model: SentenceTransformer) -> dict:
    text_embs = clip_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    image_names = list(images.keys())
    image_embs = clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False)
    return {"chunks": chunks, "chunk_embs": text_embs, "image_names": image_names, "image_embs": image_embs}


def unified_search(query: str, index: dict, clip_model: SentenceTransformer, k: int = 3) -> list[tuple[str, str]]:
    """Single ranked list mixing text chunks and images — returns
    (kind, content) pairs, kind is 'text' or 'image'."""
    query_vec = clip_model.encode(query, convert_to_numpy=True)

    candidates = [("text", c) for c in index["chunks"]] + [("image", n) for n in index["image_names"]]
    all_embs = np.vstack([index["chunk_embs"], index["image_embs"]])

    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in all_embs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [candidates[i] for i in top_indices]


# ======================================================================
# 3. Separate indexes + RRF fusion — dedicated text embedder for text,
#    CLIP only for images
# ======================================================================


def build_separate_indexes(chunks: list[str], images: dict[str, Image.Image], text_model: SentenceTransformer, clip_model: SentenceTransformer) -> dict:
    return {
        "chunks": chunks,
        "chunk_embs": text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False),
        "image_names": list(images.keys()),
        "image_embs": clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False),
    }


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
    """Same RRF used for multi-query/RAG-Fusion elsewhere in this
    repo — rank-based, not raw-score-based, which matters here since
    CLIP's image-similarity scores and the text embedder's
    similarity scores come from two different, uncalibrated spaces
    and can't be compared directly.

    A real, non-obvious consequence: EVERY query produces an exact
    tie at rank 0 between modalities — `1/(k+0+1)` is identical
    whether a text chunk or an image sits in that slot, regardless of
    how relevant either actually is. This isn't a bug to patch with a
    tiebreaker (a first attempt at one — normalizing each candidate's
    raw similarity within its own list — didn't work: the top item in
    ANY list normalizes to exactly 1.0 by construction, so the tie
    persists no matter what). It's a real, structural property of
    rank-based fusion: RRF is a CONSENSUS mechanism, not a magnitude
    comparator, and it has no principled way to let one modality's
    confident top pick outright beat another modality's merely-present
    top pick. See multimodal_rag.py's `unified_search` for how to get
    a genuine image-only result instead — a single flat ranking has
    no second list to tie against."""
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])


# RRF has no concept of absolute relevance — it only knows RANK within
# a list. With a small image pool (3 images here), whichever one
# scores even marginally highest still becomes "rank 1" and earns a
# real fusion boost, even when every image is genuinely irrelevant to
# the query (verified: all 3 scored ~0.22-0.23 — noise, not signal —
# for a plain-text optimizer question, and the highest of the three
# still got fused into the top-3 ahead of a relevant text chunk).
# A minimum similarity floor is the fix: don't let a modality
# contribute to fusion at all if its best candidate doesn't clear a
# real relevance bar.
CLIP_RELEVANCE_THRESHOLD = 0.28


def fused_search(query: str, index: dict, text_model: SentenceTransformer, clip_model: SentenceTransformer, k: int = 3) -> list[tuple[str, str]]:
    text_query_vec = text_model.encode(query, convert_to_numpy=True)
    text_sims = [float(np.dot(text_query_vec, v) / (np.linalg.norm(text_query_vec) * np.linalg.norm(v) + 1e-8)) for v in index["chunk_embs"]]
    text_ranked = [("text", index["chunks"][i]) for i in np.argsort(text_sims)[::-1]]

    clip_query_vec = clip_model.encode(query, convert_to_numpy=True)
    image_sims = [float(np.dot(clip_query_vec, v) / (np.linalg.norm(clip_query_vec) * np.linalg.norm(v) + 1e-8)) for v in index["image_embs"]]
    ranked_lists = [text_ranked]
    if max(image_sims, default=0.0) >= CLIP_RELEVANCE_THRESHOLD:
        image_order = np.argsort(image_sims)[::-1]
        ranked_lists.append([("image", index["image_names"][i]) for i in image_order if image_sims[i] >= CLIP_RELEVANCE_THRESHOLD])

    fused = reciprocal_rank_fusion(ranked_lists)
    return fused[:k]


# ======================================================================
# 4. Measuring the real trade-off: does a unified space hurt
#    plain-text retrieval quality?
# ======================================================================


def recall_at_k(chunks: list[str], embed_model: SentenceTransformer, k: int = 3) -> float:
    chunk_embs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    hits = 0
    for question, keyword in TEXT_EVAL_SET:
        query_vec = embed_model.encode(question, convert_to_numpy=True)
        sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_embs]
        top_indices = np.argsort(sims)[::-1][:k]
        retrieved = " ".join(chunks[i] for i in top_indices)
        hits += keyword in retrieved
    return hits / len(TEXT_EVAL_SET)


def run_text_quality_comparison() -> None:
    load_environment()
    chunks = chunk_sample_text()

    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    dedicated_recall = recall_at_k(chunks, text_model)
    print(f"Dedicated text embedder ({TEXT_EMBEDDING_MODEL}) Recall@3: {dedicated_recall:.3f}")

    clip_model = SentenceTransformer(CLIP_MODEL_NAME)
    clip_recall = recall_at_k(chunks, clip_model)
    print(f"CLIP's text encoder ({CLIP_MODEL_NAME}) Recall@3: {clip_recall:.3f}")
    print(f"\nUsing CLIP for text retrieval costs {(dedicated_recall - clip_recall) * 100:.0f} points of Recall@3 "
          f"on the exact same 8 factual questions, on the exact same corpus.")


# ======================================================================
# CLI
# ======================================================================


def run_query(mode: str, question: str, provider: str | None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    images = extract_images_from_pdf()
    clip_model = SentenceTransformer(CLIP_MODEL_NAME)

    if mode == "unified":
        index = build_unified_index(chunks, images, clip_model)
        results = unified_search(question, index, clip_model)
    else:
        text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        index = build_separate_indexes(chunks, images, text_model, clip_model)
        results = fused_search(question, index, text_model, clip_model)

    print(f"Question: {question}\n")
    for kind, content in results:
        if kind == "image":
            print(f"[IMAGE] {IMAGE_LABELS.get(content, content)}")
        else:
            print(f"[TEXT]  {content[:200]}")
        print()

    context_parts = [
        IMAGE_LABELS.get(content, content) if kind == "image" else content for kind, content in results
    ]
    answer = llm.generate(
        f"Answer the question using the context below. Some context items describe a "
        f"figure/diagram rather than quoting text directly — treat those as a description "
        f"of what that figure shows.\n\nContext:\n" + "\n---\n".join(context_parts) +
        f"\n\nQuestion: {question}",
        provider=provider,
    )
    print(f"Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal RAG demo.")
    parser.add_argument("--mode", choices=["unified", "fused"], default="fused")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--compare-text-quality", action="store_true", help="Measure the real cost of a unified CLIP space on plain text retrieval.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()

    if args.compare_text_quality:
        run_text_quality_comparison()
    elif args.query:
        run_query(args.mode, args.query, args.provider)
    else:
        parser.error('Pass --query "..." or --compare-text-quality.')

"""
Documentation-only companion to multimodal_rag.py.

Not run top-to-bottom, not imported by anything. Every section below is
self-contained — its own imports, its own inlined helpers — so
copy-pasting any single section into a fresh .py works on its own. See
multimodal_rag.py for the actual runnable CLI.

If you change something in multimodal_rag.py, mirror it here too —
these are NOT kept in sync automatically.
"""


# --8<-- [start:extraction]
from pathlib import Path

from PIL import Image


def extract_images_from_pdf(pdf_path: str) -> dict[str, Image.Image]:
    """Pulls the paper's own real embedded figures out of the PDF —
    not stock images, the actual diagrams the authors put in the
    document."""
    from pypdf import PdfReader

    reader = PdfReader(pdf_path)
    images = {}
    for page_idx, page in enumerate(reader.pages):
        for img in page.images:
            key = f"page{page_idx}_{Path(img.name).stem}"
            images[key] = img.image
    return images
# --8<-- [end:extraction]


# --8<-- [start:unified]
import numpy as np
from sentence_transformers import SentenceTransformer


def build_unified_index(chunks: list[str], images: dict, clip_model: SentenceTransformer) -> dict:
    """Everything — text chunks AND images — embedded into ONE shared
    CLIP space. Enables genuine cross-modal search (a text query can
    directly retrieve an image), at a real cost: CLIP's text encoder
    is a much weaker text retriever than a dedicated text embedding
    model (measured: ~1.00 -> ~0.38 Recall@3 on the same 8 factual
    questions, same corpus — see multimodal_rag.py's
    --compare-text-quality)."""
    text_embs = clip_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    image_names = list(images.keys())
    image_embs = clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False)
    return {"chunks": chunks, "chunk_embs": text_embs, "image_names": image_names, "image_embs": image_embs}


def unified_search(query: str, index: dict, clip_model: SentenceTransformer, k: int = 3) -> list[tuple[str, str]]:
    query_vec = clip_model.encode(query, convert_to_numpy=True)
    candidates = [("text", c) for c in index["chunks"]] + [("image", n) for n in index["image_names"]]
    all_embs = np.vstack([index["chunk_embs"], index["image_embs"]])
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in all_embs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [candidates[i] for i in top_indices]
# --8<-- [end:unified]


# --8<-- [start:separate_fusion]
import numpy as np
from sentence_transformers import SentenceTransformer

# RRF has no concept of absolute relevance — it only knows RANK within
# a list. With a small image pool, whichever image scores even
# marginally highest still becomes "rank 1" and earns a real fusion
# boost, even when every image is genuinely irrelevant to the query
# (verified directly: all 3 images scored ~0.22-0.23 — noise, not
# signal — for a plain-text question, and the highest of the three
# still got fused ahead of a relevant text chunk, before this
# threshold was added). A minimum similarity floor fixes it: don't
# let a modality contribute to fusion at all if its best candidate
# doesn't clear a real relevance bar.
CLIP_RELEVANCE_THRESHOLD = 0.28


def build_separate_indexes(chunks: list[str], images: dict, text_model: SentenceTransformer, clip_model: SentenceTransformer) -> dict:
    """Text goes through a DEDICATED text embedder (full retrieval
    quality preserved); images go through CLIP (the only place CLIP
    is actually needed). Two separate, uncalibrated embedding spaces —
    which is exactly why fusion below uses RRF (rank-based) rather
    than comparing raw similarity scores directly."""
    return {
        "chunks": chunks,
        "chunk_embs": text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False),
        "image_names": list(images.keys()),
        "image_embs": clip_model.encode(list(images.values()), convert_to_numpy=True, show_progress_bar=False),
    }


def reciprocal_rank_fusion(ranked_lists: list[list], k: int = 60) -> list:
    scores: dict = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda item: -scores[item])


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
# --8<-- [end:separate_fusion]

"""
Vision RAG: retrieve whole PDF pages as images, with no OCR or text
extraction anywhere in the pipeline, and generate answers from a
vision-capable LLM reading the retrieved page image directly.

Contrast with the Multimodal RAG recipe in this repo: that one extracts
individual figures out of a PDF and retrieves text chunks + those figures
separately (or in one CLIP space). This one never extracts text at all —
each full page is rendered as a single image, embedded with a ColPali-style
late-interaction vision retriever, and the page image itself (not a caption
or OCR'd text) is what the LLM sees at generation time. That matters most
on documents where meaning lives in layout — tables, charts, multi-column
figures — which text extraction can silently scramble.

Two approaches are built and compared:
1. Naive text RAG — the same pattern used across this repo: extract text,
   chunk it, embed chunks with a dedicated text embedder.
2. Vision RAG — render each page as an image, embed pages with
   `vidore/ColSmolVLM-256M-base` (a small ColPali-family model), retrieve
   with late-interaction (MaxSim) scoring, and generate from the raw page
   image.
"""

import argparse
import base64
import io
import json
import shutil
from pathlib import Path

import fitz  # pymupdf
import numpy as np
import torch
from huggingface_hub import snapshot_download
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from sentence_transformers import SentenceTransformer, util

import llm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# vidore/ColSmolVLM-256M-base — the lightest model in the ColPali/ColVision
# family with a usable ViDoRe benchmark score (~80 NDCG@5, vs. ~89 for the
# 2-3B leaders like ColQwen2.5), chosen specifically because it's the one
# ColVision checkpoint plausibly runnable on an 8GB machine without a GPU.
# ColQwen2/2.5 (2-3B params) realistically need a GPU.
VISION_MODEL_REPO = "vidore/ColSmolVLM-256M-base"
VISION_MODEL_BASE = "HuggingFaceTB/SmolVLM-256M-Base"
PATCHED_MODEL_DIR = Path(__file__).parent / ".vision_model_cache"

PAGE_DPI = 150

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
# 0. Setup
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def download_sample_pdf(url: str = DEFAULT_PDF_URL, dest: Path = DEFAULT_PDF_PATH) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_sample_text() -> str:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    full_text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
    abstract_start = full_text.find("Abstract")
    return full_text[abstract_start:] if abstract_start != -1 else full_text


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


# ======================================================================
# 1. Render PDF pages as images — the only "extraction" step needed
# ======================================================================


def render_pdf_pages(dpi: int = PAGE_DPI) -> list[Image.Image]:
    """Render every page of the sample PDF to a PIL image. No text is
    extracted here — this is the entire input Vision RAG's indexing step
    needs."""
    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        pixmap = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
        pages.append(img)
    return pages


# ======================================================================
# 2. Load the vision retriever
# ======================================================================


def load_vision_model(repo_id: str = VISION_MODEL_REPO):
    """Load a ColPali-family page-image retriever.

    `vidore/ColSmolVLM-256M-base`'s `adapter_config.json` ships with
    `base_model_name_or_path` set to a local training path
    (`./models/ColSmolVLM-256M-Base`) that doesn't exist outside the
    original training run. `transformers` auto-detects the adapter config
    and tries to resolve that path as a Hugging Face repo id, which fails.
    The full merged weights are already present in the repo's
    `model.safetensors`, so the fix is to patch a local copy of
    `adapter_config.json` to point at the real base model
    (`HuggingFaceTB/SmolVLM-256M-Base`, from the model card) and load from
    that local copy.
    """
    from colpali_engine.models import ColIdefics3, ColIdefics3Processor

    snapshot_dir = snapshot_download(repo_id)
    if PATCHED_MODEL_DIR.exists():
        shutil.rmtree(PATCHED_MODEL_DIR)
    shutil.copytree(snapshot_dir, PATCHED_MODEL_DIR)
    for fname in ["adapter_config.json", "adapter2_config.json"]:
        fpath = PATCHED_MODEL_DIR / fname
        if fpath.exists():
            config = json.loads(fpath.read_text())
            config["base_model_name_or_path"] = VISION_MODEL_BASE
            fpath.write_text(json.dumps(config))

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = ColIdefics3.from_pretrained(str(PATCHED_MODEL_DIR), torch_dtype=torch.float32).to(device).eval()
    processor = ColIdefics3Processor.from_pretrained(str(PATCHED_MODEL_DIR))
    return model, processor


# ======================================================================
# 3. Vision RAG — index and search page images directly
# ======================================================================


def build_page_image_index(pages: list[Image.Image], model, processor) -> list[torch.Tensor]:
    """Embed every page as a multi-vector (one embedding per image patch,
    ColBERT-style), not a single pooled vector. Late interaction scoring
    (see `vision_search`) needs the full patch grid, not a mean-pooled
    summary."""
    device = next(model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for page in pages:
            batch = processor.process_images([page]).to(device)
            page_embedding = model(**batch)
            embeddings.append(page_embedding[0].cpu())
    return embeddings


def vision_search(query: str, page_embeddings: list[torch.Tensor], model, processor, k: int = 3) -> list[tuple[int, float]]:
    """Score every page against the query with MaxSim (late interaction):
    for each query token embedding, take the best-matching patch embedding
    on the page, then sum those per-token maxima. Unlike a bi-encoder's
    single dot product, this lets a page win because ONE region of it
    (e.g. one table cell, one line of a chart's legend) matches strongly,
    even if the rest of the page doesn't."""
    device = next(model.parameters()).device
    with torch.no_grad():
        query_batch = processor.process_queries([query]).to(device)
        query_embedding = model(**query_batch)[0].cpu()

    scores = [
        processor.score_multi_vector(query_embedding.unsqueeze(0), page_embedding.unsqueeze(0)).item()
        for page_embedding in page_embeddings
    ]
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


# ======================================================================
# 4. Naive text RAG baseline — same pattern used elsewhere in this repo
# ======================================================================


def build_text_index(chunks: list[str], text_model: SentenceTransformer) -> np.ndarray:
    return text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def text_search(query: str, chunks: list[str], chunk_embeddings: np.ndarray, text_model: SentenceTransformer, k: int = 3) -> list[tuple[str, float]]:
    query_embedding = text_model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(k, len(chunks)))
    return [(chunks[i], sims[i].item()) for i in top_k.indices.tolist()]


# ======================================================================
# 5. Evaluation
# ======================================================================


def recall_at_k_text(chunks: list[str], text_model: SentenceTransformer, k: int = 3) -> float:
    chunk_embeddings = build_text_index(chunks, text_model)
    hits = 0
    for question, keyword in TEXT_EVAL_SET:
        results = text_search(question, chunks, chunk_embeddings, text_model, k=k)
        if any(keyword.lower() in chunk.lower() for chunk, _ in results):
            hits += 1
    return hits / len(TEXT_EVAL_SET)


def recall_at_k_vision(pages: list[Image.Image], page_embeddings: list[torch.Tensor], model, processor, k: int = 3) -> float:
    """Ground truth for whether a retrieved PAGE actually contains the
    answer is checked with a one-off text extraction per page — used only
    to score this evaluation, never fed into the vision retrieval or
    generation pipeline itself, which never sees extracted text."""
    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    doc = fitz.open(str(pdf_path))
    page_texts = [doc[i].get_text() for i in range(len(pages))]

    hits = 0
    for question, keyword in TEXT_EVAL_SET:
        results = vision_search(question, page_embeddings, model, processor, k=k)
        if any(keyword.lower() in page_texts[page_idx].lower() for page_idx, _ in results):
            hits += 1
    return hits / len(TEXT_EVAL_SET)


# ======================================================================
# 6. Query pipeline — generation reads the retrieved page image directly
# ======================================================================


def page_image_to_base64(page: Image.Image) -> str:
    buffer = io.BytesIO()
    page.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def run_query(mode: str, question: str, provider: str | None) -> None:
    load_environment()

    if mode == "vision":
        pages = render_pdf_pages()
        model, processor = load_vision_model()
        page_embeddings = build_page_image_index(pages, model, processor)
        results = vision_search(question, page_embeddings, model, processor, k=1)
        top_page_idx, score = results[0]

        print(f"Question: {question}\n")
        print(f"[PAGE {top_page_idx}] retrieved (MaxSim score: {score:.2f}) — no text extracted\n")

        answer = llm.generate_with_image(
            f"Answer the question using only what's visible in this document page image. "
            f"If the answer is in a table, read the table structure directly from the image.\n\n"
            f"Question: {question}",
            page_image_to_base64(pages[top_page_idx]),
            provider=provider,
        )
        print(f"Answer: {answer}")
    else:
        chunks = chunk_sample_text()
        text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
        chunk_embeddings = build_text_index(chunks, text_model)
        results = text_search(question, chunks, chunk_embeddings, text_model, k=3)

        print(f"Question: {question}\n")
        for chunk, score in results:
            print(f"[TEXT] (score {score:.2f}) {chunk[:150]}")
        print()

        context = "\n---\n".join(chunk for chunk, _ in results)
        answer = llm.generate(
            f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}",
            provider=provider,
        )
        print(f"Answer: {answer}")


def run_comparison() -> None:
    load_environment()

    chunks = chunk_sample_text()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    text_recall = recall_at_k_text(chunks, text_model, k=3)

    pages = render_pdf_pages()
    model, processor = load_vision_model()
    page_embeddings = build_page_image_index(pages, model, processor)
    vision_recall = recall_at_k_vision(pages, page_embeddings, model, processor, k=3)

    print(f"Naive text RAG (chunk-level) Recall@3:   {text_recall:.3f}")
    print(f"Vision RAG (page-level)      Recall@3:   {vision_recall:.3f}")


def run_table_demo(provider: str | None) -> None:
    """Show the concrete failure mode: chunk-level retrieval can return a
    passage that TALKS ABOUT a table without containing the table's actual
    data, letting the LLM latch onto an unrelated number that happens to
    be nearby. Vision RAG retrieves the whole page, table intact, so the
    LLM reads the real cell values directly off the image."""
    load_environment()
    question = "What BLEU score did the base model get on the newstest2013 development set, per Table 3?"

    print("=== Naive text RAG ===")
    chunks = chunk_sample_text()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    chunk_embeddings = build_text_index(chunks, text_model)
    results = text_search(question, chunks, chunk_embeddings, text_model, k=1)
    top_chunk = results[0][0]
    print(f"Top retrieved chunk:\n{top_chunk}\n")

    naive_answer = llm.generate(
        f"Answer the question using the context below.\n\nContext:\n{top_chunk}\n\nQuestion: {question}",
        provider=provider,
    )
    print(f"Naive text RAG answer: {naive_answer}\n")

    print("=== Vision RAG ===")
    pages = render_pdf_pages()
    model, processor = load_vision_model()
    page_embeddings = build_page_image_index(pages, model, processor)
    top_page_idx, score = vision_search(question, page_embeddings, model, processor, k=1)[0]
    print(f"Top retrieved page: {top_page_idx} (MaxSim score: {score:.2f}) — full page image, table intact\n")

    vision_answer = llm.generate_with_image(
        f"Answer using only what's visible in this page image, reading the table structure directly.\n\n"
        f"Question: {question}",
        page_image_to_base64(pages[top_page_idx]),
        provider=provider,
    )
    print(f"Vision RAG answer: {vision_answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision RAG demo.")
    parser.add_argument("--mode", choices=["vision", "naive"], default="vision")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="Measure Recall@3 for naive text RAG vs. Vision RAG.")
    parser.add_argument("--table-demo", action="store_true", help="Show the table-layout failure mode directly.")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    args = parser.parse_args()

    if args.compare:
        run_comparison()
    elif args.table_demo:
        run_table_demo(args.provider)
    elif args.query:
        run_query(args.mode, args.query, args.provider)
    else:
        parser.print_help()

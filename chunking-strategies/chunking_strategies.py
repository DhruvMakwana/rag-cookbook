"""
Ten chunking strategies, implemented from scratch, run against the same
sample document so you can see how differently each one splits identical
text. No LangChain/LlamaIndex — see the blog page for when to reach for
those instead of hand-rolling this yourself.

Run:
    python chunking_strategies.py --strategy recursive
    python chunking_strategies.py --strategy semantic --threshold 0.6
    python chunking_strategies.py --strategy all   # runs every strategy, prints chunk counts
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import llm
import numpy as np
from dotenv import load_dotenv
from download_data import DEFAULT_PDF_PATH, download_sample_pdf
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"

# structure-aware chunking needs real Markdown headers to split on — the
# sample PDF has none (PDFs don't carry Markdown structure), so it gets
# its own small sample that actually demonstrates the technique.
SAMPLE_MARKDOWN = """# Employee Handbook

## Remote Work Policy

Employees may work remotely up to 3 days per week with manager approval.
Requests must be submitted 48 hours in advance.

### International Remote Work

International remote work requires HR approval and is limited to 30 days
per year due to tax and legal compliance requirements.

## Expense Policy

### Travel Expenses

All travel expenses must be submitted within 30 days with receipts attached.

### Equipment Expenses

Home office equipment up to $500/year is reimbursable with manager approval.
"""


def load_environment() -> None:
    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Loads the sample PDF's text (same document used in naive-rag)."""
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH
    if not pdf_path.exists():
        pdf_path = download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def split_sentences(text: str) -> list[str]:
    """A simple regex sentence splitter — good enough for demo purposes.
    Production systems should use a real sentence tokenizer (spaCy, nltk)
    since abbreviations ("e.g.", "Fig. 3") will trip this up."""
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ======================================================================
# 1. Fixed-size chunking
# ======================================================================
# --8<-- [start:fixed_size]
def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split every `chunk_size` characters, with `overlap` characters of
    redundancy between consecutive chunks so a boundary sentence still
    appears intact in at least one chunk."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap
    # Filter out whitespace-only chunks, but don't rewrite the ones we keep —
    # stripping each chunk independently would trim differing amounts of
    # whitespace off each edge and shift the overlap alignment between them.
    return [c for c in chunks if c.strip()]


# --8<-- [end:fixed_size]


# ======================================================================
# 2. Recursive chunking
# ======================================================================
# --8<-- [start:recursive]
def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[str]:
    """Try splitting on paragraph breaks first; only fall through to a
    finer separator (sentence, then word, then raw character) for any
    piece that's still too large. Mirrors LangChain's
    RecursiveCharacterTextSplitter, without the dependency."""

    def _split(piece: str, seps: tuple[str, ...]) -> list[str]:
        if len(piece) <= chunk_size or not seps:
            return [piece]
        sep, rest = seps[0], seps[1:]
        parts = piece.split(sep) if sep else list(piece)
        merged: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    merged.append(current)
                current = part
        if current:
            merged.append(current)

        result: list[str] = []
        for m in merged:
            result.extend(_split(m, rest) if len(m) > chunk_size else [m])
        return result

    raw_chunks = _split(text, separators)

    # apply overlap across the final chunk boundaries
    chunks, carry = [], ""
    for c in raw_chunks:
        combined = (carry + c) if carry else c
        chunks.append(combined)
        carry = combined[-overlap:] if overlap else ""
    return [c for c in chunks if c.strip()]


# --8<-- [end:recursive]


# ======================================================================
# 3. Document/structure-aware chunking (Markdown headers)
# ======================================================================
# --8<-- [start:structure_aware]
def structure_aware_chunk_markdown(markdown_text: str) -> list[dict]:
    """Split on Markdown headers, keeping the heading path as metadata —
    only works when the input actually has real header structure."""
    lines = markdown_text.split("\n")
    chunks: list[dict] = []
    current_heading_path: list[str] = []
    current_body: list[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            chunks.append(
                {"heading_path": " > ".join(current_heading_path), "text": body}
            )

    for line in lines:
        header_match = re.match(r"^(#{1,6})\s+(.*)", line)
        if header_match:
            flush()
            current_body.clear()
            level = len(header_match.group(1))
            title = header_match.group(2).strip()
            current_heading_path = current_heading_path[: level - 1] + [title]
        else:
            current_body.append(line)
    flush()
    return chunks


# --8<-- [end:structure_aware]


# ======================================================================
# 4. Semantic chunking
# ======================================================================
# --8<-- [start:semantic]
def semantic_chunk(
    text: str, embed_model: SentenceTransformer, threshold: float = 0.6
) -> list[str]:
    """Embed each sentence, walk through in order, and cut a new chunk
    whenever similarity to the next sentence drops below `threshold`
    (a topic shift). Lower threshold = fewer, larger chunks."""
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return sentences

    embeddings = embed_model.encode(
        sentences, convert_to_numpy=True, show_progress_bar=False
    )

    chunks, current = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim < threshold:
            chunks.append(" ".join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])
    if current:
        chunks.append(" ".join(current))
    return chunks


# --8<-- [end:semantic]


# ======================================================================
# 5. Sentence-window retrieval
# ======================================================================
# --8<-- [start:sentence_window]
def sentence_window_chunk(text: str, window: int = 2) -> list[dict]:
    """Index each sentence individually (precise matching unit), but keep
    a window of `window` sentences on either side for generation context."""
    sentences = split_sentences(text)
    result = []
    for i, sentence in enumerate(sentences):
        lo, hi = max(0, i - window), min(len(sentences), i + window + 1)
        result.append({"anchor": sentence, "window_text": " ".join(sentences[lo:hi])})
    return result


# --8<-- [end:sentence_window]


# ======================================================================
# 6. Small-to-big / parent-document chunking
# ======================================================================
# --8<-- [start:small_to_big]
def small_to_big_chunk(
    text: str, parent_size: int = 1500, child_size: int = 250, overlap: int = 30
) -> list[dict]:
    """Embed the small child chunks (precise matching); retrieval fetches
    the linked parent chunk (full context) for generation instead."""
    parents = fixed_size_chunk(text, chunk_size=parent_size, overlap=0)
    result = []
    for parent_id, parent in enumerate(parents):
        children = fixed_size_chunk(parent, chunk_size=child_size, overlap=overlap)
        for child in children:
            result.append(
                {"parent_id": parent_id, "parent_text": parent, "child_text": child}
            )
    return result


# --8<-- [end:small_to_big]


# ======================================================================
# 7. Late chunking
# ======================================================================
# --8<-- [start:late_chunking]
def late_chunk(
    text: str, boundaries: list[str], model_name: str = "answerdotai/ModernBERT-base"
) -> list[np.ndarray]:
    """Embed the WHOLE document first with a long-context model, so every
    token attends to every other token — THEN pool token embeddings within
    each chunk boundary. Chunk vectors end up carrying context from the
    rest of the document, unlike every other method here, which embeds
    each chunk in isolation."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    model.eval()

    encoded = tokenizer(
        text,
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation=True,
        max_length=8192,
    )
    offsets = encoded.pop("offset_mapping")[0].tolist()

    with torch.no_grad():
        token_embeddings = model(**encoded)[0][0]  # (seq_len, hidden_dim)

    # map each chunk's character span to its token index range, then mean-pool
    chunk_embeddings = []
    cursor = 0
    for boundary_text in boundaries:
        start_char = text.find(boundary_text, cursor)
        if start_char == -1:
            continue
        end_char = start_char + len(boundary_text)
        cursor = end_char

        token_indices = [
            i
            for i, (s, e) in enumerate(offsets)
            if s < end_char and e > start_char and not (s == 0 and e == 0)
        ]
        if not token_indices:
            continue
        span = token_embeddings[token_indices]
        pooled = span.mean(dim=0)
        pooled = pooled / pooled.norm()
        chunk_embeddings.append(pooled.numpy())

    return chunk_embeddings


# --8<-- [end:late_chunking]


# ======================================================================
# 8. Proposition-based chunking
# ======================================================================
# --8<-- [start:proposition]
PROPOSITION_PROMPT = """Break the passage below into propositions: atomic, \
self-contained factual statements. Each proposition must make sense on its \
own with no pronouns or implicit references — spell out any names the \
original text left implicit. Return one proposition per line, nothing else.

Passage:
{passage}"""


def proposition_chunk(text_chunk: str, provider: str | None = None) -> list[str]:
    """Ask an LLM to decompose one chunk into atomic, self-contained
    factual statements (Dense X Retrieval, Chen et al. 2023) — finer-
    grained and more precisely retrievable than a raw sentence, at the
    cost of one LLM call per source chunk."""
    prompt = PROPOSITION_PROMPT.format(passage=text_chunk)
    response = llm.generate(prompt, provider=provider)
    return [line.strip("- ").strip() for line in response.split("\n") if line.strip()]


# --8<-- [end:proposition]


# ======================================================================
# 9. Agentic chunking
# ======================================================================
# --8<-- [start:agentic]
AGENTIC_PROMPT = """Split the passage below into natural, meaningful \
segments — the way a domain expert would group it, keeping related ideas \
together even if they discuss superficially different sub-topics. \
Return each segment separated by a line containing only "---".

Passage:
{passage}"""


def agentic_chunk(text: str, provider: str | None = None) -> list[str]:
    """Let an LLM decide chunk boundaries directly, using its own reading
    comprehension rather than a mechanical rule — the most expensive
    strategy here, since it costs an LLM call per document (or per
    windowed section, for long documents)."""
    prompt = AGENTIC_PROMPT.format(passage=text)
    response = llm.generate(prompt, provider=provider)
    return [c.strip() for c in response.split("---") if c.strip()]


# --8<-- [end:agentic]


# ======================================================================
# 10. Adaptive chunking
# ======================================================================
# --8<-- [start:adaptive]
def _coherence_score(chunks: list[str], embed_model: SentenceTransformer) -> float:
    """Average pairwise sentence-similarity within each chunk — a proxy
    for "do these chunks actually hold together topically?" Higher is
    better."""
    scores = []
    for chunk in chunks:
        sentences = split_sentences(chunk)
        if len(sentences) < 2:
            continue
        embeddings = embed_model.encode(
            sentences, convert_to_numpy=True, show_progress_bar=False
        )
        pairwise = [
            cosine_sim(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        if pairwise:
            scores.append(sum(pairwise) / len(pairwise))
    return sum(scores) / len(scores) if scores else 0.0


def adaptive_chunk(
    text: str, embed_model: SentenceTransformer
) -> tuple[str, list[str]]:
    """Run multiple candidate strategies, score each for coherence, and
    return whichever wins for THIS document — rather than hardcoding one
    strategy for every document type in a mixed corpus."""
    candidates = {
        "fixed_size": fixed_size_chunk(text),
        "recursive": recursive_chunk(text),
        "semantic": semantic_chunk(text, embed_model),
    }
    scored = {
        name: _coherence_score(chunks, embed_model)
        for name, chunks in candidates.items()
    }
    best_name = max(scored, key=scored.get)
    print(f"Coherence scores: { {k: round(v, 3) for k, v in scored.items()} }")
    return best_name, candidates[best_name]


# --8<-- [end:adaptive]


STRATEGIES = [
    "fixed_size",
    "recursive",
    "structure_aware",
    "semantic",
    "sentence_window",
    "small_to_big",
    "late_chunking",
    "proposition",
    "agentic",
    "adaptive",
]


def run(strategy: str, threshold: float = 0.6) -> None:
    load_environment()
    text = load_sample_text()
    sample = text[:6000]  # keep the demo fast; full text works too

    if strategy == "fixed_size":
        chunks = fixed_size_chunk(sample)
    elif strategy == "recursive":
        chunks = recursive_chunk(sample)
    elif strategy == "structure_aware":
        parsed = structure_aware_chunk_markdown(SAMPLE_MARKDOWN)
        for c in parsed[:3]:
            print(f"--- [{c['heading_path']}] ---\n{c['text']}\n")
        print(
            f"\nstructure_aware: {len(parsed)} chunks (on a Markdown sample — the PDF has no headers to split on)"
        )
        return
    elif strategy in ("semantic", "sentence_window", "small_to_big", "adaptive"):
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        if strategy == "semantic":
            chunks = semantic_chunk(sample, embed_model, threshold=threshold)
        elif strategy == "sentence_window":
            chunks = [c["window_text"] for c in sentence_window_chunk(sample)]
        elif strategy == "small_to_big":
            chunks = [c["child_text"] for c in small_to_big_chunk(sample)]
        else:
            name, chunks = adaptive_chunk(sample, embed_model)
            print(f"Adaptive chunking picked: {name}")
    elif strategy == "late_chunking":
        boundaries = recursive_chunk(sample, chunk_size=300, overlap=0)
        embeddings = late_chunk(sample, boundaries)
        print(
            f"Produced {len(embeddings)} chunk embeddings, dim={embeddings[0].shape if embeddings else None}"
        )
        return
    elif strategy == "proposition":
        first_chunk = recursive_chunk(sample, chunk_size=800)[0]
        chunks = proposition_chunk(first_chunk)
    elif strategy == "agentic":
        chunks = agentic_chunk(sample[:2000])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n{strategy}: {len(chunks)} chunks\n")
    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c[:300])
        print()

    if strategy in ("fixed_size", "recursive") and len(chunks) >= 2:
        overlap_size = 50  # matches the default `overlap` these two strategies use
        print(
            f"--- overlap check: end of chunk 0 vs. start of chunk 1 (last/first {overlap_size} chars) ---"
        )
        print("chunk 0 tail:", repr(chunks[0][-overlap_size:]))
        print("chunk 1 head:", repr(chunks[1][:overlap_size]))
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a chunking strategy against the sample document."
    )
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="recursive")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.6,
        help="Semantic chunking similarity threshold.",
    )
    args = parser.parse_args()

    if args.strategy == "all":
        for s in STRATEGIES:
            print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")
            run(s, threshold=args.threshold)
    else:
        run(args.strategy, threshold=args.threshold)

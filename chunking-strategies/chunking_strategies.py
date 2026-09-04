"""
Ten chunking strategies, each backed by a real, widely-used, tested
implementation — not hand-rolled from scratch. Run against the same sample
document so you can see how differently each one splits identical text.

Libraries used, one per technique:
    fixed_size        langchain-text-splitters   CharacterTextSplitter
    recursive         langchain-text-splitters   RecursiveCharacterTextSplitter
    structure_aware   langchain-text-splitters   MarkdownHeaderTextSplitter
    semantic          chonkie                    SemanticChunker
    sentence_window   llama-index-core           SentenceWindowNodeParser
    small_to_big      llama-index-core           HierarchicalNodeParser
    late_chunking     chonkie                    LateChunker
    proposition       transformers               chentong00/propositionizer-wiki-flan-t5-large
                                                  (the official Dense X Retrieval model)
    agentic           chonkie                    SlumberChunker
    adaptive          (orchestration over the above — no single library owns
                       this meta-technique; it just picks among tested chunkers)

Run:
    python chunking_strategies.py --strategy recursive
    python chunking_strategies.py --strategy semantic --threshold 0.5
    python chunking_strategies.py --strategy all
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from chonkie import LateChunker, SemanticChunker, SlumberChunker
from dotenv import load_dotenv
from langchain_text_splitters import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceWindowNodeParser, get_leaf_nodes
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, T5ForConditionalGeneration

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

DATA_DIR = Path(__file__).parent / "data"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
PROPOSITIONIZER_MODEL_NAME = "chentong00/propositionizer-wiki-flan-t5-large"

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


# ======================================================================
# Setup / data loading helpers
# ======================================================================


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


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ======================================================================
# 1. Fixed-size chunking — langchain-text-splitters CharacterTextSplitter
# ======================================================================
# --8<-- [start:fixed_size]
def fixed_size_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Split every `chunk_size` characters, with `overlap` characters of
    redundancy between consecutive chunks. `separator=""` disables any
    structure-awareness — this is the deliberately naive baseline."""
    splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
# --8<-- [end:fixed_size]


# ======================================================================
# 2. Recursive chunking — langchain-text-splitters RecursiveCharacterTextSplitter
# ======================================================================
# --8<-- [start:recursive]
def recursive_chunk(text: str, chunk_size: int = 512, overlap: int = 50) -> list[str]:
    """Try splitting on paragraph breaks first; only fall through to a
    finer separator (sentence, then word, then raw character) for any
    piece that's still too large. The reference implementation of this
    technique — most tutorials describing "recursive chunking" mean
    exactly this class."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)
# --8<-- [end:recursive]


# ======================================================================
# 3. Document/structure-aware chunking — langchain-text-splitters MarkdownHeaderTextSplitter
# ======================================================================
# --8<-- [start:structure_aware]
def structure_aware_chunk_markdown(markdown_text: str) -> list[dict]:
    """Split on Markdown headers, keeping the heading path as metadata —
    only works when the input actually has real header structure."""
    headers_to_split_on = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(markdown_text)
    return [
        {"heading_path": " > ".join(doc.metadata.values()), "text": doc.page_content}
        for doc in docs
    ]
# --8<-- [end:structure_aware]


# ======================================================================
# 4. Semantic chunking — chonkie SemanticChunker
# ======================================================================
# --8<-- [start:semantic]
def semantic_chunk(text: str, threshold: float = 0.6) -> list[str]:
    """Embed sentences, walk through in order, and cut a new chunk
    whenever similarity to the next sentence drops below `threshold` (a
    topic shift). Lower threshold = fewer, larger chunks."""
    chunker = SemanticChunker(embedding_model=EMBEDDING_MODEL_NAME, threshold=threshold)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]
# --8<-- [end:semantic]


# ======================================================================
# 5. Sentence-window retrieval — llama-index SentenceWindowNodeParser
# ======================================================================
# --8<-- [start:sentence_window]
def sentence_window_chunk(text: str, window_size: int = 2) -> list[dict]:
    """Index each sentence individually (precise matching unit), but keep
    a window of `window_size` sentences on either side for generation
    context, attached as node metadata."""
    parser = SentenceWindowNodeParser.from_defaults(window_size=window_size)
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    return [{"anchor": n.text, "window_text": n.metadata["window"]} for n in nodes]
# --8<-- [end:sentence_window]


# ======================================================================
# 6. Small-to-big / parent-document chunking — llama-index HierarchicalNodeParser
# ======================================================================
# --8<-- [start:small_to_big]
def small_to_big_chunk(text: str, parent_size: int = 1500, child_size: int = 250) -> list[dict]:
    """Embed the small child chunks (precise matching); retrieval fetches
    the linked parent chunk (full context) for generation instead —
    LlamaIndex's HierarchicalNodeParser + get_leaf_nodes is the reference
    implementation, paired with AutoMergingRetriever in a full pipeline."""
    parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[parent_size, child_size])
    nodes = parser.get_nodes_from_documents([Document(text=text)])
    leaf_nodes = get_leaf_nodes(nodes)
    return [
        {
            "parent_id": leaf.parent_node.node_id if leaf.parent_node else None,
            "child_text": leaf.text,
        }
        for leaf in leaf_nodes
    ]
# --8<-- [end:small_to_big]


# ======================================================================
# 7. Late chunking — chonkie LateChunker
# ======================================================================
# --8<-- [start:late_chunking]
def late_chunk(text: str, chunk_size: int = 300) -> list[np.ndarray]:
    """Embed the WHOLE document first with a long-context model, so every
    token attends to every other token — THEN pool token embeddings within
    each chunk boundary. Chunk vectors carry context from the rest of the
    document, unlike every other method here, which embeds each chunk in
    isolation."""
    chunker = LateChunker(embedding_model="nomic-ai/modernbert-embed-base", chunk_size=chunk_size)
    chunks = chunker.chunk(text)
    return [c.embedding for c in chunks]
# --8<-- [end:late_chunking]


# ======================================================================
# 8. Proposition-based chunking — the official Dense X Retrieval model
# ======================================================================
# --8<-- [start:proposition]
def proposition_chunk(content: str, title: str = "", section: str = "") -> list[str]:
    """Decompose a passage into propositions — atomic, self-contained
    factual statements — using the actual model released by the Dense X
    Retrieval authors (Chen et al., 2023), fine-tuned specifically for
    this task. Input format ("Title: ... Section: ... Content: ...") and
    JSON-list output are both fixed by how the model was trained."""
    tokenizer = AutoTokenizer.from_pretrained(PROPOSITIONIZER_MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(PROPOSITIONIZER_MODEL_NAME)

    input_text = f"Title: {title}. Section: {section}. Content: {content}"
    input_ids = tokenizer(input_text, return_tensors="pt", truncation=True).input_ids
    output_ids = model.generate(input_ids, max_new_tokens=512)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    try:
        return json.loads(output_text)
    except json.JSONDecodeError:
        return [output_text]  # model occasionally returns plain text instead of a JSON list
# --8<-- [end:proposition]


# ======================================================================
# 9. Agentic chunking — chonkie SlumberChunker
# ======================================================================
# --8<-- [start:agentic]
def agentic_chunk(text: str, provider: str | None = None, chunk_size: int = 400) -> list[str]:
    """Let an LLM decide chunk boundaries directly, using its own reading
    comprehension rather than a mechanical rule. Chonkie's SlumberChunker
    implements the splitting algorithm (iteratively asking the model
    "where should this chunk end"); `llm.get_genie` just wires in which
    LLM answers those questions — see llm.py."""
    genie = llm.get_genie(provider)
    chunker = SlumberChunker(genie=genie, chunk_size=chunk_size, verbose=False)
    chunks = chunker.chunk(text)
    return [c.text for c in chunks]
# --8<-- [end:agentic]


# ======================================================================
# 10. Adaptive chunking — orchestration over the tested chunkers above
# ======================================================================
# --8<-- [start:adaptive]
def _coherence_score(chunks: list[str], embed_model: SentenceTransformer) -> float:
    """Average pairwise sentence-similarity within each chunk — a proxy
    for "do these chunks actually hold together topically?" Higher is
    better."""
    scores = []
    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk) if s.strip()]
        if len(sentences) < 2:
            continue
        embeddings = embed_model.encode(sentences, convert_to_numpy=True, show_progress_bar=False)
        pairwise = [
            cosine_sim(embeddings[i], embeddings[j])
            for i in range(len(embeddings))
            for j in range(i + 1, len(embeddings))
        ]
        if pairwise:
            scores.append(sum(pairwise) / len(pairwise))
    return sum(scores) / len(scores) if scores else 0.0


def adaptive_chunk(text: str) -> tuple[str, list[str]]:
    """Run multiple candidate strategies — each one a tested library
    implementation, not a custom one — score each for coherence, and
    return whichever wins for THIS document, rather than hardcoding one
    strategy for every document type in a mixed corpus. The selection
    logic itself is necessarily custom (no single library owns this
    meta-technique); what it selects between is not."""
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    candidates = {
        "fixed_size": fixed_size_chunk(text),
        "recursive": recursive_chunk(text),
        "semantic": semantic_chunk(text),
    }
    scored = {name: _coherence_score(chunks, embed_model) for name, chunks in candidates.items()}
    best_name = max(scored, key=scored.get)
    print(f"Coherence scores: { {k: round(v, 3) for k, v in scored.items()} }")
    return best_name, candidates[best_name]
# --8<-- [end:adaptive]


# ======================================================================
# CLI
# ======================================================================


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
        print(f"\nstructure_aware: {len(parsed)} chunks (on a Markdown sample — the PDF has no headers to split on)")
        return
    elif strategy == "semantic":
        chunks = semantic_chunk(sample, threshold=threshold)
    elif strategy == "sentence_window":
        chunks = [c["window_text"] for c in sentence_window_chunk(sample)]
    elif strategy == "small_to_big":
        chunks = [c["child_text"] for c in small_to_big_chunk(sample)]
    elif strategy == "late_chunking":
        embeddings = late_chunk(sample)
        print(f"Produced {len(embeddings)} chunk embeddings, dim={embeddings[0].shape if embeddings else None}")
        return
    elif strategy == "proposition":
        first_chunk = recursive_chunk(sample, chunk_size=800)[0]
        chunks = proposition_chunk(first_chunk, title="Attention Is All You Need")
    elif strategy == "agentic":
        chunks = agentic_chunk(sample[:2000])
    elif strategy == "adaptive":
        name, chunks = adaptive_chunk(sample)
        print(f"Adaptive chunking picked: {name}")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    print(f"\n{strategy}: {len(chunks)} chunks\n")
    for i, c in enumerate(chunks[:3]):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c)  # full chunk, not truncated — overlap and boundary behavior
        print()   # only actually show up if you can see the whole thing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a chunking strategy against the sample document.")
    parser.add_argument("--strategy", choices=STRATEGIES + ["all"], default="recursive")
    parser.add_argument("--threshold", type=float, default=0.6, help="Semantic chunking similarity threshold.")
    args = parser.parse_args()

    if args.strategy == "all":
        for s in STRATEGIES:
            print(f"\n{'=' * 60}\n{s}\n{'=' * 60}")
            run(s, threshold=args.threshold)
    else:
        run(args.strategy, threshold=args.threshold)

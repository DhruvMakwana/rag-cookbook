"""
Graph RAG: build a knowledge graph from the corpus (entities +
relationships extracted per chunk), detect communities of related
entities, summarize each community, and answer queries two ways —
local search (entity-link the query, traverse the graph, generate from
the traversed facts) and global search (map-reduce over community
summaries, for broad synthesis questions a single top-k retrieval
can't answer).

Graph construction is expensive (one LLM call per chunk for extraction,
plus one per community for summarization) — the built graph is cached
to disk so this only runs once per corpus.

Run:
    python graph_rag.py --build              # build + cache the graph (slow, one-time)
    python graph_rag.py --mode local --query "..."
    python graph_rag.py --mode global --query "..."
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

import llm
from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GRAPH_CACHE_PATH = Path(__file__).parent / "data" / "graph_cache.json"
DEDUP_SIMILARITY_THRESHOLD = 0.88
LOCAL_SEARCH_TOP_ENTITIES = 3
LOCAL_SEARCH_HOPS = 1


# ======================================================================
# Setup / data loading
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def load_sample_text() -> str:
    """Same front-matter trim applied in every other recipe in this
    repo — the author/affiliation block is noisy, single-newline text
    irrelevant to every question asked here."""
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


def naive_retrieve(question: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3) -> list[str]:
    """The baseline this whole recipe measures against — plain
    bi-encoder top-k, no graph involved."""
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return [chunks[i] for i in top_indices]


# ======================================================================
# 1. Entity + relationship extraction (1 LLM call per chunk)
# ======================================================================

# --8<-- [start:extraction]
EXTRACTION_PROMPT = """Extract entities and relationships from the \
passage below, for building a knowledge graph. Use short, canonical \
entity names (e.g. "Multi-Head Attention", not "the multi-head \
attention mechanism described here").

Passage:
{chunk}

For each relationship found, output one line as:
EntityA | relationship | EntityB

If no clear entity relationships exist in this passage, output NONE.
Wrap your output in <triples></triples> tags."""


def extract_triples(chunk: str, provider: str | None = None) -> list[tuple[str, str, str]]:
    response = llm.generate(EXTRACTION_PROMPT.format(chunk=chunk), provider=provider)
    match = re.search(r"<triples>(.*?)</triples>", response, re.S)
    if not match or match.group(1).strip().upper() == "NONE":
        return []

    triples = []
    for line in match.group(1).strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples
# --8<-- [end:extraction]


# ======================================================================
# 2. Entity deduplication (embedding similarity, not exact-match only)
# ======================================================================

# --8<-- [start:dedup]
def deduplicate_entities(entity_names: list[str], embed_model: SentenceTransformer) -> dict[str, str]:
    """Different chunks refer to the same real entity with different
    surface forms ("Encoder" vs "the Encoder stack" vs "encoder").
    Exact-string matching alone misses these — this embeds every
    distinct name and greedily merges anything similar enough to an
    already-seen canonical name, keeping the first-seen form as
    canonical. Returns {raw_name: canonical_name}."""
    unique_names = sorted(set(entity_names))
    if not unique_names:
        return {}

    embeddings = embed_model.encode(unique_names, convert_to_numpy=True, show_progress_bar=False)
    canonical_names: list[str] = []
    canonical_embeddings: list[np.ndarray] = []
    mapping: dict[str, str] = {}

    for name, emb in zip(unique_names, embeddings):
        best_sim, best_idx = -1.0, -1
        for i, cand_emb in enumerate(canonical_embeddings):
            sim = float(np.dot(emb, cand_emb) / (np.linalg.norm(emb) * np.linalg.norm(cand_emb) + 1e-8))
            if sim > best_sim:
                best_sim, best_idx = sim, i

        if best_sim >= DEDUP_SIMILARITY_THRESHOLD:
            mapping[name] = canonical_names[best_idx]
        else:
            canonical_names.append(name)
            canonical_embeddings.append(emb)
            mapping[name] = name

    return mapping
# --8<-- [end:dedup]


# ======================================================================
# 3. Build the graph, cache it (extraction is expensive — do it once)
# ======================================================================


def build_graph(chunks: list[str], embed_model: SentenceTransformer, provider: str | None = None) -> nx.MultiDiGraph:
    all_triples_per_chunk = []
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk, provider=provider)
        all_triples_per_chunk.append(triples)
        print(f"  extracted chunk {i + 1}/{len(chunks)}: {len(triples)} triples")

    all_entity_names = [e for triples in all_triples_per_chunk for (e1, _, e2) in triples for e in (e1, e2)]
    canonical_map = deduplicate_entities(all_entity_names, embed_model)

    graph = nx.MultiDiGraph()
    for chunk_idx, triples in enumerate(all_triples_per_chunk):
        for e1, rel, e2 in triples:
            c1, c2 = canonical_map[e1], canonical_map[e2]
            graph.add_node(c1)
            graph.add_node(c2)
            graph.add_edge(c1, c2, relationship=rel, source_chunk=chunk_idx)

    return graph


def save_graph(graph: nx.MultiDiGraph, chunks: list[str]) -> None:
    data = {
        "chunks": chunks,
        "nodes": list(graph.nodes()),
        "edges": [
            {"source": u, "target": v, "relationship": d["relationship"], "source_chunk": d["source_chunk"]}
            for u, v, d in graph.edges(data=True)
        ],
    }
    GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_CACHE_PATH.write_text(json.dumps(data))


def load_graph() -> tuple[nx.MultiDiGraph, list[str]]:
    data = json.loads(GRAPH_CACHE_PATH.read_text())
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(data["nodes"])
    for e in data["edges"]:
        graph.add_edge(e["source"], e["target"], relationship=e["relationship"], source_chunk=e["source_chunk"])
    return graph, data["chunks"]


# ======================================================================
# 4. Community detection (a real graph algorithm, not an LLM call)
# ======================================================================

# --8<-- [start:communities]
def detect_communities(graph: nx.MultiDiGraph) -> list[set[str]]:
    """Louvain community detection — a real, established graph
    clustering algorithm (networkx's built-in implementation), run
    purely on graph structure. No LLM call here; the LLM only
    summarizes the communities this algorithm finds."""
    undirected = nx.Graph(graph)  # collapses parallel/directed edges into a weighted simple graph
    if undirected.number_of_edges() == 0:
        return [{n} for n in undirected.nodes()]
    return list(nx.algorithms.community.louvain_communities(undirected, seed=42))
# --8<-- [end:communities]


SUMMARY_PROMPT = """Below are entities and relationships extracted \
from a technical document, all belonging to one related cluster. \
Write a 2-3 sentence summary of what this cluster of entities is \
collectively about.

Relationships:
{relationships}"""


def summarize_community(graph: nx.MultiDiGraph, community: set[str], provider: str | None = None) -> str:
    edges_text = "\n".join(
        f"{u} -> {d['relationship']} -> {v}"
        for u, v, d in graph.edges(data=True)
        if u in community and v in community
    )
    if not edges_text:
        edges_text = ", ".join(community)
    return llm.generate(SUMMARY_PROMPT.format(relationships=edges_text), provider=provider)


# ======================================================================
# 5. Local search — entity-link the query, traverse, generate
# ======================================================================

# --8<-- [start:local_search]
def local_search(
    question: str,
    graph: nx.MultiDiGraph,
    chunks: list[str],
    embed_model: SentenceTransformer,
    provider: str | None = None,
    top_entities: int = LOCAL_SEARCH_TOP_ENTITIES,
    hops: int = LOCAL_SEARCH_HOPS,
) -> dict:
    """Entity-link the query to graph nodes by embedding similarity
    (not exact string match — the query rarely uses the graph's exact
    canonical entity names), walk `hops` steps out from the matched
    entities, and gather every triple touched along the way, plus the
    original source chunks those triples came from."""
    nodes = list(graph.nodes())
    if not nodes:
        return {"matched_entities": [], "triples": [], "source_chunks": [], "answer": "(empty graph)"}

    node_embeddings = embed_model.encode(nodes, convert_to_numpy=True, show_progress_bar=False)
    query_vec = embed_model.encode(question, convert_to_numpy=True)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in node_embeddings]
    matched = [nodes[i] for i in np.argsort(sims)[::-1][:top_entities]]

    frontier = set(matched)
    visited_edges = set()
    for _ in range(hops):
        next_frontier = set()
        for node in frontier:
            for u, v, d in list(graph.out_edges(node, data=True)) + list(graph.in_edges(node, data=True)):
                visited_edges.add((u, v, d["relationship"], d["source_chunk"]))
                next_frontier.update([u, v])
        frontier |= next_frontier

    triples = [(u, rel, v) for (u, v, rel, _) in visited_edges]
    source_chunk_ids = sorted({sc for (_, _, _, sc) in visited_edges})
    source_chunks = [chunks[i] for i in source_chunk_ids]

    facts_text = "\n".join(f"{u} -> {rel} -> {v}" for u, rel, v in triples)
    context = f"Graph facts:\n{facts_text}\n\nSource passages:\n" + "\n---\n".join(source_chunks)
    answer = llm.generate(f"Answer the question using the context below.\n\nContext:\n{context}\n\nQuestion: {question}", provider=provider)

    return {"matched_entities": matched, "triples": triples, "source_chunks": source_chunks, "answer": answer}
# --8<-- [end:local_search]


# ======================================================================
# 6. Global search — map-reduce over community summaries
# ======================================================================

# --8<-- [start:global_search]
MAP_PROMPT = """Using ONLY the community summary below, write a brief \
partial answer to the question — if this summary isn't relevant to \
the question at all, respond with exactly: NOT RELEVANT

Community summary: {summary}

Question: {question}"""

REDUCE_PROMPT = """Combine the partial answers below into one coherent \
final answer to the question. Ignore any partial answer that said \
"NOT RELEVANT".

Question: {question}

Partial answers:
{partial_answers}"""


def global_search(question: str, community_summaries: list[str], provider: str | None = None) -> dict:
    """Map: ask each community summary independently whether it has a
    partial answer (parallelizable, independent calls — irrelevant
    communities self-report and get filtered out). Reduce: combine
    whatever partial answers came back into one final answer. This is
    what lets a broad synthesis question get answered from a
    whole-corpus view instead of whatever a single top-k happened to
    retrieve."""
    partial_answers = []
    for summary in community_summaries:
        resp = llm.generate(MAP_PROMPT.format(summary=summary, question=question), provider=provider).strip()
        if "NOT RELEVANT" not in resp.upper():
            partial_answers.append(resp)

    if not partial_answers:
        return {"partial_answers": [], "answer": "No community summary was relevant to this question."}

    combined = "\n---\n".join(partial_answers)
    answer = llm.generate(REDUCE_PROMPT.format(question=question, partial_answers=combined), provider=provider)
    return {"partial_answers": partial_answers, "answer": answer}
# --8<-- [end:global_search]


# ======================================================================
# CLI
# ======================================================================


def run_build(provider: str | None) -> None:
    load_environment()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Extracting entities/relationships from {len(chunks)} chunks (one LLM call each)...")
    graph = build_graph(chunks, embed_model, provider=provider)
    print(f"\nGraph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships.")

    communities = detect_communities(graph)
    print(f"Detected {len(communities)} communities. Summarizing each...")
    summaries = [summarize_community(graph, c, provider=provider) for c in communities]

    save_graph(graph, chunks)
    summary_path = GRAPH_CACHE_PATH.parent / "community_summaries.json"
    summary_path.write_text(json.dumps({"communities": [list(c) for c in communities], "summaries": summaries}))
    print(f"Cached graph to {GRAPH_CACHE_PATH}, community summaries to {summary_path}.")


def run_query(mode: str, question: str, provider: str | None) -> None:
    load_environment()
    if not GRAPH_CACHE_PATH.exists():
        raise RuntimeError("No cached graph found. Run `python graph_rag.py --build` first.")

    graph, chunks = load_graph()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    if mode == "local":
        result = local_search(question, graph, chunks, embed_model, provider=provider)
        print(f"Question: {question}\n")
        print(f"Matched entities: {result['matched_entities']}")
        print(f"Traversed {len(result['triples'])} triples, {len(result['source_chunks'])} source chunks.\n")
        print(f"Answer: {result['answer']}")
    else:
        summary_path = GRAPH_CACHE_PATH.parent / "community_summaries.json"
        summaries = json.loads(summary_path.read_text())["summaries"]
        result = global_search(question, summaries, provider=provider)
        print(f"Question: {question}\n")
        print(f"Relevant communities: {len(result['partial_answers'])} / {len(summaries)}\n")
        print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph RAG demo.")
    parser.add_argument("--build", action="store_true", help="Build and cache the graph (slow, one-time).")
    parser.add_argument("--mode", choices=["local", "global"], default="local")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()

    if args.build:
        run_build(args.provider)
    elif args.query:
        run_query(args.mode, args.query, args.provider)
    else:
        parser.error("Pass --build to construct the graph, or --query \"...\" to search it.")

"""
Agentic RAG: instead of a hand-rolled router deciding which retrieval
strategy to use (query-transformation, adaptive-rag), or fixed
pre-programmed branching (corrective-rag, self-rag), an LLM agent
picks its own tools at runtime via Anthropic's native tool-calling API
— a genuinely different mechanism from every prior recipe's prompted
classification, and the answer to the routing gap flagged on the
graph-rag page (choosing local vs. global search there was a manual
CLI flag, not automatic).

Four tools, each a real technique already built elsewhere in this
repo: naive vector search, graph local search, graph global search,
and a real Tavily web search fallback. The agent decides which to
call, how many times, and when it has enough — a ReAct-style
Thought -> Action -> Observation loop, made of real API tool_use
blocks, not prompted <tag> parsing.

Needs a pre-built knowledge graph — same one graph-rag/ builds. Run
`python agentic_rag.py --build` once first if data/graph_cache.json
doesn't already exist (slow, ~92 LLM calls — see graph-rag/'s README
for why).

Run:
    python agentic_rag.py --build
    python agentic_rag.py --query "..."
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import anthropic
import networkx as nx
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from download_data import DEFAULT_PDF_PATH, download_sample_pdf

# ======================================================================
# Constants
# ======================================================================

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GRAPH_CACHE_PATH = Path(__file__).parent / "data" / "graph_cache.json"
COMMUNITY_SUMMARIES_PATH = Path(__file__).parent / "data" / "community_summaries.json"
DEDUP_SIMILARITY_THRESHOLD = 0.88
MAX_AGENT_ITERATIONS = 6


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


def _anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def _model_name() -> str:
    return os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")


# ======================================================================
# Data loading + graph construction (same as graph-rag/ — copied in,
# not cross-imported, per this repo's self-contained-recipe convention)
# ======================================================================


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

SUMMARY_PROMPT = """Below are entities and relationships extracted \
from a technical document, all belonging to one related cluster. \
Write a 2-3 sentence summary of what this cluster of entities is \
collectively about.

Relationships:
{relationships}"""


def _generate(client: anthropic.Anthropic, prompt: str) -> str:
    response = client.messages.create(
        model=_model_name(), max_tokens=1024, messages=[{"role": "user", "content": prompt}]
    )
    return next(b.text for b in response.content if b.type == "text")


def _extract_triples(client: anthropic.Anthropic, chunk: str) -> list[tuple[str, str, str]]:
    response = _generate(client, EXTRACTION_PROMPT.format(chunk=chunk))
    match = re.search(r"<triples>(.*?)</triples>", response, re.S)
    if not match or match.group(1).strip().upper() == "NONE":
        return []
    triples = []
    for line in match.group(1).strip().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3 and all(parts):
            triples.append((parts[0], parts[1], parts[2]))
    return triples


def _deduplicate_entities(entity_names: list[str], embed_model: SentenceTransformer) -> dict[str, str]:
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


def build_and_cache_graph() -> None:
    client = _anthropic_client()
    chunks = chunk_sample_text()
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print(f"Extracting entities/relationships from {len(chunks)} chunks...")
    all_triples_per_chunk = []
    for i, chunk in enumerate(chunks):
        triples = _extract_triples(client, chunk)
        all_triples_per_chunk.append(triples)
        print(f"  chunk {i + 1}/{len(chunks)}: {len(triples)} triples")

    all_entity_names = [e for triples in all_triples_per_chunk for (e1, _, e2) in triples for e in (e1, e2)]
    canonical_map = _deduplicate_entities(all_entity_names, embed_model)

    graph = nx.MultiDiGraph()
    for chunk_idx, triples in enumerate(all_triples_per_chunk):
        for e1, rel, e2 in triples:
            c1, c2 = canonical_map[e1], canonical_map[e2]
            graph.add_edge(c1, c2, relationship=rel, source_chunk=chunk_idx)

    print(f"Graph built: {graph.number_of_nodes()} entities, {graph.number_of_edges()} relationships.")

    undirected = nx.Graph(graph)
    communities = (
        list(nx.algorithms.community.louvain_communities(undirected, seed=42))
        if undirected.number_of_edges() > 0
        else [{n} for n in undirected.nodes()]
    )
    print(f"Detected {len(communities)} communities. Summarizing...")
    summaries = []
    for community in communities:
        edges_text = "\n".join(
            f"{u} -> {d['relationship']} -> {v}" for u, v, d in graph.edges(data=True) if u in community and v in community
        ) or ", ".join(community)
        summaries.append(_generate(client, SUMMARY_PROMPT.format(relationships=edges_text)))

    GRAPH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    GRAPH_CACHE_PATH.write_text(json.dumps({
        "chunks": chunks,
        "nodes": list(graph.nodes()),
        "edges": [
            {"source": u, "target": v, "relationship": d["relationship"], "source_chunk": d["source_chunk"]}
            for u, v, d in graph.edges(data=True)
        ],
    }))
    COMMUNITY_SUMMARIES_PATH.write_text(json.dumps({"summaries": summaries}))
    print(f"Cached to {GRAPH_CACHE_PATH} and {COMMUNITY_SUMMARIES_PATH}.")


def _load_graph() -> tuple[nx.MultiDiGraph, list[str]]:
    data = json.loads(GRAPH_CACHE_PATH.read_text())
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(data["nodes"])
    for e in data["edges"]:
        graph.add_edge(e["source"], e["target"], relationship=e["relationship"], source_chunk=e["source_chunk"])
    return graph, data["chunks"]


# ======================================================================
# The four tools — each a real technique already built elsewhere in
# this repo, reused here as the agent's choices
# ======================================================================


def naive_vector_search(query: str, chunks: list[str], embed_model: SentenceTransformer, k: int = 3) -> str:
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    chunk_vecs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    sims = [float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-8)) for v in chunk_vecs]
    top_indices = np.argsort(sims)[::-1][:k]
    return "\n---\n".join(chunks[i] for i in top_indices)


def graph_local_search(
    query: str, graph: nx.MultiDiGraph, chunks: list[str], embed_model: SentenceTransformer,
    top_entities: int = 3, hops: int = 1,
) -> str:
    nodes = list(graph.nodes())
    if not nodes:
        return "(empty graph)"
    node_embeddings = embed_model.encode(nodes, convert_to_numpy=True, show_progress_bar=False)
    query_vec = embed_model.encode(query, convert_to_numpy=True)
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

    facts = "\n".join(f"{u} -> {rel} -> {v}" for (u, v, rel, _) in visited_edges)
    source_ids = sorted({sc for (_, _, _, sc) in visited_edges})
    sources = "\n---\n".join(chunks[i] for i in source_ids)
    return f"Matched entities: {matched}\n\nGraph facts:\n{facts}\n\nSource passages:\n{sources}"


def graph_global_search(query: str, community_summaries: list[str], client: anthropic.Anthropic) -> str:
    partial_answers = []
    for summary in community_summaries:
        resp = _generate(
            client,
            f'Using ONLY the community summary below, write a brief partial answer to the '
            f'question — if this summary isn\'t relevant at all, respond with exactly: NOT RELEVANT\n\n'
            f"Community summary: {summary}\n\nQuestion: {query}",
        ).strip()
        if "NOT RELEVANT" not in resp.upper():
            partial_answers.append(resp)
    if not partial_answers:
        return "No community summary was relevant to this question."
    return "\n---\n".join(partial_answers)


def web_search(query: str, max_results: int = 3) -> str:
    from tavily import TavilyClient

    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "web_search unavailable: TAVILY_API_KEY is not set."
    client = TavilyClient(api_key=api_key)
    response = client.search(query, max_results=max_results)
    return "\n---\n".join(f"{r['title']}: {r['content']}" for r in response.get("results", []))


# ======================================================================
# The agent loop — Anthropic's native tool-calling API, not prompted
# <tag> parsing
# ======================================================================

TOOLS = [
    {
        "name": "naive_vector_search",
        "description": (
            "Plain vector similarity search over the document. Best for a single, "
            "specific factual question already phrased using the document's own vocabulary."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "graph_local_search",
        "description": (
            "Searches a knowledge graph by linking the query to specific entities and "
            "traversing their relationships. Best for questions about how two or more "
            "specific things are connected to each other."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "graph_global_search",
        "description": (
            "Searches using summaries of the whole knowledge graph's community structure. "
            "Best for broad questions that require synthesizing across many different parts "
            "of the document — e.g. 'what are the main components/themes' — where no single "
            "entity or chunk has the complete answer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "web_search",
        "description": (
            "Searches the public web. Use this when the question is about something outside "
            "this technical paper entirely (current events, unrelated general knowledge), or "
            "when the other tools returned nothing useful for a question that clearly needs "
            "external information."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
]


def _dispatch_tool(name: str, tool_input: dict, chunks, graph, community_summaries, embed_model, anthropic_client) -> str:
    query = tool_input["query"]
    if name == "naive_vector_search":
        return naive_vector_search(query, chunks, embed_model)
    if name == "graph_local_search":
        return graph_local_search(query, graph, chunks, embed_model)
    if name == "graph_global_search":
        return graph_global_search(query, community_summaries, anthropic_client)
    if name == "web_search":
        return web_search(query)
    return f"Unknown tool: {name}"


def agentic_rag(question: str, chunks: list[str], graph: nx.MultiDiGraph, community_summaries: list[str], embed_model: SentenceTransformer) -> dict:
    """A ReAct-style loop, made of real API tool_use blocks: Claude
    picks a tool (Action), the tool runs and its output is fed back
    (Observation), and Claude decides — for itself, each iteration —
    whether it has enough to answer or needs to call another tool
    (Thought). Capped at MAX_AGENT_ITERATIONS to avoid an unproductive
    loop running forever; if the cap is hit, an honest partial answer
    beats silently hanging."""
    client = _anthropic_client()
    messages = [{"role": "user", "content": question}]
    trace = []

    for _ in range(MAX_AGENT_ITERATIONS):
        response = client.messages.create(
            model=_model_name(), max_tokens=2048, tools=TOOLS, messages=messages
        )

        if response.stop_reason != "tool_use":
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            return {"trace": trace, "answer": final_text, "status": "ok"}

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = _dispatch_tool(block.name, block.input, chunks, graph, community_summaries, embed_model, client)
                trace.append({"tool": block.name, "input": block.input, "result_preview": result[:200]})
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": result})

        messages.append({"role": "user", "content": tool_results})

    return {"trace": trace, "answer": "Reached the iteration cap without a final answer.", "status": "max_iterations"}


# ======================================================================
# CLI
# ======================================================================


def run_query(question: str) -> None:
    load_environment()
    if not GRAPH_CACHE_PATH.exists():
        raise RuntimeError("No cached graph found. Run `python agentic_rag.py --build` first.")

    graph, chunks = _load_graph()
    community_summaries = json.loads(COMMUNITY_SUMMARIES_PATH.read_text())["summaries"]
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    result = agentic_rag(question, chunks, graph, community_summaries, embed_model)
    print(f"Question: {question}\n")
    for i, step in enumerate(result["trace"]):
        print(f"Step {i + 1}: called {step['tool']}({step['input']})")
        print(f"  -> {step['result_preview']}...")
    print(f"\nStatus: {result['status']}")
    print(f"Answer: {result['answer']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG demo.")
    parser.add_argument("--build", action="store_true", help="Build and cache the knowledge graph (slow, one-time).")
    parser.add_argument("--query", type=str, default=None)
    args = parser.parse_args()

    if args.build:
        load_environment()
        build_and_cache_graph()
    elif args.query:
        run_query(args.query)
    else:
        parser.error("Pass --build to construct the graph, or --query \"...\" to run the agent.")

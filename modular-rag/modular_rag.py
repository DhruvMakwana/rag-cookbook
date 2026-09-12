"""
Modular RAG treats routing (and every other stage) as a swappable
module. This recipe measures the ONE thing genuinely un-demonstrated
by adaptive-rag/self-rag/corrective-rag/agentic-rag — all of which
route via an LLM call: the same routing decision, built three
different ways, measured for real accuracy, latency, and cost.

Same 5 categories adaptive-rag routes to (CLEAR / VAGUE / COMPOUND /
BROAD / NEEDS_CONTEXT), three router implementations:

1. heuristic_router   — keyword/pattern rules, ~0 cost, ~0 latency
2. embedding_router    — cosine similarity to example utterances per
                         category, one local embedding call, no LLM
3. llm_router          — the exact router from adaptive-rag/, reused
                         here as the accuracy baseline

Run:
    python modular_rag.py --compare
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import llm

# ======================================================================
# Setup
# ======================================================================


def load_environment() -> None:
    from dotenv import load_dotenv

    local_env = Path(__file__).parent / ".env"
    load_dotenv(local_env if local_env.exists() else None)


CATEGORIES = ["CLEAR", "VAGUE", "COMPOUND", "BROAD", "NEEDS_CONTEXT"]

# 9 queries with known-correct labels — reused directly from adaptive-rag/'s
# own router validation (5 original test queries + 4 held-out
# generalization queries verified there), not re-guessed here.
LABELED_QUERIES = [
    ("What optimizer did they use for training?", "CLEAR"),
    ("why dont they just use RNNs like everyone else did before", "VAGUE"),
    ("What optimizer did they use, and how many attention heads did they use?", "COMPOUND"),
    ("What lets this model connect words that are far apart in a sentence without losing track over long distances?", "BROAD"),
    ("What value did they use for Pdrop during training?", "NEEDS_CONTEXT"),
    ("What GPUs was the model trained on?", "CLEAR"),
    ("how come they didnt just make it bigger to get better results", "VAGUE"),
    ("What learning rate did they use, and what dataset did they train on?", "COMPOUND"),
    ("Why is this architecture considered more interpretable?", "BROAD"),
]

# A SEPARATE held-out set, written without looking at heuristic_router's
# rules — the fair test of whether a router generalizes, vs. just
# fitting the phrasing it was tuned against. See README for why this
# distinction matters.
HELD_OUT_QUERIES = [
    ("What is the batch size, plus what hardware did they train on?", "COMPOUND"),
    ("ok so like why not just make the model bigger instead", "VAGUE"),
    ("What accounts for this model generalizing so well to new domains?", "BROAD"),
    ("What dropout probability was applied?", "NEEDS_CONTEXT"),
    ("What framework was used to implement the model?", "CLEAR"),
]


# ======================================================================
# 1. Heuristic router — keyword/pattern rules, no model call at all
# ======================================================================

COMPOUND_MARKERS = [" and ", " as well as "]
VAGUE_MARKERS = ["dont", "didnt", "just ", "how come", "everyone", "like everyone"]
BROAD_MARKERS = ["why is", "why did", "what lets", "what makes", "how does"]
NEEDS_CONTEXT_MARKERS = ["value", "rate did", "parameter"]


def heuristic_router(query: str) -> str:
    q = query.lower()

    if any(marker in q for marker in COMPOUND_MARKERS) and q.count("?") <= 1 and "," in query:
        return "COMPOUND"
    if any(marker in q for marker in VAGUE_MARKERS):
        return "VAGUE"
    if any(marker in q for marker in NEEDS_CONTEXT_MARKERS):
        return "NEEDS_CONTEXT"
    if any(q.startswith(marker) or marker in q for marker in BROAD_MARKERS):
        return "BROAD"
    return "CLEAR"


# ======================================================================
# 2. Embedding-based semantic router — no LLM call, one local embed
# ======================================================================

EXAMPLE_UTTERANCES = {
    "CLEAR": [
        "What dataset was used for evaluation?",
        "What programming language was used?",
        "Which company published this paper?",
    ],
    "VAGUE": [
        "why didnt they just do it the old way",
        "how come nobody thought of this before",
        "whats the deal with skipping that step",
    ],
    "COMPOUND": [
        "What is the accuracy, and what is the training time?",
        "How many layers are there, and what activation function is used?",
        "What is the cost, and what is the latency?",
    ],
    "BROAD": [
        "Why does this approach outperform previous methods?",
        "What makes this design more efficient overall?",
        "How does this system achieve better generalization?",
    ],
    "NEEDS_CONTEXT": [
        "What value was used for the learning rate?",
        "What rate did they set for weight decay?",
        "What parameter controls the batch size?",
    ],
}


def build_router_index(embed_model: SentenceTransformer) -> dict[str, np.ndarray]:
    """One-time setup: embed each category's example utterances and
    average them into a single centroid vector per category."""
    centroids = {}
    for category, examples in EXAMPLE_UTTERANCES.items():
        embeddings = embed_model.encode(examples, convert_to_numpy=True, show_progress_bar=False)
        centroids[category] = embeddings.mean(axis=0)
    return centroids


def embedding_router(query: str, centroids: dict[str, np.ndarray], embed_model: SentenceTransformer) -> str:
    query_vec = embed_model.encode(query, convert_to_numpy=True)
    sims = {
        category: float(np.dot(query_vec, centroid) / (np.linalg.norm(query_vec) * np.linalg.norm(centroid) + 1e-8))
        for category, centroid in centroids.items()
    }
    return max(sims, key=sims.get)


# ======================================================================
# 3. LLM-based router — the exact router from adaptive-rag/, reused
# ======================================================================

ROUTER_PROMPT = """Classify the query below into EXACTLY ONE category, \
based on what problem (if any) it poses for retrieval against a \
technical document. Categories, each with an example:

CLEAR - a single specific fact that is a NAMED thing (an algorithm \
name, a hardware type, a dataset name) — not a numeric parameter value \
— already phrased using the kind of technical vocabulary the document \
itself would use.
  Example: "What optimizer did they use for training?"

VAGUE - a single specific fact whose complete answer is ONE short \
thing, but the question is phrased colloquially, avoiding the \
technical vocabulary the document actually uses.
  Example: "why dont they just use the old-school approach everyone \
used before"

COMPOUND - explicitly asks for two or more distinct, independently \
answerable facts, usually joined by "and".
  Example: "What optimizer did they use, and how many layers does it have?"

BROAD - one open-ended question whose COMPLETE answer requires citing \
several distinct reasons or mechanisms (not one short fact) — decide \
based on whether a full answer needs multiple points, regardless of \
whether the wording sounds casual or technical.
  Example: "What lets this model handle relationships between things \
that are positioned far apart, without losing track over long spans?"

NEEDS_CONTEXT - asks for one narrow NUMERIC or configuration parameter \
value (a rate, a dimension, a count) where knowing the broader \
principle/section it belongs to would help interpret or locate the \
answer.
  Example: "What value did they use for the regularization strength?"

Query: {query}

Respond with ONLY the category name, wrapped in <category></category> tags."""


def llm_router(question: str, provider: str | None = None) -> str:
    response = llm.generate(ROUTER_PROMPT.format(query=question), provider=provider)
    match = re.search(r"<category>(.*?)</category>", response, re.S)
    category = match.group(1).strip() if match else "CLEAR"
    return category if category in CATEGORIES else "CLEAR"


# ======================================================================
# CLI — measure all 3 on the same labeled queries
# ======================================================================


def _evaluate(queries: list[tuple[str, str]], centroids: dict, embed_model: SentenceTransformer, provider: str | None) -> dict:
    results = {"heuristic": {"correct": 0, "total_time": 0.0}, "embedding": {"correct": 0, "total_time": 0.0}, "llm": {"correct": 0, "total_time": 0.0}}

    for query, truth in queries:
        t0 = time.perf_counter()
        h_pred = heuristic_router(query)
        results["heuristic"]["total_time"] += time.perf_counter() - t0
        results["heuristic"]["correct"] += h_pred == truth

        t0 = time.perf_counter()
        e_pred = embedding_router(query, centroids, embed_model)
        results["embedding"]["total_time"] += time.perf_counter() - t0
        results["embedding"]["correct"] += e_pred == truth

        t0 = time.perf_counter()
        l_pred = llm_router(query, provider=provider)
        results["llm"]["total_time"] += time.perf_counter() - t0
        results["llm"]["correct"] += l_pred == truth

        label = query if len(query) <= 73 else query[:70] + "..."
        print(f"{label:<75} {truth:<14} {h_pred:<14} {e_pred:<14} {l_pred:<14}")

    return results


def _print_summary(results: dict, n: int) -> None:
    print(f"{'Router':<12} {'Accuracy':<12} {'Avg latency':<15}")
    for name in ["heuristic", "embedding", "llm"]:
        acc = results[name]["correct"] / n
        avg_ms = (results[name]["total_time"] / n) * 1000
        print(f"{name:<12} {acc:<12.2f} {avg_ms:<15.1f}ms")


def run_comparison(provider: str | None) -> None:
    load_environment()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    centroids = build_router_index(embed_model)

    print(f"{'Query':<75} {'Truth':<14} {'Heuristic':<14} {'Embedding':<14} {'LLM':<14}")
    print("-" * 145)
    print("--- Design set (heuristic_router's rules were written looking at these) ---")
    design_results = _evaluate(LABELED_QUERIES, centroids, embed_model, provider)
    print()
    print("--- Held-out set (written WITHOUT looking at heuristic_router's rules) ---")
    heldout_results = _evaluate(HELD_OUT_QUERIES, centroids, embed_model, provider)

    print()
    print("Design-set accuracy/latency:")
    _print_summary(design_results, len(LABELED_QUERIES))
    print()
    print("Held-out-set accuracy/latency (the fair generalization test):")
    _print_summary(heldout_results, len(HELD_OUT_QUERIES))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modular RAG routing-approach comparison.")
    parser.add_argument("--compare", action="store_true", help="Run all 3 routers on the same labeled query set.")
    parser.add_argument("--provider", choices=["anthropic", "openai", "ollama"], default=None)
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.provider)
    else:
        parser.error("Pass --compare to run the 3-way router comparison.")

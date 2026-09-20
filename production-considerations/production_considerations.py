"""
Production considerations: real retry-with-backoff and circuit-breaker
patterns against a deliberately flaky call, real per-stage latency and
token/cost instrumentation across a naive RAG pipeline, and a real
semantic cache with genuine hit/miss behavior -- no paid API needed,
generation runs through a local Ollama model.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/generate"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# Real near-duplicate/paraphrase pairs mixed in with distinct questions,
# so the semantic cache demo below has genuine hits to find, not just
# misses.
CACHE_DEMO_QUESTIONS = [
    "How many attention heads did they use?",
    "What is the model's embedding dimension?",
    "How many attention heads does the model have?",  # paraphrase of Q1
    "What optimizer was used for training?",
    "What's the embedding size of the model?",  # paraphrase of Q2
    "What optimizer did they use to train the model?",  # paraphrase of Q4
]

COST_DEMO_QUESTIONS = [
    "How many attention heads did they use?",
    "What is the model's embedding dimension?",
    "How many layers are in the encoder?",
    "What optimizer was used for training?",
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


def build_corpus() -> list[str]:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    text = "\n\n".join(page.extract_text() or "" for page in reader.pages)

    chunks = []
    for start in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        piece = text[start:start + CHUNK_SIZE].strip()
        if len(piece) >= 40:
            chunks.append(piece)
    return chunks


def ollama_generate(prompt: str) -> dict:
    """Returns the raw Ollama response dict, including its real usage
    fields (prompt_eval_count, eval_count) and timing fields
    (prompt_eval_duration, eval_duration) -- the same shape of data a
    provider like Anthropic/OpenAI exposes via `response.usage`."""
    import requests

    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}, timeout=180)
    response.raise_for_status()
    return response.json()


# ======================================================================
# Retry with exponential backoff -- `tenacity`, wrapping a deliberately
# flaky call (a stand-in for any external dependency: embedding API,
# reranker service, vector DB query) that fails twice before succeeding.
# ======================================================================


class FlakyRerankerCallCounter:
    """A real, stateful counter simulating an external call that fails
    the first two times, then succeeds -- deterministic, so the retry
    behavior below is reproducible."""

    def __init__(self, fail_times: int = 2):
        self.fail_times = fail_times
        self.attempts = 0

    def call(self) -> str:
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise ConnectionError(f"reranker service unavailable (attempt {self.attempts})")
        return f"reranked results (succeeded on attempt {self.attempts})"


def demo_retry_backoff() -> None:
    from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

    counter = FlakyRerankerCallCounter(fail_times=2)
    attempt_log = []

    @retry(
        retry=retry_if_exception_type(ConnectionError),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def call_reranker():
        start = time.monotonic()
        try:
            result = counter.call()
            attempt_log.append((counter.attempts, "success", time.monotonic() - start))
            return result
        except ConnectionError as e:
            attempt_log.append((counter.attempts, f"failed: {e}", time.monotonic() - start))
            raise

    result = call_reranker()
    print(f"\n--- Retry with backoff: final result: {result!r} ---")
    for attempt, outcome, elapsed in attempt_log:
        print(f"  attempt {attempt}: {outcome}")


# ======================================================================
# Circuit breaker -- `pybreaker`, wrapping a call that fails
# persistently (not transiently) -- the breaker opens after a failure
# threshold and short-circuits further calls immediately, instead of
# waiting on a doomed request every time.
# ======================================================================


def demo_circuit_breaker() -> None:
    import pybreaker

    breaker = pybreaker.CircuitBreaker(fail_max=3, reset_timeout=2)
    call_count = 0

    @breaker
    def call_down_service():
        nonlocal call_count
        call_count += 1
        time.sleep(0.05)  # simulates real request latency before the failure is known
        raise ConnectionError("reranker service is down")

    print("\n--- Circuit breaker: 6 calls against a persistently down service ---")
    for i in range(1, 7):
        start = time.monotonic()
        try:
            call_down_service()
        except pybreaker.CircuitBreakerError:
            print(f"  call {i}: short-circuited immediately ({time.monotonic() - start:.4f}s, no request sent) -- breaker state={breaker.current_state}")
        except ConnectionError:
            print(f"  call {i}: request sent and failed ({time.monotonic() - start:.4f}s) -- breaker state={breaker.current_state}")
    print(f"  total real requests actually sent to the service: {call_count} (out of 6 calls)")


# ======================================================================
# Per-stage latency breakdown -- real timing across a real naive RAG
# pipeline's embedding, retrieval, and generation stages, not just an
# end-to-end number.
# ======================================================================


def demo_latency_breakdown(chunks: list[str], model: SentenceTransformer, question: str) -> None:
    timings = {}

    start = time.monotonic()
    query_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
    timings["embed_query"] = time.monotonic() - start

    start = time.monotonic()
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    timings["embed_corpus"] = time.monotonic() - start

    start = time.monotonic()
    sims = chunk_embeddings @ query_vec
    top_k = np.argsort(-sims)[:3]
    retrieved = [chunks[i] for i in top_k]
    timings["retrieval_search"] = time.monotonic() - start

    start = time.monotonic()
    context_block = "\n---\n".join(retrieved)
    ollama_generate(f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}")
    timings["generation"] = time.monotonic() - start

    total = sum(timings.values())
    print(f"\n--- Per-stage latency for {question!r} ---")
    for stage, seconds in timings.items():
        print(f"  {stage:<16} {seconds:>7.3f}s  ({100 * seconds / total:>5.1f}%)")
    print(f"  {'total':<16} {total:>7.3f}s")


# ======================================================================
# Per-stage cost/token tracking -- real usage fields from each call,
# aggregated per stage. Ollama exposes prompt_eval_count/eval_count;
# Anthropic/OpenAI expose the same shape of data via `response.usage`
# -- the aggregation pattern below is provider-agnostic.
# ======================================================================


def demo_cost_tracking(chunks: list[str], model: SentenceTransformer, questions: list[str]) -> None:
    usage_by_stage = {"generation": {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}}

    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)

    for question in questions:
        query_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)
        sims = chunk_embeddings @ query_vec
        top_k = np.argsort(-sims)[:3]
        context_block = "\n---\n".join(chunks[i] for i in top_k)

        response = ollama_generate(f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}")
        usage_by_stage["generation"]["prompt_tokens"] += response.get("prompt_eval_count", 0)
        usage_by_stage["generation"]["completion_tokens"] += response.get("eval_count", 0)
        usage_by_stage["generation"]["calls"] += 1

    print(f"\n--- Token usage across {len(questions)} questions ---")
    stage = usage_by_stage["generation"]
    print(f"  generation: {stage['calls']} calls, {stage['prompt_tokens']} prompt tokens, {stage['completion_tokens']} completion tokens")
    print(f"  avg prompt tokens/call: {stage['prompt_tokens'] / stage['calls']:.0f}")
    print(f"  avg completion tokens/call: {stage['completion_tokens'] / stage['calls']:.0f}")


# ======================================================================
# Semantic cache -- embed each query, check cosine similarity against
# past cached queries, serve the cached response above a similarity
# threshold instead of calling the LLM again.
# ======================================================================


def demo_semantic_cache(chunks: list[str], model: SentenceTransformer, questions: list[str], threshold: float = 0.85) -> None:
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    cache: list[dict] = []  # [{"query_vec": ..., "question": ..., "answer": ...}]
    hits, misses = 0, 0

    for question in questions:
        query_vec = model.encode(question, convert_to_numpy=True, normalize_embeddings=True)

        best_sim, best_entry = 0.0, None
        for entry in cache:
            sim = float(query_vec @ entry["query_vec"])
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_entry is not None and best_sim >= threshold:
            hits += 1
            print(f"  CACHE HIT  (sim={best_sim:.3f})  {question!r}  ->  matched cached {best_entry['question']!r}")
            continue

        misses += 1
        sims = chunk_embeddings @ query_vec
        top_k = np.argsort(-sims)[:3]
        context_block = "\n---\n".join(chunks[i] for i in top_k)
        response = ollama_generate(f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}")
        cache.append({"query_vec": query_vec, "question": question, "answer": response["response"]})
        print(f"  CACHE MISS (best_sim={best_sim:.3f})  {question!r}  ->  called the LLM, cached the result")

    print(f"\n  {hits} hits, {misses} misses out of {len(questions)} queries ({100 * hits / len(questions):.0f}% hit rate)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Production considerations demos.")
    parser.add_argument("--part", choices=["retry", "circuit-breaker", "latency", "cost", "cache", "all"], default="all")
    args = parser.parse_args()

    chunks = build_corpus()
    print(f"Corpus: {len(chunks)} chunks")

    if args.part in ("retry", "all"):
        demo_retry_backoff()
    if args.part in ("circuit-breaker", "all"):
        demo_circuit_breaker()
    if args.part in ("latency", "cost", "cache", "all"):
        model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    if args.part in ("latency", "all"):
        demo_latency_breakdown(chunks, model, "How many attention heads did they use?")
    if args.part in ("cost", "all"):
        demo_cost_tracking(chunks, model, COST_DEMO_QUESTIONS)
    if args.part in ("cache", "all"):
        demo_semantic_cache(chunks, model, CACHE_DEMO_QUESTIONS)

"""
CAG (Cache-Augmented Generation): skip retrieval entirely and load the
whole knowledge base into the LLM's context, reusing a cached prefill
across repeated queries instead of retrieving relevant chunks per query.

Scope note: the original CAG paper's actual mechanism is precomputing and
reusing an LLM's raw KV-cache -- only possible with a self-hosted,
open-weight model, since hosted APIs never expose `past_key_values`.
Running an 8B+ open-weight model locally isn't realistic on an 8GB
machine with no dedicated GPU (it needs 16GB+ RAM even quantized). This
recipe measures the closest real equivalent available via a hosted API:
Anthropic's `cache_control` prompt caching, applied to the entire sample
document. It's an honest substitution, not the literal paper mechanism --
see the README and blog page for the full distinction.

Two modes are compared on the same document, same questions:
1. fresh  -- the whole document resent as the system prompt on every
   call, no caching. This is the "just stuff it in context" baseline.
2. cached -- the document is cache-annotated; the first call pays a
   real premium to write the cache, every later call within the TTL
   reads from it instead of reprocessing the document.
"""

import argparse
from pathlib import Path

import llm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

DEFAULT_MODEL = "claude-sonnet-5"

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


def load_document_text() -> str:
    """The entire document, not chunks of it -- CAG's whole premise is
    that the corpus is small enough to load in full."""
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def run_query(mode: str, question: str, model: str) -> None:
    load_environment()
    document_text = load_document_text()

    if mode == "cached":
        answer, usage, elapsed = llm.generate_cached(document_text, question, model)
    else:
        answer, usage, elapsed = llm.generate_fresh(document_text, question, model)

    cost = llm.cost_from_usage(usage)
    print(f"Question: {question}\n")
    print(f"Answer: {answer}\n")
    print(f"Usage: {usage}")
    print(f"Latency: {elapsed:.2f}s   Cost: ${cost:.5f}")


def run_comparison(model: str) -> None:
    load_environment()
    document_text = load_document_text()

    print("=== Fresh (no caching, full document resent every call) ===")
    fresh_total_latency = 0.0
    fresh_total_cost = 0.0
    fresh_hits = 0
    for question, keyword in TEXT_EVAL_SET:
        answer, usage, elapsed = llm.generate_fresh(document_text, question, model)
        cost = llm.cost_from_usage(usage)
        fresh_total_latency += elapsed
        fresh_total_cost += cost
        hit = keyword.lower() in answer.lower()
        fresh_hits += hit
        print(f"  {'PASS' if hit else 'FAIL'}  {elapsed:5.2f}s  ${cost:.5f}  input_tokens={usage['input_tokens']:<6}  {question[:45]}")

    print("\n=== Cached (cache_control on the document, 5-minute TTL) ===")
    cached_total_latency = 0.0
    cached_total_cost = 0.0
    cached_hits = 0
    for i, (question, keyword) in enumerate(TEXT_EVAL_SET):
        answer, usage, elapsed = llm.generate_cached(document_text, question, model)
        cost = llm.cost_from_usage(usage)
        cached_total_latency += elapsed
        cached_total_cost += cost
        hit = keyword.lower() in answer.lower()
        cached_hits += hit
        label = "cache WRITE" if usage["cache_creation_input_tokens"] > 0 else "cache READ"
        print(f"  {'PASS' if hit else 'FAIL'}  {elapsed:5.2f}s  ${cost:.5f}  {label:11s} (write={usage['cache_creation_input_tokens']}, read={usage['cache_read_input_tokens']})  {question[:35]}")

    n = len(TEXT_EVAL_SET)
    print(f"\n=== Totals over {n} questions against the same document ===")
    print(f"Fresh:  total latency {fresh_total_latency:.2f}s   total cost ${fresh_total_cost:.5f}   answer accuracy {fresh_hits}/{n}")
    print(f"Cached: total latency {cached_total_latency:.2f}s   total cost ${cached_total_cost:.5f}   answer accuracy {cached_hits}/{n}")
    print(f"Cost reduction:    {(1 - cached_total_cost / fresh_total_cost) * 100:.1f}%")
    print(f"Latency reduction: {(1 - cached_total_latency / fresh_total_latency) * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CAG demo.")
    parser.add_argument("--mode", choices=["fresh", "cached"], default="cached")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--compare", action="store_true", help="Measure real cost/latency: fresh vs. cached, across 8 questions on the same document.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.model)
    elif args.query:
        run_query(args.mode, args.query, args.model)
    else:
        parser.print_help()

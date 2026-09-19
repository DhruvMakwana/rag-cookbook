"""
RAGAS: run a real naive RAG pipeline over the standard sample paper, then
score its real retrieved contexts and real generated answers with RAGAS's
four core metrics -- Faithfulness, Answer Relevancy (now `ResponseRelevancy`
in current ragas), Context Precision, Context Recall -- and see whether
the four-metric diagnostic table actually holds up against a real system.

Scope note: ragas 0.4.x ships two parallel evaluation APIs. The
`evaluate()` + `EvaluationDataset` + named-metric-class shape used here is
the one that actually produces the four-metrics-together diagnostic table
this page is about -- it still works, but every import fires a
DeprecationWarning pointing at the newer `ragas.metrics.collections` +
`.ascore()` API, which scores one metric at a time and isn't wired for
this comparison shape. Both are documented; this recipe uses the working
one for the intended demonstration.
"""

import argparse
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, util

import llm

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_PDF_PATH = DATA_DIR / "attention_is_all_you_need.pdf"
DEFAULT_PDF_URL = "https://arxiv.org/pdf/1706.03762"

TEXT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
JUDGE_MODEL = "claude-sonnet-5"

# Question, a real reference (ground-truth) answer, and the retrieval
# keyword used elsewhere in this repo -- reused here as RAGAS's Context
# Recall metric specifically needs a reference answer to check retrieved
# context against, which the standard EVAL_SET didn't carry before.
EVAL_SET = [
    {"question": "How many attention heads did they use?", "reference": "The base model used 8 attention heads (h = 8).", "keyword": "h = 8"},
    {"question": "What is the model's embedding dimension?", "reference": "The base model's embedding dimension is 512 (dmodel = 512).", "keyword": "dmodel = 512"},
    {"question": "How many layers are in the encoder?", "reference": "The encoder has 6 layers (N = 6).", "keyword": "N = 6"},
    {"question": "What optimizer was used for training?", "reference": "They used the Adam optimizer.", "keyword": "Adam"},
    {"question": "What BLEU score did they get on English-to-German translation?", "reference": "The big model achieved a BLEU score of 28.4 on English-to-German translation.", "keyword": "28.4"},
    {"question": "What GPUs was the model trained on?", "reference": "The models were trained on NVIDIA P100 GPUs.", "keyword": "P100"},
    {"question": "How long did the base model train for?", "reference": "The base model trained for 12 hours.", "keyword": "12 hours"},
    {"question": "What dropout rate did they use?", "reference": "They used a dropout rate of Pdrop = 0.1.", "keyword": "Pdrop = 0.1"},
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


# ======================================================================
# 1. A real naive RAG pipeline -- same pattern used across this repo
# ======================================================================


def load_sample_text() -> str:
    from pypdf import PdfReader

    pdf_path = DEFAULT_PDF_PATH if DEFAULT_PDF_PATH.exists() else download_sample_pdf()
    reader = PdfReader(str(pdf_path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_sample_text(chunk_size: int = 500, overlap: int = 50) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(load_sample_text())


def build_text_index(chunks: list[str], text_model: SentenceTransformer) -> np.ndarray:
    return text_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def text_search(query: str, chunks: list[str], chunk_embeddings: np.ndarray, text_model: SentenceTransformer, k: int = 3) -> list[str]:
    query_embedding = text_model.encode(query, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(k, len(chunks)))
    return [chunks[i] for i in top_k.indices.tolist()]


def run_naive_rag(provider: str | None = None) -> list[dict]:
    """Runs the real pipeline end to end for every question in EVAL_SET,
    returning question/answer/retrieved_contexts/reference dicts -- the
    real material RAGAS scores below, not synthetic examples."""
    chunks = chunk_sample_text()
    text_model = SentenceTransformer(TEXT_EMBEDDING_MODEL)
    chunk_embeddings = build_text_index(chunks, text_model)

    samples = []
    for item in EVAL_SET:
        contexts = text_search(item["question"], chunks, chunk_embeddings, text_model, k=3)
        context_block = "\n---\n".join(contexts)
        answer = llm.generate(
            f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {item['question']}",
            provider=provider,
        )
        samples.append({
            "question": item["question"],
            "answer": answer,
            "retrieved_contexts": contexts,
            "reference": item["reference"],
        })
    return samples


# ======================================================================
# 2. Score real pipeline output with RAGAS's four core metrics
# ======================================================================


def build_ragas_dataset(samples: list[dict]):
    from ragas import EvaluationDataset, SingleTurnSample

    return EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        for s in samples
    ])


def run_ragas_eval(samples: list[dict]):
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import RunConfig, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    # ragas's own `HuggingfaceEmbeddings` wrapper is broken in this release
    # -- instantiating it directly raises a pydantic ValidationError from
    # `BaseRagasEmbeddings`'s custom `__get_pydantic_core_schema__`, before
    # you even reach the (also real) missing-async-method issue. The fix
    # is to not use it: wrap a plain LangChain embeddings object with
    # `LangchainEmbeddingsWrapper` instead -- the same pattern already used
    # for the judge LLM via `LangchainLLMWrapper`.
    # Sonnet 5 (and current-generation Claude models generally) removed
    # sampling params -- temperature/top_p/top_k are no longer accepted at
    # all, replaced by adaptive thinking. `LangchainLLMWrapper.agenerate_text`
    # sets `self.langchain_llm.temperature = 0.01` before every single call
    # by default (its own hardcoded assumption that every LLM accepts
    # temperature) -- passing `temperature=0` or omitting it entirely at
    # construction doesn't help, since the wrapper overwrites it per call
    # regardless. `bypass_temperature=True` is the wrapper's own documented
    # escape hatch for exactly this case ("Certain LLMs ... do not support
    # temperature").
    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model=JUDGE_MODEL), bypass_temperature=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=f"sentence-transformers/{TEXT_EMBEDDING_MODEL}"))

    dataset = build_ragas_dataset(samples)
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    # Anthropic's per-minute limits are tighter than ragas's 16-worker
    # default assumes -- a lower worker count avoids tripping 429s on a
    # real account without a high rate-limit tier.
    result = evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=4, timeout=180))
    return result


def run_comparison(provider: str | None = None) -> None:
    load_environment()
    print(f"Running naive RAG over {len(EVAL_SET)} questions...")
    samples = run_naive_rag(provider)

    print("\n=== Naive RAG output ===")
    for s in samples:
        print(f"Q: {s['question']}")
        print(f"A: {s['answer'][:150]}")
        print()

    print("=== RAGAS scores ===")
    result = run_ragas_eval(samples)
    df = result.to_pandas()
    print(df[["user_input", "faithfulness", "answer_relevancy", "llm_context_precision_with_reference", "context_recall"]].to_string())
    print()
    print("Aggregate:", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAGAS deep dive demo.")
    parser.add_argument("--compare", action="store_true", help="Run naive RAG + score it with all 4 RAGAS metrics.")
    parser.add_argument("--provider", choices=["anthropic", "openai"], default=None)
    args = parser.parse_args()

    if args.compare:
        run_comparison(args.provider)
    else:
        parser.print_help()

"""
Documentation-only companion to ragas_deep_dive.py.

Every function below is self-contained (own imports, no cross-function
dependencies) so each one can be extracted and pasted standalone into the
blog. Not run as a script, not kept in sync with ragas_deep_dive.py
automatically.
"""


# --8<-- [start:naive_rag]
def run_naive_rag(question: str, chunks: list, reference: str) -> dict:
    """A real naive RAG call: retrieve top-3 chunks, generate an answer,
    package it into the shape RAGAS needs -- question, answer, the
    retrieved contexts (not just the final answer), and a reference
    (ground-truth) answer for the metrics that need one."""
    import anthropic
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    query_embedding = model.encode(question, convert_to_numpy=True)
    sims = util.cos_sim(query_embedding, chunk_embeddings)[0]
    top_k = sims.topk(min(3, len(chunks)))
    retrieved_contexts = [chunks[i] for i in top_k.indices.tolist()]

    client = anthropic.Anthropic()
    context_block = "\n---\n".join(retrieved_contexts)
    response = client.messages.create(
        model="claude-sonnet-5",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Answer the question using the context below.\n\nContext:\n{context_block}\n\nQuestion: {question}"}],
    )
    answer = response.content[0].text

    return {"question": question, "answer": answer, "retrieved_contexts": retrieved_contexts, "reference": reference}
# --8<-- [end:naive_rag]


# --8<-- [start:ragas_eval]
def run_ragas_eval(samples: list) -> object:
    """Scores real pipeline output with RAGAS's four core metrics.
    `samples` is a list of dicts shaped like `run_naive_rag`'s return
    value.

    Setup requirements for this ragas release: embeddings go through
    `LangchainEmbeddingsWrapper` around a real `langchain-huggingface`
    embeddings object, the same pattern used for the judge LLM via
    `LangchainLLMWrapper` -- ragas's own `HuggingfaceEmbeddings` wrapper
    fails pydantic validation on direct instantiation. The judge LLM
    wrapper also needs `bypass_temperature=True`, since current Claude
    models (Sonnet 5 and later) no longer accept a `temperature`
    parameter, which `LangchainLLMWrapper` otherwise sets on the
    underlying model before every call by default.
    """
    from langchain_anthropic import ChatAnthropic
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas import EvaluationDataset, RunConfig, SingleTurnSample, evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        ResponseRelevancy,
    )

    evaluator_llm = LangchainLLMWrapper(ChatAnthropic(model="claude-sonnet-5"), bypass_temperature=True)
    evaluator_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s["question"],
            response=s["answer"],
            retrieved_contexts=s["retrieved_contexts"],
            reference=s["reference"],
        )
        for s in samples
    ])

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    # Anthropic's per-minute rate limits are tighter than ragas's 16-worker
    # default assumes.
    return evaluate(dataset=dataset, metrics=metrics, run_config=RunConfig(max_workers=4, timeout=180))
# --8<-- [end:ragas_eval]

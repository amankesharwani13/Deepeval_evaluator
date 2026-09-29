from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecision
from src.llm_provider import get_azure_chat_open_ai_llm

from src.config import (
    RAGAS_MODEL,
)


def create_ragas_llm():
    """
    Create the LLM used by RAGAS metrics.
    """

    client = get_azure_chat_open_ai_llm(
        model_kwargs={},
        streaming=False,
        disable_streaming=True,
        callbacks=None,
    )

    return llm_factory(
        RAGAS_MODEL,
        client=client,
    )


async def evaluate_context_precision(
    question: str,
    ground_truth: str,
    retrieved_contexts: list[str],
) -> float:

    if not retrieved_contexts:
        raise ValueError(
            "Retrieved Context is required "
            "for Context Precision."
        )

    llm = create_ragas_llm()

    metric = ContextPrecision(
        llm=llm,
    )

    result = await metric.ascore(
        user_input=question,
        reference=ground_truth,
        retrieved_contexts=retrieved_contexts,
    )

    return float(result.value)

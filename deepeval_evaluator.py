from pydantic import BaseModel

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    GEval, # answer accuracy metric
    HallucinationMetric,
)

from deepeval.models.base_model import DeepEvalBaseLLM

from deepeval.test_case import (
    LLMTestCase,
    SingleTurnParams,
)

from src.llm_provider import get_azure_chat_open_ai_llm

class AzureDeepEvalLLM(DeepEvalBaseLLM):
    """
    Adapter between our LangChain AzureChatOpenAI
    model and DeepEval.
    """

    def __init__(self):
        self.model = get_azure_chat_open_ai_llm(
            model_kwargs={},
            streaming=False,
            disable_streaming=True,
            callbacks=None,
        )

    def load_model(self):
        return self.model

    def generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        chat_model = self.load_model()

        response = chat_model.invoke(prompt)

        if schema is not None:
            return schema.model_validate_json(response.content)

        return response.content

    async def a_generate(
        self,
        prompt: str,
        schema: BaseModel | None = None,
    ):
        chat_model = self.load_model()

        response = await chat_model.ainvoke(prompt)

        if schema is not None:
            return schema.model_validate_json(response.content)

        return response.content

    def get_model_name(self):
        return "Azure OpenAI"


# DeepEval-compatible LLM
llm = AzureDeepEvalLLM()


def create_test_case(
    question: str,
    ground_truth: str,
    ai_response: str,
    # context: list[str] | None = None,
) -> LLMTestCase:
    """
    Convert one Excel row into a DeepEval test case.
    """

    return LLMTestCase(
        input=question,
        actual_output=ai_response,
        expected_output=ground_truth,
        # context=context or [],
         context=[ground_truth],
    )


def evaluate_answer_relevancy(
    test_case: LLMTestCase,
) -> tuple[float | None, str | None]:

    metric = AnswerRelevancyMetric(
        model=llm,
        threshold=None,
        include_reason=True,
    )

    metric.measure(test_case)

    return (
        metric.score,
        metric.reason,
    )


def evaluate_answer_correctness(
    test_case: LLMTestCase,
) -> tuple[float | None, str | None]:

    metric = GEval(
        name="Answer Accuracy",
        model=llm,
        evaluation_steps=[
            (
                "Check whether the actual output "
                "correctly answers the question."
            ),
            (
                "Compare the factual content of the "
                "actual output with the expected output."
            ),
            (
                "Penalize factual contradictions."
            ),
            (
                "Penalize missing important information "
                "when it is necessary to answer the question."
            ),
            (
                "Do not penalize different wording when "
                "the meaning is correct."
            ),
        ],
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
        ],
        threshold=None,
    )

    metric.measure(test_case)

    return (
        metric.score,
        metric.reason,
    )


def evaluate_bias(
    test_case: LLMTestCase,
) -> tuple[float | None, str | None]:

    metric = BiasMetric(
        model=llm,
        threshold=None,
        include_reason=True,
    )

    metric.measure(test_case)

    return (
        metric.score,
        metric.reason,
    )


def evaluate_hallucination(
    test_case: LLMTestCase,
) -> tuple[float | None, str | None]:

    metric = HallucinationMetric(
        model=llm,
        threshold=None,
        include_reason=True,
    )

    metric.measure(test_case)

    return (
        metric.score,
        metric.reason,
    )

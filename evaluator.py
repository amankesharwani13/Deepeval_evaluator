import logging

from src.deepeval_evaluator import (
    create_test_case,
    evaluate_answer_correctness,
    evaluate_answer_relevancy,
    evaluate_bias,
    evaluate_hallucination,
)

from src.ragas_evaluator import (
    evaluate_context_precision,
)


logger = logging.getLogger(__name__)


async def evaluate_single_row(
    question: str,
    ground_truth: str,
    ai_response: str,
    retrieved_contexts: list[str] | None = None,
) -> dict:

    result = {
        "Answer Relevancy": None,
        "Answer Relevancy Reason": None,

        "Answer Accuracy": None,
        "Answer Accuracy Reason": None,

        "Context Precision": None,

        "Hallucination Score": None,
        "Hallucination Reason": None,

        "Bias": None,
        "Bias Reason": None,

        "Evaluation Error": None,
    }

    try:

        # Create DeepEval test case

        test_case = create_test_case(
            question=question,
            ground_truth=ground_truth,
            ai_response=ai_response,
        )

        # Answer Relevancy

        try:

            logger.info(
                "Running Answer Relevancy..."
            )

            score, reason = (
                evaluate_answer_relevancy(
                    test_case
                )
            )

            result["Answer Relevancy"] = score
            result[
                "Answer Relevancy Reason"
            ] = reason

        except Exception as exc:

            logger.exception(
                "Answer Relevancy failed."
            )

            result[
                "Answer Relevancy Reason"
            ] = f"Metric error: {exc}"

        # Answer Accuracy

        try:

            logger.info(
                "Running Answer Accuracy..."
            )

            score, reason = (
                evaluate_answer_correctness(
                    test_case
                )
            )

            result["Answer Accuracy"] = score
            result[
                "Answer Accuracy Reason"
            ] = reason

        except Exception as exc:

            logger.exception(
                "Answer Accuracy failed."
            )

            result[
                "Answer Accuracy Reason"
            ] = f"Metric error: {exc}"

        # Bias

        try:

            logger.info(
                "Running Bias..."
            )

            score, reason = evaluate_bias(
                test_case
            )

            result["Bias"] = score
            result["Bias Reason"] = reason

        except Exception as exc:

            logger.exception(
                "Bias evaluation failed."
            )

            result[
                "Bias Reason"
            ] = f"Metric error: {exc}"

        # Hallucination
        #
        # Ground Truth is already passed as
        # test_case.context in create_test_case().
        #
        # Therefore Hallucination does NOT
        # depend on Retrieved Context.

        try:

            logger.info(
                "Running Hallucination..."
            )

            score, reason = (
                evaluate_hallucination(
                    test_case
                )
            )

            result[
                "Hallucination Score"
            ] = score

            result[
                "Hallucination Reason"
            ] = reason

        except Exception as exc:

            logger.exception(
                "Hallucination evaluation failed."
            )

            result[
                "Hallucination Reason"
            ] = f"Metric error: {exc}"

        # Context Precision
        #
        # Context Precision still requires
        # actual Retrieved Context.
        #
        # Ground Truth must NOT be used here
        # as Retrieved Context.

        if retrieved_contexts:

            try:

                logger.info(
                    "Running Context Precision..."
                )

                score = (
                    await evaluate_context_precision(
                        question=question,
                        ground_truth=ground_truth,
                        retrieved_contexts=retrieved_contexts,
                    )
                )

                result[
                    "Context Precision"
                ] = score

            except Exception as exc:

                logger.exception(
                    "Context Precision failed."
                )

                result[
                    "Context Precision"
                ] = None

                result[
                    "Evaluation Error"
                ] = (
                    f"Context Precision error: {exc}"
                )

        else:

            result[
                "Context Precision"
            ] = None

        return result

    except Exception as exc:

        logger.exception(
            "Unexpected evaluation error."
        )

        result[
            "Evaluation Error"
        ] = str(exc)

        return result

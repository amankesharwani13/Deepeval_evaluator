import logging
from pathlib import Path
from src.confusion_matrix import create_accuracy_confusion_matrix

import pandas as pd

from src.config import (
    INPUT_FILE,
    OUTPUT_FILE,
)

from src.evaluator import (
    evaluate_single_row,
)

from src.excel_handler import (
    read_input_excel,
    save_output_excel,
)

from src.summary import (
    create_summary,
)

# Logging configuration

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(__name__)


def parse_context(
    context_value,
) -> list[str]:

    if pd.isna(context_value):
        return []

    context_text = str(
        context_value
    ).strip()

    if not context_text:
        return []

    # We assume contexts are separated
    # by a blank line.
    contexts = [
        item.strip()
        for item in context_text.split("\n\n")
        if item.strip()
    ]

    return contexts


async def main():

    logger.info(
        "Loading input Excel..."
    )

    df = read_input_excel(
        INPUT_FILE
    )

    logger.info(
        "Found %d rows.", # total rows
        len(df),
    )

    if "Actual Correctness" not in df.columns: # for  confusion matrix
        raise ValueError(
            "The input Excel must contain an "
            "'Actual Correctness' column to create "
            "the Answer Accuracy confusion matrix."
        )

    results = []

    success_count = 0
    failure_count = 0

    has_context = (
        "Retrieved Context"
        in df.columns
    )

    for index, row in df.iterrows():

        row_number = index + 1

        logger.info(
            "Evaluating row %d/%d...",
            row_number,
            len(df),
        )

        question = str(
            row["Question"]
        )

        ground_truth = str(
            row["Ground Truth"]
        )

        ai_response = str(
            row["AI Response"]
        )

        contexts = []

        if has_context:

            contexts = parse_context(
                row["Retrieved Context"]
            )

        result = await evaluate_single_row(
            question=question,
            ground_truth=ground_truth,
            ai_response=ai_response,
            retrieved_contexts=contexts,
        )

        if result.get(
            "Evaluation Error"
        ):

            failure_count += 1

        else:

            success_count += 1

        results.append(result)

    # Create detailed output

    result_df = pd.DataFrame(
        results
    )

    detailed_df = pd.concat(
        [
            df.reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Create summary

    summary_df = create_summary(
        detailed_df
    )

    confusion_df = create_accuracy_confusion_matrix(
        detailed_df,
        actual_column="Actual Correctness",
        score_column="Answer Accuracy",
        threshold=0.5,
   )

    # Save output

    logger.info(
        "Writing output Excel..."
    )

    save_output_excel(
        detailed_df=detailed_df,
        summary_df=summary_df,
        confusion_df=confusion_df,
        output_file=OUTPUT_FILE,
    )

    logger.info(
        "Evaluation completed."
    )

    logger.info(
        "Successful rows: %d",
        success_count,
    )

    logger.info(
        "Rows with errors: %d",
        failure_count,
    )

    logger.info(
        "Output generated: %s",
        Path(OUTPUT_FILE).resolve(),
    )


if __name__ == "__main__":

    import asyncio

    asyncio.run(main())

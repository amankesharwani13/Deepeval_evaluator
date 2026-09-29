from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Question",
    "Ground Truth",
    "AI Response",
]

OPTIONAL_COLUMNS = [
    "Retrieved Context",
    "Actual Correctness",
]


def read_input_excel(file_path: str) -> pd.DataFrame:
    """
    Read evaluation data from an Excel file.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Input Excel file not found: {file_path}"
        )

    df = pd.read_excel(path)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    return df


def save_output_excel(
    detailed_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    output_file: str,
) -> None:
    """
    Save detailed results and summary into
    separate Excel sheets.
    """

    output_path = Path(output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        detailed_df.to_excel(
            writer,
            sheet_name="Detailed Results",
            index=False,
        )

        summary_df.to_excel(
            writer,
            sheet_name="Summary",
            index=False,
        )

        confusion_df.to_excel(
            writer,
            sheet_name="Accuracy Confusion Matrix",
        )

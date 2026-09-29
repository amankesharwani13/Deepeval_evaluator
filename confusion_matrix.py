import pandas as pd


def create_accuracy_confusion_matrix(
    df: pd.DataFrame,
    actual_column: str = "Actual Correctness",
    score_column: str = "Answer Accuracy",
    threshold: float = 0.5,
) -> pd.DataFrame:

    # Check actual correctness column
    if actual_column not in df.columns:
        raise ValueError(
            f"'{actual_column}' is required to create "
            "the Answer Accuracy confusion matrix."
        )

    # Check Answer Accuracy column
    if score_column not in df.columns:
        raise ValueError(
            f"'{score_column}' is required to create "
            "the Answer Accuracy confusion matrix."
        )

    # Convert Answer Accuracy into numeric values
    scores = pd.to_numeric(df[score_column], errors="coerce")

    # Remove rows where score or actual class is missing
    valid_mask = scores.notna() & df[actual_column].notna()

    scores = scores[valid_mask]
    actual = df.loc[valid_mask, actual_column]

    # Convert Answer Accuracy score into predicted class
    predicted = scores.apply(
        lambda score: "Correct"
        if score >= threshold
        else "Incorrect"
    )

    # Create confusion matrix
    matrix = pd.crosstab(
        actual,
        predicted,
        rownames=["Actual"],
        colnames=["Predicted"],
        dropna=False,
    )

    # Make sure both classes are always visible
    for column in ["Correct", "Incorrect"]:
        if column not in matrix.columns:
            matrix[column] = 0

    matrix = matrix[["Correct", "Incorrect"]]

    for row in ["Correct", "Incorrect"]:
        if row not in matrix.index:
            matrix.loc[row] = 0

    matrix = matrix.loc[["Correct", "Incorrect"]]

    return matrix

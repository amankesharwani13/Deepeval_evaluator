import pandas as pd


METRIC_COLUMNS = [
    "Answer Relevancy",
    "Answer Accuracy",
    "Hallucination Score",
    "Bias",
]


def create_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for metric in METRIC_COLUMNS:

        if metric not in df.columns:
            continue

        values = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        valid_values = values.dropna()

        if valid_values.empty:

            rows.append(
                {
                    "Metric": metric,
                    "Average Score": None,
                    "Percentage": None,
                    "Evaluated Rows": 0,
                }
            )

            continue

        average_score = (
            valid_values.mean()
        )

        percentage = (
            average_score * 100
        )

        rows.append(
            {
                "Metric": metric,
                "Average Score": round(
                    average_score,
                    4,
                ),
                "Percentage": round(
                    percentage,
                    2,
                ),
                "Evaluated Rows": len(
                    valid_values
                ),
            }
        )

    return pd.DataFrame(rows)

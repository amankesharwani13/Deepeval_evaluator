import os

from dotenv import load_dotenv


load_dotenv()


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

RAGAS_MODEL = os.getenv(
    "RAGAS_MODEL",
    "gpt-4o-mini",
)

DEEPEVAL_MODEL = os.getenv(
    "DEEPEVAL_MODEL",
    "gpt-4o-mini",
)

INPUT_FILE = os.getenv(
    "INPUT_FILE",
    "data/input.xlsx",
)

OUTPUT_FILE = os.getenv(
    "OUTPUT_FILE",
    "output/evaluation_result.xlsx",
)


if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY is not configured. "
        "Add it to your .env file."
    )

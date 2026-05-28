"""
Dataset loading and caching for the Bitext Customer Service dataset.

The dataset is loaded once from HuggingFace and cached in-process.
Schema:
  - flags       : str  — behavioral tags
  - instruction : str  — user message / customer query
  - category    : str  — high-level topic (ACCOUNT, REFUND, SHIPPING, …)
  - intent      : str  — specific action (get_refund, track_order, …)
  - response    : str  — agent reply
"""

from __future__ import annotations

import logging
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
DATASET_SPLIT = "train"


@lru_cache(maxsize=1)
def load_dataset() -> pd.DataFrame:
    """
    Load the Bitext Customer Service dataset from HuggingFace.

    Downloads and caches the dataset on first call; subsequent calls return
    the in-process cached DataFrame instantly.

    Returns:
        pd.DataFrame with columns: flags, instruction, category, intent, response.

    Raises:
        RuntimeError: If the dataset cannot be downloaded or parsed.
    """
    try:
        from datasets import load_dataset as hf_load_dataset

        logger.info("Loading dataset '%s' from HuggingFace…", DATASET_NAME)
        hf_ds = hf_load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        df: pd.DataFrame = hf_ds.to_pandas()

        # Normalise text so lookups are case-insensitive by default
        df["category"] = df["category"].str.upper().str.strip()
        df["intent"] = df["intent"].str.lower().str.strip()
        df["instruction"] = df["instruction"].str.strip()
        df["response"] = df["response"].str.strip()

        logger.info(
            "Dataset loaded: %d records, %d categories, %d intents.",
            len(df),
            df["category"].nunique(),
            df["intent"].nunique(),
        )
        return df

    except Exception as exc:
        raise RuntimeError(
            f"Failed to load dataset '{DATASET_NAME}': {exc}"
        ) from exc


def get_dataframe() -> pd.DataFrame:
    """
    Return the cached Bitext Customer Service dataset as a DataFrame.

    Convenience wrapper around :func:`load_dataset` for use in tools.
    """
    return load_dataset()

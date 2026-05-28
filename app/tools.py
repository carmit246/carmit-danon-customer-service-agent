"""
LangChain tools for querying the Bitext Customer Service dataset.

Each tool has:
  - A descriptive name and docstring that guides the LLM's tool selection.
  - A Pydantic input schema with typed, described fields.
  - Typed return values.

Design philosophy: a few well-described tools beat many vague ones.
The descriptions are as important as the implementations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from .data_loader import get_dataframe

logger = logging.getLogger(__name__)


# ── Pydantic input schemas ────────────────────────────────────────────────────


class GetIntentsInput(BaseModel):
    """Input schema for :func:`get_intents`."""

    category: Optional[str] = Field(
        default=None,
        description=(
            "Category name to filter by (e.g. 'REFUND', 'SHIPPING'). "
            "Case-insensitive. If omitted, all intents across all categories are returned."
        ),
    )


class CountRecordsInput(BaseModel):
    """Input schema for :func:`count_records`."""

    category: Optional[str] = Field(
        default=None,
        description="Category to filter by (e.g. 'REFUND'). Case-insensitive.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent to filter by (e.g. 'get_refund'). Case-insensitive.",
    )


class ShowExamplesInput(BaseModel):
    """Input schema for :func:`show_examples`."""

    n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="How many examples to return (1–20).",
    )
    category: Optional[str] = Field(
        default=None,
        description="Category to restrict examples to. Case-insensitive.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent to restrict examples to. Case-insensitive.",
    )
    search_term: Optional[str] = Field(
        default=None,
        description=(
            "Free-text keyword to search for inside user instructions "
            "(e.g. 'money back', 'cancel', 'track'). Case-insensitive."
        ),
    )


class IntentDistributionInput(BaseModel):
    """Input schema for :func:`get_intent_distribution`."""

    category: str = Field(
        description="Category name to compute intent distribution for (e.g. 'ACCOUNT').",
    )


class GetSampleForSummaryInput(BaseModel):
    """Input schema for :func:`get_sample_for_summary`."""

    category: Optional[str] = Field(
        default=None,
        description="Category to sample from.",
    )
    intent: Optional[str] = Field(
        default=None,
        description="Intent to sample from.",
    )
    n: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Number of examples to include in the sample (5–50).",
    )
    include_responses: bool = Field(
        default=True,
        description="Whether to include agent responses alongside user instructions.",
    )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _filter_df(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> Any:
    """Return a filtered slice of the dataset DataFrame."""
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    return df


# ── Tool functions ────────────────────────────────────────────────────────────


@tool
def get_categories() -> dict[str, Any]:
    """
    Return the list of all unique top-level categories in the dataset.

    Use this tool when the user asks:
      - "What categories exist?"
      - "What topics are covered?"
      - "What kinds of issues are in the dataset?"
    It is also useful as a first step before filtering by category, to confirm
    that a category name exists and see its exact spelling.

    Returns a dict with keys 'categories' (sorted list) and 'count'.
    """
    df = get_dataframe()
    categories: list[str] = sorted(df["category"].unique().tolist())
    return {"categories": categories, "count": len(categories)}


@tool(args_schema=GetIntentsInput)
def get_intents(category: Optional[str] = None) -> dict[str, Any]:
    """
    Return all unique intents, optionally restricted to one category.

    Use this tool when the user asks:
      - "What intents exist in the SHIPPING category?"
      - "What specific actions are covered under REFUND?"
    Also call this before filtering by intent to confirm the exact intent name,
    or to list what's available when a user asks about a broad concept.

    Returns a dict with 'intents' (sorted list), 'count', and (if filtered) 'category'.
    """
    df = get_dataframe()
    if category:
        cat = category.upper().strip()
        subset = df[df["category"] == cat]
        if subset.empty:
            return {
                "error": f"Category '{category}' not found. "
                "Call get_categories() to see valid category names.",
            }
        intents = sorted(subset["intent"].unique().tolist())
        return {"category": cat, "intents": intents, "count": len(intents)}

    intents = sorted(df["intent"].unique().tolist())
    return {"intents": intents, "count": len(intents)}


@tool(args_schema=CountRecordsInput)
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict[str, Any]:
    """
    Count dataset records that match optional category and/or intent filters.

    Use this tool when the user asks:
      - "How many refund requests are there?"
      - "How many records belong to the SHIPPING category?"
      - "What is the volume of 'cancel_order' intents?"
    You can filter by category alone, intent alone, or both together.

    Returns a dict with 'count' and 'filters' showing what was applied.
    """
    subset = _filter_df(category=category, intent=intent)
    return {
        "count": int(len(subset)),
        "filters": {
            "category": category.upper() if category else None,
            "intent": intent.lower() if intent else None,
        },
    }


@tool(args_schema=ShowExamplesInput)
def show_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
    search_term: Optional[str] = None,
) -> dict[str, Any]:
    """
    Show N example records from the dataset, with optional filters.

    Use this tool when the user wants to:
      - "Show me 5 examples from the SHIPPING category."
      - "Give me samples of 'get_refund' interactions."
      - "Find examples where customers ask about 'money back' or 'cancellation'."
    For broad concept searches (e.g. 'people wanting their money back'), use
    search_term with relevant keywords; optionally combine with category/intent.

    Returns a dict with 'examples' (list of instruction/intent/category/response dicts),
    'returned', 'total_matching', and 'filters'.
    """
    df = _filter_df(category=category, intent=intent)

    if search_term:
        mask = (
            df["instruction"].str.contains(search_term, case=False, na=False)
            | df["response"].str.contains(search_term, case=False, na=False)
        )
        df = df[mask]

    if df.empty:
        return {
            "error": "No records match the given filters.",
            "examples": [],
            "returned": 0,
            "total_matching": 0,
        }

    sample = df.sample(n=min(n, len(df)))
    examples = sample[["instruction", "intent", "category", "response"]].to_dict(
        orient="records"
    )

    return {
        "examples": examples,
        "returned": len(examples),
        "total_matching": int(len(df)),
        "filters": {
            "category": category.upper() if category else None,
            "intent": intent.lower() if intent else None,
            "search_term": search_term,
        },
    }


@tool(args_schema=IntentDistributionInput)
def get_intent_distribution(category: str) -> dict[str, Any]:
    """
    Return the count and percentage breakdown of intents within a category.

    Use this tool when the user asks:
      - "What is the distribution of intents in the ACCOUNT category?"
      - "What are the most common request types under REFUND?"
      - "Break down the SHIPPING category by intent."

    Returns a dict with 'category', 'total', and 'distribution' — a list of
    {intent, count, percentage} dicts sorted by frequency descending.
    """
    df = get_dataframe()
    cat = category.upper().strip()
    subset = df[df["category"] == cat]

    if subset.empty:
        return {
            "error": f"Category '{category}' not found. "
            "Call get_categories() to see valid category names.",
        }

    counts = subset["intent"].value_counts()
    total = int(len(subset))
    distribution = [
        {
            "intent": intent,
            "count": int(cnt),
            "percentage": round(cnt / total * 100, 1),
        }
        for intent, cnt in counts.items()
    ]

    return {
        "category": cat,
        "total": total,
        "distribution": distribution,
    }


@tool(args_schema=GetSampleForSummaryInput)
def get_sample_for_summary(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 20,
    include_responses: bool = True,
) -> dict[str, Any]:
    """
    Retrieve a representative text sample to use for qualitative summarization.

    Use this tool when the user asks for:
      - "Summarize the FEEDBACK category."
      - "How do agents typically respond to cancellation requests?"
      - "What themes appear in SHIPPING complaints?"
    First call this tool to gather raw examples, then synthesize the results
    into a meaningful summary in your final response.

    Set include_responses=True (default) to see both customer messages and
    agent replies — useful for understanding how agents handle specific issues.

    Returns a dict with 'sample' (list of entries), 'sample_size',
    'total_in_scope', and 'filters'.
    """
    df = _filter_df(category=category, intent=intent)

    if df.empty:
        return {
            "error": "No records match the given filters.",
            "sample": [],
            "sample_size": 0,
        }

    sample = df.sample(n=min(n, len(df)))

    entries: list[dict[str, str]] = []
    for _, row in sample.iterrows():
        entry: dict[str, str] = {
            "instruction": row["instruction"],
            "intent": row["intent"],
        }
        if include_responses:
            entry["response"] = row["response"]
        entries.append(entry)

    return {
        "sample": entries,
        "sample_size": len(entries),
        "total_in_scope": int(len(df)),
        "filters": {
            "category": category.upper() if category else None,
            "intent": intent.lower() if intent else None,
        },
    }


# ── Public registry ───────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_categories,
    get_intents,
    count_records,
    show_examples,
    get_intent_distribution,
    get_sample_for_summary,
]

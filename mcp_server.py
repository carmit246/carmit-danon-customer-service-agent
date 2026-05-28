#!/usr/bin/env python3
"""
MCP server for the Customer Service Data Analyst Agent (Task 3).

Exposes all 6 dataset analysis tools via the Model Context Protocol using FastMCP.
Any MCP-compatible client — Claude Desktop, the MCP Inspector, a custom Python
client, etc. — can connect and call these tools directly.

Usage
─────
  # stdio transport (default — used by Claude Desktop and the MCP CLI)
  python3 mcp_server.py

  # HTTP / SSE transport — useful for web clients
  python3 mcp_server.py --sse
  python3 mcp_server.py --sse --port 9000

Connecting a client (see README for more examples)
───────────────────────────────────────────────────
  # Inspect interactively
  npx @modelcontextprotocol/inspector python3 mcp_server.py

  # Python async client (stdio)
  from fastmcp import Client
  import asyncio

  async def main():
      async with Client("python3 mcp_server.py") as client:
          result = await client.call_tool("get_categories", {})
          print(result)

  asyncio.run(main())
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# ── Bootstrap ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from fastmcp import FastMCP

from app.data_loader import get_dataframe

# ── Server instance ────────────────────────────────────────────────────────────

mcp = FastMCP(
    "Customer Service Data Analyst",
    instructions=(
        "Tools for analysing the Bitext Customer Service dataset — a labelled "
        "collection of customer support interactions. "
        "Categories include ACCOUNT, CANCELLATION_FEE, DELIVERY, FEEDBACK, "
        "INVOICE, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION, and more. "
        "Use get_categories() first to discover what is available."
    ),
)

# ── Internal helper ────────────────────────────────────────────────────────────


def _filter_df(
    category: Optional[str] = None,
    intent: Optional[str] = None,
):
    """Return a filtered slice of the dataset."""
    df = get_dataframe()
    if category:
        df = df[df["category"] == category.upper().strip()]
    if intent:
        df = df[df["intent"] == intent.lower().strip()]
    return df


# ── Tool definitions ───────────────────────────────────────────────────────────


@mcp.tool()
def get_categories() -> dict:
    """
    Return all unique top-level categories in the Bitext customer service dataset.

    Call this first to discover available categories and verify exact spelling
    before using other tools that accept a category parameter.

    Returns a dict with 'categories' (sorted list) and 'count'.
    """
    df = get_dataframe()
    categories: list[str] = sorted(df["category"].unique().tolist())
    return {"categories": categories, "count": len(categories)}


@mcp.tool()
def get_intents(category: Optional[str] = None) -> dict:
    """
    Return all unique intents in the dataset, optionally filtered to one category.

    Use this to explore what specific actions are covered under a category, or
    to confirm the exact spelling of an intent before passing it to other tools.

    Args:
        category: Category name (e.g. 'REFUND', 'SHIPPING'). Case-insensitive.
                  If omitted, returns every intent across all categories.

    Returns a dict with 'intents' (sorted list), 'count', and 'category' (if filtered).
    """
    df = get_dataframe()
    if category:
        cat = category.upper().strip()
        subset = df[df["category"] == cat]
        if subset.empty:
            return {
                "error": f"Category '{category}' not found. "
                "Call get_categories() to see valid names."
            }
        intents = sorted(subset["intent"].unique().tolist())
        return {"category": cat, "intents": intents, "count": len(intents)}
    intents = sorted(df["intent"].unique().tolist())
    return {"intents": intents, "count": len(intents)}


@mcp.tool()
def count_records(
    category: Optional[str] = None,
    intent: Optional[str] = None,
) -> dict:
    """
    Count dataset records matching optional category and/or intent filters.

    Use this to measure how many records belong to a category or intent.
    You can combine both filters for finer granularity.

    Args:
        category: Category to filter by (e.g. 'REFUND'). Case-insensitive.
        intent:   Intent to filter by (e.g. 'get_refund'). Case-insensitive.

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


@mcp.tool()
def show_examples(
    n: int = 5,
    category: Optional[str] = None,
    intent: Optional[str] = None,
    search_term: Optional[str] = None,
) -> dict:
    """
    Return N example records with optional filters or keyword search.

    Use this to browse sample customer queries and agent responses, or to find
    examples matching a concept (e.g. 'money back', 'track', 'cancel').

    Args:
        n:           Number of examples (1–20).
        category:    Restrict to this category. Case-insensitive.
        intent:      Restrict to this intent. Case-insensitive.
        search_term: Keyword to search for in customer instructions (e.g. 'money back').

    Returns a dict with 'examples' (list of {instruction, intent, category, response}),
    'returned', and 'total_matching'.
    """
    df = _filter_df(category=category, intent=intent)
    if search_term:
        mask = df["instruction"].str.contains(search_term, case=False, na=False) | \
               df["response"].str.contains(search_term, case=False, na=False)
        df = df[mask]
    if df.empty:
        return {"error": "No records match the given filters.", "examples": []}
    n = max(1, min(n, 20, len(df)))
    sample = df.sample(n=n)
    return {
        "examples": sample[["instruction", "intent", "category", "response"]].to_dict(
            orient="records"
        ),
        "returned": len(sample),
        "total_matching": int(len(df)),
    }


@mcp.tool()
def get_intent_distribution(category: str) -> dict:
    """
    Return the count and percentage breakdown of intents within a category.

    Use this to understand the composition of a category — which intents are
    most frequent and how they are distributed.

    Args:
        category: Category name (e.g. 'ACCOUNT'). Case-insensitive.

    Returns a dict with 'category', 'total', and 'distribution' — a list of
    {intent, count, percentage} dicts sorted by frequency descending.
    """
    df = get_dataframe()
    cat = category.upper().strip()
    subset = df[df["category"] == cat]
    if subset.empty:
        return {"error": f"Category '{category}' not found. Call get_categories()."}
    counts = subset["intent"].value_counts()
    total = int(len(subset))
    return {
        "category": cat,
        "total": total,
        "distribution": [
            {
                "intent": intent,
                "count": int(cnt),
                "percentage": round(cnt / total * 100, 1),
            }
            for intent, cnt in counts.items()
        ],
    }


@mcp.tool()
def get_sample_for_summary(
    category: Optional[str] = None,
    intent: Optional[str] = None,
    n: int = 20,
    include_responses: bool = True,
) -> dict:
    """
    Return a representative text sample for qualitative analysis or summarisation.

    Use this when you want to understand the language patterns, themes, or tone
    within a category or intent. Set include_responses=True to also see how
    agents typically reply.

    Args:
        category:          Category to sample from.
        intent:            Intent to sample from.
        n:                 Sample size (5–50).
        include_responses: Include agent responses alongside user instructions.

    Returns a dict with 'sample' (list of entries), 'sample_size', 'total_in_scope'.
    """
    df = _filter_df(category=category, intent=intent)
    if df.empty:
        return {"error": "No records match the given filters.", "sample": []}
    n = max(5, min(n, 50, len(df)))
    sample = df.sample(n=n)
    entries: list[dict] = []
    for _, row in sample.iterrows():
        entry: dict = {"instruction": row["instruction"], "intent": row["intent"]}
        if include_responses:
            entry["response"] = row["response"]
        entries.append(entry)
    return {
        "sample": entries,
        "sample_size": len(entries),
        "total_in_scope": int(len(df)),
    }


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MCP server for the Customer Service Data Analyst Agent"
    )
    parser.add_argument(
        "--sse",
        action="store_true",
        help="Use HTTP/SSE transport instead of the default stdio transport.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the SSE transport (default: 8000).",
    )
    args = parser.parse_args()

    if args.sse:
        print(f"Starting MCP server on http://localhost:{args.port}/sse …")
        mcp.run(transport="sse", port=args.port)
    else:
        mcp.run()  # stdio — for Claude Desktop / MCP CLI

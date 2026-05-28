"""
Query router for the Customer Service Data Analyst Agent.

Classifies every incoming user query as one of:
  - structured   : concrete, data-driven question answerable by querying the dataset
  - unstructured : open-ended question requiring qualitative analysis / summarisation
  - out_of_scope : unrelated to the dataset — must be declined politely

The router runs as the *first* node in the LangGraph before any tool selection.
"""

from __future__ import annotations

import logging
from typing import Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are a query router for a customer-service data analyst agent.

The agent works exclusively with the **Bitext Customer Service dataset** — a collection of
customer support interactions labelled by category (ACCOUNT, CANCELLATION_FEE, CONTACT,
DELIVERY, FEEDBACK, INVOICE, NEWSLETTER, ORDER, PAYMENT, REFUND, SHIPPING, SUBSCRIPTION)
and by specific intent (get_refund, track_order, change_order, etc.).

Classify the user's query into exactly one of three types:

**structured**
Questions with concrete, data-driven answers that can be resolved by querying the dataset.
Also includes questions about the conversation history or the agent's memory of the user.
Examples:
  - "What categories exist in the dataset?"
  - "How many refund requests did we get?"
  - "Show me 5 examples from the SHIPPING category."
  - "What is the distribution of intents in the ACCOUNT category?"
  - "Show me examples of people wanting their money back."
  - "What do you remember about me?"
  - "What have we talked about?"
  - "Show me 3 more." (follow-up referencing earlier results)

**unstructured**
Open-ended questions that require qualitative analysis, summarisation, or pattern recognition
over the dataset content.
Examples:
  - "Summarize the FEEDBACK category."
  - "How do agents typically respond to cancellation requests?"
  - "What themes appear in SHIPPING complaints?"
  - "Summarize how agents respond to complaint intents."

**out_of_scope**
Questions unrelated to the Bitext customer-service dataset — general knowledge, creative tasks,
software recommendations, current events, etc.
Examples:
  - "Who won the 2024 Champions League?"
  - "Write me a poem about customer service."
  - "What's the best CRM software for handling complaints?"
  - "Who is the president of France?"

Return your classification and a one-sentence reasoning.
"""


# ── Output schema ─────────────────────────────────────────────────────────────


class QueryClassification(BaseModel):
    """Structured output from the query router."""

    query_type: Literal["structured", "unstructured", "out_of_scope"] = Field(
        description="Category of the query."
    )
    reasoning: str = Field(
        description="One-sentence explanation of the classification decision."
    )


# ── Factory ───────────────────────────────────────────────────────────────────


def create_router(llm: ChatOpenAI) -> Callable[[str], QueryClassification]:
    """
    Build and return a query-classification callable backed by *llm*.

    Args:
        llm: A :class:`~langchain_openai.ChatOpenAI` instance (should have
             temperature=0 for deterministic classification).

    Returns:
        A function ``classify(query: str) -> QueryClassification``.
    """
    structured_llm = llm.with_structured_output(QueryClassification)

    def classify(query: str) -> QueryClassification:
        """
        Classify a user query as structured, unstructured, or out_of_scope.

        Args:
            query: Raw user input string.

        Returns:
            :class:`QueryClassification` with ``query_type`` and ``reasoning``.
        """
        messages = [
            SystemMessage(content=_ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=f"Classify this query: {query}"),
        ]
        result: QueryClassification = structured_llm.invoke(messages)
        logger.debug(
            "[Router] '%s…' → %s | %s",
            query[:60],
            result.query_type,
            result.reasoning,
        )
        return result

    return classify

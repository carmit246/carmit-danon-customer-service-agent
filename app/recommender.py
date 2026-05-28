"""
Query recommendation engine (Bonus B).

When the user asks "What should I query next?" (or similar), this module:
  1. Inspects the session's conversation history and user profile.
  2. Suggests a single relevant follow-up query — without executing it.
  3. Supports refinement: the user can ask for a different suggestion.
  4. Detects confirmation ("yes / do it") vs. refinement vs. moving on.

None of this logic lives inside the LangGraph graph; it runs as a pre-/post-
processing layer in the CLI (main.py) and the Streamlit UI (streamlit_app.py),
which keeps the graph clean and the recommendation flow easy to test standalone.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# How many recent text messages to include in context for LLM calls.
_CONTEXT_WINDOW = 10


# ── Pydantic output schemas ───────────────────────────────────────────────────


class _Suggestion(BaseModel):
    """A single query suggestion with brief justification."""

    query: str = Field(
        description=(
            "The suggested query, written exactly as the user would type it "
            "(e.g. 'Show me 5 examples from the REFUND category')."
        )
    )
    reasoning: str = Field(
        description="One sentence explaining why this is a good follow-up."
    )


class _RecAction(BaseModel):
    """Classification of a user's response to a pending recommendation."""

    action: Literal["confirm", "refine", "other"] = Field(
        description=(
            "'confirm' — user accepts the suggestion and wants it executed "
            "(e.g. 'yes', 'do it', 'go ahead', 'ok', 'sure', 'sounds good'); "
            "'refine'  — user wants a different or modified suggestion "
            "(e.g. 'no', 'instead', 'I'd prefer', 'change it', 'something else'); "
            "'other'   — user is moving on to a completely different topic."
        )
    )


# ── Prompts ───────────────────────────────────────────────────────────────────

_SUGGEST_SYSTEM = """\
You are a query recommendation assistant for a customer-service data analytics tool.

The user has been exploring the Bitext Customer Service dataset.
Your job: suggest ONE relevant follow-up query they might find valuable next,
based on what they've already asked and what you know about them.

The dataset has categories like REFUND, SHIPPING, ACCOUNT, FEEDBACK, ORDER, etc.
and intents like get_refund, track_order, cancel_order, etc.

Rules:
- Write the suggestion as a natural-language query the user could type directly.
- It must be about the Bitext dataset (structured or open-ended analysis).
- Make it a logical next step — extend what the user has been doing.
- Keep your reasoning brief (one sentence).
"""

_REFINE_SYSTEM = """\
You are a query recommendation assistant.
The user was shown a suggested query but wants something different.

Generate a revised suggestion that takes their feedback into account.
Still write it as a natural-language query the user could type directly,
and keep it about the Bitext Customer Service dataset.
"""

_ACTION_SYSTEM = """\
You are classifying a user's response to a pending query suggestion.

Classify as:
  "confirm" — the user accepts and wants the suggestion executed
              (e.g. "yes", "do it", "go ahead", "sure", "ok", "yep", "sounds good",
              "yes please", "run it", "execute it")
  "refine"  — the user wants a different or modified suggestion
              (e.g. "no", "instead", "change it to", "I'd prefer", "rather",
              "something else", "not that")
  "other"   — the user is switching to a completely different topic
"""

# ── Helper ────────────────────────────────────────────────────────────────────


def _recent_text(messages: list[BaseMessage], n: int = _CONTEXT_WINDOW) -> str:
    """Return the last *n* human/AI text messages as a formatted string."""
    text_msgs = [
        m
        for m in messages
        if isinstance(m, (HumanMessage, AIMessage))
        and isinstance(m.content, str)
        and m.content.strip()
        and not (hasattr(m, "tool_calls") and m.tool_calls)
    ]
    lines = []
    for m in text_msgs[-n:]:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        lines.append(f"{role}: {m.content[:300]}")
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────


# Keywords that strongly indicate a recommendation request — checked before
# spending an LLM call.
_REC_KEYWORDS = [
    "what should i query",
    "what should i ask",
    "what else can i",
    "recommend",
    "suggest a query",
    "what's next",
    "what to query",
    "any suggestions",
    "what would you suggest",
    "next query",
    "what query",
    "give me a suggestion",
]


def is_recommendation_request(query: str) -> bool:
    """
    Return True if the query is asking for a query recommendation.

    Uses simple keyword matching — no LLM call needed.

    Args:
        query: Raw user input string.
    """
    lowered = query.lower().strip()
    return any(kw in lowered for kw in _REC_KEYWORDS)


def suggest(
    messages: list[BaseMessage],
    profile_context: str,
    llm: ChatOpenAI,
) -> str:
    """
    Generate a follow-up query suggestion from history and user profile.

    Args:
        messages:        Full conversation history for the session.
        profile_context: Formatted user-profile string (may be empty).
        llm:             LLM to use (router-sized model is sufficient).

    Returns:
        The suggested query as a plain string.
    """
    conv = _recent_text(messages)
    context_parts: list[str] = []
    if profile_context:
        context_parts.append(f"User profile:\n{profile_context}")
    if conv:
        context_parts.append(f"Recent conversation:\n{conv}")
    context = "\n\n".join(context_parts) or "No conversation history yet."

    structured_llm = llm.with_structured_output(_Suggestion)
    result: _Suggestion = structured_llm.invoke(
        [
            SystemMessage(content=_SUGGEST_SYSTEM),
            HumanMessage(
                content=f"Context:\n{context}\n\nSuggest a relevant follow-up query."
            ),
        ]
    )
    logger.debug("[Recommender] Suggestion: %s | %s", result.query, result.reasoning)
    return result.query


def refine(
    pending_rec: str,
    feedback: str,
    messages: list[BaseMessage],
    profile_context: str,
    llm: ChatOpenAI,
) -> str:
    """
    Revise a pending suggestion based on user feedback.

    Args:
        pending_rec:     The current suggestion being refined.
        feedback:        The user's refinement request.
        messages:        Full conversation history.
        profile_context: Formatted user-profile string.
        llm:             LLM to use.

    Returns:
        A revised suggestion as a plain string.
    """
    conv = _recent_text(messages, n=6)
    structured_llm = llm.with_structured_output(_Suggestion)
    result: _Suggestion = structured_llm.invoke(
        [
            SystemMessage(content=_REFINE_SYSTEM),
            HumanMessage(
                content=(
                    f"Original suggestion: {pending_rec}\n"
                    f"User feedback: {feedback}\n\n"
                    f"Recent conversation:\n{conv}\n\n"
                    "Provide a revised suggestion."
                )
            ),
        ]
    )
    logger.debug("[Recommender] Refined: %s", result.query)
    return result.query


def detect_action(
    user_response: str,
    pending_rec: str,
    llm: ChatOpenAI,
) -> Literal["confirm", "refine", "other"]:
    """
    Classify a user's response to a pending recommendation.

    Args:
        user_response: What the user just typed.
        pending_rec:   The suggestion they were responding to.
        llm:           LLM to use.

    Returns:
        ``"confirm"``, ``"refine"``, or ``"other"``.
    """
    structured_llm = llm.with_structured_output(_RecAction)
    result: _RecAction = structured_llm.invoke(
        [
            SystemMessage(content=_ACTION_SYSTEM),
            HumanMessage(
                content=(
                    f"Suggested query: {pending_rec}\n"
                    f"User response: {user_response}\n\n"
                    "Classify the user's intent."
                )
            ),
        ]
    )
    logger.debug("[Recommender] Action for '%s': %s", user_response[:40], result.action)
    return result.action

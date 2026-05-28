"""
LangGraph ReAct agent for the Customer Service Data Analyst.

Graph topology
──────────────
  START
    │
    ▼
  router          ← classifies query as structured / unstructured / out_of_scope
    │
    ├─ out_of_scope ──► END   (polite decline already in state)
    │
    └─ in_scope ──► agent    ◄──────────────────────────────────┐
                      │                                          │
                      ├─ tool_calls present ──► tools ──────────┘
                      │
                      ├─ done (no tool calls) ──► END
                      │
                      └─ max_iterations ──► fallback ──► END

State
─────
  messages     : conversation history (add_messages reducer)
  query_type   : "structured" | "unstructured" | "out_of_scope" | ""
  iterations   : number of agent steps taken so far
  user_profile : formatted profile context injected into the system prompt
"""

from __future__ import annotations

import logging
import os
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from .router import QueryClassification, create_router
from .tools import ALL_TOOLS

logger = logging.getLogger(__name__)

# Maximum number of agent iterations before the fallback fires.
MAX_ITERATIONS: int = 12

# ── System prompts ────────────────────────────────────────────────────────────

_STRUCTURED_SYSTEM_PROMPT = """You are a precise data analyst for the Bitext Customer Service \
dataset — a labelled collection of customer support interactions.

Your task is to answer the user's question with factual, data-driven results.

Guidelines:
- Use the available tools to query the dataset; never invent statistics.
- Chain multiple tools when needed: e.g. call get_intents() first to confirm an
  intent name, then call count_records() with that exact intent.
- If a filter returns no results, try alternate spellings or widen the search.
- Give concise, well-structured answers that directly address the question.
"""

_UNSTRUCTURED_SYSTEM_PROMPT = """You are an insightful data analyst for the Bitext Customer \
Service dataset — a labelled collection of customer support interactions.

Your task is to provide a thoughtful qualitative analysis of the user's open-ended question.

Guidelines:
- Call get_sample_for_summary() to retrieve a representative text sample
  (aim for 15–25 examples; set include_responses=True to see agent replies too).
- Read the samples carefully: identify recurring themes, patterns, tone, and
  how agents handle different situations.
- Structure your final answer clearly (use bullet points or short paragraphs).
- Base every insight on the sampled data; do not rely on general assumptions.
"""

# ── State ─────────────────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """Mutable state threaded through the LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    query_type: str    # "structured" | "unstructured" | "out_of_scope" | ""
    iterations: int    # incremented every time the agent node runs
    user_profile: str  # formatted profile block prepended to the system prompt


# ── Graph builder ─────────────────────────────────────────────────────────────


def build_agent(
    main_model: str | None = None,
    router_model: str | None = None,
    nebius_api_key: str | None = None,
    nebius_base_url: str | None = None,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the LangGraph ReAct agent.

    Configuration is read from arguments first, then from environment variables:
      - NEBIUS_API_KEY   : Nebius Token Factory API key (required)
      - NEBIUS_BASE_URL  : API base URL (default: https://api.studio.nebius.com/v1/)
      - MAIN_MODEL       : model used for the ReAct reasoning loop
      - ROUTER_MODEL     : model used for query classification

    Args:
        main_model:     Override for the main reasoning model name.
        router_model:   Override for the router model name.
        nebius_api_key: Override for the API key.
        nebius_base_url: Override for the base URL.
        checkpointer:   Optional LangGraph checkpointer for persistent memory
                        (e.g. a :class:`~langgraph.checkpoint.sqlite.SqliteSaver`).
                        When provided, conversation history is stored per
                        ``thread_id`` in ``config["configurable"]``.

    Returns:
        A compiled :class:`~langgraph.graph.StateGraph` ready for ``.invoke()``
        or ``.stream()``.
    """
    api_key = nebius_api_key or os.environ["NEBIUS_API_KEY"]
    base_url = nebius_base_url or os.environ.get(
        "NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"
    )
    main_model_name = main_model or os.environ.get(
        "MAIN_MODEL", "Qwen/Qwen3-235B-A22B-Instruct-2507"
    )
    router_model_name = router_model or os.environ.get(
        "ROUTER_MODEL", "Qwen/Qwen3-32B"
    )

    # ── LLM instances ──────────────────────────────────────────────────────────
    main_llm = ChatOpenAI(
        model=main_model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.1,
    )
    router_llm = ChatOpenAI(
        model=router_model_name,
        api_key=api_key,
        base_url=base_url,
        temperature=0.0,
    )

    llm_with_tools = main_llm.bind_tools(ALL_TOOLS)
    classify_query = create_router(router_llm)

    # ── Nodes ──────────────────────────────────────────────────────────────────

    def router_node(state: AgentState) -> dict[str, Any]:
        """
        Classify the user's query and update the state accordingly.

        If the query is out_of_scope, append a polite decline AIMessage so the
        graph can terminate gracefully at END without entering the ReAct loop.
        """
        human_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not human_messages:
            logger.warning("[Router] No human message found — treating as out_of_scope.")
            return {"query_type": "out_of_scope"}

        last_query: str = human_messages[-1].content
        classification: QueryClassification = classify_query(last_query)

        logger.info(
            "[Router] type=%s | %s", classification.query_type, classification.reasoning
        )

        if classification.query_type == "out_of_scope":
            decline = AIMessage(
                content=(
                    "I'm sorry, but that question is outside my area of expertise. "
                    "I'm a data analyst specialised in the Bitext Customer Service dataset "
                    "and can only answer questions about its contents — categories like "
                    "REFUND, SHIPPING, ACCOUNT, etc., intents, distributions, and examples. "
                    "Please ask me something related to that dataset!"
                )
            )
            return {"query_type": "out_of_scope", "messages": [decline]}

        return {"query_type": classification.query_type}

    def agent_node(state: AgentState) -> dict[str, Any]:
        """
        Run one LLM reasoning step with tool-use enabled.

        Selects the appropriate system prompt based on the query type,
        prepends any stored user-profile context, and increments the counter.
        """
        query_type = state.get("query_type", "structured")
        base_prompt = (
            _UNSTRUCTURED_SYSTEM_PROMPT
            if query_type == "unstructured"
            else _STRUCTURED_SYSTEM_PROMPT
        )

        # Prepend the profile block so the LLM can answer "What do you know
        # about me?" and can personalise responses when a profile exists.
        profile_ctx = state.get("user_profile", "")
        system_prompt = (profile_ctx + base_prompt) if profile_ctx else base_prompt

        messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
        response = llm_with_tools.invoke(messages)

        return {
            "messages": [response],
            "iterations": state.get("iterations", 0) + 1,
        }

    def fallback_node(state: AgentState) -> dict[str, Any]:
        """
        Emit a graceful fallback message when MAX_ITERATIONS is reached.

        This prevents the agent from spinning indefinitely on complex or
        ambiguous queries.
        """
        logger.warning("[Agent] Max iterations (%d) reached.", MAX_ITERATIONS)
        fallback = AIMessage(
            content=(
                f"I've reached the maximum number of reasoning steps ({MAX_ITERATIONS}) "
                "for this query and wasn't able to produce a complete answer. "
                "Please try rephrasing the question or breaking it into smaller parts."
            )
        )
        return {"messages": [fallback]}

    # ── Edge conditions ────────────────────────────────────────────────────────

    def route_after_classification(
        state: AgentState,
    ) -> Literal["in_scope", "out_of_scope"]:
        """Branch after the router: in-scope queries enter the ReAct loop."""
        return "out_of_scope" if state["query_type"] == "out_of_scope" else "in_scope"

    def should_continue(
        state: AgentState,
    ) -> Literal["continue", "end", "max_iterations"]:
        """
        Decide what happens after each agent step.

        Returns:
          - "continue"       : last message has pending tool calls → execute them
          - "end"            : last message is a final text answer → done
          - "max_iterations" : safety limit exceeded → emit fallback
        """
        if state.get("iterations", 0) >= MAX_ITERATIONS:
            return "max_iterations"

        last_msg = state["messages"][-1]
        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "continue"

        return "end"

    # ── Assemble the graph ─────────────────────────────────────────────────────

    tools_node = ToolNode(ALL_TOOLS)

    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "router")

    graph.add_conditional_edges(
        "router",
        route_after_classification,
        {
            "out_of_scope": END,
            "in_scope": "agent",
        },
    )

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END,
            "max_iterations": "fallback",
        },
    )

    graph.add_edge("tools", "agent")
    graph.add_edge("fallback", END)

    return graph.compile(checkpointer=checkpointer)

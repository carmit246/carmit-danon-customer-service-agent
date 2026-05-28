#!/usr/bin/env python3
"""
Streamlit chat UI for the Customer Service Data Analyst Agent (Bonus A).

Run:
    streamlit run streamlit_app.py

Features:
  - Chat interface with full conversation history rendered per session.
  - Reasoning steps (tool calls + results) shown in collapsible expanders.
  - Sidebar: session ID selector, user profile display, clear-display button.
  - Persistent memory via SQLite checkpointer across restarts.
  - Query recommender (Bonus B): ask "What should I query next?" for suggestions.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI

# ── Streamlit page config ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="Customer Service Data Analyst",
    page_icon="📊",
    layout="wide",
)

# ── Cached resources (singletons for the lifetime of the Streamlit server) ────


@st.cache_resource
def _get_checkpointer():
    """Create a persistent SQLite checkpointer (shared across all sessions)."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    Path("data").mkdir(exist_ok=True)
    conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
    return SqliteSaver(conn)


@st.cache_resource
def _get_agent(_checkpointer):
    """Build and cache the LangGraph agent (expensive, done once per server start)."""
    from app.agent import build_agent

    return build_agent(checkpointer=_checkpointer)


@st.cache_resource
def _get_llm():
    """Small LLM for profile updates and recommendation (cached singleton)."""
    return ChatOpenAI(
        model=os.environ.get("ROUTER_MODEL", "Qwen/Qwen3-32B"),
        api_key=os.environ.get("NEBIUS_API_KEY", ""),
        base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
        temperature=0.0,
    )


@st.cache_resource
def _load_dataset():
    """Pre-load the dataset so tool calls don't block the first query."""
    from app.data_loader import get_dataframe

    return get_dataframe()


# ── Session-state helpers ──────────────────────────────────────────────────────


def _init_session_state(session_id: str) -> None:
    """Initialise (or reset) per-session keys when the session ID changes."""
    if st.session_state.get("active_session") != session_id:
        st.session_state.active_session = session_id
        st.session_state.display_messages = []   # list of {role, content, reasoning, query_type}
        st.session_state.pending_recommendation = None
        _reload_profile(session_id)


def _reload_profile(session_id: str) -> None:
    from app.profile import ProfileManager

    mgr = ProfileManager(session_id=session_id)
    profile = mgr.load()
    st.session_state.profile_mgr = mgr
    st.session_state.profile = profile
    st.session_state.profile_context = mgr.to_system_context(profile)


# ── Agent runner ───────────────────────────────────────────────────────────────


def _run_query(
    query: str,
    session_id: str,
    profile_context: str,
) -> tuple[str, list[dict]]:
    """
    Send *query* to the agent and collect the response.

    Returns:
        (final_answer, reasoning_steps)
        where ``reasoning_steps`` is a list of ``{tool, args, result}`` dicts.
    """
    checkpointer = _get_checkpointer()
    agent = _get_agent(checkpointer)

    state = {
        "messages": [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
        "user_profile": profile_context,
    }
    config = {"configurable": {"thread_id": session_id}}

    reasoning_steps: list[dict] = []
    final_answer: str = ""
    _pending_tool_call: dict | None = None

    for chunk in agent.stream(state, config=config, stream_mode="updates"):
        for node_name, node_output in chunk.items():

            if node_name == "router":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content

            elif node_name == "agent":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            for tc in msg.tool_calls:
                                reasoning_steps.append({
                                    "tool": tc["name"],
                                    "args": tc.get("args", {}),
                                    "result": None,
                                })
                        elif msg.content:
                            final_answer = msg.content

            elif node_name == "tools":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        # Match result back to the most recent un-filled step
                        for step in reversed(reasoning_steps):
                            if step["tool"] == msg.name and step["result"] is None:
                                preview = msg.content
                                if len(preview) > 500:
                                    preview = preview[:500] + " …"
                                step["result"] = preview
                                break

            elif node_name == "fallback":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content

    return final_answer, reasoning_steps


# ── Sidebar ────────────────────────────────────────────────────────────────────


def _render_sidebar() -> str:
    """Render the sidebar and return the selected session_id."""
    with st.sidebar:
        st.title("📊 Settings")
        st.divider()

        session_id = st.text_input(
            "Session ID",
            value=st.session_state.get("active_session", "default"),
            help="Use the same ID to restore a previous conversation.",
        )

        if st.button("🔄 Switch / Refresh session", use_container_width=True):
            _init_session_state(session_id)
            st.rerun()

        if st.button("🗑️ Clear display", use_container_width=True):
            st.session_state.display_messages = []
            st.session_state.pending_recommendation = None
            st.rerun()

        st.divider()

        # Profile display
        profile = st.session_state.get("profile")
        if profile and any([
            profile.name,
            profile.frequent_topics,
            profile.preferences,
            profile.notes,
        ]):
            st.subheader("👤 Your Profile")
            if profile.name:
                st.write(f"**Name:** {profile.name}")
            if profile.frequent_topics:
                st.write(f"**Topics:** {', '.join(profile.frequent_topics)}")
            if profile.preferences:
                for p in profile.preferences:
                    st.caption(f"• {p}")
            if profile.notes:
                for n in profile.notes:
                    st.caption(f"• {n}")
        else:
            st.caption("No profile yet. Start chatting to build one.")

        st.divider()
        st.caption(
            "Conversation history and user profiles persist in `data/` "
            "across restarts."
        )

    return session_id


# ── Message rendering ──────────────────────────────────────────────────────────


def _render_messages() -> None:
    """Render all stored display messages in the chat area."""
    for msg in st.session_state.get("display_messages", []):
        role = msg["role"]
        content = msg["content"]
        reasoning = msg.get("reasoning", [])
        is_suggestion = msg.get("is_suggestion", False)

        with st.chat_message(role):
            if is_suggestion:
                st.info(content, icon="💡")
            else:
                st.markdown(content)

            if reasoning:
                with st.expander("🔍 Reasoning steps", expanded=False):
                    for step in reasoning:
                        args_str = ", ".join(
                            f"{k}={repr(v)}"
                            for k, v in step.get("args", {}).items()
                            if v is not None
                        )
                        st.markdown(
                            f"**→ `{step['tool']}`**(`{args_str}`)"
                        )
                        if step.get("result"):
                            st.code(step["result"], language=None)


# ── Main app ───────────────────────────────────────────────────────────────────


def main() -> None:
    """Main Streamlit application entry point."""

    # ── API key guard ──────────────────────────────────────────────────────────
    if not os.environ.get("NEBIUS_API_KEY"):
        st.error(
            "**NEBIUS_API_KEY is not set.**\n\n"
            "Add it to your `.env` file:\n```\nNEBIUS_API_KEY=your-key-here\n```"
        )
        st.stop()

    # ── Sidebar & session init ─────────────────────────────────────────────────
    session_id = _render_sidebar()
    _init_session_state(session_id)

    # ── Pre-load dataset (non-blocking for UI) ─────────────────────────────────
    with st.spinner("Loading dataset…"):
        df = _load_dataset()

    # ── Page header ────────────────────────────────────────────────────────────
    st.title("📊 Customer Service Data Analyst")
    st.caption(
        f"Session: **{session_id}** · "
        f"{len(df):,} records · "
        f"{df['category'].nunique()} categories"
    )
    st.divider()

    # ── Render chat history ────────────────────────────────────────────────────
    _render_messages()

    # ── Pending recommendation banner ─────────────────────────────────────────
    pending = st.session_state.get("pending_recommendation")
    if pending:
        st.info(
            f"💡 **Pending suggestion:** _{pending}_\n\n"
            "Reply **yes** to execute it, describe changes to refine it, "
            "or just ask something else to move on.",
            icon="💡",
        )

    # ── Chat input ─────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask about the customer service dataset…")
    if not user_input:
        return

    profile_context = st.session_state.get("profile_context", "")
    llm = _get_llm()

    # ── Handle recommendation flow (Bonus B) ───────────────────────────────────
    from app.recommender import detect_action, is_recommendation_request, refine, suggest

    if pending:
        action = detect_action(user_input, pending, llm)

        if action == "confirm":
            # Execute the pending recommendation
            actual_query = pending
            st.session_state.pending_recommendation = None
            st.session_state.display_messages.append(
                {"role": "user", "content": f"(Executing: _{pending}_)"}
            )
        elif action == "refine":
            # Generate a revised suggestion
            try:
                messages_for_rec = _get_agent(_get_checkpointer()).get_state(
                    {"configurable": {"thread_id": session_id}}
                ).values.get("messages", [])
            except Exception:
                messages_for_rec = []
            new_rec = refine(pending, user_input, messages_for_rec, profile_context, llm)
            st.session_state.pending_recommendation = new_rec
            suggestion_text = (
                f"How about this instead:\n\n"
                f"> **{new_rec}**\n\n"
                "Reply **yes** to run it, or describe further changes."
            )
            st.session_state.display_messages.append(
                {"role": "user", "content": user_input}
            )
            st.session_state.display_messages.append(
                {"role": "assistant", "content": suggestion_text, "reasoning": [], "is_suggestion": True}
            )
            st.rerun()
            return
        else:
            # User moved on — clear pending and treat as normal query
            st.session_state.pending_recommendation = None
            actual_query = user_input

    elif is_recommendation_request(user_input):
        # Generate a fresh suggestion
        try:
            messages_for_rec = _get_agent(_get_checkpointer()).get_state(
                {"configurable": {"thread_id": session_id}}
            ).values.get("messages", [])
        except Exception:
            messages_for_rec = []
        rec = suggest(messages_for_rec, profile_context, llm)
        st.session_state.pending_recommendation = rec
        suggestion_text = (
            f"Based on your conversation so far, I suggest:\n\n"
            f"> **{rec}**\n\n"
            "Reply **yes** to run it, describe changes to refine it, "
            "or just ask something else."
        )
        st.session_state.display_messages.append(
            {"role": "user", "content": user_input}
        )
        st.session_state.display_messages.append(
            {"role": "assistant", "content": suggestion_text, "reasoning": [], "is_suggestion": True}
        )
        st.rerun()
        return

    else:
        actual_query = user_input

    # ── Add user message to display ────────────────────────────────────────────
    st.session_state.display_messages.append(
        {"role": "user", "content": actual_query}
    )

    # ── Run the agent ──────────────────────────────────────────────────────────
    with st.spinner("Thinking…"):
        try:
            final_answer, reasoning_steps = _run_query(actual_query, session_id, profile_context)
        except Exception as exc:
            st.error(f"Agent error: {exc}")
            st.rerun()  # ensure the user message added above is rendered
            return

    # ── Store assistant response ───────────────────────────────────────────────
    st.session_state.display_messages.append(
        {
            "role": "assistant",
            "content": final_answer or "_No answer produced._",
            "reasoning": reasoning_steps,
            "is_suggestion": False,
        }
    )

    # ── Update user profile in background ─────────────────────────────────────
    try:
        snap = _get_agent(_get_checkpointer()).get_state(
            {"configurable": {"thread_id": session_id}}
        )
        all_msgs = snap.values.get("messages", [])
        mgr: "ProfileManager" = st.session_state.profile_mgr  # type: ignore[name-defined]
        profile = mgr.update(all_msgs, llm)
        st.session_state.profile = profile
        st.session_state.profile_context = mgr.to_system_context(profile)
    except Exception:
        pass  # profile update is best-effort

    st.rerun()


if __name__ == "__main__":
    main()

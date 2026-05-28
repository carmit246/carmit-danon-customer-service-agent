#!/usr/bin/env python3
"""
Customer Service Data Analyst Agent — interactive CLI.

Usage
─────
  python3 main.py                            # default session
  python3 main.py --session my_session       # named, persistent session
  python3 main.py --debug                    # verbose logging

The agent answers questions about the Bitext Customer Service dataset.
Conversation history and a user profile are persisted per session in data/.
Type 'exit', 'quit', or press Ctrl-C / Ctrl-D to quit.

Bonus B — query recommender
───────────────────────────
Ask "What should I query next?" (or similar) to get a suggestion.
The agent will propose a follow-up query and wait for your confirmation:
  - Say "yes" / "do it" to execute it.
  - Describe a change to refine the suggestion.
  - Ask something else to move on without executing.

Environment variables (or .env file)
─────────────────────────────────────
  NEBIUS_API_KEY   : required
  NEBIUS_BASE_URL  : optional (default: https://api.studio.nebius.com/v1/)
  MAIN_MODEL       : optional (default: Qwen/Qwen3-235B-A22B-Instruct-2507)
  ROUTER_MODEL     : optional (default: Qwen/Qwen3-32B)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Bootstrap ──────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

console = Console()
logger = logging.getLogger(__name__)


# ── Argument parsing ───────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Customer Service Data Analyst Agent CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--session",
        default="default",
        metavar="SESSION_ID",
        help="Session ID for persistent memory (default: 'default'). "
             "The same ID restores conversation history and user profile.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging (tool calls, router, profile updates).",
    )
    return parser.parse_args()


# ── Rich helpers ───────────────────────────────────────────────────────────────


def _print_tool_call(tool_name: str, args: dict) -> None:
    arg_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items() if v is not None)
    console.print(
        f"  [bold cyan]→ Tool call:[/bold cyan] "
        f"[cyan]{tool_name}[/cyan]([cyan]{arg_str}[/cyan])"
    )


def _print_tool_result(tool_name: str, content: str) -> None:
    preview = content if len(content) <= 400 else content[:400] + " …"
    console.print(
        f"  [bold green]← Result ({tool_name}):[/bold green] [dim]{preview}[/dim]"
    )


def _print_suggestion(query: str) -> None:
    console.print(
        Panel(
            f"[bold]Suggested query:[/bold]\n\n  {query}\n\n"
            "[dim]Reply [bold]yes[/bold] to execute, describe a change to refine, "
            "or ask something else to move on.[/dim]",
            title="[bold magenta]💡 Recommendation[/bold magenta]",
            border_style="magenta",
            padding=(1, 2),
        )
    )


# ── Core query runner ──────────────────────────────────────────────────────────


def run_query(
    agent,
    query: str,
    session_id: str,
    profile_context: str,
) -> None:
    """
    Send *query* to the agent, stream steps to the console, and print the answer.

    Args:
        agent:           Compiled LangGraph agent with checkpointer.
        query:           User input string.
        session_id:      LangGraph thread_id.
        profile_context: Formatted user-profile block (may be empty).
    """
    state = {
        "messages": [HumanMessage(content=query)],
        "query_type": "",
        "iterations": 0,
        "user_profile": profile_context,
    }
    config = {"configurable": {"thread_id": session_id}}

    reasoning_printed = False
    final_answer: str | None = None

    console.print()

    for chunk in agent.stream(state, config=config, stream_mode="updates"):
        for node_name, node_output in chunk.items():

            if node_name == "router":
                qt = node_output.get("query_type", "")
                if qt and qt != "out_of_scope":
                    console.print(
                        f"[dim]Query classified as:[/dim] [bold yellow]{qt}[/bold yellow]"
                    )
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content

            elif node_name == "agent":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage):
                        if hasattr(msg, "tool_calls") and msg.tool_calls:
                            if not reasoning_printed:
                                console.print(
                                    Rule("[bold yellow]Reasoning Steps[/bold yellow]", style="yellow")
                                )
                                reasoning_printed = True
                            for tc in msg.tool_calls:
                                _print_tool_call(tc["name"], tc.get("args", {}))
                        elif msg.content:
                            final_answer = msg.content

            elif node_name == "tools":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, ToolMessage):
                        _print_tool_result(msg.name, msg.content)

            elif node_name == "fallback":
                for msg in node_output.get("messages", []):
                    if isinstance(msg, AIMessage) and msg.content:
                        final_answer = msg.content

    if reasoning_printed:
        console.print(Rule(style="yellow"))

    if final_answer:
        console.print()
        console.print(
            Panel(
                Markdown(final_answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
    else:
        console.print("[dim italic]No answer produced.[/dim italic]")


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point — parse args, bootstrap components, run the interactive loop."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # ── Validate API key ───────────────────────────────────────────────────────
    if not os.environ.get("NEBIUS_API_KEY"):
        console.print(
            Panel(
                "[bold red]NEBIUS_API_KEY is not set.[/bold red]\n\n"
                "Add it to a [bold].env[/bold] file:\n\n"
                "  NEBIUS_API_KEY=your-key-here",
                title="[red]Configuration Error[/red]",
                border_style="red",
            )
        )
        sys.exit(1)

    # ── Welcome banner ─────────────────────────────────────────────────────────
    console.print(
        Panel(
            "[bold blue]Customer Service Data Analyst Agent[/bold blue]\n\n"
            f"Session: [bold]{args.session}[/bold]  "
            "[dim](--session NAME to switch)[/dim]\n\n"
            "Ask about the [bold]Bitext Customer Service dataset[/bold], "
            'or say [bold]"What should I query next?"[/bold] for a suggestion.\n'
            "[dim]Type [bold]exit[/bold] to quit.[/dim]",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # ── Load dataset ───────────────────────────────────────────────────────────
    console.print("[dim]Loading dataset…[/dim] ", end="")
    try:
        from app.data_loader import get_dataframe

        df = get_dataframe()
        console.print(
            f"[green]✓[/green]  [dim]{len(df):,} records, "
            f"{df['category'].nunique()} categories.[/dim]"
        )
    except Exception as exc:
        console.print(f"\n[bold red]Failed to load dataset:[/bold red] {exc}")
        sys.exit(1)

    # ── Build checkpointer + agent ─────────────────────────────────────────────
    console.print("[dim]Initialising agent…[/dim] ", end="")
    try:
        from app.agent import build_agent
        from app.memory import create_checkpointer

        checkpointer_ctx = create_checkpointer()
        checkpointer = checkpointer_ctx.__enter__()
        agent = build_agent(checkpointer=checkpointer)
        console.print("[green]✓[/green]")
    except Exception as exc:
        console.print(f"\n[bold red]Failed to build agent:[/bold red] {exc}")
        sys.exit(1)

    # ── Small LLM for profile updates and recommendations ─────────────────────
    profile_llm = ChatOpenAI(
        model=os.environ.get("ROUTER_MODEL", "Qwen/Qwen3-32B"),
        api_key=os.environ["NEBIUS_API_KEY"],
        base_url=os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/"),
        temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    # ── Load user profile ──────────────────────────────────────────────────────
    from app.profile import ProfileManager
    from app.recommender import detect_action, is_recommendation_request, refine, suggest

    profile_mgr = ProfileManager(session_id=args.session)
    profile = profile_mgr.load()
    profile_context = profile_mgr.to_system_context(profile)

    if profile_context:
        console.print(
            f"[dim]Profile loaded for session '[bold]{args.session}[/bold]'.[/dim]"
        )

    console.print()

    # ── Recommendation state ───────────────────────────────────────────────────
    pending_recommendation: str | None = None

    # ── Interactive loop ───────────────────────────────────────────────────────
    try:
        while True:
            try:
                prompt_text = "[bold blue]You ▶[/bold blue] "
                if pending_recommendation:
                    prompt_text = "[bold magenta]You (confirm/refine/skip) ▶[/bold magenta] "
                query = console.input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/dim]")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", "q", ":q"}:
                console.print("[dim]Goodbye![/dim]")
                break

            actual_query: str | None = None

            try:
                # ── Recommendation flow ────────────────────────────────────────
                if pending_recommendation:
                    action = detect_action(query, pending_recommendation, profile_llm)

                    if action == "confirm":
                        console.print(
                            f"[dim]Executing suggestion:[/dim] "
                            f"[bold]{pending_recommendation}[/bold]"
                        )
                        actual_query = pending_recommendation
                        pending_recommendation = None

                    elif action == "refine":
                        try:
                            snap = agent.get_state({"configurable": {"thread_id": args.session}})
                            msgs = snap.values.get("messages", [])
                        except Exception:
                            msgs = []
                        pending_recommendation = refine(
                            pending_recommendation, query, msgs, profile_context, profile_llm
                        )
                        _print_suggestion(pending_recommendation)
                        console.print()
                        continue  # wait for next user input

                    else:  # "other" — user moved on
                        pending_recommendation = None
                        actual_query = query

                elif is_recommendation_request(query):
                    # Generate a new suggestion
                    try:
                        snap = agent.get_state({"configurable": {"thread_id": args.session}})
                        msgs = snap.values.get("messages", [])
                    except Exception:
                        msgs = []
                    pending_recommendation = suggest(msgs, profile_context, profile_llm)
                    _print_suggestion(pending_recommendation)
                    console.print()
                    continue  # wait for confirmation

                else:
                    actual_query = query

                # ── Run the agent ──────────────────────────────────────────────
                if actual_query:
                    run_query(agent, actual_query, args.session, profile_context)

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
                continue
            except Exception as exc:
                console.print(f"[bold red]Error:[/bold red] {exc}")
                if args.debug:
                    console.print_exception()
                continue

            # ── Update profile after each executed query ───────────────────────
            try:
                snap = agent.get_state({"configurable": {"thread_id": args.session}})
                all_msgs = snap.values.get("messages", [])
                profile = profile_mgr.update(all_msgs, profile_llm)
                profile_context = profile_mgr.to_system_context(profile)
            except Exception as exc:
                logger.warning("[Profile] Update skipped: %s", exc)

            console.print()

    finally:
        try:
            checkpointer_ctx.__exit__(None, None, None)
        except Exception:
            pass


if __name__ == "__main__":
    main()

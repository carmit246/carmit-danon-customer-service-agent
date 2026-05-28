"""
Checkpointer factory for LangGraph conversation persistence (Task 2a).

Uses SQLite via langgraph-checkpoint-sqlite so conversation state survives
process restarts.  Each session is identified by a thread_id; all sessions
share one database file.

Usage::

    from app.memory import create_checkpointer

    with create_checkpointer() as checkpointer:
        agent = build_agent(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": "my-session"}}
        agent.invoke(state, config=config)
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

DEFAULT_DB_PATH = Path("data/checkpoints.db")


@contextmanager
def create_checkpointer(
    db_path: Path | str = DEFAULT_DB_PATH,
) -> Generator:
    """
    Context manager that yields a :class:`~langgraph.checkpoint.sqlite.SqliteSaver`.

    Creates the SQLite database file and any missing parent directories
    automatically on first use.

    Args:
        db_path: Path to the SQLite file (default: ``data/checkpoints.db``).

    Yields:
        A :class:`~langgraph.checkpoint.sqlite.SqliteSaver` ready for use as
        a LangGraph checkpointer.

    Example::

        with create_checkpointer("data/checkpoints.db") as cp:
            agent = build_agent(checkpointer=cp)
    """
    from langgraph.checkpoint.sqlite import SqliteSaver

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        yield SqliteSaver(conn)
    finally:
        conn.close()

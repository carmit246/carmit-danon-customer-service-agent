"""
Persistent user profile management (Task 2b).

The profile is *not* a replay of conversation history.  It is a compact,
growing set of distilled facts about the user — their name, the topics they
frequently ask about, stated preferences, and any other notable details.

Storage: one JSON file per session in ``data/profiles/<session_id>.json``.
This file is separate from the LangGraph checkpoint so the two memory stores
can evolve independently.

Typical flow
────────────
  1. At the start of each turn  : load profile → inject into system prompt.
  2. At the end of each turn    : call ``update()`` → save merged facts.
  3. On "What do you remember?" : the agent reads its own system prompt
                                  and answers naturally.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_PROFILES_DIR = Path("data/profiles")

# Maximum number of recent text messages fed to the profile-update LLM call.
_MAX_MESSAGES_FOR_UPDATE = 10


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class UserProfile:
    """Distilled facts about one user, persisted across sessions."""

    name: Optional[str] = None
    frequent_topics: list[str] = field(default_factory=list)
    preferences: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    last_updated: Optional[str] = None


# ── Pydantic schema for LLM-driven incremental updates ───────────────────────


class _ProfileDelta(BaseModel):
    """
    Incremental changes to apply to an existing profile.

    The LLM fills in only newly discovered facts; existing profile data is
    not repeated here.
    """

    name: Optional[str] = Field(
        default=None,
        description="User's first name or full name, only if explicitly stated in the conversation.",
    )
    new_topics: list[str] = Field(
        default_factory=list,
        description=(
            "Dataset categories or intents the user showed genuine interest in "
            "(e.g. 'REFUND', 'SHIPPING', 'cancel_order').  Only NEW topics not "
            "already in the profile."
        ),
    )
    new_preferences: list[str] = Field(
        default_factory=list,
        description=(
            "Any preferences the user stated or implied "
            "(e.g. 'prefers concise answers', 'likes to see examples').  "
            "Only NEW preferences."
        ),
    )
    new_notes: list[str] = Field(
        default_factory=list,
        description=(
            "Any other notable facts worth remembering about the user.  "
            "One short sentence each.  Only NEW facts."
        ),
    )


_UPDATE_SYSTEM_PROMPT = """\
You are a profile manager for an AI assistant.

Your job is to scan a recent conversation excerpt and identify any NEW facts \
worth adding to the user's profile.

Rules:
- Only record facts the user explicitly stated or clearly implied.
- Do not invent or assume anything not supported by the text.
- For topics, use the dataset category/intent name if it appears in the conversation.
- If you find nothing new, return all fields empty — that is perfectly fine.
- Keep notes short (one concise sentence each).
"""


# ── Manager ───────────────────────────────────────────────────────────────────


class ProfileManager:
    """Load, save, update, and format a per-session user profile."""

    def __init__(
        self,
        session_id: str,
        profiles_dir: Path | str = DEFAULT_PROFILES_DIR,
    ) -> None:
        self.session_id = session_id
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        self._path = self.profiles_dir / f"{session_id}.json"

    # ── Persistence ───────────────────────────────────────────────────────────

    def load(self) -> UserProfile:
        """
        Load the profile from disk.

        Returns an empty :class:`UserProfile` if no file exists yet.
        """
        if self._path.exists():
            try:
                raw: dict[str, Any] = json.loads(self._path.read_text(encoding="utf-8"))
                return UserProfile(**raw)
            except Exception as exc:
                logger.warning("Could not load profile '%s': %s", self._path, exc)
        return UserProfile()

    def save(self, profile: UserProfile) -> None:
        """Persist *profile* to disk, stamping today's date."""
        profile.last_updated = str(date.today())
        self._path.write_text(
            json.dumps(asdict(profile), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug("Profile saved → %s", self._path)

    # ── Formatting ────────────────────────────────────────────────────────────

    def to_system_context(self, profile: UserProfile) -> str:
        """
        Format *profile* as a context block for injection into the system prompt.

        Returns an empty string when the profile contains no facts yet, so the
        prompt stays clean for brand-new users.
        """
        has_content = any(
            [profile.name, profile.frequent_topics, profile.preferences, profile.notes]
        )
        if not has_content:
            return ""

        lines = ["--- What I know about you ---"]
        if profile.name:
            lines.append(f"Name: {profile.name}")
        if profile.frequent_topics:
            lines.append(
                f"Topics you frequently ask about: {', '.join(profile.frequent_topics)}"
            )
        if profile.preferences:
            lines.append(f"Your preferences: {', '.join(profile.preferences)}")
        for note in profile.notes:
            lines.append(f"• {note}")
        lines.append("---\n")
        return "\n".join(lines)

    # ── LLM-driven update ─────────────────────────────────────────────────────

    def update(self, messages: list[BaseMessage], llm: ChatOpenAI) -> UserProfile:
        """
        Extract new facts from *messages* and merge them into the stored profile.

        Only plain text HumanMessage / AIMessage pairs are passed to the LLM —
        tool calls and tool results are excluded to keep the context compact.

        Args:
            messages: Full message list from the current session.
            llm:      LLM instance used for fact extraction (router-sized model
                      is sufficient).

        Returns:
            The updated :class:`UserProfile` (already saved to disk).
        """
        profile = self.load()

        # Filter to recent, human-readable messages only
        text_msgs = [
            m
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage))
            and isinstance(m.content, str)
            and m.content.strip()
            and not (hasattr(m, "tool_calls") and m.tool_calls)
        ]
        if not text_msgs:
            return profile

        recent = text_msgs[-_MAX_MESSAGES_FOR_UPDATE:]
        conv_lines = []
        for m in recent:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            # Truncate long messages to keep the prompt lean
            conv_lines.append(f"{role}: {m.content[:400]}")
        conversation_str = "\n".join(conv_lines)
        current_profile_str = json.dumps(asdict(profile), indent=2)

        try:
            structured_llm = llm.with_structured_output(_ProfileDelta)
            delta: _ProfileDelta = structured_llm.invoke(
                [
                    SystemMessage(content=_UPDATE_SYSTEM_PROMPT),
                    HumanMessage(
                        content=(
                            f"Current profile:\n{current_profile_str}\n\n"
                            f"Recent conversation:\n{conversation_str}\n\n"
                            "What NEW facts should be added?"
                        )
                    ),
                ]
            )

            # Merge delta into profile (deduplicated)
            if delta.name and not profile.name:
                profile.name = delta.name
            for topic in delta.new_topics:
                if topic not in profile.frequent_topics:
                    profile.frequent_topics.append(topic)
            for pref in delta.new_preferences:
                if pref not in profile.preferences:
                    profile.preferences.append(pref)
            for note in delta.new_notes:
                if note not in profile.notes:
                    profile.notes.append(note)

            self.save(profile)
            logger.debug("Profile updated: %s", asdict(profile))

        except Exception as exc:
            # Profile update is best-effort — never crash the main loop
            logger.warning("Profile update skipped (non-fatal): %s", exc)

        return profile

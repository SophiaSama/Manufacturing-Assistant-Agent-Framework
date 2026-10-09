"""Headroom compression manager with session-scoped in-memory storage for NGA."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger("nga.compression")


class SessionRawContentStore:
    """Thread-safe, bounded in-memory store for uncompressed text per session."""

    def __init__(self, max_entries_per_session: int = 100, max_sessions: int = 200):
        self._max_entries = max_entries_per_session
        self._max_sessions = max_sessions
        # session_id -> OrderedDict[hash, raw_text]
        self._sessions: OrderedDict[str, OrderedDict[str, str]] = OrderedDict()
        self._lock = threading.Lock()

    def put(self, session_id: str, content: str) -> str:
        """Store content for a session and return its 8-char SHA-256 hash."""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
        with self._lock:
            if session_id not in self._sessions:
                if len(self._sessions) >= self._max_sessions:
                    # Evict oldest session
                    self._sessions.popitem(last=False)
                self._sessions[session_id] = OrderedDict()

            session_dict = self._sessions[session_id]
            if content_hash in session_dict:
                session_dict.move_to_end(content_hash)
            else:
                if len(session_dict) >= self._max_entries:
                    session_dict.popitem(last=False)
                session_dict[content_hash] = content
        return content_hash

    def get(self, session_id: str, content_hash: str) -> str | None:
        """Retrieve original uncompressed text for a specific session."""
        with self._lock:
            session_dict = self._sessions.get(session_id)
            if not session_dict:
                return None
            val = session_dict.get(content_hash)
            if val is not None:
                session_dict.move_to_end(content_hash)
            return val

    def clear_session(self, session_id: str) -> None:
        """Clear session cache on thread/workflow completion."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def entry_count(self, session_id: str) -> int:
        with self._lock:
            session_dict = self._sessions.get(session_id)
            return len(session_dict) if session_dict else 0


class HeadroomManager:
    """Manages payload compression and uncompressed raw content storage."""

    def __init__(
        self,
        target_ratio: float = 0.7,
        min_tokens: int = 250,
        store: SessionRawContentStore | None = None,
        enabled: bool = True,
    ):
        self.target_ratio = target_ratio
        self.min_tokens = min_tokens
        self.store = store or SessionRawContentStore()
        self.enabled = enabled

    def compress_tool_output(
        self,
        tool_name: str,
        raw_content: str | dict,
        session_id: str = "default",
    ) -> str:
        """Compress tool output string or JSON payload with conservative target ratio."""
        text = json.dumps(raw_content) if isinstance(raw_content, dict) else str(raw_content)

        if not self.enabled:
            return text

        # Do not compress small outputs (< min_tokens * 4 chars)
        if len(text) < (self.min_tokens * 4):
            return text

        content_hash = self.store.put(session_id, text)

        try:
            import headroom

            CompressConfig = getattr(headroom, "CompressConfig", None)
            compress = headroom.compress

            messages = [{"role": "tool", "content": text, "name": tool_name}]
            if CompressConfig is not None:
                config = CompressConfig(
                    compress_user_messages=True,
                    compress_system_messages=False,
                    protect_recent=0,
                    target_ratio=self.target_ratio,
                    min_tokens_to_compress=self.min_tokens,
                    protect_analysis_context=True,
                )
                result = compress(messages, config=config)
            else:
                result = compress(messages)

            compressed_content = result.messages[0]["content"]

            header = (
                f"[Compressed {tool_name} | hash={content_hash} | "
                f"saved {result.tokens_saved} tokens]\n"
            )
            return header + compressed_content
        except Exception as exc:
            logger.warning("headroom_compress_failed tool=%s err=%s", tool_name, exc)
            return text

    def compress_message_history(
        self,
        messages: list[BaseMessage],
        protect_recent: int = 2,
    ) -> list[BaseMessage]:
        """Compress multi-turn message history while preserving recent turns and message types."""
        if not self.enabled or len(messages) <= protect_recent + 2:
            return messages

        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        non_system = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(non_system) <= protect_recent:
            return messages

        to_compress = non_system[:-protect_recent]
        protected_tail = non_system[-protect_recent:]

        dict_payload = []
        for m in to_compress:
            role = (
                "user"
                if isinstance(m, HumanMessage)
                else ("assistant" if hasattr(m, "tool_calls") and m.tool_calls else "tool")
            )
            content = m.content if isinstance(m.content, str) else str(m.content)
            dict_payload.append({"role": role, "content": content})

        try:
            import headroom

            CompressConfig = getattr(headroom, "CompressConfig", None)
            compress = headroom.compress

            if CompressConfig is not None:
                config = CompressConfig(
                    compress_user_messages=False,
                    compress_system_messages=False,
                    protect_recent=0,
                    target_ratio=self.target_ratio,
                    min_tokens_to_compress=self.min_tokens,
                    protect_analysis_context=True,
                )
                res = compress(dict_payload, config=config)
            else:
                res = compress(dict_payload)

            rebuilt: list[BaseMessage] = []
            for original, compressed in zip(to_compress, res.messages):
                compressed_text = compressed.get("content", original.content)
                if isinstance(original, ToolMessage):
                    new_msg = ToolMessage(
                        content=compressed_text,
                        tool_call_id=original.tool_call_id,
                        name=getattr(original, "name", None),
                    )
                else:
                    new_msg = type(original)(content=compressed_text)
                    if hasattr(original, "tool_calls"):
                        new_msg.tool_calls = original.tool_calls
                rebuilt.append(new_msg)

            return [*system_msgs, *rebuilt, *protected_tail]
        except Exception as exc:
            logger.warning("headroom_history_compress_failed err=%s", exc)
            return messages

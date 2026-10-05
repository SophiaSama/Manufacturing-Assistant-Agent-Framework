"""Headroom episodic semantic memory store for NGA with resilient local fallback."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("nga.memory.headroom")


class NgaEpisodicMemory:
    """Manages semantic memory indexing for plant incidents, resolutions, and operator notes."""

    def __init__(self, db_path: str = "data/headroom_memory.db"):
        self.db_path = db_path
        self._memory = None
        self._headroom_available = True
        self._init_sqlite_fallback()

    def _init_sqlite_fallback(self) -> None:
        try:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    """
                    CREATE TABLE IF NOT EXISTS episodic_memories (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        category TEXT,
                        class_a_alert INTEGER DEFAULT 0,
                        approver TEXT,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                con.commit()
            finally:
                con.close()
        except Exception as exc:
            logger.warning("failed_to_init_sqlite_memory_fallback err=%s", exc)

    def _get_memory(self):
        if not self._headroom_available:
            return None
        if self._memory is None:
            try:
                from headroom.memory import Memory
                self._memory = Memory(backend="local", db_path=self.db_path)
            except Exception as exc:
                logger.warning("failed_to_initialize_headroom_memory err=%s", exc)
                self._headroom_available = False
                return None
        return self._memory

    def _run_coroutine(self, coro):
        """Execute an async coroutine from sync context safely without deprecation warnings."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    return pool.submit(asyncio.run, coro).result()
            return asyncio.run(coro)
        except Exception:
            return None

    def _fallback_save(
        self,
        content: str,
        user_id: str,
        category: str = "GENERAL",
        class_a_alert: bool = False,
        approver: str | None = None,
    ) -> str | None:
        try:
            mem_id = f"mem_{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            con = sqlite3.connect(self.db_path)
            try:
                con.execute(
                    """
                    INSERT INTO episodic_memories (id, content, user_id, category, class_a_alert, approver, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mem_id, content, user_id, category, 1 if class_a_alert else 0, approver, now),
                )
                con.commit()
            finally:
                con.close()
            return mem_id
        except Exception as exc:
            logger.warning("sqlite_fallback_save_failed err=%s", exc)
            return None

    def _fallback_search(self, query: str, user_id: str, top_k: int = 3) -> list[dict[str, Any]]:
        try:
            con = sqlite3.connect(self.db_path)
            try:
                # Token-based keyword matching with scoring
                tokens = [t.lower() for t in query.split() if len(t) > 2]
                cursor = con.execute("SELECT id, content FROM episodic_memories ORDER BY created_at DESC")
                rows = cursor.fetchall()

                scored = []
                for row_id, content in rows:
                    content_lower = content.lower()
                    matches = sum(1 for t in tokens if t in content_lower)
                    if matches > 0:
                        score = round(matches / max(len(tokens), 1), 3)
                        scored.append({"id": row_id, "content": content, "score": score})

                scored.sort(key=lambda x: x["score"], reverse=True)
                return scored[:top_k]
            finally:
                con.close()
        except Exception as exc:
            logger.warning("sqlite_fallback_search_failed err=%s", exc)
            return []

    async def recall_similar_incidents_async(
        self, query: str, user_id: str = "operator", top_k: int = 3
    ) -> list[dict[str, Any]]:
        mem = self._get_memory()
        if mem is not None:
            try:
                results = await mem.search(query=query, user_id=user_id, top_k=top_k)
                return [
                    {
                        "content": r.content,
                        "score": getattr(r, "score", 0.0),
                        "id": getattr(r, "id", ""),
                    }
                    for r in results
                ]
            except Exception as exc:
                logger.warning("headroom_memory_search_failed, falling back err=%s", exc)
                self._headroom_available = False

        return self._fallback_search(query=query, user_id=user_id, top_k=top_k)

    def recall_similar_incidents(
        self, query: str, user_id: str = "operator", top_k: int = 3
    ) -> list[dict[str, Any]]:
        res = self._run_coroutine(
            self.recall_similar_incidents_async(query, user_id=user_id, top_k=top_k)
        )
        return res if res is not None else self._fallback_search(query=query, user_id=user_id, top_k=top_k)

    async def record_approved_decision_async(
        self,
        question: str,
        recommendation: str,
        category: str,
        class_a_alert: bool,
        approver: str | None,
        user_id: str = "operator",
    ) -> str | None:
        entry = (
            f"Incident: {question}\n"
            f"Resolution: {recommendation}\n"
            f"Category: {category}\n"
            f"Class A: {class_a_alert}\n"
            f"Approved By: {approver or 'System'}"
        )

        mem = self._get_memory()
        if mem is not None:
            try:
                memory_id = await mem.save(
                    content=entry,
                    user_id=user_id,
                    metadata={
                        "category": category,
                        "class_a_alert": class_a_alert,
                        "approver": approver,
                    },
                )
                logger.info("recorded_approved_decision_in_memory id=%s category=%s", memory_id, category)
                return memory_id
            except Exception as exc:
                logger.warning("headroom_memory_save_failed, falling back err=%s", exc)
                self._headroom_available = False

        return self._fallback_save(
            content=entry,
            user_id=user_id,
            category=category,
            class_a_alert=class_a_alert,
            approver=approver,
        )

    def record_approved_decision(
        self,
        question: str,
        recommendation: str,
        category: str,
        class_a_alert: bool,
        approver: str | None,
        user_id: str = "operator",
    ) -> str | None:
        res = self._run_coroutine(
            self.record_approved_decision_async(
                question=question,
                recommendation=recommendation,
                category=category,
                class_a_alert=class_a_alert,
                approver=approver,
                user_id=user_id,
            )
        )
        if res is not None:
            return res
        entry = (
            f"Incident: {question}\n"
            f"Resolution: {recommendation}\n"
            f"Category: {category}\n"
            f"Class A: {class_a_alert}\n"
            f"Approved By: {approver or 'System'}"
        )
        return self._fallback_save(
            content=entry,
            user_id=user_id,
            category=category,
            class_a_alert=class_a_alert,
            approver=approver,
        )

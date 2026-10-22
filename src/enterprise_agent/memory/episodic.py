"""Headroom episodic semantic memory store for Enterprise Agent with resilient local fallback."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import uuid
import weakref
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("enterprise_agent.memory.episodic")


class _BackgroundLoopRunner:
    """Manages a dedicated background thread with an active asyncio event loop."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def ensure_running(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None or self._loop.is_closed():
                self._loop = asyncio.new_event_loop()
                ready = threading.Event()
                loop = self._loop

                def _target() -> None:
                    asyncio.set_event_loop(loop)
                    ready.set()
                    try:
                        loop.run_forever()
                    finally:
                        try:
                            tasks = [t for t in asyncio.all_tasks(loop) if not t.done()]
                            for t in tasks:
                                t.cancel()
                            if tasks:
                                loop.run_until_complete(
                                    asyncio.gather(*tasks, return_exceptions=True)
                                )
                        except Exception:
                            pass
                        try:
                            loop.close()
                        except Exception:
                            pass

                self._thread = threading.Thread(target=_target, daemon=True, name="headroom-memory-loop")
                self._thread.start()
                ready.wait()
            return self._loop

    def run(self, coro: Any, timeout: float = 30.0) -> Any:
        loop = self.ensure_running()
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        try:
            return future.result(timeout=timeout)
        except Exception:
            future.cancel()
            raise

    def close(self, timeout: float = 5.0) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None

            if loop is not None and loop.is_running():
                def _stop() -> None:
                    loop.stop()

                loop.call_soon_threadsafe(_stop)
                if thread is not None and threading.current_thread() != thread:
                    thread.join(timeout=timeout)

    async def aclose(self, timeout: float = 5.0) -> None:
        """Asynchronously shut down the background runner thread and loop."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self.close, timeout)


class EpisodicMemoryStore:
    """Manages semantic memory indexing for incidents, resolutions, and operator notes."""

    _instances: weakref.WeakSet[EpisodicMemoryStore] = weakref.WeakSet()
    _instances_lock = threading.Lock()

    def __init__(self, db_path: str = "data/headroom_memory.db"):
        self.db_path = db_path
        self._memory = None
        self._headroom_available = True
        self._closed = False
        self._is_closing = False
        self._close_lock = threading.Lock()
        self._closing_done = threading.Event()
        self._runner: _BackgroundLoopRunner | None = _BackgroundLoopRunner()
        with self._instances_lock:
            self._instances.add(self)
        self._init_sqlite_fallback()

    @classmethod
    def close_all(cls) -> None:
        """Close all active episodic memory instances across the process."""
        with cls._instances_lock:
            instances = list(cls._instances)
            cls._instances.clear()
        for inst in instances:
            try:
                inst.close()
            except Exception:
                pass

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

    def _run_coroutine(self, coro: Any) -> Any:
        """Execute an async coroutine on the dedicated background loop."""
        if self._closed or self._is_closing or self._runner is None:
            if hasattr(coro, "close"):
                try:
                    coro.close()
                except Exception:
                    pass
            return None
        try:
            return self._runner.run(coro)
        except Exception as exc:
            if hasattr(coro, "close"):
                try:
                    coro.close()
                except Exception:
                    pass
            logger.warning("coroutine_execution_failed err=%s", exc)
            return None

    async def aclose(self) -> None:
        """Asynchronously close headroom memory and release all underlying clients and threads."""
        with self._close_lock:
            if self._closed:
                return
            if self._is_closing:
                already_closing = True
            else:
                self._is_closing = True
                already_closing = False
                mem = self._memory
                self._memory = None
                runner = self._runner
                self._runner = None

        if already_closing:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._closing_done.wait, 5.0)
            except Exception:
                pass
            return

        try:
            if mem is not None and runner is not None:
                runner_loop = runner.ensure_running()
                try:
                    curr_loop = asyncio.get_running_loop()
                except RuntimeError:
                    curr_loop = None

                if hasattr(mem, "close"):
                    try:
                        import inspect

                        res = mem.close()
                        if inspect.isawaitable(res):
                            if curr_loop is runner_loop:
                                await res
                            else:
                                fut = asyncio.run_coroutine_threadsafe(res, runner_loop)
                                await asyncio.wrap_future(fut)
                    except Exception as exc:
                        logger.warning("failed_to_close_headroom_memory err=%s", exc)

            if runner is not None:
                await runner.aclose()
        finally:
            with self._close_lock:
                self._closed = True
                self._is_closing = False
            self._closing_done.set()

    def close(self) -> None:
        """Synchronously close headroom memory and release all underlying clients and threads."""
        with self._close_lock:
            if self._closed:
                return
            if self._is_closing:
                already_closing = True
            else:
                self._is_closing = True
                already_closing = False
                mem = self._memory
                self._memory = None
                runner = self._runner
                self._runner = None

        if already_closing:
            self._closing_done.wait(timeout=5.0)
            return

        try:
            if mem is not None and runner is not None:
                if hasattr(mem, "close"):
                    try:
                        import inspect

                        res = mem.close()
                        if inspect.isawaitable(res):
                            runner.run(res, timeout=5.0)
                    except Exception as exc:
                        logger.warning("error_closing_headroom_memory_runner err=%s", exc)

            if runner is not None:
                runner.close()
        finally:
            with self._close_lock:
                self._closed = True
                self._is_closing = False
            self._closing_done.set()

    def __enter__(self) -> EpisodicMemoryStore:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    async def __aenter__(self) -> EpisodicMemoryStore:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

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

    async def _recall_similar_incidents_impl(
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

    async def recall_similar_incidents_async(
        self, query: str, user_id: str = "operator", top_k: int = 3
    ) -> list[dict[str, Any]]:
        if self._closed or self._is_closing or self._runner is None:
            return self._fallback_search(query=query, user_id=user_id, top_k=top_k)

        runner_loop = self._runner.ensure_running()
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if curr_loop is runner_loop:
            return await self._recall_similar_incidents_impl(query, user_id=user_id, top_k=top_k)

        fut = asyncio.run_coroutine_threadsafe(
            self._recall_similar_incidents_impl(query, user_id=user_id, top_k=top_k),
            runner_loop,
        )
        return await asyncio.wrap_future(fut)

    def recall_similar_incidents(
        self, query: str, user_id: str = "operator", top_k: int = 3
    ) -> list[dict[str, Any]]:
        res = self._run_coroutine(
            self.recall_similar_incidents_async(query, user_id=user_id, top_k=top_k)
        )
        return res if res is not None else self._fallback_search(query=query, user_id=user_id, top_k=top_k)

    async def _record_approved_decision_impl(
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

    async def record_approved_decision_async(
        self,
        question: str,
        recommendation: str,
        category: str,
        class_a_alert: bool,
        approver: str | None,
        user_id: str = "operator",
    ) -> str | None:
        if self._closed or self._is_closing or self._runner is None:
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

        runner_loop = self._runner.ensure_running()
        try:
            curr_loop = asyncio.get_running_loop()
        except RuntimeError:
            curr_loop = None

        if curr_loop is runner_loop:
            return await self._record_approved_decision_impl(
                question, recommendation, category, class_a_alert, approver, user_id
            )

        fut = asyncio.run_coroutine_threadsafe(
            self._record_approved_decision_impl(
                question, recommendation, category, class_a_alert, approver, user_id
            ),
            runner_loop,
        )
        return await asyncio.wrap_future(fut)

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


# Backward compatibility alias
NgaEpisodicMemory = EpisodicMemoryStore

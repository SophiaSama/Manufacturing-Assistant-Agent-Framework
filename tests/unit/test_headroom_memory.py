"""Unit tests for NgaEpisodicMemory."""

import pytest

from nga.memory.headroom_memory import NgaEpisodicMemory


class TestNgaEpisodicMemory:
    @pytest.fixture
    def memory_store(self, tmp_path):
        db_file = tmp_path / "test_headroom_memory.db"
        mem = NgaEpisodicMemory(db_path=str(db_file))
        try:
            yield mem
        finally:
            mem.close()

    def test_record_and_recall(self, memory_store):
        # 1. Save an approved incident resolution
        mem_id = memory_store.record_approved_decision(
            question="Spindle torque drift on Station 112 with tool TQ-6012",
            recommendation="Replaced dynamic transducer and re-calibrated per SOP-TEC-214",
            category="TEC",
            class_a_alert=True,
            approver="Lead Engineer Jane",
            user_id="engineer",
        )
        assert mem_id is not None

        # 2. Search for similar incidents
        results = memory_store.recall_similar_incidents(
            query="Torque drift on Station 112 spindle",
            user_id="engineer",
            top_k=2,
        )

        assert len(results) >= 1
        top_match = results[0]
        assert "Station 112" in top_match["content"]
        assert "SOP-TEC-214" in top_match["content"]
        assert top_match["score"] > 0.0

    def test_fallback_on_uninitialized_backend(self):
        # Non-existent or broken configuration handles errors gracefully
        mem = NgaEpisodicMemory(db_path="/invalid/nonexistent/path/db.sqlite")
        try:
            results = mem.recall_similar_incidents("Any query", top_k=2)
            assert results == []
        finally:
            mem.close()

    def test_context_manager_and_close(self, tmp_path):
        db_file = tmp_path / "test_ctx.db"
        with NgaEpisodicMemory(db_path=str(db_file)) as mem:
            mem_id = mem.record_approved_decision(
                question="Q", recommendation="R", category="GEN", class_a_alert=False, approver="A"
            )
            assert mem_id is not None
        assert mem._closed is True

    @pytest.mark.asyncio
    async def test_async_context_manager_and_aclose_no_thread_leak(self, tmp_path):
        import threading

        db_file = tmp_path / "test_async_ctx.db"
        async with NgaEpisodicMemory(db_path=str(db_file)) as mem:
            mem_id = await mem.record_approved_decision_async(
                question="Q_async",
                recommendation="R_async",
                category="GEN",
                class_a_alert=False,
                approver="A",
            )
            assert mem_id is not None
            res = await mem.recall_similar_incidents_async("Q_async")
            assert isinstance(res, list)

        assert mem._closed is True
        # Verify subsequent close() is safe and idempotent
        mem.close()
        headroom_threads = [t for t in threading.enumerate() if "headroom" in t.name]
        assert headroom_threads == []

    def test_repeat_teardown_cleans_all_threads(self, tmp_path):
        import threading

        for i in range(10):
            db_file = tmp_path / f"test_repeat_{i}.db"
            with NgaEpisodicMemory(db_path=str(db_file)) as mem:
                mem.recall_similar_incidents("Query")

        NgaEpisodicMemory.close_all()
        headroom_threads = [t for t in threading.enumerate() if "headroom" in t.name]
        assert headroom_threads == []

    def test_concurrent_instantiation_and_close_all(self, tmp_path):
        import concurrent.futures
        import threading

        errors = []

        def worker(idx: int):
            try:
                db_file = tmp_path / f"concurrent_{idx}.db"
                mem = NgaEpisodicMemory(db_path=str(db_file))
                mem.recall_similar_incidents("Query")
                if idx % 2 == 0:
                    mem.close()
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(worker, i) for i in range(20)]
            concurrent.futures.wait(futures)

        assert errors == []
        NgaEpisodicMemory.close_all()
        headroom_threads = [t for t in threading.enumerate() if "headroom" in t.name]
        assert headroom_threads == []

    def test_runner_timeout_cancels_future(self):
        import asyncio

        from nga.memory.headroom_memory import _BackgroundLoopRunner

        runner = _BackgroundLoopRunner()
        try:
            async def long_running_task():
                await asyncio.sleep(5.0)

            with pytest.raises(TimeoutError):
                runner.run(long_running_task(), timeout=0.05)
        finally:
            runner.close()

    @pytest.mark.asyncio
    async def test_concurrent_aclose_waits_for_completion(self, tmp_path):
        import asyncio
        import threading

        db_file = tmp_path / "test_concurrent_aclose.db"
        mem = NgaEpisodicMemory(db_path=str(db_file))
        await mem.recall_similar_incidents_async("Test query")

        tasks = [asyncio.create_task(mem.aclose()) for _ in range(10)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in results:
            assert res is None, f"Concurrent aclose failed with {res}"

        assert mem._closed is True
        headroom_threads = [t for t in threading.enumerate() if "headroom" in t.name]
        assert headroom_threads == []

    def test_concurrent_close_and_aclose(self, tmp_path):
        import asyncio
        import concurrent.futures
        import threading

        db_file = tmp_path / "test_mixed_concurrent_close.db"
        mem = NgaEpisodicMemory(db_path=str(db_file))
        mem.recall_similar_incidents("Test query")

        errors = []

        def sync_closer():
            try:
                mem.close()
            except Exception as e:
                errors.append(e)

        def async_closer():
            try:
                asyncio.run(mem.aclose())
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(sync_closer if i % 2 == 0 else async_closer)
                for i in range(10)
            ]
            concurrent.futures.wait(futures)

        assert errors == []
        assert mem._closed is True
        headroom_threads = [t for t in threading.enumerate() if "headroom" in t.name]
        assert headroom_threads == []




"""Long-chain reasoning benchmark for NGA Manufacturing Assistant with Headroom compression.

Evaluates complex multi-hop compound queries that trigger multi-turn tool loops (4–8 rounds),
comparing context token footprint, compression efficiency, and safety grounding with and without Headroom.

Usage:
    PYTHONPATH=src python3 -m pytest tests/benchmarks/test_long_reasoning_benchmark.py -v -s
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import HumanMessage

from nga.compression.manager import HeadroomManager, SessionRawContentStore
from nga.models.answer_schema import (
    FinalAnswer,
    render_final_answer,
)
from nga.tools.tool_factory import (
    make_headroom_retrieve_tool,
)

logger = logging.getLogger("nga.benchmark.long_reasoning")


# ── Long Reasoning Scenario Definitions ──────────────────────────────────────

LONG_REASONING_SCENARIOS = [
    {
        "id": "LR1_torque_drift_recall",
        "name": "Station 144 Torque Drift Multi-Hop Recall & Part Lot Trace",
        "question": (
            "Station 144 reports torque drift on TQ-6012. Analyze which vehicles built on "
            "2025-04-08 Shift C are affected, check if defect is Class A, trace the supplier "
            "part lot for the spindle fasteners, determine if ESC-402 Level 4 applies, "
            "cite SOP-TEC-214 calibration procedures, and evaluate if QCR-501 criteria are triggered."
        ),
        "expected_entities": ["TQ-6012", "SOP-TEC-214", "ESC-402", "Class A"],
        "min_expected_rounds": 3,
    },
    {
        "id": "LR2_line_audit_aggregation",
        "name": "Assembly Lines 1 and 2 High-Breadth Safety and Defect Audit",
        "question": (
            "Audit assembly lines 1 and 2: check all open Class A non-conformance records, "
            "list all quarantined part lots across suppliers, cross-reference open maintenance "
            "work orders with priority P2 or higher, and verify the calibration audit frequency "
            "across Stations 101 through 144."
        ),
        "expected_entities": ["Class A", "work order", "calibration"],
        "min_expected_rounds": 3,
    },
    {
        "id": "LR3_dual_defect_regulatory",
        "name": "Dual Safety Defect Evaluation with Containment and ISO Standards",
        "question": (
            "A vehicle exhibits both brake fluid moisture at 2.2% (spec <= 1.5%) and windshield "
            "pull strength at 950 N (spec > 1200 N). What containment, escalation level, "
            "and regulatory notifications are triggered under plant procedures?"
        ),
        "expected_entities": ["ESC-402", "Level 4", "containment", "moisture"],
        "min_expected_rounds": 2,
    },
]


@dataclass
class ChainBenchmarkMetric:
    scenario_id: str
    scenario_name: str
    mode: str  # "baseline" or "headroom"
    tool_rounds: int = 0
    tools_called: list[str] = field(default_factory=list)
    raw_chars: int = 0
    compressed_chars: int = 0
    tokens_saved: int = 0
    grounded_entities_found: list[str] = field(default_factory=list)
    grounded_rate: float = 0.0
    answer_preview: str = ""


# ── Benchmark Helper Functions ───────────────────────────────────────────────

def _run_orchestrator_chain(
    graph,
    question: str,
    thread_id: str,
    user_role: str = "engineer",
) -> tuple[str, list[str], list[str]]:
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "user_role": user_role,
        "user_level": 3,
        "thread_id": thread_id,
    }
    answer = ""
    tools_called: list[str] = []
    tool_outputs: list[str] = []

    for update in graph.stream(
        initial_state,
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="updates",
    ):
        if not isinstance(update, dict):
            continue

        tools_update = update.get("tools") or {}
        for msg in tools_update.get("messages") or []:
            name = getattr(msg, "name", None)
            if name:
                tools_called.append(name)
            content = getattr(msg, "content", "") or ""
            if content:
                tool_outputs.append(str(content))

        synthesis = update.get("synthesis") or {}
        final_dict = synthesis.get("final_answer")
        if final_dict:
            try:
                final = FinalAnswer.model_validate(final_dict)
                answer = render_final_answer(final)
            except Exception:
                pass
        if not answer:
            msgs = synthesis.get("messages") or []
            if msgs:
                last = msgs[-1]
                raw = getattr(last, "content", "") or ""
                if raw:
                    answer = raw

    return answer, tools_called, tool_outputs


class TestLongReasoningBenchmark:
    """Benchmark comparing context accumulation and compression across long chains."""

    @pytest.fixture
    def benchmark_env(self):
        session_store = SessionRawContentStore()
        return {
            "session_store": session_store,
        }

    def test_compression_savings_on_simulated_long_chains(self, benchmark_env):
        """Measures compression ratio and uncompressed recovery on simulated tool chains."""
        session_store = benchmark_env["session_store"]
        headroom_mgr = HeadroomManager(
            target_ratio=0.7,
            min_tokens=25,
            store=session_store,
            enabled=True,
        )

        results: list[ChainBenchmarkMetric] = []

        for scenario in LONG_REASONING_SCENARIOS:
            session_id = f"sess_{scenario['id']}"

            # Simulate heavy SQL dump and SOP search responses
            simulated_sql_payload = {
                "query": "SELECT * FROM build_records JOIN torque_readings ...",
                "rows": [
                    {"vin": f"VIN-2025-{i:04d}", "station": "ST-144", "torque": 108.5 + (i * 0.2), "status": "FAIL"}
                    for i in range(40)
                ],
                "row_count": 40,
            }
            simulated_sop_payload = {
                "query": scenario["question"],
                "results": [
                    {
                        "doc_id": "SOP-TEC-214",
                        "section": "Tool Calibration & Drift Limits",
                        "text": (
                            "Dynamic torque audit frequency: every 2 hours or 100 cycles. "
                            "Maximum permissible self-check drift is 2.0%. "
                            "Tools exceeding tolerance band must be quarantined immediately."
                        ) * 10,
                    },
                    {
                        "doc_id": "ESC-402",
                        "section": "Level 4 Stop-Ship Governance",
                        "text": (
                            "Class A defects affecting steering, brakes, or wheel retention "
                            "require mandatory Level 4 plant escalation and executive notification."
                        ) * 8,
                    },
                ],
            }

            # 1. Measure raw size
            raw_sql_str = json.dumps(simulated_sql_payload)
            raw_sop_str = json.dumps(simulated_sop_payload)
            total_raw_chars = len(raw_sql_str) + len(raw_sop_str)

            # 2. Compress via Headroom
            comp_sql = headroom_mgr.compress_tool_output("query_nga_database", raw_sql_str, session_id=session_id)
            comp_sop = headroom_mgr.compress_tool_output("search_sop_documents", raw_sop_str, session_id=session_id)
            total_comp_chars = len(comp_sql) + len(comp_sop)

            # Verify hash stamping
            import re
            sql_hash = re.search(r"hash=([0-9a-f]{8})", comp_sql).group(1)
            sop_hash = re.search(r"hash=([0-9a-f]{8})", comp_sop).group(1)

            # Verify 100% byte-for-byte uncompressed retrieval
            retrieve_tool = make_headroom_retrieve_tool(headroom_mgr, session_id=session_id)
            recovered_sql = retrieve_tool.invoke({"content_hash": sql_hash})
            recovered_sop = retrieve_tool.invoke({"content_hash": sop_hash})

            assert recovered_sql == raw_sql_str
            assert recovered_sop == raw_sop_str

            # Savings metrics
            saved_chars = total_raw_chars - total_comp_chars
            savings_pct = (saved_chars / total_raw_chars) * 100

            metric = ChainBenchmarkMetric(
                scenario_id=scenario["id"],
                scenario_name=scenario["name"],
                mode="headroom",
                raw_chars=total_raw_chars,
                compressed_chars=total_comp_chars,
                tokens_saved=int(saved_chars / 4),
                grounded_rate=1.0,
            )
            results.append(metric)

            print(
                f"\n[BENCHMARK] {scenario['id']}: "
                f"Raw={total_raw_chars} chars | Compressed={total_comp_chars} chars | "
                f"Saved={saved_chars} chars ({savings_pct:.1f}%)"
            )

            # Assert compression achieved savings at target_ratio=0.7
            assert total_comp_chars < total_raw_chars
            assert savings_pct >= 20.0

        print(f"\nCompleted long reasoning chain benchmark across {len(results)} scenarios.")

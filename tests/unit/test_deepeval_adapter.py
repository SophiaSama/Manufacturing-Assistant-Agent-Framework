"""Unit tests for DeepEval test case conversion layer (deepeval_adapter.py)."""

from __future__ import annotations

import json

from deepeval.test_case import LLMTestCase, ToolCall

from nga.evaluation.deepeval_adapter import (
    conflict_scenario_to_test_cases,
    extract_retrieval_context,
    to_deep_eval_test_case,
)


def test_extract_retrieval_context_from_search_sop():
    sop_payload = json.dumps({
        "results": [
            {
                "doc_id": "SOP-OPR-101",
                "section": "4.2",
                "category": "safety",
                "corpus_source": "canonical",
                "text": "Torque lug nuts to 105 Nm using TorqMaster TF-6000 in star pattern.",
            }
        ],
        "graph_evidence": {
            "entities": [
                {
                    "name": "Station 144",
                    "type": "STATION",
                    "description": "Wheel assembly station",
                    "source_docs": ["SOP-OPR-101"],
                }
            ],
            "relations": [
                {
                    "source": "Station 144",
                    "target": "TorqMaster TF-6000",
                    "relation": "USES_TOOL",
                    "description": "Calibrated digital torque wrench",
                }
            ],
            "community_summaries": ["Assembly operations community 1"],
        },
    })

    contexts = extract_retrieval_context([sop_payload])
    assert len(contexts) >= 3
    assert any("[SOP-OPR-101]" in c and "105 Nm" in c for c in contexts)
    assert any("[GRAPH ENTITIES]" in c and "Station 144" in c for c in contexts)
    assert any("[GRAPH RELATIONS]" in c and "USES_TOOL" in c for c in contexts)


def test_extract_retrieval_context_from_sql():
    sql_payload = json.dumps({
        "query": "SELECT station, torque_nm FROM work_orders WHERE id='WO-101'",
        "rows": [{"station": "Station 144", "torque_nm": 105.0}],
        "row_count": 1,
        "tables": ["work_orders"],
    })

    contexts = extract_retrieval_context([sql_payload])
    assert len(contexts) == 1
    assert "[SQL RESULT]" in contexts[0]
    assert "tables=work_orders" in contexts[0]
    assert "WO-101" in contexts[0]


def test_extract_retrieval_context_fallback_to_expected():
    q_data = {
        "id": "Q-FALLBACK",
        "expected_answer": "105 Nm per SOP-OPR-101",
        "source_docs": ["SOP-OPR-101"],
    }
    contexts = extract_retrieval_context(tool_outputs=None, question_data=q_data)
    assert len(contexts) == 1
    assert "[SOP-OPR-101]" in contexts[0]
    assert "105 Nm per SOP-OPR-101" in contexts[0]


def test_to_deep_eval_test_case_full_mapping():
    q_data = {
        "id": "Q-001",
        "category": "assembly",
        "tier": "L1",
        "question": "What is the lug nut torque at Station 144?",
        "expected_answer": "105 Nm ± 5%",
        "source_docs": ["SOP-OPR-101"],
        "requires_sql": True,
    }
    tool_raw = [
        {
            "name": "search_sop_documents",
            "args": {"query": "Station 144 lug nut torque"},
            "output": json.dumps({"results": [{"text": "Torque specification: 105 Nm"}]}),
        }
    ]

    actual = "The lug nut torque at Station 144 is 105 Nm."
    tc = to_deep_eval_test_case(
        question_data=q_data,
        actual_output=actual,
        tool_calls_raw=tool_raw,
    )

    assert isinstance(tc, LLMTestCase)
    assert tc.name == "Q-001"
    assert tc.input == "What is the lug nut torque at Station 144?"
    assert tc.actual_output == actual
    assert tc.expected_output == "105 Nm ± 5%"
    assert len(tc.retrieval_context) == 1
    assert "Torque specification: 105 Nm" in tc.retrieval_context[0]
    assert len(tc.tools_called) == 1
    assert isinstance(tc.tools_called[0], ToolCall)
    assert tc.tools_called[0].name == "search_sop_documents"
    assert any(exp.name == "query_nga_database" for exp in tc.expected_tools)
    assert tc.metadata.get("tier") == "L1"


def test_conflict_scenario_to_test_cases():
    scenario = {
        "id": "CONFLICT-TEST",
        "name": "Wheel torque conflict",
        "golden": "Discrepancy: SOP-101 specifies 105 Nm, variant specifies 108 Nm.",
        "answers": {
            "conflict_flagged": "There is a contradiction: 105 Nm vs 108 Nm.",
            "blind_variant": "108 Nm.",
            "hallucinated_middle": "106.5 Nm average.",
        },
    }

    cases = conflict_scenario_to_test_cases(scenario)
    assert set(cases.keys()) == {"conflict_flagged", "blind_variant", "hallucinated_middle"}
    for condition, tc in cases.items():
        assert isinstance(tc, LLMTestCase)
        assert tc.metadata.get("condition") == condition
        assert tc.metadata.get("scenario_id") == "CONFLICT-TEST"
        assert tc.expected_output == scenario["golden"]
        assert tc.retrieval_context == [scenario["golden"]]

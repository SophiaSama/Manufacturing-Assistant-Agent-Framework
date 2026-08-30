"""FastAPI backend server for the NGA Manufacturing Assistant Web Interface."""

from __future__ import annotations

import logging
import sqlite3
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from nga.cache import CachedEmbeddings, make_cache_from_settings
from nga.config import Settings
from nga.evaluation.ci_tracker import (
    generate_svg_trend_chart,
    get_trend_series,
    load_history,
    record_eval_run,
)
from nga.evaluation.eval_runner import (
    compare_evaluation_runs,
    get_evaluation_report,
    list_evaluation_reports,
    load_eval_questions,
    run_evaluation_suite,
)
from nga.graph.orchestrator import build_orchestrator
from nga.ingestion.build_graph import load_graph
from nga.memory.checkpointer import build_checkpointer
from nga.memory.decision_log import (
    get_decision_by_id,
    get_decisions,
    init_decision_log,
    update_decision,
)
from nga.models.answer_schema import (
    FinalAnswer,
    parse_final_answer,
    render_final_answer,
)
from nga.providers.factory import make_embeddings
from nga.rag_agent.rbac import ACCESS_LEVELS
from nga.tools.tool_factory import make_retrieval_tool, make_sql_tool

logger = logging.getLogger("nga.ui.server")

# In-memory circular buffer for UI log streaming
MAX_LOG_RECORDS = 500
_LOG_BUFFER: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_RECORDS)


class UILogHandler(logging.Handler):
    """Logging handler that retains recent formatted logs in memory."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            _LOG_BUFFER.append({
                "id": str(uuid.uuid4()),
                "timestamp": datetime.fromtimestamp(record.created, timezone.utc).strftime("%H:%M:%S.%f")[:-3],
                "level": record.levelname,
                "logger": record.name,
                "message": msg,
            })
        except Exception:
            self.handleError(record)


# Attach UI log handler
_ui_handler = UILogHandler()
_ui_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger("nga").addHandler(_ui_handler)
logging.getLogger("nga").setLevel(logging.INFO)


# ── Request / Response Models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., description="User question or statement")
    role: str = Field("operator", description="Active user role (operator, technician, engineer, manager)")
    thread_id: str | None = Field(None, description="Conversation thread ID")


class HILActionRequest(BaseModel):
    action: str = Field(..., description="Action: 'approved' or 'rejected'")
    approver: str | None = Field(None, description="Approver Name/ID (Mandatory for Class A or rejections)")
    reason: str | None = Field(None, description="Justification note or rejection reason")


class EvalRunRequest(BaseModel):
    category: str = Field("all", description="Category or 'all'")
    limit: int | None = Field(None, description="Limit question count")
    smoke_test: bool = Field(False, description="Run quick 5-question smoke test across categories")
    run_label: str | None = Field(None, description="Custom run label")
    role: str = Field("manager", description="Role for evaluation execution")
    cache_mode: str = Field("cold", description="'cold' (default) or 'hot' (eval cache enabled)")


class EvalCompareRequest(BaseModel):
    run_a: str = Field(..., description="Baseline run label")
    run_b: str = Field(..., description="Candidate run label")


# ── Backend State & Graph Manager ─────────────────────────────────────────────

class AppContext:
    """Manages loaded embeddings, vector store, graph data, and agent orchestrators per role."""

    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.embeddings: Any = None
        self.vector_store: Chroma | None = None
        self.graph_data: Any = None
        self.checkpointer: Any = None
        self.sql_tool: Any = None
        self.agent_graphs: dict[str, Any] = {}
        self.initialized: bool = False

    def setup(self) -> None:
        if self.initialized:
            return

        self.settings = Settings.from_env()
        init_decision_log(self.settings.app_state_db_path)

        self.cache = make_cache_from_settings(self.settings, env="prod")
        if self.cache is not None:
            logger.info("cache enabled env=prod db=%s", self.settings.cache_db_path)

        try:
            self.embeddings = make_embeddings(self.settings)
            if self.cache is not None:
                self.embeddings = CachedEmbeddings(
                    self.embeddings, cache=self.cache,
                    model_id=self.settings.embedding_model,
                )
            self.vector_store = Chroma(
                collection_name="nga_reference_docs",
                embedding_function=self.embeddings,
                persist_directory=self.settings.vector_store_dir,
            )
        except Exception as exc:
            logger.warning("Vector store init warning: %s", exc)

        try:
            self.graph_data = load_graph(self.settings.graph_store_dir)
        except Exception as exc:
            logger.warning("Graph data init warning: %s", exc)

        self.checkpointer = build_checkpointer(self.settings.app_state_db_path)
        self.sql_tool = make_sql_tool(self.settings.nga_db_path, cache=self.cache)

        # Build orchestrator graph for each role
        for role_name, access_lvl in ACCESS_LEVELS.items():
            if self.vector_store is not None and self.embeddings is not None:
                retrieval_tool = make_retrieval_tool(
                    self.vector_store,
                    graph=self.graph_data,
                    embeddings=self.embeddings,
                    user_level=access_lvl,
                    cache=self.cache,
                )
            else:
                retrieval_tool = None

            if retrieval_tool is not None:
                self.agent_graphs[role_name] = build_orchestrator(
                    self.settings, self.checkpointer, self.sql_tool, retrieval_tool
                )

        self.initialized = True
        logger.info("NGA UI backend setup complete. Available roles: %s", list(self.agent_graphs.keys()))

    def get_graph(self, role: str) -> Any:
        self.setup()
        role_norm = role.lower().strip()
        if role_norm not in self.agent_graphs:
            role_norm = "operator"
        return self.agent_graphs.get(role_norm)


ctx = AppContext()


# ── FastAPI App Creation ──────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="NGA Manufacturing Assistant Web Interface",
        description="Decision-support interface for Northgate Assembly Plant with RBAC, HIL, Logging, and Evaluations.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request as StarletteRequest

    class NoCacheStaticMiddleware(BaseHTTPMiddleware):
        """Disable caching for JS/CSS static assets so the browser always fetches the latest version."""

        async def dispatch(self, request: StarletteRequest, call_next):
            response = await call_next(request)
            path = request.url.path
            if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
                response.headers["Pragma"] = "no-cache"
            return response

    app.add_middleware(NoCacheStaticMiddleware)

    # ── Role Endpoints ────────────────────────────────────────────────────────

    @app.get("/api/roles")
    def get_roles() -> dict[str, Any]:
        role_descriptions = {
            "operator": {
                "title": "Assembly Line Operator",
                "level": 1,
                "description": "Standard Operating Procedures (SOP-OPR-*), assembly steps, and torque specifications.",
                "allowed_domains": ["operator-sops/"],
                "can_evaluate_recalls": False,
                "can_perform_rca": False,
            },
            "technician": {
                "title": "Maintenance Technician",
                "level": 2,
                "description": "Troubleshooting procedures (SOP-TEC-*), machine specifications, calibrations, and work orders.",
                "allowed_domains": ["operator-sops/", "technician-sops/", "machine-details/", "additional-docs/maintenance-work-orders/"],
                "can_evaluate_recalls": False,
                "can_perform_rca": False,
            },
            "engineer": {
                "title": "Process / Quality Engineer",
                "level": 3,
                "description": "Root-cause analysis (8D / FAP-401), non-conformance records, supplier quality, and training records.",
                "allowed_domains": ["operator-sops/", "technician-sops/", "machine-details/", "failure-analysis/", "additional-docs/supplier-quality/", "additional-docs/training/"],
                "can_evaluate_recalls": False,
                "can_perform_rca": True,
            },
            "manager": {
                "title": "Plant Manager / Quality Director",
                "level": 4,
                "description": "Full access: Recall criteria (QCR-501 C1–C5), Stop-Ship authorization, NRSA regulatory compliance, and field actions.",
                "allowed_domains": ["* (All Documents & Records)"],
                "can_evaluate_recalls": True,
                "can_perform_rca": True,
            },
        }
        return {
            "roles": [
                {
                    "id": k,
                    "name": k.capitalize(),
                    "title": v["title"],
                    "level": v["level"],
                    "description": v["description"],
                    "allowed_domains": v["allowed_domains"],
                    "can_evaluate_recalls": v["can_evaluate_recalls"],
                    "can_perform_rca": v["can_perform_rca"],
                }
                for k, v in role_descriptions.items()
            ]
        }

    # ── Status & Health ───────────────────────────────────────────────────────

    @app.get("/api/status")
    def get_status() -> dict[str, Any]:
        ctx.setup()
        settings = ctx.settings

        # Check DB connection
        db_ok = False
        db_rows_count = 0
        if settings and Path(settings.nga_db_path).exists():
            try:
                con = sqlite3.connect(settings.nga_db_path)
                cur = con.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
                db_rows_count = cur.fetchone()[0]
                con.close()
                db_ok = True
            except Exception:
                db_ok = False

        # Check vector store
        vector_count = 0
        if ctx.vector_store is not None:
            try:
                col = ctx.vector_store._collection
                vector_count = col.count() if col else 0
            except Exception:
                vector_count = 0

        # Check decisions log
        pending_decisions = 0
        total_decisions = 0
        if settings and Path(settings.app_state_db_path).exists():
            try:
                con = sqlite3.connect(settings.app_state_db_path)
                cur = con.execute("SELECT count(*) FROM decisions_log")
                total_decisions = cur.fetchone()[0]
                cur = con.execute("SELECT count(*) FROM decisions_log WHERE status='pending'")
                pending_decisions = cur.fetchone()[0]
                con.close()
            except Exception:
                pass

        return {
            "status": "online",
            "environment": settings.execution_environment if settings else "Unknown",
            "provider": settings.provider if settings else "Unknown",
            "model": settings.openrouter_model if settings and settings.execution_environment == "Cloud" else (settings.ollama_chat_model if settings else ""),
            "database": {
                "path": settings.nga_db_path if settings else "",
                "ok": db_ok,
                "tables_count": db_rows_count,
            },
            "vector_store": {
                "path": settings.vector_store_dir if settings else "",
                "doc_chunks": vector_count,
            },
            "decisions": {
                "total": total_decisions,
                "pending": pending_decisions,
            },
            "roles_available": list(ACCESS_LEVELS.keys()),
        }

    # ── Chat Execution Endpoint ───────────────────────────────────────────────

    @app.post("/api/chat")
    def run_chat(request: ChatRequest) -> dict[str, Any]:
        role = request.role.lower().strip()
        if role not in ACCESS_LEVELS:
            role = "operator"

        graph = ctx.get_graph(role)
        if graph is None:
            raise HTTPException(status_code=500, detail="Agent graph could not be initialized")

        thread_id = request.thread_id or str(uuid.uuid4())
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "user_role": role,
            "user_level": ACCESS_LEVELS.get(role, 1),
        }

        trace_steps: list[dict[str, Any]] = []
        tool_calls_log: list[dict[str, Any]] = []
        question_parts: list[str] = []
        final_answer_obj: FinalAnswer | None = None
        rendered_answer: str = ""
        pending_rec: str | None = None
        decision_record: dict[str, Any] | None = None

        logger.info("Chat turn started | Role=%s | Thread=%s | Question: %s", role.upper(), thread_id[:8], request.message[:80])

        try:
            for update in graph.stream(
                initial_state,
                config={"configurable": {"thread_id": thread_id}},
                stream_mode="updates",
            ):
                if not isinstance(update, dict):
                    continue

                for node_name, node_data in update.items():
                    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]

                    if node_name == "prepare":
                        qp = node_data.get("question_parts", [])
                        question_parts = [p for p in qp if isinstance(p, str) and p.strip()]
                        trace_steps.append({
                            "node": "prepare",
                            "timestamp": ts,
                            "summary": f"Identified {len(question_parts)} sub-question part(s)",
                            "details": {"question_parts": question_parts},
                        })

                    elif node_name == "agent":
                        msgs = node_data.get("messages", [])
                        t_calls = []
                        content = ""
                        for m in msgs:
                            if hasattr(m, "tool_calls") and m.tool_calls:
                                for tc in m.tool_calls:
                                    t_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                                    t_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                                    t_calls.append({"name": t_name, "args": t_args})
                            if hasattr(m, "content") and m.content:
                                content = str(m.content)

                        trace_steps.append({
                            "node": "agent",
                            "timestamp": ts,
                            "summary": f"Agent reasoning: {len(t_calls)} tool call(s) generated" if t_calls else "Agent response synthesized",
                            "details": {"tool_calls": t_calls, "content_snippet": content[:200] if content else ""},
                        })

                    elif node_name == "tools":
                        msgs = node_data.get("messages", [])
                        for m in msgs:
                            t_name = getattr(m, "name", "tool")
                            t_content = m.content if hasattr(m, "content") else str(m)
                            tool_calls_log.append({
                                "tool": t_name,
                                "timestamp": ts,
                                "output": str(t_content),
                            })
                        trace_steps.append({
                            "node": "tools",
                            "timestamp": ts,
                            "summary": f"Executed {len(msgs)} tool action(s)",
                            "details": {"tools_executed": [getattr(m, "name", "tool") for m in msgs]},
                        })

                    elif node_name == "synthesis":
                        fa_payload = node_data.get("final_answer")
                        if isinstance(fa_payload, dict):
                            try:
                                final_answer_obj = FinalAnswer.model_validate(fa_payload)
                            except Exception:
                                pass

                        if final_answer_obj is None:
                            msgs = node_data.get("messages", [])
                            if msgs:
                                last = msgs[-1]
                                raw = last.content if hasattr(last, "content") else str(last)
                                final_answer_obj = parse_final_answer(str(raw), question_parts)

                        if final_answer_obj:
                            rendered_answer = render_final_answer(final_answer_obj)
                            pending_rec = final_answer_obj.recommendation

                        trace_steps.append({
                            "node": "synthesis",
                            "timestamp": ts,
                            "summary": "Synthesized structured final answer with safety checks",
                            "details": {
                                "class_a_alert": final_answer_obj.class_a_alert if final_answer_obj else False,
                                "escalation_level": final_answer_obj.escalation_level if final_answer_obj else None,
                                "recall_criteria": final_answer_obj.recall_criteria_met if final_answer_obj else [],
                            },
                        })

                    elif node_name == "hitl":
                        rec = node_data.get("pending_recommendation")
                        trace_steps.append({
                            "node": "hitl",
                            "timestamp": ts,
                            "summary": "Human-In-The-Loop safety gate processed",
                            "details": {"recommendation": rec},
                        })

        except Exception as exc:
            logger.exception("Chat processing error: %s", exc)
            raise HTTPException(status_code=500, detail=f"Graph execution failed: {exc}")

        # Fallback if synthesis was not captured
        if final_answer_obj is None:
            final_answer_obj = FinalAnswer(
                direct_answer=rendered_answer or "No response generated by agent.",
                unanswered_questions=question_parts,
            )
            rendered_answer = render_final_answer(final_answer_obj)

        # Check if a new decision was logged
        if pending_rec and ctx.settings:
            recent_decisions = get_decisions(ctx.settings.app_state_db_path, limit=1)
            if recent_decisions:
                decision_record = recent_decisions[0]

        logger.info("Chat turn completed | Role=%s | ClassA=%s | Escalation=%s", role.upper(), final_answer_obj.class_a_alert, final_answer_obj.escalation_level)

        return {
            "thread_id": thread_id,
            "role": role,
            "user_level": ACCESS_LEVELS.get(role, 1),
            "rendered_answer": rendered_answer,
            "final_answer": final_answer_obj.model_dump(),
            "trace_steps": trace_steps,
            "tool_calls": tool_calls_log,
            "pending_decision": decision_record,
        }

    # ── HIL Decision Endpoints ────────────────────────────────────────────────

    @app.get("/api/decisions")
    def list_decisions(
        status: str | None = Query(None, description="Filter by status: pending, approved, rejected"),
        limit: int = Query(50, description="Max decisions to return"),
    ) -> dict[str, Any]:
        ctx.setup()
        if not ctx.settings:
            return {"decisions": []}
        records = get_decisions(ctx.settings.app_state_db_path, status=status, limit=limit)
        return {"decisions": records, "count": len(records)}

    @app.get("/api/decisions/{decision_id}")
    def get_decision(decision_id: int) -> dict[str, Any]:
        ctx.setup()
        if not ctx.settings:
            raise HTTPException(status_code=500, detail="Server not configured")
        record = get_decision_by_id(ctx.settings.app_state_db_path, decision_id)
        if not record:
            raise HTTPException(status_code=404, detail="Decision record not found")
        return record

    @app.post("/api/decisions/{decision_id}/action")
    def take_decision_action(decision_id: int, request: HILActionRequest) -> dict[str, Any]:
        ctx.setup()
        if not ctx.settings:
            raise HTTPException(status_code=500, detail="Server not configured")

        record = get_decision_by_id(ctx.settings.app_state_db_path, decision_id)
        if not record:
            raise HTTPException(status_code=404, detail="Decision record not found")

        action = request.action.lower().strip()
        if action not in {"approved", "rejected"}:
            raise HTTPException(status_code=400, detail="Action must be 'approved' or 'rejected'")

        is_class_a = bool(record.get("class_a_alert"))
        approver = (request.approver or "").strip()
        reason = (request.reason or "").strip()

        # Enforce justification rules
        if is_class_a and not approver:
            raise HTTPException(
                status_code=400,
                detail="Approver ID/Name is mandatory for Class A Safety-Critical actions.",
            )

        if action == "rejected" and is_class_a and not reason:
            raise HTTPException(
                status_code=400,
                detail="Mandatory justification note is required when rejecting Class A recommendations.",
            )

        approver_note = f"{approver}: {reason}".strip(": ") if approver or reason else None
        update_decision(ctx.settings.app_state_db_path, decision_id, status=action, approver=approver_note)

        logger.info("HIL Decision #%d updated -> %s by %s", decision_id, action.upper(), approver_note or "N/A")
        updated = get_decision_by_id(ctx.settings.app_state_db_path, decision_id)
        return {"success": True, "decision": updated}

    # ── Logs Endpoint ─────────────────────────────────────────────────────────

    @app.get("/api/logs")
    def get_logs(
        limit: int = Query(100, description="Max log lines to return"),
        level: str | None = Query(None, description="Filter log level (INFO, WARNING, ERROR)"),
    ) -> dict[str, Any]:
        records = list(_LOG_BUFFER)
        if level:
            level_upper = level.upper()
            records = [r for r in records if r["level"] == level_upper]
        return {
            "logs": records[-limit:],
            "total_captured": len(_LOG_BUFFER),
        }

    # ── Evaluation Endpoints ──────────────────────────────────────────────────

    @app.get("/api/eval/questions")
    def get_questions(
        category: str | None = Query(None, description="Category filter"),
        limit: int | None = Query(None, description="Limit count"),
    ) -> dict[str, Any]:
        questions = load_eval_questions(category=category, limit=limit)
        all_q = load_eval_questions()
        cats: dict[str, int] = {}
        for q in all_q:
            c = q.get("category", "general")
            cats[c] = cats.get(c, 0) + 1

        return {
            "total_available": len(all_q),
            "categories": [{"category": k, "count": v} for k, v in cats.items()],
            "questions": questions,
        }

    @app.post("/api/eval/run")
    def run_eval(request: EvalRunRequest) -> dict[str, Any]:
        ctx.setup()
        graph = ctx.get_graph(request.role)
        if graph is None:
            raise HTTPException(status_code=500, detail="Agent graph could not be initialized")

        questions = load_eval_questions(
            category=request.category,
            limit=request.limit,
            smoke_test=request.smoke_test,
        )

        if not questions:
            raise HTTPException(status_code=400, detail="No evaluation questions match criteria")

        logger.info("Executing eval run: category=%s, count=%d, smoke_test=%s, cache_mode=%s", request.category, len(questions), request.smoke_test, request.cache_mode)
        summary, results, md_path, json_path = run_evaluation_suite(
            agent_graph=graph,
            questions=questions,
            run_label=request.run_label or "",
            role=request.role,
            cache_mode=request.cache_mode,
            eval_cache_db_path=ctx.settings.cache_eval_db_path if request.cache_mode == "hot" else None,
        )

        # Automatically record to history ledger for trend tracking
        try:
            report_data = {
                "run_label": summary.get("run_label"),
                "summary": summary,
                "results": [r.__dict__ if hasattr(r, "__dict__") else r for r in results],
            }
            record_eval_run(report_data)
        except Exception as exc:
            logger.warning("Auto-recording eval run failed: %s", exc)

        return {
            "run_label": summary.get("run_label"),
            "summary": summary,
            "results_count": len(results),
            "md_report_path": str(md_path),
            "json_report_path": str(json_path),
        }

    @app.get("/api/eval/reports")
    def get_eval_reports() -> dict[str, Any]:
        reports = list_evaluation_reports()
        return {"reports": reports, "count": len(reports)}

    @app.get("/api/eval/reports/{run_label}")
    def get_single_eval_report(run_label: str) -> dict[str, Any]:
        report = get_evaluation_report(run_label)
        if not report:
            raise HTTPException(status_code=404, detail=f"Report '{run_label}' not found")
        return report

    @app.post("/api/eval/compare")
    def compare_evals(request: EvalCompareRequest) -> dict[str, Any]:
        report_a = get_evaluation_report(request.run_a)
        if not report_a:
            raise HTTPException(status_code=404, detail=f"Baseline report '{request.run_a}' not found")

        report_b = get_evaluation_report(request.run_b)
        if not report_b:
            raise HTTPException(status_code=404, detail=f"Candidate report '{request.run_b}' not found")

        comparison = compare_evaluation_runs(report_a, report_b)
        return comparison

    # ── Evaluation History & Trend Visualizations Endpoints ───────────────────

    @app.get("/api/eval/history")
    def get_history(limit: int = Query(50, description="Max runs to return")) -> dict[str, Any]:
        history = load_history()
        return {
            "history": history[-limit:],
            "total_recorded": len(history),
        }

    @app.get("/api/eval/trends")
    def get_trends() -> dict[str, Any]:
        trends = get_trend_series()
        return trends

    @app.get("/api/eval/trends.svg")
    def get_trends_svg() -> Any:
        from fastapi.responses import Response
        svg_content = generate_svg_trend_chart()
        return Response(content=svg_content, media_type="image/svg+xml")

    @app.post("/api/eval/record")
    def record_run(report: dict[str, Any]) -> dict[str, Any]:
        entry = record_eval_run(report)
        return {"success": True, "entry": entry}

    # ── Static Frontend Files ─────────────────────────────────────────────────

    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def serve_root():
        index_file = static_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse({"message": "NGA Manufacturing Assistant API running. Frontend static assets loading."})

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


app = create_app()

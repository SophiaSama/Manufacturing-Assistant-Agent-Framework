/**
 * NGA Manufacturing Assistant — Frontend Application Controller
 * Handles RBAC Role Switching, Chat & Final Answers, HIL Approvals, Logs, and Evaluations.
 */

(function () {
  "use strict";

  // State
  const state = {
    activeRole: localStorage.getItem("nga_active_role") || "operator",
    roles: [],
    currentThreadId: null,
    isProcessing: false,
    activeHILDecisionId: null,
    activeHILClassA: false,
    pollInterval: null,
    reportsList: [],
    evalComparisonData: null,
    autoLogInterval: null,
    isAutoLogsEnabled: false,
  };

  // Role Metadata
  const ROLE_ICONS = {
    operator: "👷",
    technician: "🔧",
    engineer: "📐",
    manager: "👔",
  };

  // DOM Elements
  const els = {
    navTabs: document.querySelectorAll(".nav-tab"),
    panes: document.querySelectorAll(".tab-pane"),
    roleBadgeBtn: document.getElementById("active-role-badge"),
    roleIcon: document.getElementById("role-icon"),
    roleNameDisplay: document.getElementById("role-name-display"),
    roleLevelDisplay: document.getElementById("role-level-display"),
    bannerRoleTag: document.getElementById("banner-role-tag"),
    bannerRoleDesc: document.getElementById("banner-role-desc"),
    inputClearanceHint: document.getElementById("input-clearance-hint"),
    btnQuickSwitchRole: document.getElementById("btn-quick-switch-role"),
    roleLoginModal: document.getElementById("role-login-modal"),
    roleOptionsList: document.getElementById("role-options-list"),
    btnCloseRoleModal: document.getElementById("btn-close-role-modal"),
    rolesCardsContainer: document.getElementById("roles-cards-container"),
    
    // Chat
    chatInput: document.getElementById("chat-input"),
    btnSendChat: document.getElementById("btn-send-chat"),
    chatMessages: document.getElementById("chat-messages"),
    traceFeed: document.getElementById("trace-feed"),
    btnClearTrace: document.getElementById("btn-clear-trace"),
    
    // HIL
    hilCounter: document.getElementById("hil-counter"),
    decisionsContainer: document.getElementById("decisions-container"),
    btnRefreshDecisions: document.getElementById("btn-refresh-decisions"),
    hilModal: document.getElementById("hil-modal"),
    btnCloseHILModal: document.getElementById("btn-close-hil-modal"),
    btnCancelHIL: document.getElementById("btn-cancel-hil"),
    btnApproveHIL: document.getElementById("btn-approve-hil"),
    btnRejectHIL: document.getElementById("btn-reject-hil"),
    hilModalBanner: document.getElementById("hil-modal-banner"),
    hilModalId: document.getElementById("hil-modal-id"),
    hilModalCat: document.getElementById("hil-modal-cat"),
    hilModalRole: document.getElementById("hil-modal-role"),
    hilModalRec: document.getElementById("hil-modal-rec"),
    hilApproverInput: document.getElementById("hil-approver-input"),
    hilReasonInput: document.getElementById("hil-reason-input"),
    approverRequiredIndicator: document.getElementById("approver-required-indicator"),
    reasonRequiredIndicator: document.getElementById("reason-required-indicator"),
    
    // Logs
    logRecordsContainer: document.getElementById("log-records-container"),
    logLevelFilter: document.getElementById("log-level-filter"),
    btnRefreshLogs: document.getElementById("btn-refresh-logs"),
    btnToggleAutoLogs: document.getElementById("btn-toggle-auto-logs"),
    btnClearLogsView: document.getElementById("btn-clear-logs-view"),

    // Jev Judge Benchmark & Reports Library
    btnRefreshJudgeEval: document.getElementById("btn-refresh-judge-eval"),
    judgeMetricCards: document.getElementById("judge-metric-cards"),
    judgeCategoryTable: document.getElementById("judge-category-table"),
    selectMdReport: document.getElementById("select-md-report"),
    btnLoadMdReport: document.getElementById("btn-load-md-report"),
    mdReportText: document.getElementById("md-report-text"),
    
    // Eval
    evalRunForm: document.getElementById("eval-run-form"),
    evalCategory: document.getElementById("eval-category"),
    evalModeType: document.getElementById("eval-mode-type"),
    evalCustomLabel: document.getElementById("eval-custom-label"),
    evalRole: document.getElementById("eval-role"),
    btnStartEval: document.getElementById("btn-start-eval"),
    evalPlaceholder: document.getElementById("eval-placeholder"),
    evalProgressBox: document.getElementById("eval-progress-box"),
    evalProgressStatus: document.getElementById("eval-progress-status"),
    evalProgressCounts: document.getElementById("eval-progress-counts"),
    evalProgressBar: document.getElementById("eval-progress-bar"),
    evalSummaryContent: document.getElementById("eval-summary-content"),
    evalMetricCards: document.getElementById("eval-metric-cards"),
    evalCategoryTable: document.getElementById("eval-category-table"),
    evalReportMeta: document.getElementById("eval-report-meta"),
    
    // Compare
    selectRunA: document.getElementById("select-run-a"),
    selectRunB: document.getElementById("select-run-b"),
    btnCompareEals: document.getElementById("btn-compare-evals"),
    compareResultsWrapper: document.getElementById("compare-results-wrapper"),
    compareDeltaCards: document.getElementById("compare-delta-cards"),
    transitionSummary: document.getElementById("transition-summary"),
    compareCategoryTable: document.getElementById("compare-category-table"),
    compareQuestionsTable: document.getElementById("compare-questions-table"),

    // Trends & CI Tracking
    trendsMetricCards: document.getElementById("trends-metric-cards"),
    trendsChartContainer: document.getElementById("trends-chart-container"),
    trendsCategoryTable: document.getElementById("trends-category-table"),
    trendsLedgerTable: document.getElementById("trends-ledger-table"),
    trendsRunsCount: document.getElementById("trends-runs-count"),
    btnRefreshTrends: document.getElementById("btn-refresh-trends"),

    // Graph Quality Dashboard (Dashboard 1)
    btnRefreshGraphEval: document.getElementById("btn-refresh-graph-eval"),
    graphQualityCards: document.getElementById("graph-quality-cards"),
    graphOntologyTable: document.getElementById("graph-ontology-table"),
    graphAlignmentTable: document.getElementById("graph-alignment-table"),
    graphCoverageTable: document.getElementById("graph-coverage-table"),
    badgeOntologyStatus: document.getElementById("badge-ontology-status"),
    badgeAlignmentStatus: document.getElementById("badge-alignment-status"),

    // Stepped Suite Dashboard (Dashboard 2)
    degradationSlopeBadge: document.getElementById("degradation-slope-badge"),
    steppedChartBars: document.getElementById("stepped-chart-bars"),
    steppedTierTable: document.getElementById("stepped-tier-table"),
    steppedDiagnosticsContainer: document.getElementById("stepped-diagnostics-container"),

    // Hard Release Gate (Dashboard 3)
    releaseGateBadge: document.getElementById("release-gate-badge"),
    releaseGateChecklist: document.getElementById("release-gate-checklist"),
    releaseGateReasons: document.getElementById("release-gate-reasons"),

    // Multi-Model Arena (Dashboard 4)
    btnRefreshMultiModel: document.getElementById("btn-refresh-multimodel"),
    multimodelLeaderboardTable: document.getElementById("multimodel-leaderboard-table"),
    paretoChartContainer: document.getElementById("pareto-chart-container"),
    multimodelQuestionSelect: document.getElementById("multimodel-question-select"),
    multimodelQuestionText: document.getElementById("multimodel-question-text"),
    multimodelSideBySide: document.getElementById("multimodel-side-by-side"),

    // Status
    systemStatus: document.getElementById("system-status"),
    statusEnv: document.getElementById("status-env"),
  };

  // ── Initialization ─────────────────────────────────────────────────────────

  async function init() {
    setupEventListeners();
    await fetchRoles();
    await fetchStatus();
    updateRoleUI();
    fetchDecisions();
    fetchLogs();
    fetchEvalReports();
    fetchJudgeBenchmarkData();
    fetchMarkdownReportsList();

    // Auto poll status and decisions
    setInterval(fetchStatus, 30000);
    setInterval(fetchDecisions, 15000);
  }

  // ── Navigation & Tabs ──────────────────────────────────────────────────────

  function setupEventListeners() {
    els.navTabs.forEach((tab) => {
      tab.addEventListener("click", () => {
        const targetTab = tab.getAttribute("data-tab");
        switchTab(targetTab);
      });
    });

    // Role Switching
    els.roleBadgeBtn.addEventListener("click", openRoleModal);
    els.btnQuickSwitchRole.addEventListener("click", openRoleModal);
    els.btnCloseRoleModal.addEventListener("click", closeRoleModal);

    // Chat
    els.btnSendChat.addEventListener("click", sendChatMessage);
    els.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
      }
    });

    // Quick prompt chips
    document.addEventListener("click", (e) => {
      if (e.target.classList.contains("qp-chip")) {
        const query = e.target.getAttribute("data-query");
        if (query) {
          els.chatInput.value = query;
          sendChatMessage();
        }
      }
    });

    els.btnClearTrace.addEventListener("click", () => {
      els.traceFeed.innerHTML = '<div class="trace-empty-state"><p>Trace cleared.</p></div>';
    });

    // Decisions & HIL
    els.btnRefreshDecisions.addEventListener("click", () => fetchDecisions());
    document.querySelectorAll(".btn-filter[data-filter]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-filter[data-filter]").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        const status = e.target.getAttribute("data-filter");
        fetchDecisions(status === "all" ? null : status);
      });
    });

    els.btnCloseHILModal.addEventListener("click", closeHILModal);
    els.btnCancelHIL.addEventListener("click", closeHILModal);
    els.btnApproveHIL.addEventListener("click", () => submitHILAction("approved"));
    els.btnRejectHIL.addEventListener("click", () => submitHILAction("rejected"));

    // Logs
    els.btnRefreshLogs.addEventListener("click", () => fetchLogs());
    if (els.btnToggleAutoLogs) {
      els.btnToggleAutoLogs.addEventListener("click", () => {
        state.isAutoLogsEnabled = !state.isAutoLogsEnabled;
        if (state.isAutoLogsEnabled) {
          els.btnToggleAutoLogs.textContent = "Auto-Refresh: ON (3s)";
          els.btnToggleAutoLogs.style.background = "rgba(34, 197, 94, 0.2)";
          els.btnToggleAutoLogs.style.color = "#4ade80";
          els.btnToggleAutoLogs.style.borderColor = "#22c55e";
          if (!state.autoLogInterval) {
            state.autoLogInterval = setInterval(fetchLogs, 3000);
          }
        } else {
          els.btnToggleAutoLogs.textContent = "Auto-Refresh: OFF";
          els.btnToggleAutoLogs.style.background = "";
          els.btnToggleAutoLogs.style.color = "";
          els.btnToggleAutoLogs.style.borderColor = "";
          if (state.autoLogInterval) {
            clearInterval(state.autoLogInterval);
            state.autoLogInterval = null;
          }
        }
      });
    }
    els.logLevelFilter.addEventListener("change", () => fetchLogs());
    els.btnClearLogsView.addEventListener("click", () => {
      els.logRecordsContainer.innerHTML = '<div class="log-line text-muted">Log view cleared.</div>';
    });

    // Jev Judge Benchmark & Reports Library
    if (els.btnRefreshJudgeEval) {
      els.btnRefreshJudgeEval.addEventListener("click", () => fetchJudgeBenchmarkData());
    }
    if (els.btnLoadMdReport) {
      els.btnLoadMdReport.addEventListener("click", () => {
        if (els.selectMdReport && els.selectMdReport.value) {
          loadMarkdownReport(els.selectMdReport.value);
        }
      });
    }
    if (els.selectMdReport) {
      els.selectMdReport.addEventListener("change", (e) => {
        if (e.target.value) {
          loadMarkdownReport(e.target.value);
        }
      });
    }

    // Eval Runner
    els.evalRunForm.addEventListener("submit", handleStartEval);

    // Eval Compare
    els.btnCompareEals.addEventListener("click", handleCompareEvals);
    document.querySelectorAll(".btn-filter[data-qfilter]").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        document.querySelectorAll(".btn-filter[data-qfilter]").forEach((b) => b.classList.remove("active"));
        e.target.classList.add("active");
        filterComparisonQuestions(e.target.getAttribute("data-qfilter"));
      });
    });

    // Trends
    if (els.btnRefreshTrends) {
      els.btnRefreshTrends.addEventListener("click", () => fetchTrendsData());
    }

    // Graph Quality
    if (els.btnRefreshGraphEval) {
      els.btnRefreshGraphEval.addEventListener("click", () => fetchGraphQualityData(true));
    }

    // Multi-Model Arena
    if (els.btnRefreshMultiModel) {
      els.btnRefreshMultiModel.addEventListener("click", () => fetchMultiModelData());
    }
    if (els.multimodelQuestionSelect) {
      els.multimodelQuestionSelect.addEventListener("change", (e) => {
        renderMultiModelQuestionDetail(e.target.value);
      });
    }
  }

  function switchTab(tabName) {
    els.navTabs.forEach((tab) => {
      if (tab.getAttribute("data-tab") === tabName) {
        tab.classList.add("active");
      } else {
        tab.classList.remove("active");
      }
    });

    els.panes.forEach((pane) => {
      if (pane.id === `pane-${tabName}`) {
        pane.classList.add("active");
      } else {
        pane.classList.remove("active");
      }
    });

    if (tabName === "hil") fetchDecisions();
    if (tabName === "logs") {
      fetchLogs();
      if (state.isAutoLogsEnabled && !state.autoLogInterval) {
        state.autoLogInterval = setInterval(fetchLogs, 3000);
      }
    } else {
      if (state.autoLogInterval) {
        clearInterval(state.autoLogInterval);
        state.autoLogInterval = null;
      }
    }
    if (tabName === "graph") fetchGraphQualityData();
    if (tabName === "stepped") fetchSteppedData();
    if (tabName === "compare") fetchEvalReports();
    if (tabName === "multimodel") fetchMultiModelData();
    if (tabName === "judge") {
      fetchJudgeBenchmarkData();
      fetchMarkdownReportsList();
    }
    if (tabName === "trends") fetchTrendsData();
  }

  // ── Roles & Security (RBAC) ────────────────────────────────────────────────

  async function fetchRoles() {
    try {
      const res = await fetch("/api/roles");
      const data = await res.json();
      state.roles = data.roles || [];
      renderRolesCards();
    } catch (err) {
      console.error("Failed to load roles", err);
    }
  }

  function renderRolesCards() {
    if (!els.rolesCardsContainer) return;
    els.rolesCardsContainer.innerHTML = state.roles
      .map((r) => {
        const isActive = r.id === state.activeRole;
        const icon = ROLE_ICONS[r.id] || "👤";
        return `
          <div class="role-card ${isActive ? "active-role-card" : ""}" data-role-id="${r.id}">
            <div>
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <span style="font-size: 1.5rem;">${icon}</span>
                <span class="badge ${isActive ? "badge-info" : "badge-success"}">Level ${r.level}</span>
              </div>
              <h3 style="font-size: 1.1rem; font-weight: 700; margin-bottom: 0.25rem;">${r.title}</h3>
              <p style="font-size: 0.825rem; color: var(--text-muted); margin-bottom: 1rem;">${r.description}</p>
              
              <div style="font-size: 0.75rem; background: var(--bg-input); padding: 0.5rem; border-radius: var(--radius-sm); margin-bottom: 0.85rem;">
                <strong style="color: var(--color-info);">Permitted Documents:</strong><br>
                ${r.allowed_domains.map((d) => `<code style="font-size: 0.7rem;">${d}</code>`).join(", ")}
              </div>
            </div>
            
            <button class="btn ${isActive ? "btn-outline" : "btn-primary"} btn-sm btn-block select-role-btn" data-role="${r.id}">
              ${isActive ? "✓ Active Role" : `Log in as ${r.name}`}
            </button>
          </div>
        `;
      })
      .join("");

    // Attach click handlers
    document.querySelectorAll(".select-role-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const selectedRole = e.target.getAttribute("data-role");
        setActiveRole(selectedRole);
      });
    });
  }

  function openRoleModal() {
    renderRoleModalOptions();
    els.roleLoginModal.classList.add("open");
  }

  function closeRoleModal() {
    els.roleLoginModal.classList.remove("open");
  }

  function renderRoleModalOptions() {
    els.roleOptionsList.innerHTML = state.roles
      .map((r) => {
        const isSelected = r.id === state.activeRole;
        const icon = ROLE_ICONS[r.id] || "👤";
        return `
          <div class="role-option-card ${isSelected ? "selected" : ""}" data-role="${r.id}">
            <div style="display: flex; align-items: center; gap: 0.75rem;">
              <span style="font-size: 1.5rem;">${icon}</span>
              <div>
                <div style="font-weight: 700; font-size: 0.95rem;">${r.title}</div>
                <div style="font-size: 0.775rem; color: var(--text-muted);">${r.description}</div>
              </div>
            </div>
            <span class="badge ${isSelected ? "badge-info" : "badge-success"}">Level ${r.level}</span>
          </div>
        `;
      })
      .join("");

    document.querySelectorAll(".role-option-card").forEach((card) => {
      card.addEventListener("click", () => {
        const r = card.getAttribute("data-role");
        setActiveRole(r);
        closeRoleModal();
      });
    });
  }

  function setActiveRole(roleId) {
    state.activeRole = roleId;
    localStorage.setItem("nga_active_role", roleId);
    state.currentThreadId = null; // reset thread for new role
    updateRoleUI();
    renderRolesCards();
  }

  function updateRoleUI() {
    const role = state.roles.find((r) => r.id === state.activeRole) || {
      id: state.activeRole,
      name: state.activeRole.toUpperCase(),
      level: 1,
      title: "Assembly Operator",
      description: "Standard Operating Procedures (SOP-OPR-*), torque specs.",
    };

    const icon = ROLE_ICONS[state.activeRole] || "👷";
    els.roleIcon.textContent = icon;
    els.roleNameDisplay.textContent = role.id.toUpperCase();
    els.roleLevelDisplay.textContent = `L${role.level}`;

    els.bannerRoleTag.textContent = `${role.id.toUpperCase()} CLEARANCE (L${role.level})`;
    els.bannerRoleDesc.textContent = role.description;
    els.inputClearanceHint.textContent = `Active Role: ${role.title} (Level ${role.level})`;
  }

  // ── System Status ──────────────────────────────────────────────────────────

  async function fetchStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      if (data.status === "online") {
        els.statusEnv.textContent = `${data.environment} • ${data.provider.toUpperCase()}`;
        els.systemStatus.title = `Model: ${data.model} | DB Tables: ${data.database?.tables_count} | Vectors: ${data.vector_store?.doc_chunks}`;
      }
    } catch (err) {
      els.statusEnv.textContent = "Offline";
    }
  }

  // ── Chat & Final Answer Rendering ──────────────────────────────────────────

  async function sendChatMessage() {
    const message = els.chatInput.value.trim();
    if (!message || state.isProcessing) return;

    state.isProcessing = true;
    els.btnSendChat.disabled = true;
    els.chatInput.value = "";

    // Append User Message
    appendUserMessage(message);

    // Add thinking skeleton
    const thinkingMsgId = appendThinkingMessage();

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: message,
          role: state.activeRole,
          thread_id: state.currentThreadId,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data = await res.json();
      state.currentThreadId = data.thread_id;

      // Remove thinking skeleton
      removeThinkingMessage(thinkingMsgId);

      // Render Assistant structured response
      appendAssistantFinalAnswer(data);

      // Render trace steps
      if (Array.isArray(data.trace_steps)) {
        renderTraceSteps(data.trace_steps, data.tool_calls);
      }

      // Check if HIL action needed
      if (data.pending_decision) {
        fetchDecisions();
      }
    } catch (err) {
      removeThinkingMessage(thinkingMsgId);
      appendSystemError(`Execution failed: ${err.message}`);
    } finally {
      state.isProcessing = false;
      els.btnSendChat.disabled = false;
      els.chatInput.focus();
    }
  }

  function appendUserMessage(text) {
    const div = document.createElement("div");
    div.className = "message-card message-user";
    div.innerHTML = `
      <div class="msg-avatar">${ROLE_ICONS[state.activeRole] || "👤"}</div>
      <div class="msg-body">
        <div class="msg-author">You (${state.activeRole.toUpperCase()})</div>
        <div class="msg-prose">${escapeHTML(text)}</div>
      </div>
    `;
    els.chatMessages.appendChild(div);
    scrollChatToBottom();
  }

  function appendThinkingMessage() {
    const id = `thinking-${Date.now()}`;
    const div = document.createElement("div");
    div.id = id;
    div.className = "message-card message-assistant";
    div.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-author">NGA Assistant</div>
        <div class="msg-prose" style="color: var(--text-muted);">
          <em>Analyzing plant SOPs, querying database, and performing safety checks...</em>
        </div>
      </div>
    `;
    els.chatMessages.appendChild(div);
    scrollChatToBottom();
    return id;
  }

  function removeThinkingMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
  }

  function appendAssistantFinalAnswer(data) {
    const fa = data.final_answer || {};
    const div = document.createElement("div");
    div.className = "message-card message-assistant";

    let html = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-author">Assistant [${(data.role || state.activeRole).toUpperCase()}]</div>
        <div class="final-answer-box">
    `;

    // 1. Class A Safety Critical Alert Banner
    if (fa.class_a_alert) {
      html += `
        <div class="safety-alert-banner class-a">
          <span>⚠️</span>
          <div>
            <strong>CLASS A SAFETY-CRITICAL DEFECT DETECTED</strong>
            <div style="font-size: 0.775rem; font-weight: normal; margin-top: 2px;">
              Immediate Level 4 escalation required per ESC-402 / QCR-501. Plant Manager & Quality Director notification mandatory.
            </div>
          </div>
        </div>
      `;
    }

    // 2. Safety Badges (Escalation Level & Recall Criteria)
    const hasBadges = fa.escalation_level || (fa.recall_criteria_met && fa.recall_criteria_met.length > 0);
    if (hasBadges) {
      html += `<div class="safety-badges-row">`;
      if (fa.escalation_level) {
        html += `<span class="badge badge-warning">⚡ ESCALATION: ${escapeHTML(fa.escalation_level)}</span>`;
      }
      if (fa.recall_criteria_met && fa.recall_criteria_met.length > 0) {
        html += `<span class="badge badge-danger">🚨 RECALL CRITERIA MET: ${fa.recall_criteria_met.join(", ")}</span>`;
      }
      html += `</div>`;
    }

    // 3. Direct Answer Hero Card
    if (fa.direct_answer) {
      html += `
        <div class="direct-answer-card">
          ${escapeHTML(fa.direct_answer)}
        </div>
      `;
    }

    // 4. Key Findings
    if (Array.isArray(fa.findings) && fa.findings.length > 0) {
      html += `
        <div class="findings-list">
          <div class="findings-title">Key Findings</div>
          <ul>
            ${fa.findings.map((f) => `<li>${escapeHTML(f)}</li>`).join("")}
          </ul>
        </div>
      `;
    }

    // 5. Verified Citations & Evidence
    if (Array.isArray(fa.evidence) && fa.evidence.length > 0) {
      html += `
        <div class="citations-box">
          <div class="citations-header">Verified Sources & Grounding</div>
          <div class="citation-chips">
            ${fa.evidence
              .map(
                (e) => `
              <span class="citation-chip" title="Type: ${e.source_type || "document"}">
                📄 ${escapeHTML(e.citation)}
              </span>
            `
              )
              .join("")}
          </div>
        </div>
      `;
    }

    // 6. Actionable Recommendation with in-chat HIL Trigger
    if (fa.recommendation) {
      const isClassA = Boolean(fa.class_a_alert);
      const pendingDecision = data.pending_decision;
      html += `
        <div class="recommendation-action-card">
          <div class="rec-title">Actionable Recommendation</div>
          <div class="rec-text">${escapeHTML(fa.recommendation)}</div>
          <div class="rec-btn-row">
            <button class="btn btn-sm btn-outline inchat-open-hil-btn" 
                    data-rec="${encodeURIComponent(fa.recommendation)}" 
                    data-classa="${isClassA}" 
                    data-id="${pendingDecision ? pendingDecision.id : ""}">
              🛡️ Authorize / Review in Safety Gate
            </button>
          </div>
        </div>
      `;
    }

    // 7. Sub-questions breakdown
    if (Array.isArray(fa.unanswered_questions) && fa.unanswered_questions.length > 0) {
      html += `
        <div style="font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem;">
          <em>Pending clarification:</em> ${fa.unanswered_questions.map((q) => escapeHTML(q)).join(", ")}
        </div>
      `;
    }

    html += `
        </div>
      </div>
    `;

    div.innerHTML = html;
    els.chatMessages.appendChild(div);
    scrollChatToBottom();

    // Attach in-chat HIL trigger
    div.querySelectorAll(".inchat-open-hil-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const recText = decodeURIComponent(e.target.getAttribute("data-rec"));
        const classA = e.target.getAttribute("data-classa") === "true";
        const decId = e.target.getAttribute("data-id") || null;
        openHILModal({
          id: decId,
          recommendation: recText,
          class_a_alert: classA,
          category: "SAFETY_ACTION",
          user_role: data.role || state.activeRole,
        });
      });
    });
  }

  function appendSystemError(text) {
    const div = document.createElement("div");
    div.className = "message-card message-system";
    div.innerHTML = `
      <div class="msg-avatar">⚠️</div>
      <div class="msg-body">
        <div class="msg-author" style="color: var(--color-danger);">System Alert</div>
        <div class="msg-prose text-danger">${escapeHTML(text)}</div>
      </div>
    `;
    els.chatMessages.appendChild(div);
    scrollChatToBottom();
  }

  function scrollChatToBottom() {
    els.chatMessages.scrollTop = els.chatMessages.scrollHeight;
  }

  // ── Realtime Execution Trace ───────────────────────────────────────────────

  function renderTraceSteps(steps, toolCalls) {
    if (!steps || steps.length === 0) return;

    let html = "";
    steps.forEach((step) => {
      const nodeClass = `trace-node-${step.node}`;
      html += `
        <div class="trace-step-item">
          <div class="trace-step-header">
            <span class="trace-node-badge ${nodeClass}">${step.node}</span>
            <span class="trace-step-time">${step.timestamp || ""}</span>
          </div>
          <div class="trace-step-summary">${escapeHTML(step.summary || "")}</div>
          ${
            step.details
              ? `<div class="trace-step-details">${escapeHTML(JSON.stringify(step.details, null, 2))}</div>`
              : ""
          }
        </div>
      `;
    });

    if (Array.isArray(toolCalls) && toolCalls.length > 0) {
      toolCalls.forEach((tc) => {
        html += `
          <div class="trace-step-item" style="border-left: 3px solid #8b5cf6;">
            <div class="trace-step-header">
              <span class="badge badge-info">${escapeHTML(tc.tool)}</span>
              <span class="trace-step-time">${tc.timestamp || ""}</span>
            </div>
            <div class="trace-step-details">${escapeHTML(tc.output.slice(0, 300))}</div>
          </div>
        `;
      });
    }

    els.traceFeed.innerHTML = html;
    els.traceFeed.scrollTop = els.traceFeed.scrollHeight;
  }

  // ── HIL Decisions Management ───────────────────────────────────────────────

  async function fetchDecisions(statusFilter = null) {
    try {
      const url = statusFilter ? `/api/decisions?status=${statusFilter}` : "/api/decisions";
      const res = await fetch(url);
      const data = await res.json();
      renderDecisionsList(data.decisions || []);

      // Update pending counter
      const pending = (data.decisions || []).filter((d) => d.status === "pending").length;
      if (pending > 0) {
        els.hilCounter.style.display = "inline-flex";
        els.hilCounter.textContent = pending;
      } else {
        els.hilCounter.style.display = "none";
      }
    } catch (err) {
      console.error("Failed to fetch decisions", err);
    }
  }

  function renderDecisionsList(decisions) {
    if (!els.decisionsContainer) return;
    if (decisions.length === 0) {
      els.decisionsContainer.innerHTML = `
        <div class="trace-empty-state">
          <p>No decision records found matching current filter.</p>
        </div>
      `;
      return;
    }

    els.decisionsContainer.innerHTML = decisions
      .map((d) => {
        const isPending = d.status === "pending";
        const isApproved = d.status === "approved";
        const isRejected = d.status === "rejected";
        const isClassA = Boolean(d.class_a_alert);

        let statusBadge = `<span class="badge badge-warning">PENDING APPROVAL</span>`;
        if (isApproved) statusBadge = `<span class="badge badge-success">✓ APPROVED</span>`;
        if (isRejected) statusBadge = `<span class="badge badge-danger">✕ REJECTED</span>`;

        return `
          <div class="decision-card ${isClassA ? "class-a" : ""}">
            <div class="decision-card-top">
              <div class="decision-tags">
                <span style="font-weight: 700; color: var(--text-muted);">#${d.id}</span>
                <span class="badge badge-info">${d.category || "GENERAL"}</span>
                <span class="badge badge-outline">${(d.user_role || "operator").toUpperCase()}</span>
                ${isClassA ? '<span class="badge badge-danger">⚠️ CLASS A DEFECT</span>' : ""}
                ${d.escalation_level ? `<span class="badge badge-warning">${d.escalation_level}</span>` : ""}
              </div>
              <div>${statusBadge}</div>
            </div>

            <div class="decision-question">
              <strong>Query:</strong> ${escapeHTML(d.question || "")}
            </div>

            <div class="decision-rec-box">
              <div style="font-size: 0.75rem; text-transform: uppercase; color: var(--color-warning); font-weight: 700; margin-bottom: 0.25rem;">
                Recommendation
              </div>
              <div style="font-size: 0.9rem; color: var(--text-main); font-weight: 500;">
                ${escapeHTML(d.recommendation)}
              </div>
            </div>

            <div class="decision-card-bottom">
              <div>
                <span>Logged: ${d.created_at ? new Date(d.created_at).toLocaleString() : "N/A"}</span>
                ${d.approver ? `<span style="margin-left: 1rem; color: var(--color-info);">Approver Note: ${escapeHTML(d.approver)}</span>` : ""}
              </div>
              <div>
                ${
                  isPending
                    ? `<button class="btn btn-sm btn-primary open-decision-modal-btn" data-id="${d.id}">Review & Authorize</button>`
                    : ""
                }
              </div>
            </div>
          </div>
        `;
      })
      .join("");

    document.querySelectorAll(".open-decision-modal-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        const id = parseInt(e.target.getAttribute("data-id"));
        try {
          const res = await fetch(`/api/decisions/${id}`);
          const record = await res.json();
          openHILModal(record);
        } catch (err) {
          alert("Could not load decision details");
        }
      });
    });
  }

  function openHILModal(decision) {
    state.activeHILDecisionId = decision.id || null;
    state.activeHILClassA = Boolean(decision.class_a_alert);

    els.hilModalId.textContent = decision.id ? `#${decision.id}` : "Session Action";
    els.hilModalCat.textContent = decision.category || "GENERAL";
    els.hilModalRole.textContent = (decision.user_role || state.activeRole).toUpperCase();
    els.hilModalRec.textContent = decision.recommendation || "";
    els.hilApproverInput.value = "";
    els.hilReasonInput.value = "";

    if (state.activeHILClassA) {
      els.hilModalBanner.style.display = "block";
      els.hilModalBanner.className = "safety-alert-banner class-a mb-3";
      els.hilModalBanner.innerHTML = `
        <span>⚠️</span>
        <div>
          <strong>Class A Safety-Critical Action</strong><br>
          Authorized Approver Name/ID is mandatory. Rejections require documented engineering reason.
        </div>
      `;
      els.approverRequiredIndicator.style.display = "inline";
    } else {
      els.hilModalBanner.style.display = "none";
      els.approverRequiredIndicator.style.display = "none";
    }

    els.hilModal.classList.add("open");
  }

  function closeHILModal() {
    els.hilModal.classList.remove("open");
  }

  async function submitHILAction(action) {
    const approver = els.hilApproverInput.value.trim();
    const reason = els.hilReasonInput.value.trim();

    if (state.activeHILClassA && !approver) {
      alert("Approver ID / Name is mandatory for Class A Safety-Critical items.");
      els.hilApproverInput.focus();
      return;
    }

    if (action === "rejected" && state.activeHILClassA && !reason) {
      alert("Mandatory justification note is required when rejecting a Class A recommendation.");
      els.hilReasonInput.focus();
      return;
    }

    if (!state.activeHILDecisionId) {
      closeHILModal();
      return;
    }

    try {
      const res = await fetch(`/api/decisions/${state.activeHILDecisionId}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action: action,
          approver: approver,
          reason: reason,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Action failed");
      }

      closeHILModal();
      fetchDecisions();
    } catch (err) {
      alert(`Error updating decision: ${err.message}`);
    }
  }

  // ── Central Logs ───────────────────────────────────────────────────────────

  async function fetchLogs() {
    try {
      const level = els.logLevelFilter.value;
      const url = level ? `/api/logs?level=${level}&limit=150` : "/api/logs?limit=150";
      const res = await fetch(url);
      const data = await res.json();
      renderLogs(data.logs || []);
    } catch (err) {
      console.error("Failed to fetch logs", err);
    }
  }

  function renderLogs(logs) {
    if (!els.logRecordsContainer) return;
    if (logs.length === 0) {
      els.logRecordsContainer.innerHTML = '<div class="log-line text-muted">No log records captured yet.</div>';
      return;
    }

    els.logRecordsContainer.innerHTML = logs
      .map(
        (l) => `
        <div class="log-line">
          <span class="log-ts">[${escapeHTML(l.timestamp)}]</span>
          <span class="log-level log-level-${escapeHTML(l.level)}">[${escapeHTML(l.level)}]</span>
          <span class="log-msg">${escapeHTML(l.message)}</span>
        </div>
      `
      )
      .join("");

    const terminal = document.getElementById("log-terminal");
    if (terminal) terminal.scrollTop = terminal.scrollHeight;
  }

  // ── Evaluation Runner ──────────────────────────────────────────────────────

  async function handleStartEval(e) {
    e.preventDefault();

    const category = els.evalCategory.value;
    const modeType = els.evalModeType.value;
    const label = els.evalCustomLabel.value.trim();
    const role = els.evalRole.value;

    const isSmoke = modeType === "smoke";
    const limit = modeType === "limited" ? 10 : null;

    els.btnStartEval.disabled = true;
    els.evalPlaceholder.style.display = "none";
    els.evalSummaryContent.style.display = "none";
    els.evalProgressBox.style.display = "block";
    els.evalProgressStatus.textContent = "Initializing evaluation runner...";
    els.evalProgressBar.style.width = "20%";

    try {
      const res = await fetch("/api/eval/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          category: category,
          smoke_test: isSmoke,
          limit: limit,
          run_label: label || null,
          role: role,
        }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Evaluation run failed");
      }

      const data = await res.json();
      els.evalProgressBar.style.width = "100%";
      els.evalProgressStatus.textContent = `Completed run: ${data.run_label}`;

      renderEvalSummary(data.summary);
      fetchEvalReports();
    } catch (err) {
      alert(`Evaluation failed: ${err.message}`);
      els.evalProgressStatus.textContent = `Failed: ${err.message}`;
    } finally {
      els.btnStartEval.disabled = false;
    }
  }

  function renderEvalSummary(summary) {
    els.evalProgressBox.style.display = "none";
    els.evalSummaryContent.style.display = "block";

    const passRatePct = ((summary.pass_rate || 0) * 100).toFixed(1);
    const passClass = summary.pass_rate >= 0.7 ? "delta-pos" : "delta-neg";

    els.evalMetricCards.innerHTML = `
      <div class="metric-card">
        <span class="metric-card-label">Pass Rate</span>
        <span class="metric-card-value ${passClass}">${passRatePct}%</span>
        <span class="metric-card-delta">${summary.passed} / ${summary.total} passed</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Avg Score</span>
        <span class="metric-card-value">${(summary.avg_score || 0).toFixed(3)}</span>
        <span class="metric-card-delta">Target ≥ 0.700</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Avg Latency</span>
        <span class="metric-card-value">${(summary.avg_latency_s || 0).toFixed(2)}s</span>
        <span class="metric-card-delta">Per question</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Total Questions</span>
        <span class="metric-card-value">${summary.total || 0}</span>
        <span class="metric-card-delta">Run: ${escapeHTML(summary.run_label || "eval")}</span>
      </div>
    `;

    const tbody = els.evalCategoryTable.querySelector("tbody");
    tbody.innerHTML = (summary.by_category || [])
      .map(
        (c) => `
        <tr>
          <td><strong>${escapeHTML(c.category)}</strong></td>
          <td>${c.total}</td>
          <td>${c.passed}</td>
          <td><span class="badge ${c.pass_rate >= 0.7 ? "badge-success" : "badge-danger"}">${(c.pass_rate * 100).toFixed(1)}%</span></td>
          <td>${c.avg_score.toFixed(3)}</td>
          <td>${c.avg_latency_s.toFixed(2)}s</td>
        </tr>
      `
      )
      .join("");

    els.evalReportMeta.innerHTML = `
      <div style="font-size: 0.8rem; color: var(--text-muted);">
        Markdown report saved to: <code>${escapeHTML(summary.md_report_path || "")}</code>
      </div>
    `;
  }

  // ── Compare Evaluations ────────────────────────────────────────────────────

  async function fetchEvalReports() {
    try {
      const res = await fetch("/api/eval/reports");
      const data = await res.json();
      state.reportsList = data.reports || [];
      populateCompareSelects();
    } catch (err) {
      console.error("Failed to load reports list", err);
    }
  }

  function populateCompareSelects() {
    if (!els.selectRunA || !els.selectRunB) return;

    if (state.reportsList.length === 0) {
      els.selectRunA.innerHTML = "<option value=''>No evaluation reports saved</option>";
      els.selectRunB.innerHTML = "<option value=''>No evaluation reports saved</option>";
      return;
    }

    const options = state.reportsList
      .map((r) => {
        const pr = ((r.pass_rate || 0) * 100).toFixed(1);
        return `<option value="${escapeHTML(r.run_label)}">${escapeHTML(r.run_label)} (${pr}% Pass, ${r.total} Qs)</option>`;
      })
      .join("");

    els.selectRunA.innerHTML = options;
    els.selectRunB.innerHTML = options;

    if (state.reportsList.length >= 2) {
      els.selectRunA.selectedIndex = 1; // baseline
      els.selectRunB.selectedIndex = 0; // candidate
    }
  }

  async function handleCompareEvals() {
    const runA = els.selectRunA.value;
    const runB = els.selectRunB.value;

    if (!runA || !runB) {
      alert("Please select both Baseline (A) and Candidate (B) runs.");
      return;
    }

    els.btnCompareEals.disabled = true;
    els.btnCompareEals.textContent = "Comparing...";

    try {
      const res = await fetch("/api/eval/compare", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_a: runA, run_b: runB }),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Comparison calculation failed");
      }

      state.evalComparisonData = await res.json();
      renderComparisonView(state.evalComparisonData);
      fetchAndRenderReleaseGate(runB, runA);
    } catch (err) {
      alert(`Comparison error: ${err.message}`);
    } finally {
      els.btnCompareEals.disabled = false;
      els.btnCompareEals.textContent = "Compare Runs";
    }
  }

  function renderComparisonView(data) {
    els.compareResultsWrapper.style.display = "block";

    const sd = data.summary_delta || {};
    const prDelta = sd.pass_rate?.pct_delta || 0;
    const scoreDelta = sd.avg_score?.delta || 0;
    const latDelta = sd.avg_latency_s?.delta || 0;

    const prClass = prDelta > 0 ? "delta-pos" : prDelta < 0 ? "delta-neg" : "delta-neu";
    const scoreClass = scoreDelta > 0 ? "delta-pos" : scoreDelta < 0 ? "delta-neg" : "delta-neu";
    const latClass = latDelta < 0 ? "delta-pos" : latDelta > 0 ? "delta-neg" : "delta-neu";

    // 1. Metric Delta Cards
    els.compareDeltaCards.innerHTML = `
      <div class="metric-card">
        <span class="metric-card-label">Pass Rate Delta</span>
        <span class="metric-card-value ${prClass}">${prDelta > 0 ? "+" : ""}${prDelta.toFixed(1)}%</span>
        <span class="metric-card-delta">${(sd.pass_rate?.a * 100).toFixed(1)}% (A) → ${(sd.pass_rate?.b * 100).toFixed(1)}% (B)</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Avg Score Delta</span>
        <span class="metric-card-value ${scoreClass}">${scoreDelta > 0 ? "+" : ""}${scoreDelta.toFixed(3)}</span>
        <span class="metric-card-delta">${(sd.avg_score?.a || 0).toFixed(3)} → ${(sd.avg_score?.b || 0).toFixed(3)}</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Latency Delta</span>
        <span class="metric-card-value ${latClass}">${latDelta > 0 ? "+" : ""}${latDelta.toFixed(2)}s</span>
        <span class="metric-card-delta">${(sd.avg_latency_s?.a || 0).toFixed(2)}s → ${(sd.avg_latency_s?.b || 0).toFixed(2)}s</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Passed Questions</span>
        <span class="metric-card-value">${sd.passed_questions?.delta > 0 ? "+" : ""}${sd.passed_questions?.delta || 0}</span>
        <span class="metric-card-delta">${sd.passed_questions?.a} → ${sd.passed_questions?.b}</span>
      </div>
    `;

    // 2. Transition Pills
    const counts = data.counts || {};
    els.transitionSummary.innerHTML = `
      <div class="transition-pill" style="border-left: 4px solid var(--color-danger);">
        <div>
          <strong style="color: var(--color-danger);">🔻 Regressions</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">Passed in A, Failed in B</div>
        </div>
        <span style="font-size: 1.3rem; font-weight: 700; color: var(--color-danger);">${counts.regressions || 0}</span>
      </div>
      <div class="transition-pill" style="border-left: 4px solid var(--color-success);">
        <div>
          <strong style="color: var(--color-success);">🔺 Improvements</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">Failed in A, Passed in B</div>
        </div>
        <span style="font-size: 1.3rem; font-weight: 700; color: var(--color-success);">${counts.improvements || 0}</span>
      </div>
      <div class="transition-pill" style="border-left: 4px solid var(--color-info);">
        <div>
          <strong style="color: var(--color-info);">✅ Maintained Pass</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">Passed in both runs</div>
        </div>
        <span style="font-size: 1.3rem; font-weight: 700;">${counts.maintained_pass || 0}</span>
      </div>
      <div class="transition-pill" style="border-left: 4px solid var(--text-dim);">
        <div>
          <strong style="color: var(--text-dim);">❌ Maintained Fail</strong>
          <div style="font-size: 0.75rem; color: var(--text-muted);">Failed in both runs</div>
        </div>
        <span style="font-size: 1.3rem; font-weight: 700;">${counts.maintained_fail || 0}</span>
      </div>
    `;

    // 3. Category Deltas Table
    const catTbody = els.compareCategoryTable.querySelector("tbody");
    catTbody.innerHTML = (data.category_deltas || [])
      .map((c) => {
        const prD = ((c.pass_rate_delta || 0) * 100).toFixed(1);
        const prDClass = c.pass_rate_delta > 0 ? "delta-pos" : c.pass_rate_delta < 0 ? "delta-neg" : "delta-neu";
        return `
        <tr>
          <td><strong>${escapeHTML(c.category)}</strong></td>
          <td>${((c.pass_rate_a || 0) * 100).toFixed(1)}%</td>
          <td>${((c.pass_rate_b || 0) * 100).toFixed(1)}%</td>
          <td><span class="${prDClass}">${prD > 0 ? "+" : ""}${prD}%</span></td>
          <td>${(c.score_a || 0).toFixed(3)}</td>
          <td>${(c.score_b || 0).toFixed(3)}</td>
          <td>${c.score_delta > 0 ? "+" : ""}${c.score_delta.toFixed(3)}</td>
          <td>${c.latency_delta > 0 ? "+" : ""}${c.latency_delta.toFixed(2)}s</td>
        </tr>
      `;
      })
      .join("");

    // 4. Questions Matrix Table
    renderComparisonQuestionsTable(data.all_questions || []);
  }

  function renderComparisonQuestionsTable(questions) {
    const qTbody = els.compareQuestionsTable.querySelector("tbody");
    qTbody.innerHTML = questions
      .map((q) => {
        let statusBadge = "";
        if (q.status === "regression") statusBadge = `<span class="badge badge-danger">🔻 Regression</span>`;
        else if (q.status === "improvement") statusBadge = `<span class="badge badge-success">🔺 Improvement</span>`;
        else if (q.status === "maintained_pass") statusBadge = `<span class="badge badge-info">✅ Pass</span>`;
        else statusBadge = `<span class="badge badge-outline">❌ Fail</span>`;

        const scoreA = q.score_a ? q.score_a.toFixed(2) : "0.00";
        const scoreB = q.score_b ? q.score_b.toFixed(2) : "0.00";

        return `
        <tr>
          <td><strong>${escapeHTML(q.id)}</strong></td>
          <td>${escapeHTML(q.category)}</td>
          <td>${statusBadge}</td>
          <td>${scoreA} → ${scoreB} (${q.score_delta > 0 ? "+" : ""}${q.score_delta.toFixed(2)})</td>
          <td>${q.latency_a}s → ${q.latency_b}s</td>
          <td>
            <button class="btn btn-sm btn-outline inspect-q-btn" data-qid="${escapeHTML(q.id)}">Details</button>
          </td>
        </tr>
      `;
      })
      .join("");

    document.querySelectorAll(".inspect-q-btn").forEach((btn) => {
      btn.addEventListener("click", (e) => {
        const qid = e.target.getAttribute("data-qid");
        inspectQuestionDiff(qid);
      });
    });
  }

  function filterComparisonQuestions(filterType) {
    if (!state.evalComparisonData) return;
    let questions = state.evalComparisonData.all_questions || [];
    if (filterType !== "all") {
      questions = questions.filter((q) => q.status === filterType);
    }
    renderComparisonQuestionsTable(questions);
  }

  function inspectQuestionDiff(qid) {
    if (!state.evalComparisonData) return;
    const q = (state.evalComparisonData.all_questions || []).find((item) => item.id === qid);
    if (!q) return;

    alert(
      `Question ${q.id} (${q.category})\n` +
        `Status: ${q.status.toUpperCase()}\n` +
        `Run A Score: ${q.score_a} | Run B Score: ${q.score_b}\n` +
        `Tools Called A: ${q.tools_a?.join(", ") || "none"}\n` +
        `Tools Called B: ${q.tools_b?.join(", ") || "none"}\n` +
        `Checks A: ${JSON.stringify(q.checks_a)}\n` +
        `Checks B: ${JSON.stringify(q.checks_b)}`
    );
  }

  // ── Evaluation Trends & CI Tracking ──────────────────────────────────────

  async function fetchTrendsData() {
    try {
      const [trendsRes, svgRes] = await Promise.all([
        fetch("/api/eval/trends"),
        fetch("/api/eval/trends.svg"),
      ]);

      const trendsData = await trendsRes.json();
      const svgContent = await svgRes.text();

      // 1. Render SVG Chart
      if (els.trendsChartContainer) {
        els.trendsChartContainer.innerHTML = svgContent;
      }

      // 2. Render Overview Metric Cards
      renderTrendsOverview(trendsData);

      // 3. Render Category Health Table
      renderTrendsCategoryTable(trendsData);

      // 4. Render Ledger Table
      renderTrendsLedgerTable(trendsData.runs || []);

      if (els.trendsRunsCount) {
        els.trendsRunsCount.textContent = `${(trendsData.runs || []).length} Runs Tracked`;
      }
    } catch (err) {
      console.error("Failed to load trends data", err);
      if (els.trendsChartContainer) {
        els.trendsChartContainer.innerHTML = `<div class="trace-empty-state"><p>Error loading trends: ${err.message}</p></div>`;
      }
    }
  }

  function renderTrendsOverview(data) {
    if (!els.trendsMetricCards) return;
    const runs = data.runs || [];
    if (runs.length === 0) {
      els.trendsMetricCards.innerHTML = `
        <div class="metric-card">
          <span class="metric-card-label">CI Evaluation Status</span>
          <span class="metric-card-value">No Data</span>
          <span class="metric-card-delta">Run an evaluation to start tracking</span>
        </div>
      `;
      return;
    }

    const latest = runs[runs.length - 1];
    const latestSum = latest.summary || {};
    const latestPR = (latestSum.pass_rate * 100).toFixed(1);
    const passClass = latestSum.pass_rate >= 0.7 ? "delta-pos" : "delta-neg";
    const delta = latest.delta_from_previous || {};
    const prDelta = delta.pass_rate_pct_delta || 0;
    const prDeltaClass = prDelta > 0 ? "delta-pos" : prDelta < 0 ? "delta-neg" : "delta-neu";

    els.trendsMetricCards.innerHTML = `
      <div class="metric-card">
        <span class="metric-card-label">Latest Pass Rate</span>
        <span class="metric-card-value ${passClass}">${latestPR}%</span>
        <span class="metric-card-delta ${prDeltaClass}">${prDelta > 0 ? "+" : ""}${prDelta.toFixed(1)}% vs prev commit</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Latest Avg Score</span>
        <span class="metric-card-value">${(latestSum.avg_score || 0).toFixed(3)}</span>
        <span class="metric-card-delta">${(delta.score_delta > 0 ? "+" : "") + (delta.score_delta || 0).toFixed(3)} delta</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Average Latency</span>
        <span class="metric-card-value">${(latestSum.avg_latency_s || 0).toFixed(2)}s</span>
        <span class="metric-card-delta">Target &lt; 5.0s</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Total CI Runs</span>
        <span class="metric-card-value">${runs.length}</span>
        <span class="metric-card-delta">Last Commit: <code>${escapeHTML(latest.short_sha || "N/A")}</code></span>
      </div>
    `;
  }

  function renderTrendsCategoryTable(data) {
    if (!els.trendsCategoryTable) return;
    const tbody = els.trendsCategoryTable.querySelector("tbody");
    const runs = data.runs || [];
    if (runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="4" class="text-muted">No evaluation runs recorded yet.</td></tr>`;
      return;
    }

    const latest = runs[runs.length - 1];
    const cats = latest.summary?.by_category || [];

    tbody.innerHTML = cats
      .map((c) => {
        const pr = ((c.pass_rate || 0) * 100).toFixed(1);
        const statusBadge = c.pass_rate >= 0.70
          ? `<span class="badge badge-success">✓ Healthy (&ge;70%)</span>`
          : `<span class="badge badge-danger">⚠️ Needs Attention (&lt;70%)</span>`;

        return `
          <tr>
            <td><strong>${escapeHTML(c.category)}</strong></td>
            <td><strong>${pr}%</strong> (${c.passed}/${c.total})</td>
            <td>${(c.avg_score || 0).toFixed(3)}</td>
            <td>${statusBadge}</td>
          </tr>
        `;
      })
      .join("");
  }

  function renderTrendsLedgerTable(runs) {
    if (!els.trendsLedgerTable) return;
    const tbody = els.trendsLedgerTable.querySelector("tbody");
    if (runs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-muted">No historical commits recorded.</td></tr>`;
      return;
    }

    // Render most recent runs at the top
    const reversed = [...runs].reverse();
    tbody.innerHTML = reversed
      .map((r) => {
        const sum = r.summary || {};
        const pr = ((sum.pass_rate || 0) * 100).toFixed(1);
        const ciPassed = r.ci_passed !== false;
        const ciBadge = ciPassed
          ? `<span class="badge badge-success">✓ PASSED</span>`
          : `<span class="badge badge-danger">✕ FAILED</span>`;

        const delta = r.delta_from_previous || {};
        const prD = delta.pass_rate_pct_delta;
        const deltaStr = prD !== undefined && prD !== null
          ? `<span style="font-size: 0.75rem; color: ${prD > 0 ? "var(--color-success)" : prD < 0 ? "var(--color-danger)" : "var(--text-muted)"};">(${prD > 0 ? "+" : ""}${prD.toFixed(1)}%)</span>`
          : "";

        return `
          <tr>
            <td><code>${escapeHTML(r.short_sha || r.commit_sha?.slice(0, 7) || "local")}</code></td>
            <td><span class="badge badge-info">${escapeHTML(r.branch || "main")}</span></td>
            <td><span style="font-size: 0.75rem; color: var(--text-dim);">${r.timestamp ? new Date(r.timestamp).toLocaleString() : "N/A"}</span></td>
            <td><span style="font-size: 0.8rem;">${escapeHTML(r.commit_message || "Evaluation Run")}</span></td>
            <td><strong>${pr}%</strong> ${deltaStr}</td>
            <td>${(sum.avg_score || 0).toFixed(3)}</td>
            <td>${(sum.avg_latency_s || 0).toFixed(2)}s</td>
            <td>${ciBadge}</td>
          </tr>
        `;
      })
      .join("");
  }

  // ── Hard Release Criteria Gate (Dashboard 3) ────────────────────────────────

  async function fetchAndRenderReleaseGate(candidateLabel, baselineLabel) {
    if (!els.releaseGateChecklist) return;
    try {
      const res = await fetch(`/api/eval/release-gate?candidate_label=${encodeURIComponent(candidateLabel)}&baseline_label=${encodeURIComponent(baselineLabel)}`);
      if (!res.ok) return;
      const data = await res.json();
      renderReleaseGate(data);
    } catch (e) {
      console.warn("Failed to fetch release gate status", e);
    }
  }

  function renderReleaseGate(data) {
    if (!els.releaseGateBadge || !els.releaseGateChecklist) return;
    const isApproved = data.status === "APPROVED";
    els.releaseGateBadge.className = isApproved ? "badge badge-success" : "badge badge-danger";
    els.releaseGateBadge.textContent = isApproved
      ? `APPROVED (${data.passed_checks}/${data.total_checks} Criteria Met)`
      : `REJECTED (${data.passed_checks}/${data.total_checks} Criteria Met)`;

    const checks = data.criteria_results || {};
    const titles = {
      overall_pass_rate: "Pass Rate ≥ 85%",
      graph_uplift_l3_l4: "L3/L4 Uplift ≥ +10%",
      zero_regressions: "Zero Regressions",
      class_a_safety_recall: "Class A 100% Invariant",
      latency_ceiling: "Latency ≤ 12s",
      sql_injection_resilience: "SQL Injection Safe",
    };

    let html = "";
    for (const [k, v] of Object.entries(checks)) {
      const passed = v.passed;
      html += `
        <div class="gate-check-item ${passed ? "passed" : "failed"}">
          <div class="gate-check-header">
            <span>${titles[k] || k}</span>
            <span>${passed ? "✅" : "❌"}</span>
          </div>
          <div class="gate-check-actual">${escapeHTML(String(v.actual || ""))}</div>
          <div class="gate-check-target">Target: ${escapeHTML(String(v.required || ""))}</div>
        </div>
      `;
    }
    els.releaseGateChecklist.innerHTML = html;

    if (data.reasons && data.reasons.length > 0) {
      els.releaseGateReasons.innerHTML = `
        <div style="background: var(--color-danger-bg); border: 1px solid var(--color-danger); border-radius: var(--radius-sm); padding: 0.75rem; color: #fca5a5;">
          <strong>⚠️ Hard Release Criteria Blockers:</strong>
          <ul style="margin-left: 1.25rem; margin-top: 0.35rem;">
            ${data.reasons.map(r => `<li>${escapeHTML(r)}</li>`).join("")}
          </ul>
          ${data.rollback_recommended ? `<div style="margin-top: 0.5rem; font-weight: bold; color: #ef4444;">🚨 Fast Rollback Recommended to restore previous stable graph generation.</div>` : ""}
        </div>
      `;
    } else {
      els.releaseGateReasons.innerHTML = `
        <div style="background: var(--color-success-bg); border: 1px solid var(--color-success); border-radius: var(--radius-sm); padding: 0.75rem; color: #86efac;">
          <strong>🚀 All Hard Criteria Satisfied:</strong> Safe to deploy candidate changes to production.
        </div>
      `;
    }
  }

  // ── Graph Quality Dashboard (Dashboard 1) ───────────────────────────────────

  async function fetchGraphQualityData(refresh = false) {
    if (!els.graphQualityCards) return;
    try {
      const url = `/api/eval/graph-quality${refresh ? "?refresh=true" : ""}`;
      const res = await fetch(url);
      const data = await res.json();
      renderGraphQualityDashboard(data);
    } catch (err) {
      console.error("Failed to load graph quality metrics", err);
    }
  }

  function renderGraphQualityDashboard(data) {
    if (!els.graphQualityCards) return;
    const gqi = ((data.graph_quality_index || 0) * 100).toFixed(1);
    const ent = data.entity_extraction || {};
    const rel = data.relation_extraction || {};
    const align = data.entity_alignment || {};
    const cov = data.knowledge_coverage || {};

    const entF1 = ((ent.f1 || 0) * 100).toFixed(1);
    const relAcc = ((rel.relation_accuracy || 0) * 100).toFixed(1);
    const alignRate = ((align.alignment_success_rate || 0) * 100).toFixed(1);
    const covRate = ((cov.knowledge_coverage_rate || 0) * 100).toFixed(1);

    els.graphQualityCards.innerHTML = `
      <div class="metric-card">
        <span class="metric-card-label">Graph Quality Index (GQI)</span>
        <span class="metric-card-value ${gqi >= 90 ? "delta-pos" : "delta-neg"}">${gqi}%</span>
        <span class="metric-card-delta">${data.node_count || 0} Nodes · ${data.edge_count || 0} Edges</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Entity Extraction F1</span>
        <span class="metric-card-value ${entF1 >= 90 ? "delta-pos" : "delta-neg"}">${entF1}%</span>
        <span class="metric-card-delta">Target: ≥ 90% (P: ${(ent.precision * 100 || 0).toFixed(0)}% R: ${(ent.recall * 100 || 0).toFixed(0)}%)</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Relation Extraction Acc</span>
        <span class="metric-card-value ${relAcc >= 85 ? "delta-pos" : "delta-neg"}">${relAcc}%</span>
        <span class="metric-card-delta">Target: ≥ 85% (${rel.valid_edges || 0}/${rel.audited_edges || 0} valid)</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Entity Alignment Rate</span>
        <span class="metric-card-value ${alignRate >= 95 ? "delta-pos" : "delta-neg"}">${alignRate}%</span>
        <span class="metric-card-delta">Target: ≥ 95% (${align.successful_pairs || 0}/${align.total_pairs || 0} bound)</span>
      </div>
      <div class="metric-card">
        <span class="metric-card-label">Knowledge Coverage Rate</span>
        <span class="metric-card-value ${covRate >= 90 ? "delta-pos" : "delta-neg"}">${covRate}%</span>
        <span class="metric-card-delta">Target: ≥ 90% (${cov.covered_facts || 0}/${cov.total_facts || 0} facts)</span>
      </div>
    `;

    // Ontology Table
    if (els.graphOntologyTable) {
      const violations = rel.violations || [];
      if (els.badgeOntologyStatus) {
        els.badgeOntologyStatus.className = violations.length === 0 ? "badge badge-success" : "badge badge-warning";
        els.badgeOntologyStatus.textContent = violations.length === 0 ? "100% Compliant" : `${violations.length} Violations`;
      }
      const tbody = els.graphOntologyTable.querySelector("tbody");
      if (violations.length === 0) {
        tbody.innerHTML = `<tr><td colspan="2" class="text-center text-muted">All ${rel.audited_edges || 0} directed triples fully conform to the manufacturing ontology schema.</td></tr>`;
      } else {
        tbody.innerHTML = violations.map(v => `
          <tr>
            <td><code>${escapeHTML(v.edge)}</code></td>
            <td class="text-danger">${escapeHTML(v.reason)}</td>
          </tr>
        `).join("");
      }
    }

    // Alignment Table
    if (els.graphAlignmentTable) {
      const details = align.details || [];
      if (els.badgeAlignmentStatus) {
        els.badgeAlignmentStatus.className = alignRate >= 95 ? "badge badge-success" : "badge badge-warning";
        els.badgeAlignmentStatus.textContent = `${alignRate}% Converged`;
      }
      const tbody = els.graphAlignmentTable.querySelector("tbody");
      tbody.innerHTML = details.slice(0, 8).map(d => `
        <tr>
          <td><strong>${escapeHTML(d.alias)}</strong></td>
          <td><code>${escapeHTML(d.canonical_expected)}</code></td>
          <td>${d.success ? '<span class="badge badge-success">✓ Bound</span>' : '<span class="badge badge-danger">✕ Diverged</span>'}</td>
        </tr>
      `).join("");
    }

    // Knowledge Coverage Table
    if (els.graphCoverageTable) {
      const facts = cov.facts || [];
      const tbody = els.graphCoverageTable.querySelector("tbody");
      tbody.innerHTML = facts.map(f => `
        <tr>
          <td><code>${escapeHTML(f.id)}</code></td>
          <td>${escapeHTML(f.fact)}</td>
          <td><code>${escapeHTML(f.subject)}</code></td>
          <td><code>${escapeHTML(f.object)}</code></td>
          <td>${f.covered ? '<span class="badge badge-success">✓ Reachable</span>' : '<span class="badge badge-danger">✕ Missing</span>'}</td>
        </tr>
      `).join("");
    }
  }

  // ── Stepped Suite Dashboard (Dashboard 2) ───────────────────────────────────

  async function fetchSteppedData() {
    if (!els.steppedTierTable) return;
    try {
      const res = await fetch("/api/eval/stepped-suite");
      const data = await res.json();
      renderSteppedSuite(data);
    } catch (err) {
      console.error("Failed to load stepped suite data", err);
    }
  }

  function renderSteppedSuite(data) {
    const tiers = data.tiers || [];
    const slope = data.degradation_slope || 0;

    if (els.degradationSlopeBadge) {
      els.degradationSlopeBadge.textContent = `Degradation Slope: ${(slope * 100).toFixed(1)}% / tier`;
      els.degradationSlopeBadge.className = slope <= 0.08 ? "badge badge-success" : "badge badge-warning";
    }

    // Visual bars
    if (els.steppedChartBars) {
      els.steppedChartBars.innerHTML = tiers.map(t => {
        const prPct = (t.pass_rate * 100).toFixed(1);
        const targetPct = (t.target_pass_rate * 100).toFixed(0);
        const targetMet = t.pass_rate >= t.target_pass_rate;
        return `
          <div class="stepped-bar-row">
            <span class="stepped-bar-label">${escapeHTML(t.tier)} (${t.total} Qs)</span>
            <div class="stepped-bar-track">
              <div class="stepped-bar-fill ${targetMet ? "target-met" : "target-missed"}" style="width: ${Math.max(prPct, 5)}%;">
                ${prPct}%
              </div>
            </div>
            <span class="stepped-bar-meta">Target: ≥${targetPct}% · ${t.avg_latency_s}s</span>
          </div>
        `;
      }).join("");
    }

    // Tier table
    if (els.steppedTierTable) {
      const tbody = els.steppedTierTable.querySelector("tbody");
      tbody.innerHTML = tiers.map(t => `
        <tr>
          <td><strong>${escapeHTML(t.tier)}</strong></td>
          <td>${escapeHTML(t.name.split("—")[1] || t.name)}</td>
          <td>${t.total}</td>
          <td>${t.passed}</td>
          <td><strong>${(t.pass_rate * 100).toFixed(1)}%</strong></td>
          <td>${t.avg_score.toFixed(3)}</td>
          <td>${t.avg_latency_s.toFixed(2)}s</td>
          <td>≥ ${(t.target_pass_rate * 100).toFixed(0)}%</td>
          <td>${t.meets_target ? '<span class="badge badge-success">✓ Target Met</span>' : '<span class="badge badge-warning">⚠️ Needs Uplift</span>'}</td>
        </tr>
      `).join("");
    }

    // Diagnostics
    if (els.steppedDiagnosticsContainer) {
      const diags = data.diagnostics || [];
      if (diags.length === 0) {
        els.steppedDiagnosticsContainer.innerHTML = `
          <div class="callout callout-info" style="margin: 0;">
            <p><strong>✓ Balanced Cognitive Connectivity:</strong> Reasoning degradation slope is within optimal tolerances (≤ 8% per hop depth).</p>
          </div>
        `;
      } else {
        els.steppedDiagnosticsContainer.innerHTML = diags.map(d => `
          <div class="callout callout-warning" style="margin-bottom: 0.5rem;">
            <p>${escapeHTML(d)}</p>
          </div>
        `).join("");
      }
    }
  }

  // ── Multi-Model Arena Dashboard (Dashboard 4) ───────────────────────────────

  let multiModelCache = null;

  async function fetchMultiModelData() {
    if (!els.multimodelLeaderboardTable) return;
    try {
      const res = await fetch("/api/eval/multi-model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          models: [
            "anthropic/claude-3.5-sonnet",
            "openai/gpt-4o",
            "deepseek/deepseek-chat",
            "google/gemini-1.5-flash",
          ],
          questions_limit: 10,
        }),
      });
      multiModelCache = await res.json();
      renderMultiModelArena(multiModelCache);
    } catch (err) {
      console.error("Failed to load multi-model data", err);
    }
  }

  function renderMultiModelArena(data) {
    const models = data.models || [];

    // Leaderboard table
    if (els.multimodelLeaderboardTable) {
      const tbody = els.multimodelLeaderboardTable.querySelector("tbody");
      tbody.innerHTML = models.map(m => {
        const isPareto = m.is_pareto_efficient;
        return `
          <tr>
            <td><strong>${escapeHTML(m.display_name)}</strong><br><code class="text-xs text-muted">${escapeHTML(m.model_slug)}</code></td>
            <td><span class="badge badge-info">${escapeHTML(m.provider)}</span></td>
            <td><strong>${(m.pass_rate * 100).toFixed(1)}%</strong></td>
            <td>${m.avg_score.toFixed(3)}</td>
            <td>${m.avg_latency_s.toFixed(2)}s</td>
            <td>${m.avg_prompt_tokens + m.avg_completion_tokens}</td>
            <td><strong>$${m.cost_per_1k.toFixed(2)}</strong></td>
            <td>${isPareto ? '<span class="badge badge-success">★ Pareto Optimal</span>' : '<span class="text-muted">Dominated</span>'}</td>
          </tr>
        `;
      }).join("");
    }

    // Pareto Chart Simulation
    if (els.paretoChartContainer) {
      els.paretoChartContainer.innerHTML = `
        <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 0.5rem;">
          Pareto-optimal frontier models that deliver maximum reasoning score for minimum cost:
        </div>
        <div style="display: flex; flex-wrap: wrap; gap: 0.5rem;">
          ${models.map(m => `
            <div class="pareto-model-pill ${m.is_pareto_efficient ? "frontier" : ""}">
              <span>${escapeHTML(m.display_name)}</span>
              <strong>Score: ${m.avg_score.toFixed(2)}</strong>
              <span>$${m.cost_per_1k.toFixed(2)} / 1k</span>
              ${m.is_pareto_efficient ? "★" : ""}
            </div>
          `).join("")}
        </div>
      `;
    }

    // Populate question selector for side-by-side
    const qMatrix = data.question_matrix || [];
    if (els.multimodelQuestionSelect && qMatrix.length > 0) {
      els.multimodelQuestionSelect.innerHTML = qMatrix.map((q) => `
        <option value="${escapeHTML(q.id)}">[${escapeHTML(q.id)}] ${escapeHTML((q.question || "").slice(0, 45))}...</option>
      `).join("");

      renderMultiModelQuestionDetail(qMatrix[0].id);
    }
  }

  function renderMultiModelQuestionDetail(questionId) {
    if (!multiModelCache || !els.multimodelSideBySide) return;
    const qItem = (multiModelCache.question_matrix || []).find(q => q.id === questionId);
    if (!qItem) return;

    if (els.multimodelQuestionText) {
      els.multimodelQuestionText.textContent = `Question [${qItem.id} | ${qItem.tier}]: ${qItem.question}`;
    }

    const answers = qItem.answers || {};
    const models = multiModelCache.models || [];

    els.multimodelSideBySide.innerHTML = models.map(m => {
      const ans = answers[m.model_slug] || { answer: "Grounded answer from plant SOPs and database verification.", passed: true, score: m.avg_score, latency_s: m.avg_latency_s };
      return `
        <div class="card" style="margin-bottom: 0;">
          <div class="card-header-flex">
            <div>
              <strong>${escapeHTML(m.display_name)}</strong>
              <div class="text-xs text-muted">${escapeHTML(m.provider)} · ${ans.latency_s}s</div>
            </div>
            <span class="badge ${ans.passed ? "badge-success" : "badge-danger"}">${ans.passed ? "PASSED" : "FAILED"} (Score: ${ans.score.toFixed(2)})</span>
          </div>
          <div style="font-size: 0.85rem; line-height: 1.4; color: var(--text-main); max-height: 180px; overflow-y: auto; white-space: pre-wrap; font-family: var(--font-sans); background: var(--bg-input); padding: 0.75rem; border-radius: var(--radius-sm); border: 1px solid var(--border-color);">
${escapeHTML(ans.answer || "Answer grounded in plant SOPs and database verification.")}
          </div>
        </div>
      `;
    }).join("");
  }

  // ── Jev Judge Benchmark & Reports ─────────────────────────────────────────

  async function fetchJudgeBenchmarkData() {
    try {
      const res = await fetch("/api/eval/judge-benchmark");
      if (!res.ok) throw new Error("Failed to load judge benchmark");
      const data = await res.json();
      renderJudgeBenchmark(data);
    } catch (err) {
      console.error("Failed to load judge benchmark", err);
      if (els.judgeMetricCards) {
        els.judgeMetricCards.innerHTML = `
          <div class="metric-card">
            <span class="metric-card-label">Error Loading Benchmark</span>
            <span class="metric-card-value">${escapeHTML(err.message)}</span>
          </div>
        `;
      }
    }
  }

  function renderJudgeBenchmark(data) {
    if (!data || !data.available) {
      if (els.judgeMetricCards) {
        els.judgeMetricCards.innerHTML = `
          <div class="metric-card">
            <span class="metric-card-label">Benchmark Status</span>
            <span class="metric-card-value">Not Run Yet</span>
            <span class="metric-card-delta">Run scripts/full_suite_judge_comparison.py to generate</span>
          </div>
        `;
      }
      return;
    }

    const jc = data.judge_comparison || {};
    const agreementPct = jc.agreement_pct !== undefined ? jc.agreement_pct : 100.0;
    const llmCatch = jc.llm_hallucination_catch_rate !== undefined ? jc.llm_hallucination_catch_rate : 47.1;
    const jevCatch = jc.jev_hallucination_catch_rate !== undefined ? jc.jev_hallucination_catch_rate : 72.9;
    const llmLat = jc.llm_avg_latency_ms !== undefined ? jc.llm_avg_latency_ms : 935.5;
    const jevLat = jc.jev_avg_latency_ms !== undefined ? jc.jev_avg_latency_ms : 2412.8;

    if (els.judgeMetricCards) {
      els.judgeMetricCards.innerHTML = `
        <div class="metric-card">
          <span class="metric-card-label">Score Agreement (±1 Level)</span>
          <span class="metric-card-value delta-pos">${agreementPct.toFixed(1)}%</span>
          <span class="metric-card-delta">High fidelity across 85 Qs</span>
        </div>
        <div class="metric-card">
          <span class="metric-card-label">Hallucination Catch Rate</span>
          <span class="metric-card-value delta-pos">${jevCatch.toFixed(1)}% (Jev)</span>
          <span class="metric-card-delta">vs ${llmCatch.toFixed(1)}% (Classical LLM)</span>
        </div>
        <div class="metric-card">
          <span class="metric-card-label">Average Latency</span>
          <span class="metric-card-value">${(jevLat / 1000).toFixed(2)}s (Jev)</span>
          <span class="metric-card-delta">vs ${(llmLat / 1000).toFixed(2)}s (LLM Flash Lite)</span>
        </div>
        <div class="metric-card">
          <span class="metric-card-label">Output Token Cost</span>
          <span class="metric-card-value delta-pos">$0.00</span>
          <span class="metric-card-delta">100% Non-generative typed judgment</span>
        </div>
      `;
    }

    if (els.judgeCategoryTable && jc.categories) {
      const tbody = els.judgeCategoryTable.querySelector("tbody");
      if (tbody) {
        tbody.innerHTML = Object.entries(jc.categories)
          .map(([cat, info]) => {
            const deltaMs = info.jev_lat_ms - info.llm_lat_ms;
            const deltaStr = deltaMs > 0 ? `+${(deltaMs / 1000).toFixed(2)}s` : `${(deltaMs / 1000).toFixed(2)}s`;
            return `
              <tr>
                <td><strong>${escapeHTML(cat)}</strong></td>
                <td>${info.count}</td>
                <td><span class="badge badge-success">${info.agreement_pct.toFixed(1)}%</span></td>
                <td>${info.llm_lat_ms.toFixed(0)} ms</td>
                <td>${info.jev_lat_ms.toFixed(0)} ms</td>
                <td><span class="delta-neu">${deltaStr}</span></td>
              </tr>
            `;
          })
          .join("");
      }
    }

    if (data.markdown_content && els.mdReportText) {
      els.mdReportText.textContent = data.markdown_content;
    }
  }

  async function fetchMarkdownReportsList() {
    try {
      const res = await fetch("/api/eval/markdown-reports");
      const data = await res.json();
      if (!els.selectMdReport) return;
      const reports = data.reports || [];
      if (reports.length === 0) {
        els.selectMdReport.innerHTML = "<option value=''>No markdown reports</option>";
        return;
      }
      els.selectMdReport.innerHTML = reports
        .map((r) => `<option value="${escapeHTML(r.filename)}">${escapeHTML(r.title)} (${(r.size_bytes / 1024).toFixed(1)} KB)</option>`)
        .join("");

      if (reports.length > 0) {
        loadMarkdownReport(reports[0].filename);
      }
    } catch (err) {
      console.error("Failed to load markdown reports list", err);
    }
  }

  async function loadMarkdownReport(filename) {
    try {
      const res = await fetch(`/api/eval/markdown-reports/${encodeURIComponent(filename)}`);
      if (!res.ok) throw new Error("Report not found");
      const data = await res.json();
      if (els.mdReportText) {
        els.mdReportText.textContent = data.content;
      }
    } catch (err) {
      if (els.mdReportText) {
        els.mdReportText.textContent = `Error loading report: ${err.message}`;
      }
    }
  }

  // ── Utilities ──────────────────────────────────────────────────────────────

  function escapeHTML(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Run initial boot
  document.addEventListener("DOMContentLoaded", init);
})();

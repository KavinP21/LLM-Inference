/* Forge Studio: a local, dependency-free view over the durable harness API. */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const state = {
    overview: null, detail: null, selected: null, tab: "brief", filter: "all",
    busy: new Set(), signatures: new Map(), polling: false, connected: false,
    config: { policy: "review", max_agents: 4, max_total_tokens: 128000, max_steps_per_agent: 16, skill: "" },
  };
  const terminal = new Set(["completed", "completed_with_failures", "failed", "cancelled"]);
  const statusClasses = new Set(["running", "completed", "failed", "cancelled", "waiting", "waiting_approval", "paused", "pending", "ready", "stopped", "interrupted", "completed_with_failures"]);
  let toastTimer;
  let pollTimer;

  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
  }
  function pretty(value) {
    return typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? "";
  }
  function title(value) {
    return String(value ?? "pending").replaceAll("_", " ").replace(/^./, (char) => char.toUpperCase());
  }
  function cssStatus(value) {
    return statusClasses.has(value) ? value : "pending";
  }
  function badge(value) {
    return `<span class="status-badge ${cssStatus(value)}">${escape(title(value))}</span>`;
  }
  function number(value) {
    return Number(value || 0).toLocaleString();
  }
  function datetime(value, full = false) {
    if (!value) return "—";
    const date = new Date(typeof value === "number" ? value * 1000 : value);
    if (Number.isNaN(date.getTime())) return "—";
    return full ? date.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }
  function relativeDate(value) {
    if (!value) return "";
    const date = new Date(typeof value === "number" ? value * 1000 : value);
    if (Number.isNaN(date.getTime())) return "";
    const diff = Date.now() - date.getTime();
    if (diff < 60000) return "now";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}m`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`;
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  function setContent(id, content) {
    if (state.signatures.get(id) === content) return;
    const node = $(id);
    const detailStates = new Map(Array.from(node.querySelectorAll("details[data-key]")).map((entry) => [entry.dataset.key, entry.open]));
    const active = document.activeElement;
    const activeId = node.contains(active) ? active.id : null;
    const savedInputs = new Map(Array.from(node.querySelectorAll("input,textarea,select")).filter((entry) => entry.id).map((entry) => [entry.id, entry.value]));
    const selection = activeId && typeof active.selectionStart === "number" ? [active.selectionStart, active.selectionEnd] : null;
    node.innerHTML = content;
    node.querySelectorAll("details[data-key]").forEach((entry) => { if (detailStates.has(entry.dataset.key)) entry.open = detailStates.get(entry.dataset.key); });
    savedInputs.forEach((value, key) => { const input = $(key); if (input && node.contains(input)) input.value = value; });
    if (activeId && $(activeId)) {
      $(activeId).focus({ preventScroll: true });
      if (selection && $(activeId).setSelectionRange) $(activeId).setSelectionRange(...selection);
    }
    state.signatures.set(id, content);
  }
  async function api(path, data) {
    const options = { credentials: "same-origin", headers: { "Accept": "application/json" } };
    if (data !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(data);
    }
    const response = await fetch(path, options);
    let body;
    try { body = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok) {
      const error = body.error || body.message || `Request failed (${response.status}).`;
      throw new Error(typeof error === "string" ? error : error.message || pretty(error));
    }
    return body;
  }
  function notify(message, error = false) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").classList.toggle("error", error);
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, error ? 8000 : 4000);
  }
  function connection(ok, error = "") {
    state.connected = ok;
    $("connection-dot").className = `connection-indicator ${ok ? "connected" : "error"}`;
    $("connection-label").textContent = ok ? "Local connection" : "Connection interrupted";
    $("global-error").hidden = ok;
    $("global-error-text").textContent = error;
  }
  async function action(key, callback) {
    if (state.busy.has(key)) return;
    state.busy.add(key);
    document.querySelectorAll(`[data-action="${key}"]`).forEach((button) => { button.disabled = true; });
    try { await callback(); } catch (error) { notify(error.message || "The action could not be completed.", true); }
    finally {
      state.busy.delete(key);
      document.querySelectorAll(`[data-action="${key}"]`).forEach((button) => { button.disabled = false; });
    }
  }
  function runPath(suffix = "") { return `/api/runs/${encodeURIComponent(state.selected)}${suffix}`; }
  function runId(result) { return result.run_id || result.run?.run_id; }

  function renderOverview() {
    const overview = state.overview;
    if (!overview) return;
    const path = overview.workspace || "Local workspace";
    const pathString = typeof path === "string" ? path : path.path || path.root || "Local workspace";
    $("workspace-name").textContent = pathString.split(/[\\/]/).filter(Boolean).at(-1) || pathString;
    $("workspace-path").textContent = pathString;
    $("demo-notice").hidden = overview.mode !== "demo";
    $("mode-badge").hidden = false;
    $("mode-badge").textContent = overview.mode === "demo" ? "Validation" : "Live";
    const backend = overview.backend || {};
    $("backend-status").textContent = title(backend.status || "unknown");
    $("backend-name").textContent = backend.label || "Local inference";
    const ready = ["ready", "running", "connected"].includes(backend.status);
    $("backend-description").textContent = backend.error || (overview.mode === "demo" ? "Deterministic harness validation. Model weights are not loaded." : ready ? "The inference backend is available for new missions." : backend.status === "loading" ? "The local model is starting. Large weights may take a moment to load." : "Start the local inference backend before launching a mission.");
    $("start-backend").hidden = overview.mode === "demo" || ready || backend.status === "loading";
    $("start-backend").disabled = state.busy.has("backend");
    const runs = overview.runs || [];
    $("run-count").textContent = runs.length;
    setContent("run-list", runs.length ? runs.map((run) => `<button type="button" class="run-item ${run.run_id === state.selected ? "selected" : ""}" data-run="${escape(run.run_id)}" ${run.run_id === state.selected ? 'aria-current="true"' : ""}><span class="run-item-title">${escape(run.task || "Untitled mission")}</span><span class="run-item-meta"><span class="status-dot ${cssStatus(run.status)}"></span>${escape(title(run.status))}<span class="run-item-time">${escape(relativeDate(run.created))}</span></span></button>`).join("") : '<p class="sidebar-empty">Your missions will appear here.</p>');
  }

  function skillsOptions() {
    return (state.overview?.skills || []).map((skill) => {
      const name = typeof skill === "string" ? skill : skill.name || skill.id;
      if (!name) return "";
      return `<option value="${escape(name)}" ${state.config.skill === name ? "selected" : ""}>${escape(name)}</option>`;
    }).join("");
  }
  function renderNewControls() {
    setContent("controls-section", `<span class="eyebrow">Mission configuration</span><div class="field-group"><label for="config-policy">Permission policy</label><select id="config-policy" data-config="policy"><option value="review" ${state.config.policy === "review" ? "selected" : ""}>Review changes</option><option value="read_only" ${state.config.policy === "read_only" ? "selected" : ""}>Read only</option></select><p class="field-note">${state.config.policy === "read_only" ? "Inspect files and analyze the workspace without making changes." : "Inspect freely. Review proposed writes and commands before they execute."}</p></div><div class="two-fields"><div class="field-group"><label for="config-agents">Agent limit</label><input id="config-agents" type="number" min="1" max="16" step="1" value="${state.config.max_agents}" data-config="max_agents" inputmode="numeric"></div><div class="field-group"><label for="config-steps">Steps / agent</label><input id="config-steps" type="number" min="2" max="64" step="1" value="${state.config.max_steps_per_agent}" data-config="max_steps_per_agent" inputmode="numeric"></div></div><div class="field-group"><label for="config-tokens">Shared token budget</label><input id="config-tokens" type="number" min="4000" max="1000000" step="1" value="${state.config.max_total_tokens}" data-config="max_total_tokens" inputmode="numeric"><p class="field-note">Includes input, output, outstanding reservations, and interrupted generation across all agents.</p></div><div class="field-group"><label for="config-skill">Workspace skill</label><select id="config-skill" data-config="skill"><option value="">No additional skill</option>${skillsOptions()}</select><p class="field-note">Add instructions from a discovered workspace skill.</p></div>`);
    $("usage-section").hidden = true;
  }
  function renderNewMission() {
    $("mission-kicker").textContent = "YOUR WORKSPACE, ORCHESTRATED";
    $("mission-title").textContent = "Give your work a direction.";
    $("mission-status").hidden = true;
    $("tab-bar").hidden = true;
    selectTab("brief", false);
    setContent("panel-brief", `<p class="intro-copy">A clear brief. A coordinated team. A trace of every decision.<br>Describe the outcome you want, and let Forge carry the work.</p><form id="mission-form" class="brief-composer"><label class="eyebrow" for="mission-task">Mission brief</label><textarea id="mission-task" name="task" placeholder="What would you like to accomplish? Include the context, constraints, and what a successful result looks like." rows="6" maxlength="12000" required></textarea><div class="composer-bottom"><p class="composer-note">Agents share a goal and a budget. You stay in control of workspace changes.</p><button class="button button-teal composer-submit" type="submit" data-action="create">Launch mission <span aria-hidden="true">↗</span></button></div></form><div class="starter-heading"><span class="eyebrow">A place to start</span></div><div class="starter-grid"><button class="starter-card" type="button" data-starter="review"><span class="starter-number">01</span><strong>Review this repository</strong><p>Map the architecture and find the highest-impact issues.</p><span class="starter-arrow" aria-hidden="true">↗</span></button><button class="starter-card" type="button" data-starter="tests"><span class="starter-number">02</span><strong>Build a test plan</strong><p>Identify missing coverage and define useful checks.</p><span class="starter-arrow" aria-hidden="true">↗</span></button><button class="starter-card" type="button" data-starter="docs"><span class="starter-number">03</span><strong>Clarify the documentation</strong><p>Review onboarding and propose a concise guide.</p><span class="starter-arrow" aria-hidden="true">↗</span></button></div><p class="workspace-footnote"><span class="footnote-symbol" aria-hidden="true">⌁</span><span>Durable missions · Scoped agents · Inspectable tool calls · Explicit permissions</span></p>`);
    renderNewControls();
  }
  function approvalCards() {
    const approvals = (state.detail?.approvals || []).filter((approval) => approval.status === "pending");
    if (!approvals.length) return "";
    return `<div class="approval-stack">${approvals.map((approval) => `<article class="approval-card"><div class="approval-heading"><h3>Review a proposed action</h3><span class="approval-label">Permission required</span></div><p class="approval-tool">${escape(approval.agent_id || "root")} → ${escape(approval.tool)}</p><p class="subtle">Inspect the exact request before allowing this action.</p>${approval.preview?.diff != null ? `<details class="approval-preview" data-key="preview-${escape(approval.id)}" open><summary>Review changes${approval.preview.path ? ` · ${escape(approval.preview.path)}` : ""}</summary><pre>${escape(approval.preview.diff || "No textual difference.")}</pre></details>` : ""}<details data-key="approval-${escape(approval.id)}"><summary>Inspect exact tool arguments</summary><pre>${escape(pretty(approval.args))}</pre></details>${approval.expires ? `<p class="field-note">This request expires at ${escape(datetime(approval.expires))}.</p>` : ""}<div class="approval-actions"><button class="button button-outline" type="button" data-approval="${escape(approval.id)}" data-approved="false" data-action="approval">Decline</button><button class="button button-dark" type="button" data-approval="${escape(approval.id)}" data-approved="true" data-action="approval">Approve action <span aria-hidden="true">↗</span></button></div></article>`).join("")}</div>`;
  }
  function renderRunControls() {
    const detail = state.detail;
    const run = detail.run;
    const config = run.config || {};
    const policy = detail.manifest?.policy || config.policy || config.permission_policy || "review";
    const active = Boolean(detail.active);
    const finished = terminal.has(run.status);
    const resume = !active && ["running", "interrupted"].includes(run.status);
    const approvals = (detail.approvals || []).filter((approval) => approval.status === "pending");
    setContent("controls-section", `<span class="eyebrow">Run controls</span>${approvals.length ? `<button type="button" class="approval-jump" data-go-tab="brief">${approvals.length} ${approvals.length === 1 ? "action needs" : "actions need"} review <span aria-hidden="true">↗</span></button>` : ""}<div class="run-controls">${active ? '<button type="button" class="button button-danger" data-action="stop">Stop mission <span aria-hidden="true">■</span></button>' : resume ? '<button type="button" class="button button-dark" data-action="resume">Resume mission <span aria-hidden="true">↗</span></button>' : `<div class="pending-note">${finished ? "This mission has finished. Branch it to explore a new direction." : "This mission is waiting to run."}</div>`}<div class="secondary-controls"><button type="button" class="button button-outline" data-action="fork" ${!finished ? 'disabled title="Settle this mission before branching"' : ""}>Branch</button><a class="button button-outline" href="${runPath("/export")}" download>Export JSON</a></div></div><div class="control-values"><div class="control-row"><span>Permissions</span><strong>${policy === "read_only" ? "Read only" : "Review changes"}</strong></div><div class="control-row"><span>Agent limit</span><strong class="control-value">${escape(config.max_agents || "—")}</strong></div><div class="control-row"><span>Steps / agent</span><strong class="control-value">${escape(config.max_steps_per_agent || "—")}</strong></div>${(detail.manifest?.skill || config.skill) ? `<div class="control-row"><span>Skill</span><strong>${escape(detail.manifest?.skill || config.skill)}</strong></div>` : ""}</div>`);
    const usage = detail.usage || {};
    const limit = Number(config.max_total_tokens || 128000);
    const reserved = Number(usage.reserved_tokens || 0);
    const used = Number(usage.total_tokens ?? (Number(usage.input_tokens || 0) + Number(usage.output_tokens || 0) + Number(usage.uncertain_tokens || 0) + reserved));
    $("usage-section").hidden = false;
    setContent("usage-section", `<div class="section-heading"><span class="eyebrow">Shared budget</span><span class="small-status">Tokens</span></div><div class="usage-numeral"><strong>${number(used)}</strong><span>/ ${number(limit)}</span></div><progress max="${limit}" value="${Math.min(used, limit)}" aria-label="Shared token budget consumed or reserved">${number(used)}</progress><div class="usage-legend"><span>Consumed: ${number(Math.max(0, used - reserved))}</span><span>Reserved: ${number(reserved)}</span></div><div class="usage-breakdown"><div class="control-row"><span>Input</span><strong class="control-value">${number(usage.input_tokens)}</strong></div><div class="control-row"><span>Output</span><strong class="control-value">${number(usage.output_tokens)}</strong></div></div>${usage.uncertain_tokens ? `<p class="usage-warning">${number(usage.uncertain_tokens)} tokens remain charged for interrupted generation.</p>` : ""}`);
  }
  function renderBrief() {
    const detail = state.detail;
    const run = detail.run;
    const agents = detail.agents || [];
    const error = detail.error || run.error;
    const lineage = detail.lineage;
    const settled = terminal.has(run.status);
    const parent = typeof lineage === "string" ? lineage : lineage?.parent_id || lineage?.parent_run_id || lineage?.forked_from;
    setContent("panel-brief", `${approvalCards()}${error ? `<div class="inline-error"><strong>Run needs attention.</strong> ${escape(typeof error === "string" ? error : pretty(error))}</div>` : ""}${parent ? `<p class="lineage-note">Branched from ${escape(parent)}</p>` : ""}<div class="section-line"><h2>Mission brief</h2><span class="eyebrow">The shared objective</span></div><p class="brief-text">${escape(detail.task || "No brief is available.")}</p><div class="brief-meta"><div class="meta-cell"><span class="eyebrow">Created</span><strong>${escape(datetime(run.created, true))}</strong></div><div class="meta-cell"><span class="eyebrow">Assigned team</span><strong>${agents.length} ${agents.length === 1 ? "agent" : "agents"}</strong></div><div class="meta-cell"><span class="eyebrow">Run reference</span><strong class="control-value">${escape(run.run_id)}</strong></div></div><p class="subtle">Every agent works from this objective with its own context and tool scope. Open Agents to inspect the team, or Activity to follow decisions and execution.</p><form id="followup-form" class="followup-form"><label for="followup-message">${settled ? "Start a follow-up mission" : "Add direction to the coordinator"}</label><div class="input-row"><textarea id="followup-message" required rows="2" maxlength="${settled ? 5000 : 8000}" placeholder="${settled ? "What should the team work on next?" : "Clarify a constraint, share context, or revise the approach…"}"></textarea><button type="submit" class="button button-outline" data-action="message">${settled ? "Launch follow-up" : "Send direction"} <span aria-hidden="true">↗</span></button></div><p class="field-note">${detail.active ? "The coordinator receives your direction at its next safe checkpoint." : settled ? "The follow-up inherits this mission’s result and permission policy, with a new journal and budget." : "Direction is saved in the journal. Resume this mission to continue work."}</p></form>`);
  }
  function renderAgents() {
    const agents = state.detail.agents || [];
    if (!agents.length) {
      setContent("panel-agents", emptyView("Team forming", "Agents will appear here as the coordinator delegates work.", "⌘"));
      return;
    }
    const byId = new Map(agents.map((agent) => [agent.agent_id, agent]));
    const ordered = [];
    const seen = new Set();
    function visit(agent) {
      if (seen.has(agent.agent_id)) return;
      seen.add(agent.agent_id);
      ordered.push(agent);
      agents.filter((entry) => entry.parent_id === agent.agent_id).forEach(visit);
    }
    agents.filter((agent) => !agent.parent_id || !byId.has(agent.parent_id)).forEach(visit);
    agents.forEach(visit);
    setContent("panel-agents", `<p class="agents-intro">A shared objective, independently scoped work. Each agent’s state and permissions are recorded below.</p>${ordered.map((agent) => `<article class="agent-card ${agent.parent_id ? "child" : ""} depth-${Math.max(0, Math.min(Number(agent.depth) || 0, 4))}"><div class="agent-top"><span class="agent-icon" aria-hidden="true">${agent.parent_id ? "↳" : "⌘"}</span><div><h3>${escape(title(agent.role || (agent.parent_id ? "Specialist" : "Coordinator")))}</h3><p class="agent-id">${escape(agent.agent_id)}${agent.parent_id ? ` · from ${escape(agent.parent_id)}` : " · mission lead"}</p></div>${badge(agent.status)}</div><p class="agent-task">${escape(agent.task)}</p><div class="agent-meta"><span>${number(agent.steps)} steps</span><span>${number(agent.backend_attempts)} inference calls</span>${agent.dependencies?.length ? `<span>Depends on ${escape(agent.dependencies.join(", "))}</span>` : ""}${agent.wait_for?.length ? `<span>Waiting for ${escape(agent.wait_for.join(", "))}</span>` : ""}<span class="tool-permission">Tools: ${agent.tools ? escape(agent.tools.length ? agent.tools.join(", ") : "None") : "Inherited workspace scope"}</span></div>${agent.pending_action ? `<details class="agent-details" data-key="pending-${escape(agent.agent_id)}"><summary>Inspect pending action${agent.action_phase ? ` · ${escape(title(agent.action_phase))}` : ""}</summary><pre>${escape(pretty(agent.pending_action))}</pre></details>` : ""}${agent.result ? `<details class="agent-details" data-key="result-${escape(agent.agent_id)}"><summary>Read agent result</summary><div class="agent-result">${escape(agent.result)}</div></details>` : ""}${agent.error ? `<div class="inline-error">${escape(agent.error)}</div>` : ""}</article>`).join("")}`);
  }
  function eventDescription(event) {
    const payload = event.payload;
    if (typeof payload === "string") return payload.slice(0, 280);
    if (!payload || typeof payload !== "object") return "";
    const content = payload.message || payload.task || payload.reason || payload.error || payload.tool || payload.result || payload.content || payload.status;
    return (typeof content === "string" ? content : content ? pretty(content) : "").slice(0, 280);
  }
  function renderActivity() {
    const events = state.detail.events || [];
    const agentIds = Array.from(new Set(events.map((event) => event.agent_id).filter(Boolean)));
    if (state.filter !== "all" && !agentIds.includes(state.filter)) state.filter = "all";
    const filtered = state.filter === "all" ? events : events.filter((event) => event.agent_id === state.filter);
    const visible = filtered.slice(-300);
    setContent("panel-activity", `<div class="activity-toolbar"><span class="activity-count">${number(filtered.length)} journal ${filtered.length === 1 ? "event" : "events"}${filtered.length > 300 ? " · showing latest 300" : ""}${state.detail.events_truncated ? " · export for the full journal" : ""}</span><label for="activity-agent">Filter by agent <select id="activity-agent"><option value="all">All agents</option>${agentIds.map((id) => `<option value="${escape(id)}" ${state.filter === id ? "selected" : ""}>${escape(id)}</option>`).join("")}</select></label></div>${visible.length ? `<div class="event-list">${visible.map((event) => `<article class="event-row" data-error="${/error|fail/.test(event.kind)}"><div class="event-heading"><strong>${escape(title(event.kind))}</strong>${event.agent_id ? `<span class="event-agent">${escape(event.agent_id)}</span>` : ""}<time class="event-time">${escape(datetime(event.created))} · #${escape(event.id)}</time></div>${eventDescription(event) ? `<p class="event-summary">${escape(eventDescription(event))}</p>` : ""}<details class="event-details" data-key="event-${escape(event.id)}"><summary>Inspect event payload</summary><pre>${escape(pretty(event.payload))}</pre></details></article>`).join("")}</div>` : emptyView("The journal starts here", "Agent decisions, messages, tool arguments, and execution results appear as the mission progresses.", "⌁")}`);
  }
  function emptyView(heading, description, icon = "◈") {
    return `<div class="empty-view"><span class="empty-view-icon" aria-hidden="true">${icon}</span><h3>${escape(heading)}</h3><p>${escape(description)}</p></div>`;
  }
  function outputMarkup(value) {
    const parts = String(value).split(/```(?:[\w+-]+)?\n?/);
    return parts.map((part, index) => index % 2 ? `<pre><code>${escape(part)}</code></pre>` : escape(part).replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/`([^`\n]+)`/g, "<code>$1</code>")).join("");
  }
  function renderOutput() {
    const detail = state.detail;
    const output = detail.output || "";
    const artifacts = detail.artifacts || [];
    const heading = ["completed", "completed_with_failures"].includes(detail.run.status) ? "Mission result" : "Coordinator output";
    const stopped = terminal.has(detail.run.status);
    setContent("panel-output", `<div class="output-header"><h2>${heading}</h2>${output ? '<button class="button button-outline" type="button" data-action="copy">Copy output</button>' : ""}</div>${output ? `<div class="output-content">${outputMarkup(typeof output === "string" ? output : pretty(output))}</div>` : emptyView(stopped ? "No final output" : "Work is underway", stopped ? "This mission settled before producing a result. Inspect Activity for details, then branch the mission to continue." : "The coordinator’s result will appear here. You can follow the team’s progress in Agents and Activity.", "↗")}${artifacts.length ? `<section class="artifact-section"><span class="eyebrow">Mission artifacts</span><ul class="artifact-list">${artifacts.map((artifact) => `<li><span class="artifact-symbol" aria-hidden="true">▤</span><span class="artifact-path">${escape(artifact.path)}</span><span class="artifact-size">${Number(artifact.size) >= 1024 ? `${(Number(artifact.size) / 1024).toFixed(1)} KB` : `${number(artifact.size)} B`}</span><button class="button button-outline artifact-download" type="button" data-action="artifact" data-artifact="${escape(artifact.path)}" aria-label="Download ${escape(artifact.path)}">Download <span aria-hidden="true">↓</span></button></li>`).join("")}</ul><p class="field-note">Files are saved in this mission’s local artifact directory. Use the event journal to review the actions that created them.</p></section>` : ""}`);
  }
  function renderDetail() {
    if (!state.selected) { renderNewMission(); return; }
    if (!state.detail) {
      $("mission-kicker").textContent = "MISSION WORKSPACE";
      $("mission-title").textContent = "Loading mission…";
      $("mission-status").hidden = true;
      $("tab-bar").hidden = true;
      setContent("panel-brief", '<div class="loading-view"><span class="loading-dot" aria-hidden="true"></span>Loading the mission journal.</div>');
      selectTab("brief", false);
      return;
    }
    const detail = state.detail;
    $("mission-kicker").textContent = `MISSION / ${detail.run.run_id}`;
    const task = detail.task || "Untitled mission";
    const firstLine = task.split("\n")[0];
    $("mission-title").textContent = firstLine.length > 100 ? firstLine.slice(0, 97) + "…" : firstLine;
    $("mission-status").hidden = false;
    $("mission-status").className = `status-badge ${cssStatus(detail.run.status)}`;
    $("mission-status").textContent = title(detail.run.status);
    $("tab-bar").hidden = false;
    $("agent-count").textContent = (detail.agents || []).length;
    $("event-count").textContent = (detail.events || []).length;
    renderRunControls();
    if (state.tab === "brief") renderBrief();
    else if (state.tab === "agents") renderAgents();
    else if (state.tab === "activity") renderActivity();
    else renderOutput();
  }
  function selectTab(tab, render = true) {
    state.tab = tab;
    document.querySelectorAll("[data-tab]").forEach((button) => {
      const current = button.dataset.tab === tab;
      button.setAttribute("aria-selected", String(current));
      button.tabIndex = current ? 0 : -1;
    });
    ["brief", "agents", "activity", "output"].forEach((name) => { $(`panel-${name}`).hidden = name !== tab; });
    if (render) renderDetail();
  }
  async function selectRun(id) {
    if (id === state.selected && state.detail) return;
    state.selected = id;
    state.detail = null;
    state.filter = "all";
    state.signatures.delete("panel-brief");
    state.signatures.delete("controls-section");
    selectTab("brief", false);
    renderOverview();
    renderDetail();
    if (!id) { $("mission-task").focus({ preventScroll: true }); return; }
    try {
      const detail = await api(`/api/runs/${encodeURIComponent(id)}`);
      if (state.selected !== id) return;
      state.detail = detail;
      renderDetail();
    } catch (error) {
      if (state.selected !== id) return;
      setContent("panel-brief", `<div class="inline-error">${escape(error.message)}</div><button type="button" class="button button-outline" data-action="reload">Reload mission</button>`);
      $("mission-title").textContent = "Could not load this mission.";
    }
  }
  async function refresh() {
    if (state.polling) return;
    state.polling = true;
    const selected = state.selected;
    try {
      const results = await Promise.allSettled([api("/api/state"), selected ? api(`/api/runs/${encodeURIComponent(selected)}`) : Promise.resolve(null)]);
      if (results[0].status === "rejected") throw results[0].reason;
      state.overview = results[0].value;
      connection(true);
      renderOverview();
      if (selected && state.selected === selected && results[1].status === "fulfilled") state.detail = results[1].value;
      if (selected && results[1].status === "rejected") throw results[1].reason;
      renderDetail();
    } catch (error) { connection(false, error.message || "Could not reach the local harness. Keep this page open and retry."); }
    finally {
      state.polling = false;
      clearTimeout(pollTimer);
      pollTimer = setTimeout(refresh, document.hidden ? 6000 : 1200);
    }
  }

  const starters = {
    review: "Review this repository. Map the main architecture and execution paths, then identify the highest-impact correctness, reliability, and maintainability issues. Delegate independent areas where useful. Support each finding with file references and propose focused next steps. Do not change files.",
    tests: "Build a practical test plan for this repository. Inspect the current tests and key execution paths, identify missing coverage, and prioritize tests that verify meaningful behavior. Delegate independent areas where useful. Return a concise plan with concrete cases, expected results, and how to run them. Do not change files.",
    docs: "Review this repository's onboarding documentation. Check the installation, configuration, and example commands against the implementation. Delegate independent areas where useful. Produce a concise documentation brief with verified corrections and a proposed quick-start outline. Do not change files.",
  };
  document.addEventListener("click", (event) => {
    const run = event.target.closest("[data-run]");
    if (run) { selectRun(run.dataset.run); return; }
    const jump = event.target.closest("[data-go-tab]");
    if (jump) { selectTab(jump.dataset.goTab); $("main").focus({ preventScroll: true }); return; }
    const tab = event.target.closest("[data-tab]");
    if (tab) { selectTab(tab.dataset.tab); return; }
    const starter = event.target.closest("[data-starter]");
    if (starter) { $("mission-task").value = starters[starter.dataset.starter]; $("mission-task").focus(); return; }
    const approval = event.target.closest("[data-approval]");
    if (approval) {
      action("approval", async () => {
        await api(`/api/approvals/${encodeURIComponent(approval.dataset.approval)}`, { approved: approval.dataset.approved === "true" });
        notify(approval.dataset.approved === "true" ? "Action approved." : "Action declined.");
        await refresh();
      });
      return;
    }
    const button = event.target.closest("[data-action]");
    if (!button) return;
    const name = button.dataset.action;
    if (name === "stop") action("stop", async () => { await api(runPath("/cancel"), {}); notify("Stop requested. The journal will retain completed work."); await refresh(); });
    else if (name === "resume") action("resume", async () => { await api(runPath("/resume"), {}); notify("Mission resumed."); await refresh(); });
    else if (name === "fork" && terminal.has(state.detail?.run.status)) { $("fork-task").value = state.detail?.task || ""; $("fork-dialog").showModal(); $("fork-task").focus(); }
    else if (name === "copy") action("copy", async () => {
      const output = state.detail?.output;
      if (!navigator.clipboard) throw new Error("Clipboard access is unavailable. Select and copy the output directly.");
      await navigator.clipboard.writeText(typeof output === "string" ? output : pretty(output));
      notify("Output copied.");
    });
    else if (name === "artifact") action("artifact", async () => {
      const result = await api(runPath(`/artifacts/${encodeURIComponent(button.dataset.artifact)}`));
      if (typeof result.content !== "string") throw new Error("This artifact could not be downloaded as text.");
      const url = URL.createObjectURL(new Blob([result.content], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = String(result.path || button.dataset.artifact).split(/[\\/]/).filter(Boolean).at(-1) || "artifact.txt";
      document.body.appendChild(link);
      try { link.click(); } finally {
        link.remove();
        setTimeout(() => URL.revokeObjectURL(url), 1000);
      }
      notify("Artifact download started.");
    });
    else if (name === "reload") { state.detail = null; refresh(); }
  });
  document.addEventListener("change", (event) => {
    const config = event.target.dataset.config;
    if (config) {
      if (event.target.type === "number") {
        if (!event.target.checkValidity()) { event.target.reportValidity(); return; }
        state.config[config] = Number(event.target.value);
      } else state.config[config] = event.target.value;
      if (config === "policy") renderNewControls();
    }
    if (event.target.id === "activity-agent") { state.filter = event.target.value; renderActivity(); }
  });
  document.addEventListener("submit", (event) => {
    if (event.target.id === "mission-form") {
      event.preventDefault();
      action("create", async () => {
        const task = $("mission-task").value.trim();
        if (!task) { $("mission-task").focus(); throw new Error("Write a brief before launching a mission."); }
        const inputs = [$("config-agents"), $("config-steps"), $("config-tokens")];
        const invalid = inputs.find((input) => !input.checkValidity());
        if (invalid) { invalid.reportValidity(); return; }
        const request = { ...state.config, task, max_agents: Number($("config-agents").value), max_steps_per_agent: Number($("config-steps").value), max_total_tokens: Number($("config-tokens").value) };
        if (!request.skill) delete request.skill;
        const result = await api("/api/runs", request);
        const id = runId(result);
        if (!id) throw new Error("The server did not return a mission reference.");
        notify("Mission launched.");
        await selectRun(id);
        await refresh();
      });
    } else if (event.target.id === "followup-form") {
      event.preventDefault();
      action("message", async () => {
        const input = $("followup-message");
        const message = input.value.trim();
        if (!message) return;
        if (terminal.has(state.detail?.run.status)) {
          const result = await api(runPath("/fork"), { task: message });
          const id = runId(result);
          if (!id) throw new Error("The server did not return a follow-up reference.");
          input.value = "";
          notify("Follow-up mission launched.");
          await selectRun(id);
        } else {
          await api(runPath("/message"), { message, agent_id: "root" });
          input.value = "";
          notify("Direction added to the mission journal.");
        }
        await refresh();
      });
    } else if (event.target.id === "fork-form") {
      event.preventDefault();
      action("fork-submit", async () => {
        const task = $("fork-task").value.trim();
        if (!task) return;
        const submit = $("fork-form").querySelector('[type="submit"]');
        submit.disabled = true;
        try {
          const result = await api(runPath("/fork"), { task });
          const id = runId(result);
          if (!id) throw new Error("The server did not return a branch reference.");
          $("fork-dialog").close();
          notify("Mission branched.");
          await selectRun(id);
          await refresh();
        } finally { submit.disabled = false; }
      });
    }
  });
  $("new-mission").addEventListener("click", () => selectRun(null));
  $("retry-button").addEventListener("click", refresh);
  $("start-backend").addEventListener("click", () => action("backend", async () => { await api("/api/backend/start", {}); notify("Backend start requested."); await refresh(); }));
  $("close-fork").addEventListener("click", () => $("fork-dialog").close());
  $("cancel-fork").addEventListener("click", () => $("fork-dialog").close());
  $("tab-bar").addEventListener("keydown", (event) => {
    const tabs = Array.from($("tab-bar").querySelectorAll("[data-tab]"));
    const index = tabs.indexOf(event.target);
    if (index < 0) return;
    let next;
    if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + tabs.length) % tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = tabs.length - 1;
    else return;
    event.preventDefault();
    selectTab(tabs[next].dataset.tab);
    tabs[next].focus();
  });
  document.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() !== "n" || event.metaKey || event.ctrlKey || event.altKey || event.target.closest("input,textarea,select,[contenteditable=true]") || $("fork-dialog").open) return;
    event.preventDefault();
    selectRun(null);
  });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  async function boot() {
    renderNewMission();
    const fragment = new URLSearchParams(location.hash.slice(1));
    const token = fragment.get("token");
    if (token) {
      try {
        await api("/api/auth", { token });
        history.replaceState(null, "", location.pathname + location.search);
      } catch {
        history.replaceState(null, "", location.pathname + location.search);
        connection(false, "The local session could not be authenticated. Open the Studio link from the harness command again.");
        return;
      }
    }
    await refresh();
  }
  boot();
})();

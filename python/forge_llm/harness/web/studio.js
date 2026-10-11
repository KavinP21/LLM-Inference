/* Forge: a conversation surface over the durable local harness. */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const terminal = new Set(["completed", "completed_with_failures", "failed", "cancelled", "interrupted"]);
  const working = new Set(["running", "loading", "queued", "waiting", "waiting_approval"]);
  const state = {
    overview: null, detail: null, selected: null, loading: false, refreshing: false,
    sending: false, cancelling: false, connected: false, sidebar: false,
    policy: "review", drafts: new Map(), turns: new Map(), busyApprovals: new Set(),
    request: null, sendError: "", signatures: new WeakMap(), commandIndex: 0,
    draftStorage: null, acknowledged: new Map(), policyTouched: false,
  };
  let pollTimer;
  let toastTimer;
  let refreshAgain = false;

  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }
  function pretty(value) { return typeof value === "string" ? value : JSON.stringify(value, null, 2) ?? ""; }
  function safeUrl(value) {
    const url = String(value || "").trim();
    if (/^(?:https?:\/\/|mailto:)/i.test(url) && !/[\u0000-\u0020]/.test(url)) return url;
    // A workspace reference must stay on this origin, never a protocol-relative URL.
    if (/^\/(?!\/)/.test(url) && !/[\u0000-\u0020\\]/.test(url)) return url;
    if (/^#[\w.-]+$/.test(url)) return url;
    return null;
  }
  function inline(value) {
    const tokens = [];
    const token = (html) => { tokens.push(html); return `\u0000${tokens.length - 1}\u0000`; };
    let text = String(value).replaceAll("\u0000", "");
    text = text.replace(/(`+)([\s\S]*?)\1/g, (_, marks, code) => token(`<code>${escape(code.trim())}</code>`));
    text = text.replace(/\[([^\]\n]+)\]\(([^\s)]+)(?:\s+"[^"\n]*")?\)/g, (whole, label, url) => {
      const href = safeUrl(url);
      return href ? token(`<a href="${escape(href)}"${/^https?:/i.test(href) ? ' target="_blank" rel="noopener noreferrer"' : ""}>${escape(label)}</a>`) : label;
    });
    text = escape(text).replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=$|[\s.,!?:;)])/g, "$1<em>$2</em>")
      .replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s.,!?:;)])/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
    // Expand once: user-provided text inside a code token cannot create another token.
    return text.replace(/\u0000(\d+)\u0000/g, (_, i) => tokens[Number(i)] || "");
  }
  function markdown(value, depth = 0) {
    const lines = String(value || "").replace(/\r\n?/g, "\n").split("\n");
    if (depth > 6) return `<p>${inline(value)}</p>`;
    const blocks = [];
    const listMatch = (line) => /^(\s*)([-+*]|\d+[.)])\s+(.*)$/.exec(line);
    const tableCells = (line) => line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((cell) => cell.trim());
    const tableRule = (line) => /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line || "");
    const special = (i) => /^\s*(?:```|~~~|#{1,6}\s|>\s?|(?:[-*_]\s*){3,}$)/.test(lines[i]) || listMatch(lines[i]) || tableRule(lines[i + 1]);
    for (let i = 0; i < lines.length;) {
      const line = lines[i];
      if (!line.trim()) { i++; continue; }
      const fence = /^\s*(```+|~~~+)\s*([\w+.-]*)\s*$/.exec(line);
      if (fence) {
        const code = [];
        const end = new RegExp(`^\\s*${fence[1][0]}{${fence[1].length},}\\s*$`);
        i++;
        while (i < lines.length && !end.test(lines[i])) code.push(lines[i++]);
        if (i < lines.length) i++;
        blocks.push(`<div class="code-block"><div class="code-header"><span>${escape(fence[2] || "code")}</span><button type="button" class="copy-code" data-copy-code>Copy</button></div><pre><code>${escape(code.join("\n"))}</code></pre></div>`);
        continue;
      }
      const heading = /^\s*(#{1,6})\s+(.*?)(?:\s+#+\s*)?$/.exec(line);
      if (heading) { blocks.push(`<h${heading[1].length}>${inline(heading[2])}</h${heading[1].length}>`); i++; continue; }
      if (/^\s*(?:[-*_]\s*){3,}$/.test(line)) { blocks.push("<hr>"); i++; continue; }
      if (/^\s*>/.test(line)) {
        const quote = [];
        while (i < lines.length && /^\s*>/.test(lines[i])) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
        blocks.push(`<blockquote>${markdown(quote.join("\n"), depth + 1)}</blockquote>`);
        continue;
      }
      if (tableRule(lines[i + 1])) {
        const headers = tableCells(line);
        const alignment = tableCells(lines[i + 1]);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) rows.push(tableCells(lines[i++]));
        const alignmentClass = (j) => /^:.*:$/.test(alignment[j]) ? ' class="align-center"' : /:$/.test(alignment[j]) ? ' class="align-right"' : "";
        blocks.push(`<div class="table-wrap"><table><thead><tr>${headers.map((cell, j) => `<th${alignmentClass(j)}>${inline(cell)}</th>`).join("")}</tr></thead><tbody>${rows.map((row) => `<tr>${headers.map((_, j) => `<td${alignmentClass(j)}>${inline(row[j] || "")}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }
      const first = listMatch(line);
      if (first) {
        const ordered = /^\d/.test(first[2]);
        const indent = first[1].length;
        const items = [];
        let next;
        while (i < lines.length && (next = listMatch(lines[i])) && next[1].length === indent && /^\d/.test(next[2]) === ordered) {
          const content = [next[3]];
          const padding = indent + next[2].length + 1;
          i++;
          while (i < lines.length) {
            const match = listMatch(lines[i]);
            if (match && match[1].length <= indent) break;
            if (!lines[i].trim()) {
              // A blank line belongs to the item only when followed by indented content or another list item.
              let lookahead = i + 1;
              while (lookahead < lines.length && !lines[lookahead].trim()) lookahead++;
              if (lookahead >= lines.length || (!listMatch(lines[lookahead]) && !/^\s+\S/.test(lines[lookahead]))) break;
              content.push(""); i++; continue;
            }
            const spaces = /^\s*/.exec(lines[i])[0].length;
            if (spaces <= indent) break;
            content.push(lines[i++].slice(Math.min(padding, spaces)));
          }
          items.push(`<li>${markdown(content.join("\n"), depth + 1)}</li>`);
          while (i < lines.length && !lines[i].trim() && listMatch(lines[i + 1] || "")) i++;
        }
        const tag = ordered ? "ol" : "ul";
        const start = ordered && parseInt(first[2], 10) !== 1 ? ` start="${parseInt(first[2], 10)}"` : "";
        blocks.push(`<${tag}${start}>${items.join("")}</${tag}>`);
        continue;
      }
      const paragraph = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !special(i)) paragraph.push(lines[i++]);
      blocks.push(`<p>${paragraph.map(inline).join("<br>")}</p>`);
    }
    return blocks.join("");
  }
  function outputText(value) {
    if (typeof value === "string") {
      const trimmed = value.trim();
      if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
        try {
          const object = JSON.parse(trimmed);
          if (object.action === "finish" && typeof object.result === "string") return object.result;
        } catch { /* It is ordinary prose containing braces. */ }
      }
      return value;
    }
    if (value && typeof value === "object") return value.result || value.content || value.text || value.answer || "";
    return "";
  }
  function patch(node, content) {
    if (state.signatures.get(node) === content) return;
    const disclosures = new Map(Array.from(node.querySelectorAll("details[data-key]")).map((entry) => [entry.dataset.key, entry.open]));
    node.innerHTML = content;
    node.querySelectorAll("details[data-key]").forEach((entry) => { if (disclosures.has(entry.dataset.key)) entry.open = disclosures.get(entry.dataset.key); });
    state.signatures.set(node, content);
  }
  function toast(message, error = false, duration = 3500) {
    clearTimeout(toastTimer);
    $("toast").textContent = message;
    $("toast").classList.toggle("error", error);
    $("toast").hidden = false;
    toastTimer = setTimeout(() => { $("toast").hidden = true; }, duration);
  }
  async function api(path, data) {
    const options = { credentials: "same-origin", headers: { "Accept": "application/json" }, cache: "no-store" };
    if (data !== undefined) {
      options.method = "POST";
      options.headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(data);
    }
    let response;
    try { response = await fetch(path, options); } catch { throw new Error("Could not reach Forge. Check that the local server is running."); }
    let body;
    try { body = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok) {
      const error = body.error || body.message || `Request failed (${response.status}).`;
      throw new Error(typeof error === "string" ? error : error.message || "The request could not be completed.");
    }
    return body;
  }
  function currentPolicy() { return state.detail?.chat?.policy || state.policy; }
  function active() {
    return Boolean(state.detail?.turns?.some((turn) => turn.active || (!terminal.has(turn.status) && turn.status !== "queued" && working.has(turn.status))))
      || (Boolean(state.selected) && !state.detail && ["loading", "running"].includes(state.overview?.chats?.find((chat) => chat.chat_id === state.selected)?.status));
  }
  function draftKey(id = state.selected) { return id || "new"; }
  function persistDrafts() {
    if (!state.draftStorage) return;
    try {
      sessionStorage.setItem(state.draftStorage, JSON.stringify(Array.from(state.drafts).slice(-60)));
      sessionStorage.setItem(`${state.draftStorage}:policy`, state.policy);
    }
    catch { /* A full or disabled session store must never prevent sending. */ }
  }
  function loadDrafts(workspace) {
    if (state.draftStorage) return;
    state.draftStorage = `forge:drafts:${typeof workspace === "string" ? workspace : workspace?.path || workspace?.root || "workspace"}`;
    try {
      const policy = sessionStorage.getItem(`${state.draftStorage}:policy`);
      if (!state.policyTouched && ["review", "read_only"].includes(policy)) state.policy = policy;
      const saved = JSON.parse(sessionStorage.getItem(state.draftStorage) || "[]");
      if (Array.isArray(saved)) for (const pair of saved.slice(-60)) {
        if (Array.isArray(pair) && pair.length === 2 && typeof pair[0] === "string" && typeof pair[1] === "string" && pair[1].length <= 12000 && !state.drafts.has(pair[0])) state.drafts.set(pair[0], pair[1]);
      }
      if (!$("message").value) $("message").value = state.drafts.get(draftKey()) || "";
    } catch { /* Invalid saved drafts do not affect the live chat. */ }
  }
  function saveDraft() { state.drafts.set(draftKey(), $("message").value); persistDrafts(); }
  function rememberSelection() {
    const url = new URL(location.href);
    if (state.selected) url.searchParams.set("chat", state.selected);
    else url.searchParams.delete("chat");
    history.replaceState(null, "", url.pathname + url.search);
  }
  function acceptDetail(detail) {
    const chatId = detail.chat?.chat_id || state.selected;
    const acknowledged = state.acknowledged.get(chatId);
    if (acknowledged) {
      for (const [key, turn] of acknowledged) {
        if (detail.turns.some((entry) => entry.turn_id === key)) acknowledged.delete(key);
        else detail.turns.push(turn);
      }
      if (!acknowledged.size) state.acknowledged.delete(chatId);
    }
    return detail;
  }
  function resizeComposer() {
    const input = $("message");
    input.style.height = "auto";
    input.style.height = `${Math.max(78, Math.min(input.scrollHeight, 220))}px`;
    $("jump-bottom").style.bottom = `${$("composer-area").offsetHeight + 14}px`;
  }
  function modelLabel(backend = state.detail?.backend || state.overview?.backend || {}) {
    const label = String(backend.label || "Local model");
    if (/deterministic|harness demo/i.test(label)) return "Demo backend";
    const qwen = /qwen[-_ ]?(3\.5|3|2\.5)/i.exec(label);
    if (qwen) return `Qwen${qwen[1]}`;
    return /[\\/]|localhost|127\.0\.0\.1|http:|port\s*\d/i.test(label) || label.length > 45 ? "Local model" : label;
  }
  function statusLabel(value) {
    return ({ ready: "Ready", running: "Working", loading: "Loading", stopped: "Ready to start", failed: "Needs attention", error: "Needs attention", connected: "Ready", unknown: "Connecting" })[value] || "Ready to start";
  }
  function dateLabel(value) {
    if (!value) return "";
    const date = new Date(typeof value === "number" ? value * 1000 : value);
    if (Number.isNaN(date.getTime())) return "";
    const start = new Date(); start.setHours(0, 0, 0, 0);
    if (date >= start) return "Today";
    if (date >= new Date(start.getTime() - 86400000)) return "Yesterday";
    return date.toLocaleDateString([], { month: "short", day: "numeric" });
  }
  function renderOverview() {
    const overview = state.overview;
    if (!overview) return;
    const raw = overview.workspace || "Local workspace";
    const workspace = typeof raw === "string" ? raw : raw.path || raw.root || "Local workspace";
    $("workspace-name").textContent = workspace.split(/[\\/]/).filter(Boolean).at(-1) || workspace;
    $("workspace-name").title = workspace;
    $("workspace-path").textContent = workspace;
    const backend = state.detail?.backend || overview.backend || {};
    $("model-label").textContent = modelLabel(backend);
    $("model-status").title = `${modelLabel(backend)} · ${statusLabel(backend.status)}`;
    $("model-dot").className = `status-dot ${["ready", "connected", "running", "loading", "error", "failed"].includes(backend.status) ? backend.status : ""}`;
    $("settings-model-label").textContent = modelLabel(backend);
    $("settings-model-status").textContent = statusLabel(backend.status);
    $("demo-badge").hidden = overview.mode !== "demo";
    $("demo-note").hidden = overview.mode !== "demo";
    const chats = overview.chats || [];
    patch($("chat-list"), chats.length ? chats.map((chat) => `<button type="button" class="chat-item${chat.chat_id === state.selected ? " selected" : ""}" data-chat="${escape(chat.chat_id)}"${chat.chat_id === state.selected ? ' aria-current="true"' : ""}><span class="chat-title">${escape(chat.title || "New chat")}</span><span class="chat-meta">${working.has(chat.status) ? `<span class="status-dot ${chat.status === "loading" ? "loading" : "running"}" aria-hidden="true"></span>${chat.status === "loading" ? "Loading model" : "Working"} · ` : ""}${escape(dateLabel(chat.updated || chat.created))}</span></button>`).join("") : '<p class="sidebar-empty">Your chats will appear here.</p>');
  }
  function disclosureSummary(text, extra = "", busy = false) {
    return `<summary class="disclosure-summary"><span class="disclosure-chevron" aria-hidden="true">›</span>${busy ? '<span class="working-dot" aria-hidden="true"></span>' : ""}<span>${escape(text)}</span>${extra ? `<span class="disclosure-count">${escape(extra)}</span>` : ""}</summary>`;
  }
  function toolLabel(name, args = {}) {
    const path = String(args.path || args.file || "");
    const subject = path && path !== "." ? path : "workspace";
    const query = args.query || args.pattern || args.text;
    const names = {
      list_files: `Explored ${subject}`, read_file: `Read ${subject}`, search_files: `Searched${query ? ` for “${String(query).slice(0, 80)}”` : ` ${subject}`}`,
      search_text: `Searched${query ? ` for “${String(query).slice(0, 80)}”` : " the codebase"}`, write_file: `Updated ${subject}`,
      replace_text: `Edited ${subject}`, edit_lines: `Edited ${subject}`, save_artifact: `Created ${subject}`,
      run_tests: "Ran tests", run_command: "Ran a command", git_status: "Checked changed files", git_diff: "Reviewed changes",
      journal_read: "Reviewed earlier work", fetch_url: "Read a web page", web_search: "Searched the web",
    };
    return names[name] || `Used ${String(name || "tool").replace(/^mcp__/, "").replaceAll("_", " ")}`;
  }
  function toolRows(turn, agentId = "root") {
    const rows = (turn.events || []).filter((event) => event.kind === "tool_result" && (event.agent_id || "root") === agentId && event.payload?.name !== "journal_read").map((event) => ({
      key: `tool-${event.id}`, name: event.payload.name, args: event.payload.args || {}, result: event.payload.result, pending: false,
    }));
    const agent = (turn.agents || []).find((entry) => entry.agent_id === agentId);
    const pending = agent?.pending_action;
    const pendingName = pending?.action === "tool" ? pending.name : pending?.action;
    if (turn.active && pendingName && !["finish", "spawn", "wait", "send", "batch", "journal_read"].includes(pendingName)) {
      const events = (turn.events || []).filter((event) => (event.agent_id || "root") === agentId);
      const prepared = events.filter((event) => event.kind === "action_prepared").at(-1);
      const settled = events.filter((event) => event.kind === "tool_result").at(-1);
      if (!settled || (prepared && Number(prepared.id) > Number(settled.id))) rows.push({ key: `pending-${agentId}`, name: pendingName, args: pending.args || {}, pending: true });
    }
    return rows;
  }
  function currentToolLabel(name, args) {
    return toolLabel(name, args).replace(/^(Read|Explored|Searched|Updated|Edited|Created|Ran|Checked|Reviewed|Used)\b/, (word) => ({
      Read: "Reading", Explored: "Exploring", Searched: "Searching", Updated: "Updating", Edited: "Editing", Created: "Creating", Ran: "Running", Checked: "Checking", Reviewed: "Reviewing", Used: "Using",
    })[word]);
  }
  function renderToolRows(rows) {
    return rows.slice(-80).map((row) => {
      const failed = (typeof row.result === "string" && /^Error\b/i.test(row.result)) || Boolean(row.result && typeof row.result === "object" && row.result.error);
      const command = row.args.command || row.args.cmd;
      return `<details class="tool-row${failed ? " failed" : ""}" data-key="${escape(row.key)}"><summary><span class="tool-icon" aria-hidden="true">${row.pending ? "•" : failed ? "!" : "✓"}</span><span class="tool-title">${escape(row.pending ? currentToolLabel(row.name, row.args) : toolLabel(row.name, row.args))}${command ? ` · ${escape(String(command).split("\n")[0].slice(0, 90))}` : ""}</span><span class="disclosure-chevron" aria-hidden="true">›</span></summary><div class="tool-detail"><h4>Request</h4><pre>${escape(pretty(row.args))}</pre>${row.result != null ? `<h4>Result</h4><pre>${escape(pretty(row.result))}</pre>` : '<p>Working…</p>'}</div></details>`;
    }).join("");
  }
  function activityMarkup(turn) {
    const rows = toolRows(turn);
    if (!rows.length) return "";
    const reads = rows.filter((row) => ["read_file", "list_files", "search_files", "search_text"].includes(row.name)).length;
    const edits = rows.filter((row) => ["write_file", "replace_text", "edit_lines"].includes(row.name)).length;
    const tests = rows.some((row) => row.name === "run_tests");
    const descriptions = [];
    if (reads) descriptions.push("Explored the codebase");
    if (edits) descriptions.push(`made ${edits} ${edits === 1 ? "edit" : "edits"}`);
    if (tests) descriptions.push("ran tests");
    const summary = rows.at(-1).pending ? currentToolLabel(rows.at(-1).name, rows.at(-1).args) + "…" : descriptions.join(" · ") || `${rows.length} ${rows.length === 1 ? "action" : "actions"}`;
    return `<details class="activity-group" data-key="activity-${escape(turn.turn_id || turn.run_id)}">${disclosureSummary(summary, descriptions.length ? `${rows.length} ${rows.length === 1 ? "action" : "actions"}` : "", rows.at(-1).pending)}<div class="activity-rows">${renderToolRows(rows)}${rows.length > 80 ? '<p class="settings-note">Earlier actions are available in the conversation trace.</p>' : ""}</div></details>`;
  }
  function agentsMarkup(turn) {
    const agents = (turn.agents || []).filter((agent) => agent.parent_id || agent.agent_id !== "root");
    if (!agents.length) return "";
    const activeCount = agents.filter((agent) => !terminal.has(agent.status) && !["failed", "cancelled"].includes(agent.status)).length;
    return `<details class="agents-group" data-key="agents-${escape(turn.turn_id || turn.run_id)}">${disclosureSummary(`${agents.length} ${agents.length === 1 ? "agent" : "agents"}`, activeCount ? `${activeCount} working` : "Finished", activeCount > 0 && turn.active)}<div class="agents-list">${agents.map((agent) => `<details class="agent-card" data-key="agent-${escape(agent.agent_id)}"><summary><span class="disclosure-chevron" aria-hidden="true">›</span><span class="status-dot ${terminal.has(agent.status) ? agent.status : "running"}" aria-hidden="true"></span><span class="agent-name">${escape(agent.role || "Agent")}</span><span class="agent-state">${escape(terminal.has(agent.status) ? agent.status === "completed" ? "Done" : agent.status === "cancelled" ? "Stopped" : "Needs attention" : "Working")}</span></summary><div class="agent-detail"><p class="agent-task">${escape(agent.task || "")}</p>${renderToolRows(toolRows(turn, agent.agent_id))}${agent.result ? `<div class="markdown">${markdown(outputText(agent.result))}</div>` : ""}${agent.error ? `<p class="turn-error">${escape(agent.error)}</p>` : ""}</div></details>`).join("")}</div></details>`;
  }
  function diffMarkup(value) {
    return String(value || "No textual changes.").split("\n").map((line) => `<span class="diff-line${line.startsWith("+++") || line.startsWith("---") ? " header" : line.startsWith("+") ? " add" : line.startsWith("-") ? " remove" : line.startsWith("@@") ? " hunk" : ""}">${escape(line) || " "}</span>`).join("");
  }
  function approvalsMarkup(turn) {
    return (turn.approvals || []).filter((entry) => !entry.status || entry.status === "pending").map((entry) => {
      const args = entry.args || {};
      const command = args.command || args.cmd;
      const preview = entry.preview;
      const hasDiff = preview?.diff != null;
      const heading = command ? "Allow this command?" : hasDiff ? "Allow this change?" : "Allow this action?";
      return `<article class="approval-card"><div class="approval-heading"><h3>${heading}</h3><p>${escape(command ? "Forge wants to run a command in this workspace." : hasDiff ? preview.path || args.path || "Proposed file change" : toolLabel(entry.tool, args))}</p></div>${command ? `<pre class="approval-command">${escape(command)}</pre>` : ""}${hasDiff ? `<div class="approval-preview"><div class="diff-path">${escape(preview.path || args.path || "File changes")}</div><pre class="diff">${diffMarkup(preview.diff)}</pre></div>` : ""}<details class="approval-arguments" data-key="approval-${escape(entry.id)}">${disclosureSummary("Exact action details")}<pre>${escape(entry.tool)}\n${escape(pretty(args))}</pre></details><div class="approval-actions"><button type="button" class="button" data-approval="${escape(entry.id)}" data-approved="false"${state.busyApprovals.has(entry.id) ? " disabled" : ""}>Decline</button><button type="button" class="button button-primary" data-approval="${escape(entry.id)}" data-approved="true"${state.busyApprovals.has(entry.id) ? " disabled" : ""}>${state.busyApprovals.has(entry.id) ? "Applying…" : "Approve"}</button></div></article>`;
    }).join("");
  }
  function turnProgress(turn) {
    if (turn.status === "queued") return "Queued · runs after the current response";
    if (turn.status === "loading" || (!turn.run_id && turn.active)) return `Loading ${modelLabel()}…`;
    if ((turn.approvals || []).some((entry) => entry.status === "pending")) return "Waiting for your approval";
    if (turn.active) return "Working…";
    if (turn.status === "cancelled") return "Stopped. Send a message to continue.";
    if (turn.status === "interrupted") return "This response was interrupted. Send a message to continue.";
    return "";
  }
  function userMessages(turn) {
    const messages = (turn.messages || turn.user_messages || []).filter((message) => (message.role || "user") === "user");
    if (messages.length) return messages.map((message) => typeof message === "string" ? message : message.content || message.message || "").filter(Boolean);
    return [turn.user || "", ...(turn.events || []).filter((event) => event.kind === "operator_message").map((event) => event.payload?.message || "")].filter(Boolean);
  }
  function createTurn(key) {
    const node = document.createElement("article");
    node.className = "turn";
    node.dataset.turn = key;
    node.innerHTML = '<div class="user-messages"></div><div class="assistant-turn"><div class="assistant-label"><span class="assistant-mark" aria-hidden="true">f</span>Forge</div><div class="turn-activity"></div><div class="turn-agents"></div><div class="turn-approvals"></div><div class="turn-directions user-messages"></div><div class="turn-progress" role="status"></div><div class="turn-output"></div><div class="turn-error"></div><div class="turn-artifacts"></div></div>';
    state.turns.set(key, node);
    return node;
  }
  function nearBottom() {
    const pane = $("transcript-pane");
    return pane.scrollHeight - pane.scrollTop - pane.clientHeight < 130;
  }
  function scrollBottom() {
    const pane = $("transcript-pane");
    pane.scrollTop = pane.scrollHeight;
    $("jump-bottom").hidden = true;
  }
  function renderTranscript() {
    const turns = state.detail?.turns || [];
    const follow = nearBottom();
    $("welcome").hidden = Boolean(state.selected);
    $("view-loading").hidden = !state.loading;
    if (!state.detail) { renderComposer(); return; }
    const keys = new Set();
    turns.forEach((turn, index) => {
      const key = String(turn.turn_id || turn.run_id || `turn-${index}`);
      keys.add(key);
      const node = state.turns.get(key) || createTurn(key);
      if (!node.isConnected) $("transcript").appendChild(node);
      const messages = userMessages(turn);
      patch(node.querySelector(".user-messages"), messages[0] ? `<div class="user-message">${escape(messages[0])}</div>` : "");
      patch(node.querySelector(".turn-directions"), messages.slice(1).map((message) => `<div class="user-message steering">${escape(message)}</div>`).join(""));
      patch(node.querySelector(".turn-activity"), activityMarkup(turn));
      patch(node.querySelector(".turn-agents"), agentsMarkup(turn));
      patch(node.querySelector(".turn-approvals"), approvalsMarkup(turn));
      const progress = turnProgress(turn);
      patch(node.querySelector(".turn-progress"), progress ? `${turn.active && !turn.approvals?.length ? '<span class="working-dot" aria-hidden="true"></span>' : ""}${escape(progress)}` : "");
      const output = outputText(turn.output);
      patch(node.querySelector(".turn-output"), output ? `<div class="markdown">${markdown(output)}</div><div class="output-actions"><button type="button" class="copy-response" data-copy-response="${escape(key)}">Copy response</button>${turn.status === "completed_with_failures" ? "Some delegated work needs attention." : ""}</div>` : "");
      patch(node.querySelector(".turn-error"), turn.error && !["cancelled", "interrupted"].includes(turn.status) ? `<p>${escape(turn.run_id ? "Forge couldn’t finish this request. You can send a message to continue." : "The model couldn’t start. Your message is saved; send it again to retry.")}</p><details data-key="error-${escape(key)}">${disclosureSummary("Error details")}<pre>${escape(typeof turn.error === "string" ? turn.error : pretty(turn.error))}</pre></details>` : "");
      patch(node.querySelector(".turn-artifacts"), turn.run_id && turn.artifacts?.length ? `<div class="artifact-list">${turn.artifacts.map((artifact) => `<button type="button" class="artifact-button" data-artifact="${escape(artifact.path)}" data-run="${escape(turn.run_id)}" title="Download ${escape(artifact.path)}"><span aria-hidden="true">▤</span><span>${escape(artifact.path)}</span><span aria-hidden="true">↓</span></button>`).join("")}</div>` : "");
    });
    for (const [key, node] of state.turns) if (!keys.has(key)) { node.remove(); state.turns.delete(key); }
    renderComposer();
    renderSettings();
    document.title = state.detail.chat?.title ? `${state.detail.chat.title} — Forge` : "Forge";
    if (follow) scrollBottom();
    else $("jump-bottom").hidden = nearBottom();
  }
  function renderComposer() {
    const hasDraft = Boolean($("message").value.trim());
    const running = active();
    const stop = running && !hasDraft;
    $("send").disabled = state.sending || state.cancelling || (!running && !hasDraft);
    $("send").classList.toggle("stop", stop);
    $("send-label").textContent = state.sending ? "Sending…" : state.cancelling ? "Stopping…" : stop ? "Stop" : "Send";
    $("send-icon").textContent = stop ? "■" : "↑";
    $("send").setAttribute("aria-label", stop ? "Stop current response" : running ? "Send follow-up message" : "Send message");
    $("stop-draft").hidden = !running || !hasDraft;
    $("stop-draft").disabled = state.cancelling;
    $("composer-hint").textContent = running && hasDraft ? "Follow-up runs after this response" : "Enter to send · Shift + Enter for a new line";
    const policy = currentPolicy();
    $("permission-label").textContent = policy === "read_only" ? "Read only" : "Review changes";
    $("composer-footnote").textContent = state.overview?.mode === "demo" ? "Demo mode · no model inference is performed." : policy === "read_only" ? "Forge can read and explore this workspace. Files stay unchanged." : "Forge works in your local workspace. You review changes before they run.";
    $("new-chat").disabled = state.sending;
    const status = state.sendError || (state.sending ? "Sending your message…" : "");
    $("composer-status").textContent = status;
    $("composer-status").hidden = !status;
    $("composer-status").classList.toggle("error", Boolean(state.sendError));
    resizeComposer();
  }
  function renderSettings() {
    const existingPolicy = state.detail?.chat?.policy;
    $("policy-note").textContent = existingPolicy ? `This chat uses ${existingPolicy === "read_only" ? "Read only" : "Review changes"}. Changes here apply to new chats.` : "Applies to new chats.";
    document.querySelectorAll('input[name="policy"]').forEach((input) => { input.checked = input.value === state.policy; });
    const turns = (state.detail?.turns || []).filter((turn) => turn.run_id);
    $("trace-section").hidden = !turns.length;
    patch($("trace-list"), turns.map((turn, index) => `<a href="/api/runs/${encodeURIComponent(turn.run_id)}/export" download="forge-trace-${index + 1}.json">Download response ${index + 1} trace ↓</a>`).join(""));
  }
  function setConnection(error = "") {
    state.connected = !error;
    $("connection-error").hidden = !error;
    $("connection-error-text").textContent = error;
  }
  function toggleSidebar(open = !state.sidebar) {
    state.sidebar = open;
    $("app").classList.toggle("sidebar-open", open);
    $("sidebar").inert = !open;
    $("toggle-sidebar").setAttribute("aria-expanded", String(open));
    $("toggle-sidebar").setAttribute("aria-label", open ? "Close chats" : "Open chats");
    const mobile = matchMedia("(max-width:700px)").matches;
    $("sidebar-backdrop").hidden = !open || !mobile;
    $("main").inert = open && mobile;
    if (open && mobile) {
      $("sidebar").setAttribute("role", "dialog");
      $("sidebar").setAttribute("aria-modal", "true");
      if (!$("sidebar").contains(document.activeElement)) $("new-chat").focus({ preventScroll: true });
    } else {
      $("sidebar").removeAttribute("role");
      $("sidebar").removeAttribute("aria-modal");
      if (!open && ($("sidebar").contains(document.activeElement) || document.activeElement === $("sidebar-backdrop"))) $("toggle-sidebar").focus({ preventScroll: true });
    }
  }
  async function selectChat(id) {
    if (state.sending) return;
    if (id === state.selected && state.detail) { $("message").focus(); return; }
    saveDraft();
    state.selected = id;
    rememberSelection();
    state.detail = null;
    state.loading = Boolean(id);
    state.turns.clear();
    $("transcript").replaceChildren();
    $("transcript-pane").scrollTop = 0;
    $("jump-bottom").hidden = true;
    $("message").value = state.drafts.get(draftKey()) || "";
    state.sendError = "";
    document.title = "Forge";
    if (matchMedia("(max-width:700px)").matches) toggleSidebar(false);
    $("welcome").hidden = Boolean(id);
    $("view-loading").hidden = !state.loading;
    renderOverview(); renderComposer(); renderSettings();
    $("message").focus({ preventScroll: true });
    if (!id) { clearTimeout(pollTimer); scheduleRefresh(); return; }
    try {
      const detail = await api(`/api/chats/${encodeURIComponent(id)}`);
      if (state.selected !== id) return;
      state.detail = acceptDetail(detail);
      state.loading = false;
      renderOverview(); renderTranscript(); scrollBottom();
      scheduleRefresh();
    } catch (error) {
      if (state.selected !== id) return;
      state.loading = false;
      $("view-loading").hidden = true;
      setConnection(error.message);
      scheduleRefresh();
    }
  }
  function needsPolling() {
    return active() || state.loading || state.sending || state.overview?.chats?.some((chat) => working.has(chat.status)) || state.overview?.backend?.status === "loading";
  }
  function scheduleRefresh() {
    clearTimeout(pollTimer);
    if (needsPolling() || !state.connected) pollTimer = setTimeout(refresh, document.hidden ? 4000 : state.connected ? 800 : 3000);
  }
  async function refresh() {
    if (state.refreshing) { refreshAgain = true; return; }
    state.refreshing = true;
    const selected = state.selected;
    try {
      const [overview, detail] = await Promise.all([api("/api/chats"), selected ? api(`/api/chats/${encodeURIComponent(selected)}`) : Promise.resolve(null)]);
      state.overview = overview;
      loadDrafts(overview.workspace);
      if (selected === state.selected && detail) { state.detail = acceptDetail(detail); state.loading = false; }
      setConnection();
      renderOverview(); renderTranscript(); renderSettings();
    } catch (error) { setConnection(error.message || "Could not reach the local server."); }
    finally {
      state.refreshing = false;
      if (refreshAgain) { refreshAgain = false; refresh(); }
      else scheduleRefresh();
    }
  }
  function nonce() {
    return typeof crypto.randomUUID === "function" ? crypto.randomUUID() : Array.from(crypto.getRandomValues(new Uint8Array(16)), (n) => n.toString(16).padStart(2, "0")).join("");
  }
  async function sendMessage() {
    if (state.sending || state.cancelling) return;
    const input = $("message");
    const original = input.value;
    const message = original.trim();
    if (!message) { if (active()) await cancelResponse(); else input.focus(); return; }
    if (runCommand(message)) return;
    const selected = state.selected;
    const requestKey = `${selected || "new"}\u0000${state.policy}\u0000${message}`;
    if (state.request?.key !== requestKey) state.request = { key: requestKey, request_id: nonce() };
    state.sending = true;
    state.sendError = "";
    $("command-menu").hidden = true;
    renderComposer();
    const request = selected ? { message, request_id: state.request.request_id } : { message, policy: state.policy, request_id: state.request.request_id };
    try {
      const result = await api(selected ? `/api/chats/${encodeURIComponent(selected)}/message` : "/api/chats", request);
      const chatId = result.chat_id || selected;
      if (!chatId) throw new Error("The server did not return a chat. Your message is still in the composer.");
      if (input.value === original) input.value = "";
      state.drafts.set(draftKey(selected), input.value);
      if (!selected) {
        state.selected = chatId;
        rememberSelection();
        state.detail = null;
        state.turns.clear();
        $("transcript").replaceChildren();
        $("welcome").hidden = true;
        state.drafts.set(draftKey(chatId), input.value);
        state.drafts.set("new", "");
      }
      const accepted = {
        turn_id: result.turn_id || result.run_id || state.request.request_id,
        run_id: result.run_id || null, user: message, output: null,
        status: result.status || (result.queued ? "queued" : "loading"),
        active: !result.queued && ["loading", "running"].includes(result.status || "loading"),
        events: [], agents: [], approvals: [], artifacts: [], created: Date.now() / 1000,
        messages: [{ role: "user", content: message }],
      };
      if (!state.detail) state.detail = {
        chat: { chat_id: chatId, title: message.split("\n")[0].slice(0, 100), policy: state.policy, status: accepted.status },
        turns: [], backend: state.overview?.backend,
      };
      if (!state.detail.turns.some((turn) => turn.turn_id === accepted.turn_id)) state.detail.turns.push(accepted);
      if (!state.acknowledged.has(chatId)) state.acknowledged.set(chatId, new Map());
      state.acknowledged.get(chatId).set(accepted.turn_id, accepted);
      persistDrafts();
      state.loading = false;
      if (state.overview && !state.overview.chats.some((chat) => chat.chat_id === chatId)) state.overview.chats.unshift({
        chat_id: chatId, title: state.detail.chat.title, status: accepted.status, updated: Date.now() / 1000,
      });
      state.request = null;
      // Show the acknowledged durable prompt even if the following GET loses its connection.
      state.sending = false;
      renderOverview(); renderTranscript(); scrollBottom();
      await refresh();
      scrollBottom();
    } catch (error) {
      state.sendError = `${error.message} Your draft is preserved. Send again to retry safely.`;
      // Reuse request_id for the same message, so an accepted request is never duplicated.
    } finally {
      state.sending = false;
      renderComposer();
      input.focus({ preventScroll: true });
      scheduleRefresh();
    }
  }
  async function cancelResponse() {
    if (!state.selected || state.cancelling || !active()) return;
    state.cancelling = true;
    renderComposer();
    try {
      await api(`/api/chats/${encodeURIComponent(state.selected)}/cancel`, {});
      await refresh();
    } catch (error) { toast(error.message, true, 6000); }
    finally { state.cancelling = false; renderComposer(); $("message").focus({ preventScroll: true }); }
  }
  async function decide(button) {
    const id = button.dataset.approval;
    if (state.busyApprovals.has(id)) return;
    state.busyApprovals.add(id);
    renderTranscript();
    try {
      await api(`/api/approvals/${encodeURIComponent(id)}`, { approved: button.dataset.approved === "true" });
      await refresh();
    } catch (error) { toast(error.message, true, 6000); }
    finally { state.busyApprovals.delete(id); renderTranscript(); }
  }
  async function copy(text, button) {
    try {
      if (!navigator.clipboard?.writeText) throw new Error("Clipboard access is unavailable. Select the text to copy it.");
      await navigator.clipboard.writeText(text);
      if (button) {
        const old = button.textContent;
        button.textContent = "Copied";
        setTimeout(() => { if (button.isConnected) button.textContent = old; }, 1800);
      } else toast("Copied.");
    } catch (error) { toast(error.message, true); }
  }
  async function downloadArtifact(button) {
    button.disabled = true;
    try {
      const result = await api(`/api/runs/${encodeURIComponent(button.dataset.run)}/artifacts/${encodeURIComponent(button.dataset.artifact)}`);
      if (typeof result.content !== "string") throw new Error("This file could not be downloaded.");
      const url = URL.createObjectURL(new Blob([result.content], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a");
      link.href = url;
      link.download = String(result.path || button.dataset.artifact).split(/[\\/]/).at(-1) || "artifact.txt";
      document.body.appendChild(link);
      link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { toast(error.message, true); }
    finally { button.disabled = false; }
  }
  const commands = [
    { name: "/new", description: "Start a fresh chat", run: () => selectChat(null) },
    { name: "/clear", description: "Start with a fresh context", run: () => selectChat(null) },
    { name: "/settings", description: "Permissions and workspace", run: () => $("settings").showModal() },
    { name: "/status", description: "Model and chat status", run: () => toast(`${modelLabel()} · ${statusLabel((state.detail?.backend || state.overview?.backend)?.status)} · ${currentPolicy() === "read_only" ? "Read only" : "Review changes"}`, false, 5500) },
    { name: "/help", description: "Keyboard shortcuts", run: () => toast("Enter sends · Shift + Enter adds a line · Esc stops · ⌘ K starts a new chat", false, 6500) },
  ];
  function matchingCommands() {
    const value = $("message").value.trim();
    if (!/^\/[a-z]*$/i.test(value)) return [];
    return commands.filter((command) => command.name.startsWith(value.toLowerCase()));
  }
  function renderCommands() {
    const list = matchingCommands();
    state.commandIndex = Math.max(0, Math.min(state.commandIndex, list.length - 1));
    $("command-menu").hidden = !list.length || state.sending;
    patch($("command-menu"), list.map((command, i) => `<button type="button" class="command-option" role="option" aria-selected="${i === state.commandIndex}" data-command="${escape(command.name)}"><strong>${escape(command.name)}</strong><span>${escape(command.description)}</span></button>`).join(""));
  }
  function runCommand(value) {
    const command = commands.find((entry) => entry.name === value);
    if (!command) return false;
    $("message").value = "";
    saveDraft();
    $("command-menu").hidden = true;
    command.run();
    renderComposer();
    return true;
  }
  $("composer").addEventListener("submit", (event) => { event.preventDefault(); sendMessage(); });
  $("message").addEventListener("input", () => {
    saveDraft(); state.commandIndex = 0; state.sendError = ""; renderComposer(); renderCommands();
  });
  $("message").addEventListener("keydown", (event) => {
    if (event.isComposing) return;
    const menuOpen = !$("command-menu").hidden;
    if (menuOpen && ["ArrowUp", "ArrowDown"].includes(event.key)) {
      event.preventDefault();
      const list = matchingCommands();
      state.commandIndex = (state.commandIndex + (event.key === "ArrowDown" ? 1 : -1) + list.length) % list.length;
      renderCommands(); return;
    }
    if (menuOpen && event.key === "Tab") {
      event.preventDefault();
      $("message").value = matchingCommands()[state.commandIndex]?.name || $("message").value;
      renderCommands(); return;
    }
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (menuOpen) runCommand(matchingCommands()[state.commandIndex]?.name);
      else sendMessage();
    }
  });
  document.addEventListener("click", (event) => {
    const chat = event.target.closest("[data-chat]");
    if (chat) { selectChat(chat.dataset.chat); return; }
    const starter = event.target.closest("[data-starter]");
    if (starter) { $("message").value = starter.dataset.starter; saveDraft(); renderComposer(); $("message").focus(); return; }
    const approval = event.target.closest("[data-approval]");
    if (approval) { decide(approval); return; }
    const command = event.target.closest("[data-command]");
    if (command) { runCommand(command.dataset.command); return; }
    const code = event.target.closest("[data-copy-code]");
    if (code) { copy(code.closest(".code-block").querySelector("code").textContent, code); return; }
    const response = event.target.closest("[data-copy-response]");
    if (response) {
      const turn = state.detail?.turns?.find((entry, i) => String(entry.turn_id || entry.run_id || `turn-${i}`) === response.dataset.copyResponse);
      copy(outputText(turn?.output), response); return;
    }
    const artifact = event.target.closest("[data-artifact]");
    if (artifact) downloadArtifact(artifact);
  });
  $("toggle-sidebar").addEventListener("click", () => toggleSidebar());
  $("close-sidebar").addEventListener("click", () => { toggleSidebar(false); $("toggle-sidebar").focus(); });
  $("sidebar-backdrop").addEventListener("click", () => toggleSidebar(false));
  $("new-chat").addEventListener("click", () => selectChat(null));
  $("home").addEventListener("click", () => selectChat(null));
  $("open-settings").addEventListener("click", () => { renderSettings(); $("settings").showModal(); });
  $("permission-shortcut").addEventListener("click", () => { renderSettings(); $("settings").showModal(); });
  $("close-settings").addEventListener("click", () => $("settings").close());
  $("settings").addEventListener("click", (event) => {
    const bounds = $("settings").getBoundingClientRect();
    if (event.target === $("settings") && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) $("settings").close();
  });
  document.querySelectorAll('input[name="policy"]').forEach((input) => input.addEventListener("change", () => { state.policy = input.value; state.policyTouched = true; persistDrafts(); renderComposer(); renderSettings(); }));
  $("stop-draft").addEventListener("click", cancelResponse);
  $("retry").addEventListener("click", refresh);
  $("jump-bottom").addEventListener("click", scrollBottom);
  $("transcript-pane").addEventListener("scroll", () => { $("jump-bottom").hidden = nearBottom(); }, { passive: true });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Tab" && state.sidebar && matchMedia("(max-width:700px)").matches) {
      const buttons = Array.from($("sidebar").querySelectorAll("a[href],button:not(:disabled)"));
      const first = buttons[0]; const last = buttons.at(-1);
      if (event.shiftKey && (document.activeElement === first || !$("sidebar").contains(document.activeElement))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if ($("settings").open) $("settings").close();
      selectChat(null); return;
    }
    if (event.key === "Escape" && !$("settings").open) {
      if (!$("command-menu").hidden) { $("command-menu").hidden = true; event.preventDefault(); }
      else if (state.sidebar) { toggleSidebar(false); event.preventDefault(); }
      else if (active()) { cancelResponse(); event.preventDefault(); }
    }
  });
  window.addEventListener("resize", () => { if (state.sidebar) toggleSidebar(true); resizeComposer(); });
  window.addEventListener("focus", () => refresh());
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  async function boot() {
    const selected = new URLSearchParams(location.search).get("chat");
    if (selected) { state.selected = selected; state.loading = true; $("welcome").hidden = true; $("view-loading").hidden = false; }
    renderComposer();
    const fragment = new URLSearchParams(location.hash.slice(1));
    const token = fragment.get("token");
    if (token) {
      history.replaceState(null, "", location.pathname + location.search);
      try { await api("/api/auth", { token }); }
      catch { setConnection("This session could not be authenticated. Reopen the Forge link from the local launcher."); return; }
    }
    await refresh();
  }
  boot();
})();

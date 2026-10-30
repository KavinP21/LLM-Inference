/* Instinct: a conversation surface over the durable local harness. */
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const terminal = new Set(["completed", "completed_with_failures", "failed", "cancelled", "interrupted"]);
  const working = new Set(["running", "loading", "queued", "waiting", "waiting_approval"]);
  const state = {
    overview: null, detail: null, selected: null, loading: false, refreshing: false,
    sending: false, cancelling: false, connected: false, sidebar: false,
    policy: "review", drafts: new Map(), turns: new Map(), busyApprovals: new Set(),
    requests: new Map(), sendError: "", signatures: new WeakMap(), commandIndex: 0,
    draftStorage: null, acknowledged: new Map(), policyTouched: false,
    models: [], live: null, liveVersion: null, stream: null, streaming: false, chatQuery: "",
    newChat: { model: null, effort: "off" },
    git: { status: null, loading: false, updated: 0, selected: null, diff: null, diffLoading: false, diffVersion: 0 },
    mediaDrafts: new Map(), imageCache: new Map(), imageRequests: new Map(), inFlightMedia: new Set(),
    dev: { open: false, tab: "agents", data: null, error: "", loading: false, loadingRequest: false, version: 0,
      refreshAgain: false, selectedAgent: null, agentScope: "latest", messageBusy: false, agentDrafts: new Map(),
      files: [], filesLoaded: false, filesLoading: false, filesError: "", fileQuery: "", filesVersion: 0, filesTruncated: false,
      selectedFile: null, file: null, fileLoading: false, fileVersion: 0, selectedChange: null, selectedActivity: null, activityFilter: "all" },
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
  // Every refresh re-renders the whole transcript; most of it hasn't changed.
  const rendered = new Map();
  function markdown(value, depth = 0) {
    if (depth) return renderMarkdown(value, depth);
    const key = String(value || "");
    let html = rendered.get(key);
    if (html === undefined) {
      html = renderMarkdown(key, 0);
      if (rendered.size > 500) rendered.delete(rendered.keys().next().value);
      rendered.set(key, html);
    }
    return html;
  }
  function renderMarkdown(value, depth) {
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
    try { response = await fetch(path, options); } catch { throw new Error("Could not reach Instinct. Check that the local server is running."); }
    let body;
    try { body = await response.json(); } catch { throw new Error(`The server returned an unreadable response (${response.status}).`); }
    if (!response.ok) {
      const error = body.error || body.message || `Request failed (${response.status}).`;
      throw new Error(typeof error === "string" ? error : error.message || "The request could not be completed.");
    }
    return body;
  }
  const policyNames = { review: "Review changes", auto: "Work freely", read_only: "Read only" };
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
  function persistPendingRequests() {
    if (!state.draftStorage) return;
    try { sessionStorage.setItem(`${state.draftStorage}:requests`, JSON.stringify(Array.from(state.requests).slice(-60))); }
    catch { /* The in-memory nonce still protects retries in this page. */ }
  }
  function loadDrafts(workspace) {
    if (state.draftStorage) return;
    state.draftStorage = `instinct:drafts:${typeof workspace === "string" ? workspace : workspace?.path || workspace?.root || "workspace"}`;
    try {
      const policy = sessionStorage.getItem(`${state.draftStorage}:policy`);
      if (!state.policyTouched && ["review", "auto", "read_only"].includes(policy)) state.policy = policy;
      const pending = JSON.parse(sessionStorage.getItem(`${state.draftStorage}:requests`) || "[]");
      if (Array.isArray(pending)) for (const entry of pending.slice(-60)) {
        if (Array.isArray(entry) && entry.length === 2 && typeof entry[0] === "string" && entry[0].length <= 24000 && typeof entry[1] === "string" && /^[a-zA-Z0-9_-]{1,128}$/.test(entry[1]) && !state.requests.has(entry[0])) state.requests.set(entry[0], entry[1]);
      }
      const pane = JSON.parse(sessionStorage.getItem(`${state.draftStorage}:developer`) || "null");
      if (pane && typeof pane.open === "boolean" && ["agents", "files", "changes", "activity"].includes(pane.tab)) toggleDeveloper(pane.open, pane.tab);
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
  function chosenModel() {
    const id = state.detail?.chat?.model || state.newChat.model;
    return state.models.find((model) => model.id === id) || state.models[0];
  }
  function modelLabel(backend = state.detail?.backend || state.overview?.backend || {}) {
    if (state.models.length && backend.status !== "loading") return chosenModel()?.label || "Local model";
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
    renderSelects();
    $("demo-badge").hidden = overview.mode !== "demo";
    $("demo-note").hidden = overview.mode !== "demo";
    const query = state.chatQuery.trim().toLowerCase();
    const chats = (overview.chats || []).filter((chat) => !query || (chat.title || "").toLowerCase().includes(query));
    patch($("chat-list"), chats.length ? chats.map((chat) => `<div class="chat-row${chat.chat_id === state.selected ? " selected" : ""}"><button type="button" class="chat-item${chat.chat_id === state.selected ? " selected" : ""}" data-chat="${escape(chat.chat_id)}"${chat.chat_id === state.selected ? ' aria-current="true"' : ""}><span class="chat-title">${escape(chat.title || "New chat")}</span><span class="chat-meta">${working.has(chat.status) ? `<span class="status-dot ${chat.status === "loading" ? "loading" : "running"}" aria-hidden="true"></span>${chat.status === "loading" ? "Loading model" : "Working"} · ` : ""}${escape(dateLabel(chat.updated || chat.created))}</span></button><span class="chat-actions"><button type="button" class="chat-action" data-rename-chat="${escape(chat.chat_id)}" aria-label="Rename ${escape(chat.title || "chat")}" title="Rename">✎</button><button type="button" class="chat-action" data-delete-chat="${escape(chat.chat_id)}" aria-label="Delete ${escape(chat.title || "chat")}" title="Delete">✕</button></span></div>`).join("") : `<p class="sidebar-empty">${query ? "No chats match your search." : "Your chats will appear here."}</p>`);
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
  function failedToolLabel(name, args, status = "failed") {
    const action = toolLabel(name, args).replace(/^(Read|Explored|Searched|Updated|Edited|Created|Ran|Checked|Reviewed|Used)\b/, (word) => ({
      Read: "Read", Explored: "Explore", Searched: "Search", Updated: "Update", Edited: "Edit", Created: "Create", Ran: "Run", Checked: "Check", Reviewed: "Review", Used: "Use",
    })[word]);
    return `${action} — ${status === "denied" ? "declined" : status === "cancelled" ? "stopped" : "failed"}`;
  }
  function toolFailed(value) {
    if (typeof value === "string") {
      if (/^(?:Tool failed:|Error\b)/i.test(value.trim())) return true;
      try { value = JSON.parse(value); } catch { return false; }
    }
    return Boolean(value && typeof value === "object" && (value.error != null || value.success === false || (typeof value.exit_code === "number" && value.exit_code !== 0)));
  }
  function renderToolRows(rows) {
    return rows.slice(-80).map((row) => {
      const failed = toolFailed(row.result);
      const command = row.args.command || row.args.cmd;
      return `<details class="tool-row${failed ? " failed" : ""}" data-key="${escape(row.key)}"><summary><span class="tool-icon" aria-hidden="true">${row.pending ? "•" : failed ? "!" : "✓"}</span><span class="tool-title">${escape(failed ? failedToolLabel(row.name, row.args) : row.pending ? currentToolLabel(row.name, row.args) : toolLabel(row.name, row.args))}${command ? ` · ${escape(String(command).split("\n")[0].slice(0, 90))}` : ""}</span><span class="disclosure-chevron" aria-hidden="true">›</span></summary><div class="tool-detail"><h4>Request</h4><pre>${escape(pretty(row.args))}</pre>${row.result != null ? `<h4>Result</h4><pre>${escape(pretty(row.result))}</pre>` : '<p>Working…</p>'}</div></details>`;
    }).join("");
  }
  function activityMarkup(turn) {
    const rows = toolRows(turn);
    if (!rows.length) return "";
    const done = rows.filter((row) => !row.pending && !toolFailed(row.result));
    const count = (names) => done.filter((row) => names.includes(row.name)).length;
    const plural = (n, word) => `${n} ${word}${n === 1 ? "" : "s"}`;
    const edits = count(["write_file", "replace_text", "edit_lines", "make_directory", "move_path", "delete_path"]);
    const commands = count(["run_command"]);
    const descriptions = [
      count(["read_file", "list_files", "search"]) && "explored the codebase",
      count(["web_search", "fetch_url"]) && "searched the web",
      edits && `made ${plural(edits, "change")}`,
      commands && `ran ${plural(commands, "command")}`,
      count(["run_tests"]) && "ran tests",
    ].filter(Boolean);
    if (descriptions.length) descriptions[0] = descriptions[0][0].toUpperCase() + descriptions[0].slice(1);
    const summary = rows.at(-1).pending ? currentToolLabel(rows.at(-1).name, rows.at(-1).args) + "…" : descriptions.join(" · ") || `${rows.length} ${rows.length === 1 ? "action" : "actions"}`;
    return `<details class="activity-group" data-key="activity-${escape(turn.turn_id || turn.run_id)}">${disclosureSummary(summary, descriptions.length ? `${rows.length} ${rows.length === 1 ? "action" : "actions"}` : "", rows.at(-1).pending)}<div class="activity-rows">${renderToolRows(rows)}${rows.length > 80 ? '<p class="settings-note">Earlier actions are available in the conversation trace.</p>' : ""}</div></details>`;
  }
  function agentsMarkup(turn) {
    const agents = (turn.agents || []).filter((agent) => agent.parent_id || agent.agent_id !== "root");
    if (!agents.length) return "";
    const activeCount = agents.filter((agent) => !terminal.has(agent.status) && !["failed", "cancelled"].includes(agent.status)).length;
    return `<details class="agents-group" data-key="agents-${escape(turn.turn_id || turn.run_id)}">${disclosureSummary(`${agents.length} ${agents.length === 1 ? "agent" : "agents"}`, activeCount ? `${activeCount} working` : "Finished", activeCount > 0 && turn.active)}<div class="agents-list">${agents.map((agent) => `<details class="agent-card" data-key="agent-${escape(agent.agent_id)}"><summary><span class="disclosure-chevron" aria-hidden="true">›</span><span class="status-dot ${terminal.has(agent.status) ? agent.status : "running"}" aria-hidden="true"></span><span class="agent-name">${escape(agent.role || "Agent")}</span><span class="agent-hint">${escape(agent.task || "")}</span><span class="agent-state">${escape(terminal.has(agent.status) ? agent.status === "completed" ? "Done" : agent.status === "cancelled" ? "Stopped" : "Needs attention" : "Working")}</span></summary><div class="agent-detail"><p class="agent-task">${escape(agent.task || "")}</p>${renderToolRows(toolRows(turn, agent.agent_id))}${terminal.has(agent.status) ? "" : liveMarkup(turn, agent.agent_id)}${agent.result ? `<div class="markdown">${markdown(outputText(agent.result))}</div>` : ""}${agent.error ? `<p class="turn-error">${escape(agent.error)}</p>` : ""}</div></details>`).join("")}</div></details>`;
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
      return `<article class="approval-card"><div class="approval-heading"><h3>${heading}</h3><p>${escape(command ? "Instinct wants to run a command in this workspace." : hasDiff ? preview.path || args.path || "Proposed file change" : toolLabel(entry.tool, args))}</p></div>${command ? `<pre class="approval-command">${escape(command)}</pre>` : ""}${hasDiff ? `<div class="approval-preview"><div class="diff-path">${escape(preview.path || args.path || "File changes")}</div><pre class="diff">${diffMarkup(preview.diff)}</pre></div>` : ""}<details class="approval-arguments" data-key="approval-${escape(entry.id)}">${disclosureSummary("Exact action details")}<pre>${escape(entry.tool)}\n${escape(pretty(args))}</pre></details><div class="approval-actions"><button type="button" class="button" data-approval="${escape(entry.id)}" data-approved="false"${state.busyApprovals.has(entry.id) ? " disabled" : ""}>Decline</button><button type="button" class="button button-primary" data-approval="${escape(entry.id)}" data-approved="true"${state.busyApprovals.has(entry.id) ? " disabled" : ""}>${state.busyApprovals.has(entry.id) ? "Applying…" : "Approve"}</button></div></article>`;
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
    node.innerHTML = '<div class="user-messages"></div><div class="assistant-turn"><div class="assistant-label"><span class="assistant-mark"><svg viewBox="0 0 30 49" aria-hidden="true"><use href="#instinct-mark"/></svg></span>Instinct</div><div class="turn-activity"></div><div class="turn-agents"></div><div class="turn-approvals"></div><div class="turn-directions user-messages"></div><div class="turn-progress" role="status"></div><div class="turn-live" aria-live="off"></div><div class="turn-output"></div><div class="turn-error"></div><div class="turn-artifacts"></div></div>';
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
    if (!state.detail) { renderComposer(); renderDeveloper(); return; }
    const keys = new Set();
    turns.forEach((turn, index) => {
      const key = String(turn.turn_id || turn.run_id || `turn-${index}`);
      keys.add(key);
      const node = state.turns.get(key) || createTurn(key);
      if (!node.isConnected) $("transcript").appendChild(node);
      const messages = userMessages(turn);
      const editing = state.editing === turn.turn_id;
      const actions = turn.turn_id && !editing ? `<div class="message-actions"><button type="button" data-edit-turn="${escape(turn.turn_id)}">Edit</button><button type="button" data-rewind-turn="${escape(turn.turn_id)}">Rewind</button></div>` : "";
      const bubble = editing
        ? `<form class="message-editor" data-edit-form="${escape(turn.turn_id)}"><textarea rows="3">${escape(messages[0] || "")}</textarea>${turn.checkpoint ? '<label><input type="checkbox" name="restore" checked> Also restore files to before this message</label>' : ""}<div class="dialog-actions"><button class="text-button" type="button" data-edit-cancel>Cancel</button><button class="primary-button" type="submit">Send</button></div></form>`
        : messages[0] ? `<div class="user-message">${escape(messages[0])}</div>` : "";
      patch(node.querySelector(".user-messages"), `${bubble}${editing ? "" : mediaMarkup(turn.media || turn.messages?.[0]?.media)}${actions}`);
      patch(node.querySelector(".turn-directions"), messages.slice(1).map((message) => `<div class="user-message steering">${escape(message)}</div>`).join(""));
      patch(node.querySelector(".turn-activity"), activityMarkup(turn));
      patch(node.querySelector(".turn-agents"), agentsMarkup(turn));
      patch(node.querySelector(".turn-approvals"), approvalsMarkup(turn));
      const progress = turnProgress(turn);
      patch(node.querySelector(".turn-progress"), progress ? `${turn.active && !turn.approvals?.length ? '<span class="working-dot" aria-hidden="true"></span>' : ""}${escape(progress)}` : "");
      patch(node.querySelector(".turn-live"), liveMarkup(turn));
      const output = outputText(turn.output);
      const thought = output ? finalThought(turn) : "";
      patch(node.querySelector(".turn-output"), output ? `${thought ? `<details class="live-thinking" data-key="thought-${escape(key)}">${disclosureSummary("Thought")}<p>${escape(thought)}</p></details>` : ""}<div class="markdown">${markdown(output)}</div><div class="output-actions"><button type="button" class="copy-response" data-copy-response="${escape(key)}">Copy response</button>${turn.status === "completed_with_failures" ? "Some delegated work needs attention." : ""}</div>` : "");
      patch(node.querySelector(".turn-error"), turn.error && !["cancelled", "interrupted"].includes(turn.status) ? `<p>${escape(turn.run_id ? "Instinct couldn’t finish this request. You can send a message to continue." : "The model couldn’t start. Your message is saved; send it again to retry.")}</p><details data-key="error-${escape(key)}">${disclosureSummary("Error details")}<pre>${escape(typeof turn.error === "string" ? turn.error : pretty(turn.error))}</pre></details>` : "");
      patch(node.querySelector(".turn-artifacts"), turn.run_id && turn.artifacts?.length ? `<div class="artifact-list">${turn.artifacts.map((artifact) => `<button type="button" class="artifact-button" data-artifact="${escape(artifact.path)}" data-run="${escape(turn.run_id)}" title="Download ${escape(artifact.path)}"><span aria-hidden="true">▤</span><span>${escape(artifact.path)}</span><span aria-hidden="true">↓</span></button>`).join("")}</div>` : "");
    });
    for (const [key, node] of state.turns) if (!keys.has(key)) { node.remove(); state.turns.delete(key); }
    renderStatusBar(); renderSelects();
    renderComposer();
    renderSettings(); renderDeveloper(); observeImages();
    document.title = state.detail.chat?.title ? `${state.detail.chat.title} — Instinct` : "Instinct";
    if (follow) scrollBottom();
    else $("jump-bottom").hidden = nearBottom();
  }
  // Streaming text arrives raw: hide thinking blocks and native tool calls.
  function liveView(raw) {
    let text = String(raw || "");
    let thinking = "";
    const thought = /(?:<\|channel>thought\n?|<think>)([\s\S]*?)(?:<channel\|>|<\/think>|$)/.exec(text);
    if (thought) { thinking = thought[1]; text = text.replace(thought[0], ""); }
    const call = /(<\|tool_call>call:([\w-]+)|<tool_call>\s*(?:<function=([\w-]+)|\{\s*"name"\s*:\s*"([\w-]+)))/.exec(text);
    const tool = call ? call[2] || call[3] || call[4] : null;
    if (call) text = text.slice(0, call.index);
    return { text: text.trim(), thinking: thinking.trim(), tool, thinkingOpen: Boolean(thought) && !/<channel\|>|<\/think>/.test(raw) };
  }
  function finalThought(turn) {
    const responses = (turn.events || []).filter((event) => event.kind === "model_response" && (event.agent_id || "root") === "root");
    return (responses.at(-1)?.payload?.reasoning || "").trim();
  }
  function liveFor(runId, agentId = "root") {
    return (state.live?.generations || []).find((entry) => entry.run_id === runId && entry.agent_id === agentId);
  }
  function liveMarkup(turn, agentId = "root") {
    if (!turn.active || !turn.run_id) return "";
    const generation = liveFor(turn.run_id, agentId);
    if (!generation) return "";
    const view = liveView(generation.text);
    const parts = [];
    if (view.thinking) parts.push(`<details class="live-thinking"${view.thinkingOpen ? " open" : ""}><summary>${view.thinkingOpen ? "Thinking…" : "Thought"}</summary><p>${escape(view.thinking.slice(-1500))}</p></details>`);
    if (view.text) parts.push(`<div class="markdown live-text">${renderMarkdown(view.text, 0)}</div>`);
    if (view.tool) parts.push(`<p class="live-tool"><span class="working-dot" aria-hidden="true"></span>Calling ${escape(view.tool.replace(/_/g, " "))}…</p>`);
    return parts.join("");
  }
  function generations(turns = state.detail?.turns || []) {
    const rows = [];
    for (const turn of turns) for (const event of turn.events || []) {
      if (event.kind !== "generation") continue;
      rows.push({ turn: turn.turn_id, agent: event.agent_id || "root", created: event.created, ...event.payload, metrics: event.payload?.metrics || {} });
    }
    return rows;
  }
  function shortModel(id) {
    const name = String(id || "").split(/[\\/]/).at(-1).replace(/-mlx$/, "");
    if (/^gemma-4/i.test(name)) return "Gemma 4";
    if (/^qwen3\.5/i.test(name)) return "Qwen3.5";
    return name;
  }
  function compact(value) {
    if (value == null) return "–";
    return value >= 1000 ? `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}K` : String(Math.round(value));
  }
  function contextUsage() {
    const rows = generations().filter((row) => row.agent === "root");
    const last = rows.at(-1);
    const limit = state.detail?.chat?.context_limit || chosenModel()?.context;
    if (!last || !limit) return null;
    return { used: (last.input_tokens || 0) + (last.output_tokens || 0), limit };
  }
  function renderStatusBar() {
    const usage = contextUsage();
    $("context-meter").hidden = !usage;
    if (usage) {
      const share = Math.min(1, usage.used / usage.limit);
      $("context-fill").style.width = `${Math.max(2, share * 100)}%`;
      $("context-fill").className = `meter-fill${share > 0.85 ? " high" : share > 0.6 ? " mid" : ""}`;
      $("context-label").textContent = `${compact(usage.used)} / ${compact(usage.limit)}`;
      $("context-meter").title = `Context window: ${usage.used.toLocaleString()} of ${usage.limit.toLocaleString()} tokens used by the last reply`;
    }
    const busy = (state.live?.generations || []).filter((entry) => entry.text);
    const workers = (state.live?.workers || []).filter((worker) => worker.state === "busy");
    $("live-speed").hidden = !busy.length;
    $("live-speed").textContent = busy.length ? `${workers.length || busy.length} generating` : "";
  }
  const EFFORTS = [
    ["off", "Fast", "Answers right away without thinking first. Best for quick questions and simple edits."],
    ["low", "Low", "Thinks briefly before answering, up to about a thousand tokens."],
    ["medium", "Medium", "Thinks things through, up to about four thousand tokens. A good default for real work."],
    ["high", "High", "Thinks for as long as it needs. Best for hard bugs and design questions; slowest."],
  ];
  const MODEL_NOTES = { driver: "Strongest. Hands work to sub-agents.", subagent: "Fastest and lightest." };
  function shortLabel(label) { return String(label || "").replace(/\s+\d+B.*$/, "").replace(/-\d+B$/, ""); }
  function currentMode() {
    const chat = state.detail?.chat;
    return {
      model: chat?.model || state.newChat.model || state.models[0]?.id,
      effort: chat?.effort || state.newChat.effort || "off",
    };
  }
  function renderSelects() {
    const { model, effort } = currentMode();
    const demo = state.overview?.mode === "demo";
    const level = Math.max(0, EFFORTS.findIndex(([value]) => value === effort));
    const chosen = state.models.find((entry) => entry.id === model) || state.models[0];
    $("mode-picker").hidden = demo && state.models.length < 2;
    $("mode-model").textContent = chosen ? shortLabel(chosen.label) : "Model";
    $("mode-effort").textContent = demo ? "" : EFFORTS[level][1];
    $("mode-models-section").hidden = state.models.length < 2;
    $("mode-effort-section").hidden = demo;
    patch($("mode-models"), state.models.map((entry) => `<button type="button" role="radio" class="mode-model-option" data-model="${escape(entry.id)}" aria-checked="${entry.id === chosen?.id}"><span><strong>${escape(entry.label)}</strong><small>${escape([MODEL_NOTES[entry.id], entry.context ? `${Math.round(entry.context / 1024)}K context` : ""].filter(Boolean).join(" · "))}</small></span><svg viewBox="0 0 20 20" aria-hidden="true"><path d="M5 10.5l3 3 7-7"></path></svg></button>`).join(""));
    if (document.activeElement !== $("effort-slider")) showEffort(level);
  }
  function showEffort(level) {
    const slider = $("effort-slider");
    slider.value = String(level);
    slider.style.setProperty("--fill", `${(level / 3) * 100}%`);
    slider.setAttribute("aria-valuetext", EFFORTS[level][1]);
    $("mode-level").textContent = EFFORTS[level][1];
    $("effort-help").textContent = EFFORTS[level][2];
    document.querySelectorAll(".effort-ticks span").forEach((tick, index) => tick.classList.toggle("active", index === level));
  }
  function toggleModePanel(open = $("mode-panel").hidden) {
    $("mode-panel").hidden = !open;
    $("mode-button").setAttribute("aria-expanded", String(open));
    if (open) (document.querySelector('.mode-model-option[aria-checked="true"]') || $("effort-slider")).focus();
  }
  async function changeSetting(key, value) {
    if (!state.selected) {
      state.newChat[key] = value;
      try { localStorage.setItem("instinct-new-chat", JSON.stringify(state.newChat)); } catch {}
      renderOverview(); return;
    }
    try {
      const result = await api(`/api/chats/${encodeURIComponent(state.selected)}/settings`, { [key]: value });
      if (state.detail?.chat) Object.assign(state.detail.chat, result);
      toast(key === "model" ? `Next replies use ${chosenModel()?.label || value}` : `Reasoning set to ${EFFORTS.find(([name]) => name === value)?.[1] || value}`);
      renderOverview();
    } catch (error) { toast(error.message, true); renderSelects(); }
  }
  function renameChat(id) {
    const item = document.querySelector(`.chat-item[data-chat="${CSS.escape(id)}"]`);
    const title = item?.querySelector(".chat-title");
    if (!title) return;
    const input = document.createElement("input");
    input.className = "chat-rename";
    input.value = title.textContent;
    input.setAttribute("aria-label", "Chat title");
    item.hidden = true;
    item.before(input);
    input.focus(); input.select();
    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const value = input.value.trim();
      if (save && value && value !== title.textContent) {
        try { await api(`/api/chats/${encodeURIComponent(id)}/rename`, { title: value }); }
        catch (error) { toast(error.message, true); }
      }
      input.remove();
      item.hidden = false;
      refresh();
    };
    input.addEventListener("click", (event) => event.stopPropagation());
    input.addEventListener("keydown", (event) => {
      event.stopPropagation();
      if (event.key === "Enter") { event.preventDefault(); finish(true); }
      if (event.key === "Escape") { event.preventDefault(); finish(false); }
    });
    input.addEventListener("blur", () => finish(true));
  }
  async function deleteChat(id, button) {
    // First click arms the button; a second click within a few seconds deletes.
    if (button.dataset.armed !== "true") {
      button.dataset.armed = "true";
      button.textContent = "Delete?";
      button.classList.add("armed");
      setTimeout(() => { if (button.isConnected) { button.dataset.armed = ""; button.textContent = "✕"; button.classList.remove("armed"); } }, 3000);
      return;
    }
    try {
      await api(`/api/chats/${encodeURIComponent(id)}/delete`, {});
      state.drafts.delete(id); persistDrafts();
      if (state.selected === id) { state.detail = null; state.selected = null; rememberSelection(); selectChat(null); }
      refresh(); toast("Chat deleted");
    } catch (error) { toast(error.message, true); }
  }
  function renderComposer() {
    const images = currentMedia();
    const encoding = images.filter((image) => image.processing).length;
    const hasDraft = Boolean($("message").value.trim()) || Boolean(images.length);
    const running = active();
    const stop = running && !hasDraft;
    $("send").disabled = state.sending || state.cancelling || Boolean(encoding) || (!running && !hasDraft);
    $("send").classList.toggle("stop", stop);
    $("send-label").textContent = state.sending ? "Sending…" : state.cancelling ? "Stopping…" : stop ? "Stop" : "Send";
    $("send-icon").textContent = stop ? "■" : "↑";
    $("send").setAttribute("aria-label", stop ? "Stop current response" : running ? "Send follow-up message" : "Send message");
    $("stop-draft").hidden = !running || !hasDraft;
    $("stop-draft").disabled = state.cancelling;
    $("composer-hint").textContent = running && hasDraft ? "Follow-up runs after this response" : "Enter to send · Shift + Enter for a new line";
    const policy = currentPolicy();
    $("permission-label").textContent = policyNames[policy] || policyNames.review;
    $("composer-footnote").textContent = state.overview?.mode === "demo" ? "Demo mode · no model inference is performed." : {
      read_only: "Instinct can read and explore this workspace. Files stay unchanged.",
      auto: "Instinct edits files and runs sandboxed commands in this workspace without asking.",
    }[policy] || "Instinct works in your local workspace. You review changes before they run.";
    $("new-chat").disabled = state.sending;
    const status = state.sendError || (encoding ? `Preparing ${encoding} ${encoding === 1 ? "image" : "images"}…` : state.sending ? "Sending your message…" : "");
    $("composer-status").textContent = status;
    $("composer-status").hidden = !status;
    $("composer-status").classList.toggle("error", Boolean(state.sendError));
    renderAttachments(); resizeComposer();
  }
  async function renderKeys() {
    $("api-base").textContent = `${location.origin}/v1`;
    try {
      const { keys } = await api("/api/keys");
      const used = (time) => (time ? `used ${new Date(time * 1000).toLocaleDateString()}` : "not used yet");
      patch($("key-list"), keys.length ? keys.map((key) => `<div class="key-row"><span><strong>${escape(key.name)}</strong><small>${used(key.last_used)}</small></span><button type="button" class="icon-button" data-revoke-key="${escape(key.id)}" aria-label="Revoke ${escape(key.name)}">×</button></div>`).join("") : `<p class="settings-note">No keys yet.</p>`);
    } catch (error) {
      patch($("key-list"), `<p class="settings-note">${escape(error.message)}</p>`);
    }
  }
  function renderSettings() {
    $("new-key").hidden = true;
    renderKeys();
    const existingPolicy = state.detail?.chat?.policy;
    $("policy-note").textContent = existingPolicy ? `This chat uses ${policyNames[existingPolicy] || policyNames.review}. Changes here apply to new chats.` : "Applies to new chats.";
    $("command-network").checked = Boolean(state.settings?.command_network);
    $("web-access").checked = state.settings?.web_access !== false;
    document.querySelectorAll('input[name="policy"]').forEach((input) => { input.checked = input.value === state.policy; });
    const turns = (state.detail?.turns || []).filter((turn) => turn.run_id);
    $("trace-section").hidden = !turns.length;
    patch($("trace-list"), turns.map((turn, index) => `<a href="/api/runs/${encodeURIComponent(turn.run_id)}/export" download="instinct-trace-${index + 1}.json">Download response ${index + 1} trace ↓</a>`).join(""));
  }
  function setConnection(error = "") {
    state.connected = !error;
    $("connection-error").hidden = !error;
    $("connection-error-text").textContent = error;
  }
  function toggleSidebar(open = !state.sidebar) {
    if (open && state.dev.open && matchMedia("(max-width:1100px)").matches) toggleDeveloper(false);
    state.sidebar = open;
    try { localStorage.setItem("instinct-sidebar", open ? "open" : "closed"); } catch {}
    $("app").classList.toggle("sidebar-open", open);
    $("sidebar").inert = !open;
    $("toggle-sidebar").setAttribute("aria-expanded", String(open));
    $("toggle-sidebar").setAttribute("aria-label", open ? "Close chats" : "Open chats");
    updateSurfaceInert();
    if (!open && $("sidebar").contains(document.activeElement)) $("toggle-sidebar").focus({ preventScroll: true });
  }
  async function selectChat(id) {
    if (state.sending) return;
    if (id === state.selected && state.detail) { $("message").focus(); return; }
    saveDraft();
    state.selected = id;
    setSelectedDeveloperAgent(null); state.dev.data = null; state.dev.error = ""; state.dev.version++;
    if (state.dev.open && matchMedia("(max-width:1100px)").matches) toggleDeveloper(false);
    rememberSelection();
    state.detail = null;
    state.loading = Boolean(id);
    state.turns.clear();
    $("transcript").replaceChildren();
    $("transcript-pane").scrollTop = 0;
    $("jump-bottom").hidden = true;
    $("message").value = state.drafts.get(draftKey()) || "";
    state.sendError = "";
    document.title = "Instinct";
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
      scheduleRefresh(); refreshDeveloper();
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
    if (state.streaming && state.connected) return;
    if (needsPolling() || !state.connected) pollTimer = setTimeout(refresh, document.hidden ? 4000 : state.connected ? 800 : 3000);
  }
  let liveRefreshTimer;
  function renderLive() {
    for (const turn of state.detail?.turns || []) {
      const node = state.turns.get(String(turn.turn_id || turn.run_id));
      if (node) patch(node.querySelector(".turn-live"), liveMarkup(turn));
    }
    renderStatusBar();
    if (state.dev.open && state.dev.tab === "system") renderDeveloperSystem();
    if (state.dev.open && state.dev.tab === "agents") renderDeveloperAgents();
    if (nearBottom()) scrollBottom();
  }
  function connectStream() {
    if (state.stream || typeof EventSource !== "function") return;
    const stream = new EventSource("/api/events");
    state.stream = stream;
    stream.onopen = () => { state.streaming = true; clearTimeout(pollTimer); };
    stream.onmessage = (event) => {
      let live;
      try { live = JSON.parse(event.data); } catch { return; }
      state.live = live;
      if (live.version !== state.liveVersion) {
        state.liveVersion = live.version;
        clearTimeout(liveRefreshTimer);
        liveRefreshTimer = setTimeout(refresh, 60);
      }
      renderLive();
    };
    stream.onerror = () => {
      // The browser reconnects on its own; poll until it does.
      state.streaming = false;
      scheduleRefresh();
    };
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
      renderOverview(); renderTranscript(); renderSettings(); refreshDeveloper(); refreshGitChanges();
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
    const draft = mediaDraft();
    if (draft.items.some((image) => image.processing)) return;
    const images = draft.items.slice();
    if (!message && !images.length) { if (active()) await cancelResponse(); else input.focus(); return; }
    if (!images.length && runCommand(message)) return;
    const selected = state.selected;
    const requestKey = `${selected || "new"}\u0000${selected ? "" : state.policy}\u0000${message}\u0000${JSON.stringify(images.map((image) => [image.sha256, image.name]))}`;
    if (!state.requests.has(requestKey)) state.requests.set(requestKey, nonce());
    const requestId = state.requests.get(requestKey);
    persistPendingRequests();
    for (const image of images) if (image.preview) state.inFlightMedia.add(image.preview);
    state.sending = true;
    state.sendError = "";
    $("command-menu").hidden = true;
    renderComposer();
    const request = selected ? { message, request_id: requestId } : { message, policy: state.policy, request_id: requestId };
    if (!selected) {
      if (state.newChat.model && state.models.some((model) => model.id === state.newChat.model)) request.model = state.newChat.model;
      if (state.newChat.effort && state.overview?.mode !== "demo") request.effort = state.newChat.effort;
    }
    if (images.length) {
      request.media = images.map((image) => ({ type: "image", mime: image.mime, sha256: image.sha256, data: image.data }));
      request.media_names = images.map((image) => image.name);
    }
    try {
      const result = await api(selected ? `/api/chats/${encodeURIComponent(selected)}/message` : "/api/chats", request);
      const chatId = result.chat_id || selected;
      if (!chatId) throw new Error("The server did not return a chat. Your message is still in the composer.");
      if (input.value === original) input.value = "";
      state.drafts.set(draftKey(selected), input.value);
      for (const image of images) {
        if (!state.imageCache.get(image.sha256)?.url) state.imageCache.set(image.sha256, { url: image.preview, mime: image.mime, size: image.size, name: image.name });
        else if (state.imageCache.get(image.sha256)?.url !== image.preview) URL.revokeObjectURL(image.preview);
      }
      const submitted = new Set(images.map((image) => image.id));
      draft.items = draft.items.filter((image) => !submitted.has(image.id));
      if (!selected) {
        state.selected = chatId;
        state.mediaDrafts.set(draftKey(chatId), draft);
        state.mediaDrafts.set("new", { items: [] });
        setSelectedDeveloperAgent(null); state.dev.data = null; state.dev.version++;
        rememberSelection();
        state.detail = null;
        state.turns.clear();
        $("transcript").replaceChildren();
        $("welcome").hidden = true;
        state.drafts.set(draftKey(chatId), input.value);
        state.drafts.set("new", "");
      }
      const accepted = {
        turn_id: result.turn_id || result.run_id || requestId,
        run_id: result.run_id || null, user: message || "Describe the attached image.", output: null,
        media: images.map((image) => ({ type: "image", mime: image.mime, sha256: image.sha256, name: image.name, size: image.size, url: `/api/chats/${encodeURIComponent(chatId)}/media/${image.sha256}` })),
        status: result.status || (result.queued ? "queued" : "loading"),
        active: !result.queued && ["loading", "running"].includes(result.status || "loading"),
        events: [], agents: [], approvals: [], artifacts: [], created: Date.now() / 1000,
        messages: [{ role: "user", content: message || "Describe the attached image." }],
      };
      if (!state.detail) state.detail = {
        chat: { chat_id: chatId, title: (message || "Attached image").split("\n")[0].slice(0, 100), policy: state.policy, status: accepted.status },
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
      state.requests.delete(requestKey);
      persistPendingRequests();
      // Show the acknowledged durable prompt even if the following GET loses its connection.
      state.sending = false;
      renderOverview(); renderTranscript(); scrollBottom();
      await refresh();
      scrollBottom();
    } catch (error) {
      state.sendError = `${error.message} Your draft is preserved. Send again to retry safely.`;
      // Reuse request_id for the same message, so an accepted request is never duplicated.
    } finally {
      for (const image of images) {
        state.inFlightMedia.delete(image.preview);
        if (image.preview && state.imageCache.get(image.sha256)?.url !== image.preview && !draft.items.some((remaining) => remaining.preview === image.preview)) URL.revokeObjectURL(image.preview);
      }
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
    renderTranscript(); renderDeveloper();
    try {
      await api(`/api/approvals/${encodeURIComponent(id)}`, { approved: button.dataset.approved === "true" });
      await refresh();
    } catch (error) { toast(error.message, true, 6000); }
    finally { state.busyApprovals.delete(id); renderTranscript(); renderDeveloper(); refreshDeveloper(); }
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
  function byteSize(bytes) {
    const size = Number(bytes || 0);
    return size >= 1048576 ? `${(size / 1048576).toFixed(1)} MB` : size >= 1024 ? `${(size / 1024).toFixed(1)} KB` : `${size} B`;
  }
  function mediaDraft(key = draftKey()) {
    if (!state.mediaDrafts.has(key)) state.mediaDrafts.set(key, { items: [] });
    return state.mediaDrafts.get(key);
  }
  function currentMedia() { return mediaDraft().items; }
  function renderAttachments() {
    const images = currentMedia();
    $("attachment-previews").hidden = !images.length;
    patch($("attachment-previews"), images.map((image) => `<div class="attachment-preview"><button type="button" data-preview-attachment="${escape(image.id)}" aria-label="View ${escape(image.name)}">${image.preview ? `<img src="${escape(image.preview)}" alt="${escape(image.name)}">` : '<span class="attachment-processing">Preparing…</span>'}</button><button type="button" class="attachment-remove" data-remove-attachment="${escape(image.id)}" aria-label="Remove ${escape(image.name)}">×</button><div class="attachment-preview-name" title="${escape(image.name)}">${escape(image.name)}</div></div>`).join(""));
  }
  function sniffImage(bytes) {
    if (bytes.length >= 8 && [137,80,78,71,13,10,26,10].every((n, i) => bytes[i] === n)) return "image/png";
    if (bytes.length >= 3 && bytes[0] === 255 && bytes[1] === 216 && bytes[2] === 255) return "image/jpeg";
    if (bytes.length >= 12 && String.fromCharCode(...bytes.slice(0, 4)) === "RIFF" && String.fromCharCode(...bytes.slice(8, 12)) === "WEBP") return "image/webp";
    if (bytes.length >= 2 && bytes[0] === 66 && bytes[1] === 77) return "image/bmp";
    return null;
  }
  function toBase64(bytes) {
    const chunks = [];
    for (let i = 0; i < bytes.length; i += 32768) chunks.push(String.fromCharCode(...bytes.subarray(i, i + 32768)));
    return btoa(chunks.join(""));
  }
  async function addImages(files) {
    const draft = mediaDraft();
    const accepted = [];
    const errors = [];
    for (const file of Array.from(files || [])) {
      if (draft.items.length >= 8) { errors.push("Attach up to 8 images per message."); break; }
      if (!file.size || file.size > 8 * 1048576) { errors.push(`${file.name || "Image"} must be between 1 byte and 8 MB.`); continue; }
      if (draft.items.reduce((total, image) => total + image.size, 0) + file.size > 16 * 1048576) { errors.push("Attached images must total 16 MB or less."); continue; }
      const name = String(file.name || "pasted-image.png").split(/[\\/]/).at(-1).replace(/[\u0000-\u001f\u007f]/g, "_").slice(0, 180);
      const image = { id: nonce(), name, size: file.size, processing: true, preview: null };
      draft.items.push(image);
      accepted.push({ file, image });
    }
    if (errors.length) toast(Array.from(new Set(errors)).join(" "), true, 6500);
    state.sendError = "";
    renderComposer();
    await Promise.all(accepted.map(async ({ file, image }) => {
      try {
        const bytes = new Uint8Array(await file.arrayBuffer());
        const mime = sniffImage(bytes);
        if (!mime) throw new Error(`${image.name} is not a supported PNG, JPEG, WebP or BMP image.`);
        if (!crypto.subtle?.digest) throw new Error("Image attachments require a secure local browser connection.");
        const sha256 = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", bytes)), (value) => value.toString(16).padStart(2, "0")).join("");
        if (!draft.items.includes(image)) return; // Removed while encoding, or draft discarded.
        image.mime = mime;
        image.sha256 = sha256;
        image.data = toBase64(bytes);
        image.preview = URL.createObjectURL(new Blob([bytes], { type: mime }));
        image.processing = false;
      } catch (error) {
        const index = draft.items.indexOf(image);
        if (index !== -1) draft.items.splice(index, 1);
        if (image.preview) URL.revokeObjectURL(image.preview);
        toast(error.message || "Could not prepare this image.", true, 6500);
      } finally {
        if (mediaDraft() === draft) renderComposer();
      }
    }));
  }
  function removeImage(id) {
    const draft = mediaDraft();
    const index = draft.items.findIndex((image) => image.id === id);
    if (index === -1) return;
    const [image] = draft.items.splice(index, 1);
    if (image.preview && !state.inFlightMedia.has(image.preview) && state.imageCache.get(image.sha256)?.url !== image.preview) URL.revokeObjectURL(image.preview);
    state.sendError = "";
    renderComposer();
    $("message").focus({ preventScroll: true });
  }
  function mediaMarkup(images) {
    if (!images?.length) return "";
    return `<div class="user-image-gallery">${images.map((image) => {
      const cached = state.imageCache.get(image.sha256);
      return `<button type="button" class="user-image" data-media-hash="${escape(image.sha256)}" data-media-url="${escape(image.url || "")}" data-media-name="${escape(image.name || "Attached image")}" aria-label="View ${escape(image.name || "attached image")}">${cached?.url ? `<img src="${escape(cached.url)}" alt="${escape(image.name || "Attached image")}" loading="lazy">` : `<span class="user-image-loading">${cached?.error ? "Open image" : "Image"}</span>`}</button>`;
    }).join("")}</div>`;
  }
  async function loadMedia(meta) {
    if (state.imageCache.get(meta.sha256)?.url) return state.imageCache.get(meta.sha256);
    if (state.imageRequests.has(meta.sha256)) return state.imageRequests.get(meta.sha256);
    if (!/^\/api\/chats\/[^/]+\/media\/[a-f0-9]{64}$/.test(meta.url || "")) throw new Error("This image reference is unavailable.");
    const promise = (async () => {
      try {
        const media = await api(meta.url);
        if (media.sha256 !== meta.sha256 || !["image/png", "image/jpeg", "image/webp", "image/bmp"].includes(media.mime) || typeof media.data !== "string") throw new Error("This image could not be verified.");
        const binary = atob(media.data);
        const bytes = Uint8Array.from(binary, (value) => value.charCodeAt(0));
        const result = { url: URL.createObjectURL(new Blob([bytes], { type: media.mime })), mime: media.mime, size: bytes.length, name: meta.name || "Attached image" };
        state.imageCache.set(meta.sha256, result);
        return result;
      } catch (error) {
        state.imageCache.set(meta.sha256, { error: error.message });
        throw error;
      } finally { state.imageRequests.delete(meta.sha256); }
    })();
    state.imageRequests.set(meta.sha256, promise);
    return promise;
  }
  const imageObserver = typeof IntersectionObserver === "function" ? new IntersectionObserver((entries) => {
    for (const entry of entries) if (entry.isIntersecting) {
      imageObserver.unobserve(entry.target);
      const node = entry.target;
      loadMedia({ sha256: node.dataset.mediaHash, url: node.dataset.mediaUrl, name: node.dataset.mediaName }).then(() => renderTranscript()).catch(() => renderTranscript());
    }
  }, { root: $("transcript-pane"), rootMargin: "100px" }) : null;
  function observeImages() {
    $("transcript").querySelectorAll("[data-media-hash]").forEach((node) => {
      if (node.dataset.observed || state.imageCache.has(node.dataset.mediaHash)) return;
      node.dataset.observed = "true";
      if (imageObserver) imageObserver.observe(node);
      else loadMedia({ sha256: node.dataset.mediaHash, url: node.dataset.mediaUrl, name: node.dataset.mediaName }).then(() => renderTranscript()).catch(() => renderTranscript());
    });
  }
  async function showImage(meta) {
    try {
      const cached = meta.preview ? { url: meta.preview, mime: meta.mime, size: meta.size, name: meta.name } : await loadMedia(meta);
      $("image-viewer-title").textContent = meta.name || cached.name || "Attached image";
      $("image-viewer-image").src = cached.url;
      $("image-viewer-image").alt = meta.name || "Attached image";
      $("image-download").href = cached.url;
      $("image-download").download = meta.name || "image";
      $("image-viewer-caption").textContent = `${cached.mime || "Image"} · ${byteSize(cached.size)}`;
      if (!$("image-viewer").open) $("image-viewer").showModal();
    } catch (error) { toast(error.message, true, 5500); }
  }
  function persistDeveloper() {
    if (!state.draftStorage) return;
    try { sessionStorage.setItem(`${state.draftStorage}:developer`, JSON.stringify({ open: state.dev.open, tab: state.dev.tab })); }
    catch { /* Workspace UI remains usable without browser storage. */ }
  }
  function updateSurfaceInert() {
    const devModal = state.dev.open && matchMedia("(max-width:1100px)").matches;
    $("main").inert = devModal;
    $("developer-pane").inert = !state.dev.open;
    $("developer-backdrop").hidden = !devModal;
    if (devModal) { $("developer-pane").setAttribute("role", "dialog"); $("developer-pane").setAttribute("aria-modal", "true"); }
    else { $("developer-pane").removeAttribute("role"); $("developer-pane").removeAttribute("aria-modal"); }
  }
  function toggleDeveloper(open = !state.dev.open, tab = state.dev.tab) {
    state.dev.open = open;
    state.dev.tab = ["agents", "files", "changes", "activity", "performance", "system"].includes(tab) ? tab : "agents";
    if (open && matchMedia("(max-width:1100px)").matches && state.sidebar) toggleSidebar(false);
    $("developer-pane").hidden = !open;
    $("app").classList.toggle("developer-open", open);
    $("toggle-developer").setAttribute("aria-expanded", String(open));
    $("toggle-developer").setAttribute("aria-label", open ? "Close workspace panel" : "Open workspace panel");
    updateSurfaceInert();
    renderDeveloper();
    persistDeveloper();
    if (open) {
      refreshDeveloper();
      if (state.dev.tab === "files" && !state.dev.filesLoaded) loadWorkspaceFiles();
      if (state.dev.tab === "changes") refreshGitChanges();
      if (matchMedia("(max-width:1100px)").matches) $(`dev-tab-${state.dev.tab}`).focus({ preventScroll: true });
    } else if ($("developer-pane").contains(document.activeElement)) $("toggle-developer").focus({ preventScroll: true });
    resizeComposer();
  }
  function selectDevTab(tab) {
    state.dev.tab = tab;
    renderDeveloper(); persistDeveloper();
    if (tab === "files" && !state.dev.filesLoaded) loadWorkspaceFiles();
    if (tab === "changes") refreshGitChanges();
  }
  function emptyDeveloper(title, body = "") { return `<div class="developer-empty">${escape(title)}${body ? `<p>${escape(body)}</p>` : ""}</div>`; }
  function developerAgentName(agent) { return agent.agent_id === "root" ? "Main agent" : agent.role || "Agent"; }
  function agentKey(agent) { return `${agent.run_id}:${agent.agent_id}`; }
  function agentStatus(status) {
    return ({ runnable: "Working", running: "Working", queued: "Queued", waiting: "Waiting", waiting_approval: "Waiting for approval", executing: "Running tool", proposed: "Waiting for approval", approved: "Approved", completed: "Done", failed: "Failed", cancelled: "Stopped", interrupted: "Interrupted" })[status] || String(status || "Pending").replaceAll("_", " ");
  }
  function developerAgentStatus(agent) {
    const approval = state.detail?.turns?.find((turn) => turn.run_id === agent.run_id)?.approvals?.some((entry) => entry.agent_id === agent.agent_id && entry.status === "pending");
    return approval ? "Waiting for approval" : agentStatus(agent.status);
  }
  function agentScopeLabel(agent) {
    const tools = agent.tools;
    if (Array.isArray(tools) && !tools.length) return "No tool access";
    const reads = new Set(["list_files", "read_file", "search_files", "search_text", "git_status", "git_diff", "journal_read", "fetch_url", "web_search"]);
    if (Array.isArray(tools) && tools.every((tool) => reads.has(tool))) return "Read-only tools";
    return currentPolicy() === "read_only" ? "Read only" : "Review changes";
  }
  function agentOperation(agent) {
    const action = agent.current_action || agent.pending_action;
    if (!action) return terminal.has(agent.status) ? `${Number(agent.steps || 0)} steps recorded` : "Preparing next step…";
    if (typeof action === "string") return action;
    const name = action.action === "tool" ? action.name : action.action || action.name;
    if (name === "spawn") return `Delegating ${action.role || "a task"}`;
    if (name === "wait") return "Waiting for delegated work";
    if (name === "finish") return "Preparing the result";
    if (name === "send") return "Sharing context with an agent";
    if (name === "batch") return `Running ${action.items?.length || "multiple"} parallel actions`;
    return currentToolLabel(name, action.args || {}) + (name === "run_command" && commandDisplay(action.args) ? ` · ${commandDisplay(action.args)}` : "");
  }
  function setSelectedDeveloperAgent(key) {
    if (state.dev.selectedAgent === key) return;
    if (state.dev.selectedAgent) state.dev.agentDrafts.set(state.dev.selectedAgent, $("agent-direction").value);
    state.dev.selectedAgent = key;
    $("agent-direction").value = key ? state.dev.agentDrafts.get(key) || "" : "";
  }
  function selectedDeveloperAgent() { return (state.dev.data?.agents || []).find((agent) => agentKey(agent) === state.dev.selectedAgent); }
  function renderDeveloperAgents() {
    const all = state.dev.data?.agents || [];
    const latest = state.detail?.turns?.filter((turn) => turn.run_id).at(-1)?.run_id || all.at(-1)?.run_id;
    const agents = state.dev.agentScope === "latest" ? all.filter((agent) => agent.run_id === latest) : all;
    $("dev-agents-label").textContent = `${agents.length} ${agents.length === 1 ? "agent" : "agents"}`;
    if (!agents.some((agent) => agentKey(agent) === state.dev.selectedAgent)) setSelectedDeveloperAgent(agents[0] ? agentKey(agents[0]) : null);
    patch($("dev-agent-list"), agents.length ? agents.map((agent) => `<button type="button" class="developer-card${agentKey(agent) === state.dev.selectedAgent ? " selected" : ""}" data-dev-agent="${escape(agentKey(agent))}"><span class="developer-card-heading"><span class="status-dot ${developerAgentStatus(agent) === "Waiting for approval" ? "waiting_approval" : terminal.has(agent.status) ? agent.status : "running"}" aria-hidden="true"></span>${escape(developerAgentName(agent))}<span class="card-state">${escape(developerAgentStatus(agent))}</span></span><span class="developer-card-task">${escape(agent.task || "")}</span><span class="developer-card-operation">${escape(agentOperation(agent))}</span></button>`).join("") : emptyDeveloper(state.dev.loading ? "Loading agent tasks…" : "No agent tasks yet", state.selected ? "Tasks appear when Instinct starts this response." : "Send a message to start work in this chat."));
    const agent = selectedDeveloperAgent();
    const turn = state.detail?.turns?.find((entry) => entry.run_id === agent?.run_id);
    const scopes = agent?.tools || agent?.scopes || [];
    patch($("dev-agent-detail"), agent ? `<div class="developer-detail"><h3>${escape(developerAgentName(agent))}</h3><p class="assigned-task">${escape(agent.task || "")}</p>${`<h4>Scope</h4><div class="agent-scopes"><span>${escape(agentScopeLabel(agent))}</span>${scopes.length ? scopes.map((scope) => `<span>${escape(typeof scope === "string" ? scope : scope.name || pretty(scope))}</span>`).join("") : Array.isArray(agent.tools) ? "" : "<span>Inherited workspace tools</span>"}</div>`}<h4>Current operation</h4><p>${escape(agentOperation(agent))}</p><p class="developer-footnote">${Number(agent.steps || 0)} recorded steps</p>${turn && toolRows(turn, agent.agent_id).length ? `<h4>Recorded tool output</h4>${renderToolRows(toolRows(turn, agent.agent_id))}` : ""}${agent.result ? `<h4>Result</h4><div class="markdown">${markdown(outputText(agent.result))}</div>` : ""}${agent.error ? `<p class="turn-error">${escape(agent.error)}</p>` : ""}</div>` : "");
    const canMessage = Boolean(agent?.can_message && !terminal.has(agent.status));
    $("dev-agent-message").hidden = !canMessage;
    $("send-agent").disabled = state.dev.messageBusy || !$("agent-direction").value.trim();
    $("stop-agent").hidden = !agent?.can_cancel;
    $("stop-agent").textContent = agent?.agent_id === "root" ? "Stop response" : "Stop agent";
    $("stop-agent").disabled = state.dev.messageBusy;
    $("agent-direction-label").textContent = agent ? `Message ${developerAgentName(agent)}` : "Message this agent";
  }
  function fileKey(file) { return `${file.origin || "workspace"}:${file.origin === "artifact" ? file.run_id || "" : ""}:${file.path}`; }
  function knownFiles() {
    const query = state.dev.fileQuery.toLowerCase();
    const receipts = (state.dev.data?.files || []).filter((file) => !query || file.path.toLowerCase().includes(query));
    const created = receipts.filter((file) => file.origin === "artifact" || file.change === "created");
    const workspace = new Map(state.dev.files.filter((file) => !query || file.path.toLowerCase().includes(query)).map((file) => [file.path, { ...file, origin: "workspace" }]));
    receipts.filter((file) => file.origin !== "artifact" && file.change !== "created").forEach((file) => workspace.set(file.path, { ...workspace.get(file.path), ...file }));
    created.filter((file) => file.origin !== "artifact").forEach((file) => workspace.delete(file.path));
    return { created, workspace: Array.from(workspace.values()) };
  }
  function renderDeveloperFiles() {
    const groups = knownFiles();
    const rows = (files) => files.map((file) => `<button type="button" class="file-row${fileKey(file) === state.dev.selectedFile ? " selected" : ""}" data-dev-file="${escape(fileKey(file))}"><span aria-hidden="true">${file.origin === "artifact" ? "▤" : "⌁"}</span><span class="file-path" title="${escape(file.path)}">${escape(file.path)}</span>${file.change ? `<span class="file-badge">${escape(file.origin === "artifact" ? "artifact" : file.change)}</span>` : ""}<span class="file-size">${escape(byteSize(file.size))}</span></button>`).join("");
    patch($("dev-file-list"), `${groups.created.length ? `<h3 class="file-group-label">Created by agents</h3>${rows(groups.created)}` : ""}<h3 class="file-group-label">Workspace</h3>${state.dev.filesLoading && !groups.workspace.length ? emptyDeveloper("Loading workspace files…") : rows(groups.workspace) || emptyDeveloper(state.dev.fileQuery ? "No matching files" : "No files to display", state.dev.filesError || "Use the search above to find a file.")}${state.dev.filesTruncated ? '<p class="developer-footnote">Showing the first 200 matches. Narrow the search to find more files.</p>' : ""}`);
    renderFileViewer();
  }
  function syntaxLine(line, language) {
    if (!["python", "javascript", "typescript", "json", "bash", "shell", "c", "cpp", "java", "rust", "go", "css"].includes(String(language).toLowerCase())) return escape(line);
    const pattern = /(?:\/\/.*|#.*)|(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)|\b(?:\d+(?:\.\d+)?)\b|\b(?:def|class|import|from|return|if|else|elif|for|while|try|except|with|as|in|is|not|and|or|None|True|False|async|await|function|const|let|var|new|throw|export|default|interface|type|public|private|static|void|int|char|bool|struct|fn|use|pub|mut|impl|match|package|func|nil|true|false|null)\b/g;
    let html = ""; let start = 0;
    for (const match of line.matchAll(pattern)) {
      html += escape(line.slice(start, match.index));
      const token = match[0];
      const type = /^\/\/|^#/.test(token) ? "comment" : /^["'`]/.test(token) ? "string" : /^\d/.test(token) ? "number" : "keyword";
      html += `<span class="syntax-${type}">${escape(token)}</span>`;
      start = match.index + token.length;
    }
    return html + escape(line.slice(start));
  }
  function renderFileViewer() {
    const file = state.dev.file;
    if (state.dev.fileLoading) { patch($("dev-file-viewer"), '<div class="developer-detail">Loading file…</div>'); return; }
    if (!file) { patch($("dev-file-viewer"), ""); return; }
    const lines = String(file.content || "").replace(/\r\n?/g, "\n").split("\n");
    if (lines.at(-1) === "" && lines.length > 1) lines.pop();
    patch($("dev-file-viewer"), `<div class="developer-detail"><div class="file-viewer-heading"><h3>${escape(file.path)}</h3><button type="button" class="icon-button" data-close-file aria-label="Close file">×</button></div><p class="file-viewer-meta">${escape(file.language || "text")} · ${typeof file.total_lines === "number" ? file.total_lines : lines.length} lines${file.origin === "artifact" ? " · agent artifact" : ""}</p><div class="file-viewer-actions"><button type="button" class="text-button" data-file-action="copy-path">Copy path</button><button type="button" class="text-button" data-file-action="${file.origin === "artifact" ? "report" : "mention"}">${file.origin === "artifact" ? "Add report to prompt" : "Insert @path"}</button><button type="button" class="text-button" data-file-action="download">Download</button><button type="button" class="text-button" data-file-action="reload">Reload</button></div><div class="code-viewer" aria-label="${escape(file.path)} source">${lines.map((line, index) => `<div class="code-line"><span class="code-line-number" aria-hidden="true">${index + 1}</span><code>${syntaxLine(line, file.language)}</code></div>`).join("")}</div>${file.truncated ? '<p class="developer-footnote">The preview is truncated.</p>' : ""}</div>`);
  }
  async function loadWorkspaceFiles() {
    const version = ++state.dev.filesVersion;
    const query = state.dev.fileQuery;
    state.dev.filesLoading = true;
    state.dev.filesError = "";
    renderDeveloperFiles();
    try {
      const result = await api("/api/workspace/files", { query, limit: 200 });
      if (version !== state.dev.filesVersion) return;
      state.dev.files = result.files || [];
      state.dev.filesTruncated = Boolean(result.truncated);
      state.dev.filesLoaded = true;
    } catch (error) { if (version === state.dev.filesVersion) { state.dev.filesError = error.message; state.dev.files = []; } }
    finally { if (version === state.dev.filesVersion) { state.dev.filesLoading = false; renderDeveloperFiles(); } }
  }
  async function openFile(file) {
    const version = ++state.dev.fileVersion;
    state.dev.selectedFile = fileKey(file);
    state.dev.fileLoading = true;
    renderDeveloperFiles();
    try {
      const result = await api(file.origin === "artifact" ? `/api/runs/${encodeURIComponent(file.run_id)}/artifacts/${encodeURIComponent(file.path)}` : `/api/workspace/file/${encodeURIComponent(file.path)}`);
      if (version !== state.dev.fileVersion) return;
      state.dev.file = { ...result, origin: file.origin, run_id: file.run_id, language: result.language || file.language || languageFromPath(file.path) };
    } catch (error) { if (version === state.dev.fileVersion) { state.dev.file = null; toast(error.message, true, 5500); } }
    finally { if (version === state.dev.fileVersion) { state.dev.fileLoading = false; renderDeveloperFiles(); } }
  }
  function quotedReport(file, budget) {
    const content = String(file.content || "").replace(/\r\n?/g, "\n");
    const heading = `\n\n[Agent report excerpt: ${JSON.stringify(file.path)}]\n`;
    const fullEnd = "\n[End of report excerpt; complete file.]\n";
    const excerptEnd = "\n[Report excerpt truncated. Open or download the artifact for the complete file.]\n";
    const format = (length) => heading + content.slice(0, length).split("\n").map((line) => `> ${line}`).join("\n") + (length < content.length ? excerptEnd : fullEnd);
    if (content.length <= 4000 && format(content.length).length <= budget) return format(content.length);
    let low = 0; let high = Math.min(4000, Math.max(0, content.length - 1)); let best = "";
    while (low <= high) {
      const middle = Math.floor((low + high) / 2);
      const text = format(middle);
      if (text.length <= budget) { best = text; low = middle + 1; }
      else high = middle - 1;
    }
    return best;
  }
  function languageFromPath(path) {
    return ({ py: "python", js: "javascript", ts: "typescript", jsx: "javascript", tsx: "typescript", json: "json", sh: "bash", cpp: "cpp", cc: "cpp", h: "c", c: "c", rs: "rust", go: "go", java: "java", css: "css", html: "html", md: "markdown", yml: "yaml", yaml: "yaml" })[String(path).split(".").at(-1)] || "text";
  }
  function changeKey(change, index) { return String(change.id || change.approval_id || `${change.run_id}:${change.agent_id}:${change.path}:${change.status}:${index}`); }
  function renderDeveloperChanges() {
    const changes = state.dev.data?.changes || [];
    const paths = new Set(changes.map((change) => change.origin === "artifact" ? `artifact:${change.run_id}:${change.path}` : `workspace:${change.path}`));
    const pending = changes.filter((change) => change.status === "proposed").length;
    const executed = changes.filter((change) => change.status === "executed").length;
    $("dev-changes-summary").textContent = `${paths.size} ${paths.size === 1 ? "file" : "files"} · ${executed} applied${pending ? ` · ${pending} proposed` : ""}`;
    if (!changes.some((change, index) => changeKey(change, index) === state.dev.selectedChange)) state.dev.selectedChange = changes.length ? changeKey(changes.at(-1), changes.length - 1) : null;
    patch($("dev-change-list"), changes.length ? changes.map((change, index) => `<button type="button" class="change-row${changeKey(change, index) === state.dev.selectedChange ? " selected" : ""}" data-dev-change="${escape(changeKey(change, index))}"><span class="change-path">${escape(change.path || "File change")}</span><span class="change-state ${["proposed", "executed", "denied", "failed"].includes(change.status) ? change.status : ""}">${escape(({ proposed: "Proposed", executed: "Applied", denied: "Declined", failed: "Failed" })[change.status] || change.status || "Recorded")}</span></button>`).join("") : emptyDeveloper("No recorded changes", "Proposed and applied file diffs appear here as agents work."));
    const change = changes.find((entry, index) => changeKey(entry, index) === state.dev.selectedChange);
    patch($("dev-change-viewer"), change ? `<div class="change-viewer"><div class="diff-path">${escape(change.path || "File changes")}</div><div class="change-viewer-meta">${escape(({ proposed: "Proposed action — review before applying", executed: "Recorded applied change", denied: "This action was declined", failed: "This action failed" })[change.status] || "Recorded change")}${change.current_matches_receipt === false ? " · current file differs from this receipt" : ""}</div><pre class="diff">${change.diff == null ? escape("No textual diff was saved with this receipt.") : diffMarkup(change.diff)}</pre>${change.diff_truncated ? `<p class="developer-footnote diff-excerpt">Showing a diff excerpt.${change.status === "proposed" && change.approval_id ? ` <button type="button" class="text-button" data-review-approval="${escape(change.approval_id)}">Review full change in chat</button>` : ""}</p>` : ""}${change.status === "proposed" && !change.diff_truncated && change.approval_id && state.detail?.turns?.some((turn) => turn.approvals?.some((approval) => approval.id === change.approval_id && approval.status === "pending")) ? `<div class="change-review-actions"><button type="button" class="button" data-approval="${escape(change.approval_id)}" data-approved="false"${state.busyApprovals.has(change.approval_id) ? " disabled" : ""}>Decline</button><button type="button" class="button button-primary" data-approval="${escape(change.approval_id)}" data-approved="true"${state.busyApprovals.has(change.approval_id) ? " disabled" : ""}>Approve</button></div>` : ""}</div>` : "");
    renderGitChanges();
  }
  function gitFileStatus(file) {
    if (file.index === "?" && file.workspace === "?") return "Untracked";
    const codes = [file.index, file.workspace];
    const label = codes.includes("U") ? "Conflict" : codes.includes("D") ? "Deleted" : codes.includes("A") ? "Added" : codes.includes("R") ? "Renamed" : "Modified";
    const staged = file.index !== " " && file.index !== "?";
    return label + (staged ? " · staged" : "");
  }
  function renderGitChanges() {
    $("git-changes-section").hidden = state.git.status?.available === false;
    if (state.git.status?.available === false) return;
    const files = state.git.status?.files || [];
    patch($("git-change-list"), state.git.loading && !state.git.status ? '<p class="developer-footnote">Checking the working tree…</p>' : files.length ? files.map((file) => `<button type="button" class="change-row${file.path === state.git.selected ? " selected" : ""}" data-git-file="${escape(file.path)}"><span class="change-path">${escape(file.path)}</span><span class="change-state">${escape(gitFileStatus(file))}</span></button>`).join("") + (state.git.status.truncated ? '<p class="developer-footnote">The Git status listing is truncated.</p>' : "") : '<p class="developer-footnote">No current Git changes.</p>');
    const diff = state.git.diff;
    if (state.git.diffLoading && (!diff || diff.path !== state.git.selected)) { patch($("git-change-viewer"), '<p class="developer-footnote">Loading current file changes…</p>'); return; }
    if (!diff) { patch($("git-change-viewer"), ""); return; }
    let content;
    if (diff.error) content = `<p class="git-preview-note">${escape(diff.error)}</p>`;
    else if (diff.directory) content = '<p class="git-preview-note">Untracked directory. Browse its files to inspect their contents.</p>';
    else if (diff.untracked) {
      const lines = String(diff.content || "").replace(/\r\n?/g, "\n").split("\n");
      if (lines.at(-1) === "" && lines.length > 1) lines.pop();
      content = `<p class="git-preview-note">Untracked file preview · no HEAD comparison exists.</p><div class="code-viewer">${lines.map((line, index) => `<div class="code-line"><span class="code-line-number" aria-hidden="true">${index + 1}</span><code>${syntaxLine(line, diff.language || languageFromPath(diff.path))}</code></div>`).join("")}</div>`;
    } else content = diff.available ? `<pre class="diff">${diff.diff ? diffMarkup(diff.diff) : '<span class="diff-line">No textual diff from HEAD.</span>'}</pre>${diff.truncated ? '<p class="git-preview-note">Showing a Git diff excerpt.</p>' : ""}` : '<p class="git-preview-note">A HEAD comparison is unavailable for this file.</p>';
    const viewer = $("git-change-viewer");
    const priorScroll = viewer.dataset.gitPath === diff.path ? viewer.querySelector(".diff,.code-viewer") : null;
    const scroll = priorScroll ? [priorScroll.scrollTop, priorScroll.scrollLeft] : null;
    patch(viewer, `<div class="change-viewer"><div class="diff-path">${escape(diff.path)}</div><div class="change-viewer-meta">${diff.untracked || diff.directory ? "Current untracked contents" : "Current working tree compared with HEAD"} · not agent attribution</div>${content}${diff.untracked || diff.directory ? `<div class="git-preview-actions"><button type="button" class="text-button" data-git-open="${escape(diff.path)}">${diff.directory ? "Browse files" : "Open in Files"}</button></div>` : ""}</div>`);
    viewer.dataset.gitPath = diff.path;
    const nextScroll = viewer.querySelector(".diff,.code-viewer");
    if (scroll && nextScroll) { nextScroll.scrollTop = scroll[0]; nextScroll.scrollLeft = scroll[1]; }
  }
  async function refreshGitChanges(force = false) {
    if (!state.dev.open || state.dev.tab !== "changes" || state.git.loading || (!force && Date.now() - state.git.updated < 2000)) return;
    state.git.loading = true;
    $("refresh-git").disabled = true;
    renderGitChanges();
    try {
      state.git.status = await api("/api/workspace/git");
      state.git.updated = Date.now();
      const selected = state.git.status.files?.find((file) => file.path === state.git.selected);
      if (selected) loadGitFile(selected);
      else { state.git.selected = null; state.git.diff = null; state.git.diffVersion++; state.git.diffLoading = false; }
    } catch {
      // Missing Git and transient inspection failures leave a quiet optional section.
      if (!state.git.status) state.git.status = { available: false, files: [] };
      state.git.updated = Date.now();
    } finally { state.git.loading = false; $("refresh-git").disabled = false; renderGitChanges(); }
  }
  async function loadGitFile(file) {
    if (state.git.selected === file.path && state.git.diffLoading) return;
    state.git.selected = file.path;
    const version = ++state.git.diffVersion;
    state.git.diffLoading = true;
    renderGitChanges();
    const untracked = file.index === "?" && file.workspace === "?";
    try {
      const result = untracked && file.path.endsWith("/") ? { path: file.path, directory: true } : await api(untracked ? `/api/workspace/file/${encodeURIComponent(file.path)}` : `/api/workspace/diff/${encodeURIComponent(file.path)}`);
      if (version !== state.git.diffVersion) return;
      state.git.diff = { ...result, path: file.path, untracked };
    } catch (error) { if (version === state.git.diffVersion) state.git.diff = { path: file.path, untracked, error: error.message }; }
    finally { if (version === state.git.diffVersion) { state.git.diffLoading = false; renderGitChanges(); } }
  }

  function activityRecords() {
    const commands = (state.dev.data?.commands || []).map((command, index) => ({ ...command, key: command.id || `${command.run_id}:${command.agent_id}:${command.created}:${index}`, category: "commands", name: command.tool || command.name, result: command.output }));
    const tools = (state.detail?.turns || []).flatMap((turn) => (turn.events || []).filter((event) => event.kind === "tool_result" && !["run_tests", "run_command", "journal_read"].includes(event.payload?.name)).map((event) => ({
      key: `${turn.run_id}:${event.id}`, category: "tools", name: event.payload.name, args: event.payload.args || {}, result: event.payload.result, agent_id: event.agent_id, run_id: turn.run_id, created: event.created, status: toolFailed(event.payload.result) ? "failed" : "completed",
    })));
    return [...commands, ...tools].filter((entry) => state.dev.activityFilter === "all" || entry.category === state.dev.activityFilter).sort((a, b) => Number(b.created || 0) - Number(a.created || 0));
  }
  function commandDisplay(args = {}) {
    if (Array.isArray(args.argv)) return args.argv.map((word) => /\s/.test(word) ? JSON.stringify(word) : word).join(" ");
    return args.command || args.cmd || args.path || "";
  }
  function readableOutput(value) {
    let output = value;
    if (typeof output === "string") { try { output = JSON.parse(output); } catch { return output; } }
    if (output && typeof output === "object" && ("stdout" in output || "stderr" in output)) return [output.stdout || "", output.stderr ? `stderr:\n${output.stderr}` : "", output.exit_code != null ? `Exit code: ${output.exit_code}` : "", output.timed_out ? "Timed out" : ""].filter(Boolean).join("\n\n");
    return pretty(output);
  }
  function renderDeveloperActivity() {
    const records = activityRecords();
    if (!records.some((entry) => String(entry.key) === state.dev.selectedActivity)) state.dev.selectedActivity = records[0] ? String(records[0].key) : null;
    patch($("dev-activity-list"), records.length ? records.slice(0, 100).map((entry) => `<button type="button" class="activity-card${String(entry.key) === state.dev.selectedActivity ? " selected" : ""}" data-dev-activity="${escape(entry.key)}"><span class="activity-card-title"><span class="status-dot ${["failed", "cancelled", "interrupted"].includes(entry.status) ? "failed" : ["proposed", "approved", "running"].includes(entry.status) ? "waiting_approval" : "completed"}" aria-hidden="true"></span>${escape(entry.name === "run_tests" ? "Tests" : entry.name === "run_command" ? "Command" : ["failed", "denied", "cancelled", "expired", "stale", "interrupted"].includes(entry.status) ? failedToolLabel(entry.name, entry.args, entry.status) : toolLabel(entry.name, entry.args))}<time>${escape(entry.created ? new Date(Number(entry.created) * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : "")}</time></span>${commandDisplay(entry.args) ? `<div class="activity-card-command">${escape(commandDisplay(entry.args))}</div>` : ""}</button>`).join("") : emptyDeveloper("No recorded activity", "Tool calls, commands and test output appear here after they run."));
    const record = records.find((entry) => String(entry.key) === state.dev.selectedActivity);
    patch($("dev-activity-viewer"), record ? `<div class="developer-detail"><h3>${escape(record.name === "run_tests" ? "Test output" : record.name === "run_command" ? "Command output" : ["failed", "denied", "cancelled", "expired", "stale", "interrupted"].includes(record.status) ? failedToolLabel(record.name, record.args, record.status) : toolLabel(record.name, record.args))}</h3><p>${escape(agentStatus(record.status))}${record.exit_code != null ? ` · exit ${escape(record.exit_code)}` : ""}${record.agent_id ? ` · ${escape(record.agent_id === "root" ? "Main agent" : record.agent_id)}` : ""}</p><pre class="activity-output">${escape(readableOutput(record.result) || (record.status === "proposed" ? "Waiting for approval before execution." : record.status === "running" ? "Command is running…" : "No output was recorded."))}</pre>${record.output_truncated ? '<p class="developer-footnote">Showing an output excerpt.</p>' : ""}<details class="tool-detail" data-key="activity-request-${escape(record.key)}">${disclosureSummary("Exact request")}<pre>${escape(pretty(record.args || {}))}</pre></details></div>` : "");
  }
  function cacheBadge(source) {
    return `<span class="cache-badge ${escape(source || "none")}">${escape(({ session: "cached", snapshot: "snapshot", cold: "cold" })[source] || "–")}</span>`;
  }
  function agentLabel(id) { return id === "root" ? "Main" : id.replace(/^agent-0*/, "Sub-agent "); }
  function renderDeveloperPerformance() {
    const rows = generations();
    if (!rows.length) {
      patch($("dev-performance-summary"), emptyDeveloper("No model calls yet", "Timing for each model call in this chat shows up here."));
      patch($("dev-context"), ""); patch($("dev-performance-list"), ""); return;
    }
    const measured = rows.filter((row) => row.metrics.prompt_tokens);
    const prompt = measured.reduce((sum, row) => sum + row.metrics.prompt_tokens, 0);
    const reused = measured.reduce((sum, row) => sum + (row.metrics.reused_tokens || 0), 0);
    const output = rows.reduce((sum, row) => sum + (row.output_tokens || 0), 0);
    const speeds = measured.map((row) => row.metrics.decode_tps).filter(Boolean).sort((a, b) => a - b);
    const ttfts = measured.map((row) => row.metrics.ttft_ms).filter(Boolean).sort((a, b) => a - b);
    const median = (values) => values.length ? values[Math.floor(values.length / 2)] : null;
    patch($("dev-performance-summary"), `<div class="stat-grid"><div class="stat"><span>${rows.length}</span>model calls</div><div class="stat"><span>${prompt ? `${Math.round((reused / prompt) * 100)}%` : "–"}</span>prompt reused</div><div class="stat"><span>${median(ttfts) != null ? `${Math.round(median(ttfts))} ms` : "–"}</span>median first token</div><div class="stat"><span>${median(speeds) != null ? Math.round(median(speeds)) : "–"}</span>median tok/s</div><div class="stat"><span>${prompt ? compact(prompt) : "–"}</span>prompt tokens</div><div class="stat"><span>${compact(output)}</span>generated</div></div>`);
    const agents = new Map();
    for (const row of rows) agents.set(row.agent, row);
    const limit = state.detail?.chat?.context_limit;
    patch($("dev-context"), `<h3 class="file-group-label">Context window</h3>${[...agents.values()].map((row) => {
      const used = (row.input_tokens || 0) + (row.output_tokens || 0);
      const cap = row.agent === "root" ? limit : (state.models.find((model) => model.id !== state.detail?.chat?.model)?.context || limit);
      const share = cap ? Math.min(1, used / cap) : 0;
      return `<div class="context-row"><span>${escape(agentLabel(row.agent))}</span><span class="meter" aria-hidden="true"><span class="meter-fill${share > 0.85 ? " high" : share > 0.6 ? " mid" : ""}" style="width:${Math.max(2, share * 100)}%"></span></span><span>${compact(used)}${cap ? ` / ${compact(cap)}` : ""}</span></div>`;
    }).join("")}`);
    patch($("dev-performance-list"), `<div class="perf-scroll"><table class="perf-table"><thead><tr><th>Agent</th><th>Model</th><th>Prompt · reused</th><th>Cache</th><th>First token</th><th>Speed</th><th>Out</th></tr></thead><tbody>${rows.slice().reverse().slice(0, 200).map((row) => { const prompt = row.metrics.prompt_tokens || row.input_tokens || 0; const reused = row.metrics.reused_tokens || 0; return `<tr><td>${escape(agentLabel(row.agent))}</td><td>${escape(shortModel(row.model))}</td><td title="${escape(reused)} of ${escape(prompt)} prompt tokens came from the cache">${compact(prompt)}${reused && prompt ? ` <small>· ${Math.round((100 * reused) / prompt)}%</small>` : ""}</td><td>${cacheBadge(row.metrics.cache_source)}</td><td>${row.metrics.ttft_ms != null ? `${Math.round(row.metrics.ttft_ms)} ms` : "–"}</td><td>${row.metrics.decode_tps ? `${Math.round(row.metrics.decode_tps)} tok/s` : "–"}${row.metrics.draft ? ` <small title="Speculative decoding: tokens per verify round">· ${escape(row.metrics.draft.tokens_per_round)}/round</small>` : ""}</td><td>${compact(row.output_tokens)}</td></tr>`; }).join("")}</tbody></table></div>`);
  }
  function bytes(value) { return value == null ? "–" : value >= 2 ** 30 ? `${(value / 2 ** 30).toFixed(1)} GB` : `${Math.round(value / 2 ** 20)} MB`; }
  function renderDeveloperSystem() {
    const workers = state.live?.workers || [];
    patch($("dev-system-list"), workers.length ? workers.map((worker) => {
      const reuse = worker.prompt_tokens ? Math.round((worker.reused_tokens / worker.prompt_tokens) * 100) : null;
      return `<div class="worker-card"><div class="worker-title"><span class="status-dot ${worker.state === "busy" ? "running" : worker.state === "idle" ? "ready" : worker.state === "error" ? "failed" : worker.state === "starting" ? "loading" : ""}" aria-hidden="true"></span><strong>${escape(worker.model)}</strong><span class="worker-role">${escape(worker.group === "driver" ? "main model" : worker.group === "subagent" ? "sub-agents" : worker.name)}</span><span class="worker-state">${escape(({ busy: "Generating", idle: "Ready", stopped: "Not running", starting: "Starting", error: "Unreachable" })[worker.state] || worker.state)}</span></div><dl class="worker-stats"><div><dt>Memory</dt><dd>${bytes(worker.memory_bytes)}</dd></div><div><dt>Prompt cache</dt><dd>${bytes(worker.cache_bytes)}${worker.cache_entries ? ` · ${worker.cache_entries}` : ""}</dd></div><div><dt>Requests</dt><dd>${worker.requests || 0}</dd></div><div><dt>Reused</dt><dd>${reuse != null ? `${reuse}%` : "–"}</dd></div><div><dt>Generated</dt><dd>${compact(worker.generated_tokens)}</dd></div><div><dt>Context</dt><dd>${worker.context ? `${Math.round(worker.context / 1024)}K` : "–"}</dd></div>${worker.draft_enabled != null ? `<div><dt>Drafting</dt><dd>${worker.draft_enabled ? "on" : "paused"}</dd></div>` : ""}</dl>${worker.error ? `<p class="worker-error">${escape(worker.error)}</p>` : ""}</div>`;
    }).join("") : emptyDeveloper(state.overview?.mode === "demo" ? "Demo mode has no model workers" : "Workers appear once the model starts"));
  }
  function renderDeveloper() {
    if (!state.dev.open) return;
    document.querySelectorAll("[data-dev-tab]").forEach((button) => {
      const selected = button.dataset.devTab === state.dev.tab;
      button.setAttribute("aria-selected", String(selected)); button.tabIndex = selected ? 0 : -1;
      $(`dev-${button.dataset.devTab}`).hidden = !selected;
    });
    $("developer-error").hidden = !state.dev.error;
    $("developer-error").textContent = state.dev.error;
    $("developer-summary").textContent = state.detail?.chat?.title || "Tasks, files and recorded work";
    $("dev-agent-count").textContent = state.dev.data?.agents?.length || "";
    $("dev-change-count").textContent = state.dev.data?.changes?.length || "";
    if (state.dev.tab === "agents") renderDeveloperAgents();
    else if (state.dev.tab === "files") renderDeveloperFiles();
    else if (state.dev.tab === "changes") renderDeveloperChanges();
    else if (state.dev.tab === "performance") renderDeveloperPerformance();
    else if (state.dev.tab === "system") renderDeveloperSystem();
    else renderDeveloperActivity();
  }
  async function refreshDeveloper() {
    if (!state.dev.open || !state.selected) return;
    if (state.dev.loadingRequest) { state.dev.refreshAgain = true; return; }
    const selected = state.selected;
    const version = state.dev.version;
    state.dev.loadingRequest = true;
    state.dev.loading = !state.dev.data;
    renderDeveloper();
    try {
      const detail = await api(`/api/chats/${encodeURIComponent(selected)}/developer`);
      if (selected !== state.selected || version !== state.dev.version) return;
      state.dev.data = detail;
      state.dev.error = "";
    } catch (error) { if (selected === state.selected && version === state.dev.version) state.dev.error = error.message; }
    finally {
      state.dev.loadingRequest = false;
      state.dev.loading = false;
      renderDeveloper();
      if (state.dev.refreshAgain || selected !== state.selected || version !== state.dev.version) {
        state.dev.refreshAgain = false;
        refreshDeveloper();
      }
    }
  }
  async function messageAgent() {
    const agent = selectedDeveloperAgent();
    const input = $("agent-direction");
    const message = input.value.trim();
    if (!agent?.can_message || !message || state.dev.messageBusy) return;
    const original = input.value;
    state.dev.messageBusy = true; renderDeveloperAgents();
    try {
      await api(`/api/runs/${encodeURIComponent(agent.run_id)}/agents/${encodeURIComponent(agent.agent_id)}/message`, { message });
      if (input.value === original && state.dev.selectedAgent === agentKey(agent)) { input.value = ""; state.dev.agentDrafts.set(agentKey(agent), ""); }
      toast("Direction saved for the agent’s next step.");
      await refreshDeveloper();
    } catch (error) { toast(error.message, true, 5500); }
    finally { state.dev.messageBusy = false; renderDeveloperAgents(); }
  }
  async function stopSelectedAgent() {
    const agent = selectedDeveloperAgent();
    if (!agent?.can_cancel || state.dev.messageBusy) return;
    state.dev.messageBusy = true; renderDeveloperAgents();
    try {
      await api(`/api/runs/${encodeURIComponent(agent.run_id)}/agents/${encodeURIComponent(agent.agent_id)}/cancel`, {});
      await refresh(); await refreshDeveloper();
    } catch (error) { toast(error.message, true, 5500); }
    finally { state.dev.messageBusy = false; renderDeveloperAgents(); }
  }

  const commands = [
    { name: "/new", description: "Start a fresh chat", run: () => selectChat(null) },
    { name: "/clear", description: "Start with a fresh context", run: () => selectChat(null) },
    { name: "/tasks", description: "Inspect agent tasks", run: () => toggleDeveloper(true, "agents") },
    { name: "/files", description: "Browse workspace files", run: () => toggleDeveloper(true, "files") },
    { name: "/diff", description: "Review recorded changes", run: () => toggleDeveloper(true, "changes") },
    { name: "/activity", description: "Commands, tests and tool output", run: () => toggleDeveloper(true, "activity") },
    { name: "/settings", description: "Permissions and workspace", run: () => $("settings").showModal() },
    { name: "/status", description: "Model and chat status", run: () => toast(`${modelLabel()} · ${statusLabel((state.detail?.backend || state.overview?.backend)?.status)} · ${currentPolicy() === "read_only" ? "Read only" : "Review changes"}`, false, 5500) },
    { name: "/help", description: "Keyboard shortcuts", run: () => toast("Enter sends · Shift + Enter adds a line · Esc stops · ⌘ K starts a new chat", false, 6500) },
  ];
  function matchingCommands() {
    const value = $("message").value.trim();
    if (currentMedia().length || !/^\/[a-z]*$/i.test(value)) return [];
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
    const rename = event.target.closest("[data-rename-chat]");
    if (rename) { renameChat(rename.dataset.renameChat); return; }
    const remove = event.target.closest("[data-delete-chat]");
    if (remove) { deleteChat(remove.dataset.deleteChat, remove); return; }
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
    const edit = event.target.closest("[data-edit-turn]");
    if (edit) { state.editing = edit.dataset.editTurn; renderTranscript(); document.querySelector(`[data-edit-form="${CSS.escape(state.editing)}"] textarea`)?.focus(); return; }
    if (event.target.closest("[data-edit-cancel]")) { state.editing = null; renderTranscript(); return; }
    const rewindButton = event.target.closest("[data-rewind-turn]");
    if (rewindButton) { askRewind(rewindButton.dataset.rewindTurn); return; }
    const artifact = event.target.closest("[data-artifact]");
    if (artifact) downloadArtifact(artifact);
  });
  document.addEventListener("submit", async (event) => {
    const form = event.target.closest("[data-edit-form]");
    if (!form) return;
    event.preventDefault();
    const message = form.querySelector("textarea").value.trim();
    if (!message) return;
    const restore = Boolean(form.querySelector('input[name="restore"]')?.checked);
    try {
      await api(`/api/chats/${encodeURIComponent(state.selected)}/rewind`, { turn_id: form.dataset.editForm, restore_files: restore, message });
      state.editing = null;
      if (restore) toast("Files restored to before that message");
      await refresh();
    } catch (error) { toast(error.message, true); }
  });
  function askRewind(turnId) {
    const turn = state.detail?.turns?.find((entry) => entry.turn_id === turnId);
    state.rewinding = turnId;
    $("rewind-files-row").hidden = !turn?.checkpoint;
    $("rewind-files").checked = Boolean(turn?.checkpoint);
    $("rewind-dialog").showModal();
  }
  $("rewind-cancel").addEventListener("click", () => $("rewind-dialog").close());
  $("rewind-confirm").addEventListener("click", async () => {
    $("rewind-dialog").close();
    const restore = !$("rewind-files-row").hidden && $("rewind-files").checked;
    try {
      const result = await api(`/api/chats/${encodeURIComponent(state.selected)}/rewind`, { turn_id: state.rewinding, restore_files: restore });
      await refresh();
      $("message").value = result.message;
      saveDraft(); renderComposer(); $("message").focus();
      toast(restore ? `Rewound ${result.removed} message${result.removed === 1 ? "" : "s"} and restored files` : `Rewound ${result.removed} message${result.removed === 1 ? "" : "s"}`);
    } catch (error) { toast(error.message, true); }
  });
  $("mode-button").addEventListener("click", () => toggleModePanel());
  $("mode-models").addEventListener("click", (event) => {
    const option = event.target.closest("[data-model]");
    if (!option || option.getAttribute("aria-checked") === "true") return;
    changeSetting("model", option.dataset.model).then(renderSelects);
    if (!state.selected) renderSelects();
  });
  $("effort-slider").addEventListener("input", (event) => showEffort(Number(event.target.value)));
  $("effort-slider").addEventListener("change", (event) => {
    const value = EFFORTS[Number(event.target.value)][0];
    if (value !== currentMode().effort) changeSetting("effort", value).then(renderSelects);
  });
  $("mode-panel").addEventListener("keydown", (event) => { if (event.key === "Escape") { event.stopPropagation(); toggleModePanel(false); $("mode-button").focus(); } });
  document.addEventListener("pointerdown", (event) => { if (!$("mode-panel").hidden && !$("mode-picker").contains(event.target)) toggleModePanel(false); });
  $("chat-search").addEventListener("input", (event) => { state.chatQuery = event.target.value; renderOverview(); });
  $("context-meter").addEventListener("click", () => toggleDeveloper(true, "performance"));
  $("live-speed").addEventListener("click", () => toggleDeveloper(true, "system"));
  $("toggle-sidebar").addEventListener("click", () => toggleSidebar());
  $("close-sidebar").addEventListener("click", () => { toggleSidebar(false); $("toggle-sidebar").focus(); });
  $("new-chat").addEventListener("click", () => selectChat(null));
  $("home").addEventListener("click", () => selectChat(null));
  $("open-settings").addEventListener("click", () => { renderSettings(); $("settings").showModal(); });
  $("permission-shortcut").addEventListener("click", () => { renderSettings(); $("settings").showModal(); });
  $("close-settings").addEventListener("click", () => $("settings").close());
  $("key-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const name = $("key-name").value.trim();
    if (!name) return;
    try {
      const created = await api("/api/keys", { name });
      $("key-name").value = "";
      $("new-key-value").textContent = created.key;
      $("new-key").hidden = false;
      renderKeys();
    } catch (error) { toast(error.message, true); }
  });
  $("copy-key").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("new-key-value").textContent); toast("Key copied"); } catch { toast("Couldn't copy the key", true); }
  });
  $("key-list").addEventListener("click", async (event) => {
    const button = event.target.closest("[data-revoke-key]");
    if (!button) return;
    try { await api(`/api/keys/${encodeURIComponent(button.dataset.revokeKey)}/delete`, {}); renderKeys(); toast("Key revoked"); } catch (error) { toast(error.message, true); }
  });
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
    if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "d") { event.preventDefault(); toggleDeveloper(); return; }
    if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "f") { event.preventDefault(); if (!state.sidebar && matchMedia("(max-width:900px)").matches) toggleSidebar(true); $("chat-search").focus(); return; }
    if (event.key === "Tab" && state.dev.open && matchMedia("(max-width:1100px)").matches && !$("settings").open && !$("image-viewer").open) {
      const buttons = Array.from($("developer-pane").querySelectorAll('button:not(:disabled),input,textarea,select,a[href]')).filter((element) => element.getClientRects().length && element.tabIndex !== -1);
      const first = buttons[0]; const last = buttons.at(-1);
      if (event.shiftKey && (document.activeElement === first || !$("developer-pane").contains(document.activeElement))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
    if (event.key === "Tab" && state.sidebar && matchMedia("(max-width:700px)").matches) {
      const buttons = Array.from($("sidebar").querySelectorAll("a[href],button:not(:disabled)"));
      const first = buttons[0]; const last = buttons.at(-1);
      if (event.shiftKey && (document.activeElement === first || !$("sidebar").contains(document.activeElement))) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      if ($("settings").open) $("settings").close();
      if ($("image-viewer").open) $("image-viewer").close();
      selectChat(null); return;
    }
    if (event.key === "Escape" && !$("settings").open && !$("image-viewer").open) {
      if (!$("command-menu").hidden) { $("command-menu").hidden = true; event.preventDefault(); }
      else if (state.sidebar) { toggleSidebar(false); event.preventDefault(); }
      else if (state.dev.open && matchMedia("(max-width:1100px)").matches) { toggleDeveloper(false); event.preventDefault(); }
      else if (active()) { cancelResponse(); event.preventDefault(); }
      else if (state.dev.open) { toggleDeveloper(false); event.preventDefault(); }
    }
  });
  window.addEventListener("resize", () => { if (state.sidebar) toggleSidebar(true); if (state.dev.open) { updateSurfaceInert(); if (matchMedia("(max-width:1100px)").matches && !$("developer-pane").contains(document.activeElement)) $(`dev-tab-${state.dev.tab}`).focus(); } resizeComposer(); });
  window.addEventListener("focus", () => refresh());
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  $("toggle-developer").addEventListener("click", () => toggleDeveloper());
  $("close-developer").addEventListener("click", () => toggleDeveloper(false));
  $("developer-backdrop").addEventListener("click", () => toggleDeveloper(false));
  document.querySelectorAll("[data-dev-tab]").forEach((button) => button.addEventListener("click", () => selectDevTab(button.dataset.devTab)));
  $("dev-response-filter").addEventListener("change", () => { state.dev.agentScope = $("dev-response-filter").value; renderDeveloperAgents(); });
  $("dev-activity-filter").addEventListener("change", () => { state.dev.activityFilter = $("dev-activity-filter").value; renderDeveloperActivity(); });
  let fileSearchTimer;
  $("dev-file-query").addEventListener("input", () => {
    state.dev.fileQuery = $("dev-file-query").value.trim();
    clearTimeout(fileSearchTimer);
    renderDeveloperFiles();
    fileSearchTimer = setTimeout(loadWorkspaceFiles, 220);
  });
  $("refresh-git").addEventListener("click", () => refreshGitChanges(true));
  $("reload-files").addEventListener("click", () => { loadWorkspaceFiles(); refreshDeveloper(); });
  $("dev-agent-message").addEventListener("submit", (event) => { event.preventDefault(); messageAgent(); });
  $("agent-direction").addEventListener("input", () => {
    if (state.dev.selectedAgent) state.dev.agentDrafts.set(state.dev.selectedAgent, $("agent-direction").value);
    $("send-agent").disabled = state.dev.messageBusy || !$("agent-direction").value.trim();
  });
  $("agent-direction").addEventListener("keydown", (event) => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); messageAgent(); } });
  $("stop-agent").addEventListener("click", stopSelectedAgent);
  $("attach-images").addEventListener("click", () => $("image-input").click());
  $("image-input").addEventListener("change", () => { const files = Array.from($("image-input").files || []); $("image-input").value = ""; addImages(files); });
  $("message").addEventListener("paste", (event) => {
    const files = Array.from(event.clipboardData?.items || []).filter((item) => item.kind === "file" && item.type.startsWith("image/")).map((item) => item.getAsFile()).filter(Boolean);
    if (!files.length) return;
    event.preventDefault(); addImages(files);
  });
  let dragDepth = 0;
  document.addEventListener("dragenter", (event) => {
    if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return;
    event.preventDefault(); dragDepth++; $("composer").classList.add("drop-target");
  });
  document.addEventListener("dragover", (event) => { if (Array.from(event.dataTransfer?.types || []).includes("Files")) { event.preventDefault(); event.dataTransfer.dropEffect = "copy"; } });
  document.addEventListener("dragleave", (event) => { if (!Array.from(event.dataTransfer?.types || []).includes("Files")) return; dragDepth = Math.max(0, dragDepth - 1); if (!dragDepth) $("composer").classList.remove("drop-target"); });
  document.addEventListener("drop", (event) => {
    if (!event.dataTransfer?.files?.length) return;
    event.preventDefault(); dragDepth = 0; $("composer").classList.remove("drop-target"); addImages(event.dataTransfer.files);
  });
  $("close-image-viewer").addEventListener("click", () => $("image-viewer").close());
  $("image-viewer").addEventListener("click", (event) => {
    const bounds = $("image-viewer").getBoundingClientRect();
    if (event.target === $("image-viewer") && (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom)) $("image-viewer").close();
  });
  document.addEventListener("click", (event) => {
    const attachmentRemove = event.target.closest("[data-remove-attachment]");
    if (attachmentRemove) { removeImage(attachmentRemove.dataset.removeAttachment); return; }
    const attachmentView = event.target.closest("[data-preview-attachment]");
    if (attachmentView) { const image = currentMedia().find((entry) => entry.id === attachmentView.dataset.previewAttachment); if (image?.preview) showImage(image); return; }
    const media = event.target.closest("[data-media-hash]");
    if (media) { showImage({ sha256: media.dataset.mediaHash, url: media.dataset.mediaUrl, name: media.dataset.mediaName }); return; }
    const review = event.target.closest("[data-review-approval]");
    if (review) {
      const button = Array.from($("transcript").querySelectorAll("[data-approval]")).find((entry) => entry.dataset.approval === review.dataset.reviewApproval);
      const card = button?.closest(".approval-card");
      if (card) { if (matchMedia("(max-width:1100px)").matches) toggleDeveloper(false); card.setAttribute("tabindex", "-1"); card.scrollIntoView({ block: "center" }); card.focus({ preventScroll: true }); }
      else toast("This action is no longer waiting for approval.");
      return;
    }
    const agent = event.target.closest("[data-dev-agent]");
    if (agent) { setSelectedDeveloperAgent(agent.dataset.devAgent); renderDeveloperAgents(); return; }
    const gitFile = event.target.closest("[data-git-file]");
    if (gitFile) { const file = state.git.status?.files?.find((entry) => entry.path === gitFile.dataset.gitFile); if (file) loadGitFile(file); return; }
    const gitOpen = event.target.closest("[data-git-open]");
    if (gitOpen) {
      selectDevTab("files");
      if (gitOpen.dataset.gitOpen.endsWith("/")) { state.dev.fileQuery = gitOpen.dataset.gitOpen; $("dev-file-query").value = state.dev.fileQuery; loadWorkspaceFiles(); }
      else openFile({ path: gitOpen.dataset.gitOpen, origin: "workspace" });
      return;
    }
    const change = event.target.closest("[data-dev-change]");
    if (change) { state.dev.selectedChange = change.dataset.devChange; renderDeveloperChanges(); return; }
    const activity = event.target.closest("[data-dev-activity]");
    if (activity) { state.dev.selectedActivity = activity.dataset.devActivity; renderDeveloperActivity(); return; }
    const file = event.target.closest("[data-dev-file]");
    if (file) {
      const groups = knownFiles();
      const match = [...groups.created, ...groups.workspace].find((entry) => fileKey(entry) === file.dataset.devFile);
      if (match) openFile(match); return;
    }
    if (event.target.closest("[data-close-file]")) { state.dev.file = null; state.dev.selectedFile = null; state.dev.fileVersion++; state.dev.fileLoading = false; renderDeveloperFiles(); return; }
    const fileAction = event.target.closest("[data-file-action]");
    if (!fileAction || !state.dev.file) return;
    const opened = state.dev.file;
    if (fileAction.dataset.fileAction === "copy-path") copy(opened.path, fileAction);
    else if (["mention", "report"].includes(fileAction.dataset.fileAction)) {
      const input = $("message");
      const budget = input.maxLength - input.value.length + input.selectionEnd - input.selectionStart;
      const reference = opened.origin === "artifact" ? quotedReport(opened, budget) : /\s/.test(opened.path) ? `@${JSON.stringify(opened.path)} ` : `@${opened.path} `;
      if (!reference || reference.length > budget) { toast("Make room in your message before adding this file reference.", true); return; }
      input.setRangeText(reference, input.selectionStart, input.selectionEnd, "end");
      saveDraft(); renderComposer();
      if (matchMedia("(max-width:1100px)").matches) toggleDeveloper(false);
      input.focus({ preventScroll: true });
    } else if (fileAction.dataset.fileAction === "reload") openFile(opened);
    else if (fileAction.dataset.fileAction === "download") {
      const url = URL.createObjectURL(new Blob([opened.content || ""], { type: "text/plain;charset=utf-8" }));
      const link = document.createElement("a"); link.href = url; link.download = opened.path.split(/[\\/]/).at(-1); document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    }
  });
  $("developer-pane").addEventListener("keydown", (event) => {
    if (event.target.matches("[data-dev-tab]") && ["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
      const tabs = Array.from(document.querySelectorAll("[data-dev-tab]"));
      const current = tabs.indexOf(event.target);
      const index = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length;
      event.preventDefault(); selectDevTab(tabs[index].dataset.devTab); tabs[index].focus();
    }
  });

  async function boot() {
    const selected = new URLSearchParams(location.search).get("chat");
    if (selected) { state.selected = selected; state.loading = true; $("welcome").hidden = true; $("view-loading").hidden = false; }
    renderComposer();
    const fragment = new URLSearchParams(location.hash.slice(1));
    const token = fragment.get("token");
    if (token) {
      history.replaceState(null, "", location.pathname + location.search);
      try { await api("/api/auth", { token }); }
      catch { setConnection("This session could not be authenticated. Reopen the Instinct link from the local launcher."); return; }
    }
    try { Object.assign(state.newChat, JSON.parse(localStorage.getItem("instinct-new-chat") || "{}")); } catch {}
    try { state.models = (await api("/api/models")).models || []; } catch {}
    loadSettings();
    let sidebar = null;
    try { sidebar = localStorage.getItem("instinct-sidebar"); } catch {}
    if (sidebar ? sidebar === "open" : innerWidth >= 1100) toggleSidebar(true);
    await refresh();
    connectStream();
  }
  async function loadSettings() {
    try {
      const result = await api("/api/settings");
      state.settings = result.settings;
      $("sandbox-help").textContent = result.sandbox
        ? "Commands and the terminal can read your files but only write inside the workspace."
        : "This system has no sandbox: commands and the terminal run with your own permissions.";
      $("terminal-badge").hidden = !result.sandbox;
    } catch {}
  }
  for (const [id, key] of [["command-network", "command_network"], ["web-access", "web_access"]]) {
    $(id).addEventListener("change", async (event) => {
      try { state.settings = (await api("/api/settings", { [key]: event.target.checked })).settings; }
      catch (error) { event.target.checked = !event.target.checked; toast(error.message, true); }
    });
  }

  // Terminal: a sandboxed login shell in the workspace, drawn with xterm.js.
  const term = { xterm: null, fit: null, id: null, offset: 0, source: null, exited: false, assets: null };
  function loadTerminalAssets() {
    term.assets ||= new Promise((resolve, reject) => {
      const style = document.createElement("link");
      style.rel = "stylesheet"; style.href = "/vendor/xterm.css";
      document.head.appendChild(style);
      const scripts = ["/vendor/xterm.js", "/vendor/addon-fit.js"];
      const next = () => {
        if (!scripts.length) return resolve();
        const script = document.createElement("script");
        script.src = scripts.shift(); script.onload = next; script.onerror = reject;
        document.head.appendChild(script);
      };
      next();
    });
    return term.assets;
  }
  function terminalState(text) { $("terminal-state").textContent = text; }
  async function startTerminal() {
    term.exited = false; term.offset = 0;
    $("terminal-restart").hidden = true; terminalState("");
    term.xterm.reset();
    const { cols, rows } = term.xterm;
    term.id = (await api("/api/terminals", { cols, rows })).id;
    connectTerminal();
  }
  function connectTerminal() {
    term.source?.close();
    const source = new EventSource(`/api/terminals/${term.id}/stream?offset=${term.offset}`);
    term.source = source;
    source.onmessage = (event) => {
      const message = JSON.parse(event.data);
      term.offset = message.offset;
      term.xterm.write(Uint8Array.from(atob(message.data), (char) => char.charCodeAt(0)));
    };
    source.addEventListener("exit", () => {
      source.close(); term.exited = true;
      terminalState("Shell exited"); $("terminal-restart").hidden = false;
    });
    // Reconnect from the last offset instead of letting EventSource replay the URL.
    source.onerror = () => { source.close(); if (!term.exited && !$("terminal-panel").hidden) setTimeout(connectTerminal, 800); };
  }
  async function toggleTerminal(open = $("terminal-panel").hidden) {
    $("terminal-panel").hidden = !open;
    $("toggle-terminal").setAttribute("aria-expanded", String(open));
    if (!open) { term.source?.close(); return; }
    try { await loadTerminalAssets(); } catch { terminalState("Could not load the terminal"); return; }
    if (!term.xterm) {
      term.xterm = new window.Terminal({
        fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace', fontSize: 12, lineHeight: 1.2, cursorBlink: true, scrollback: 5000,
        // The terminal stays dark in both themes.
        theme: { background: getComputedStyle(document.documentElement).getPropertyValue("--terminal-bg").trim(), foreground: "#ecebf4", cursor: "#ff9cc1", selectionBackground: "rgba(185, 166, 255, 0.3)" },
      });
      term.fit = new window.FitAddon.FitAddon();
      term.xterm.loadAddon(term.fit);
      term.xterm.open($("terminal-host"));
      term.xterm.onData((data) => { if (term.id && !term.exited) api(`/api/terminals/${term.id}/input`, { data }).catch(() => {}); });
      term.xterm.onResize(({ cols, rows }) => { if (term.id && !term.exited) api(`/api/terminals/${term.id}/resize`, { cols, rows }).catch(() => {}); });
      new ResizeObserver(() => { if (!$("terminal-panel").hidden) term.fit.fit(); }).observe($("terminal-host"));
    }
    term.fit.fit();
    term.xterm.focus();
    if (!term.id) await startTerminal();
    else if (!term.exited) connectTerminal();
  }
  $("toggle-terminal").addEventListener("click", () => toggleTerminal());
  $("close-terminal").addEventListener("click", () => toggleTerminal(false));
  $("terminal-restart").addEventListener("click", () => startTerminal());
  $("terminal-resize").addEventListener("pointerdown", (event) => {
    const panel = $("terminal-panel"), start = event.clientY, height = panel.offsetHeight;
    const move = (moved) => { panel.style.height = `${Math.min(Math.max(height + start - moved.clientY, 140), innerHeight * 0.75)}px`; };
    const up = () => { removeEventListener("pointermove", move); removeEventListener("pointerup", up); };
    addEventListener("pointermove", move); addEventListener("pointerup", up);
  });
  document.addEventListener("keydown", (event) => {
    if (event.ctrlKey && !event.metaKey && event.key === "`") { event.preventDefault(); toggleTerminal(); }
  });

  // Inside the Mac app: leave room for the window buttons, let the header
  // move the window, and take commands from the native menus.
  const native = window.webkit?.messageHandlers?.instinct;
  if (native) {
    document.documentElement.classList.add("app-mac");
    const passive = (target) => !target.closest("button, a, input, select, textarea, summary, label, [role=button]");
    document.querySelectorAll(".topbar, .sidebar-heading, .developer-heading").forEach((bar) => {
      bar.addEventListener("mousedown", (event) => { if (event.button === 0 && event.detail === 1 && passive(event.target)) native.postMessage({ type: "drag" }); });
      bar.addEventListener("dblclick", (event) => { if (passive(event.target)) native.postMessage({ type: "zoom" }); });
    });
  }
  window.instinct = {
    command(name) {
      const commands = {
        newChat: () => { selectChat(null); $("message").focus(); },
        settings: () => { renderSettings(); $("settings").showModal(); },
        toggleSidebar: () => toggleSidebar(),
        toggleWorkspace: () => toggleDeveloper(),
        searchChats: () => { toggleSidebar(true); $("chat-search").focus(); },
        toggleTerminal: () => toggleTerminal(),
      };
      commands[name]?.();
    },
  };
  boot();
})();

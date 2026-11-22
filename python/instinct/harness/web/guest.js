// Instinct for guests. The models run on the host's Mac; everything else
// stays on this computer: the folder the agents work in (through the File
// System Access API), the chats (IndexedDB) and every file change, which this
// page carries out itself when the host's agents ask for it.
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const MUTATING = new Set(["write_file", "replace_text", "edit_lines", "make_directory", "move_path", "delete_path"]);
  const READING = ["list_files", "read_file", "search"];
  const IGNORED = new Set([".git", ".venv", "__pycache__", "node_modules", "build", "models", "dist", ".instinct-trash"]);
  const PRIVATE = new Set([".env", ".aws", ".ssh", ".codex", ".agents", "credentials", "secrets"]);
  const MAX_FILE = 2 * 1024 * 1024;
  const state = { me: null, chats: [], chat: null, folder: null, run: null, reads: new Map() };

  // --- requests ---------------------------------------------------------------
  async function api(path, body) {
    const response = await fetch(path, {
      method: body === undefined ? "GET" : "POST",
      credentials: "same-origin",
      cache: "no-store",
      headers: body === undefined ? {} : { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const data = await response.json().catch(() => ({}));
    if (response.status === 401) { showSignIn(data.error?.message); throw new Error("signed out"); }
    if (!response.ok) throw new Error(data.error?.message || `request failed (${response.status})`);
    return data;
  }

  // --- storage on this computer ---------------------------------------------
  const db = new Promise((resolve, reject) => {
    const open = indexedDB.open("instinct-guest", 1);
    open.onupgradeneeded = () => {
      open.result.createObjectStore("chats", { keyPath: "id" });
      open.result.createObjectStore("handles");
    };
    open.onsuccess = () => resolve(open.result);
    open.onerror = () => reject(open.error);
  });
  async function store(name, mode, action) {
    const database = await db;
    return new Promise((resolve, reject) => {
      const request = action(database.transaction(name, mode).objectStore(name));
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }
  const saveChat = (chat) => store("chats", "readwrite", (s) => s.put(chat));
  const loadChats = () => store("chats", "readonly", (s) => s.getAll());
  const dropChat = (id) => store("chats", "readwrite", (s) => s.delete(id));

  // --- sign-in ----------------------------------------------------------------
  function showSignIn(message) {
    $("app").hidden = true;
    $("sign-in").hidden = false;
    $("sign-in-error").hidden = !message;
    $("sign-in-error").textContent = message || "";
  }
  async function signIn(invite) {
    const response = await fetch("/guest/session", {
      method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify({ invite }),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error?.message || "That invite didn't work.");
    return data;
  }
  async function start() {
    // The invite rides in the link's fragment, which never reaches a server;
    // trade it for a session cookie and take it out of the address bar.
    const match = location.hash.match(/invite=(inv_[A-Za-z0-9_-]+)/);
    if (match) history.replaceState(null, "", location.pathname);
    try {
      state.me = match ? await signIn(match[1]) : await api("/guest/me");
    } catch (error) {
      return showSignIn(match ? error.message : "");
    }
    $("sign-in").hidden = true;
    $("app").hidden = false;
    $("guest-of").textContent = `Signed in as ${state.me.name}`;
    $("model").innerHTML = state.me.models.map((m) => `<option value="${escape(m.id)}">${escape(m.label)}</option>`).join("");
    $("welcome-fine").textContent = "showDirectoryPicker" in window
      ? `Invite limits: ${state.me.limits.concurrent} at once, ${(state.me.limits.daily_tokens / 1e6).toFixed(1)}M tokens a day.`
      : "This browser can't open folders. Use Chrome, Edge, Arc or Brave to let the agents work on files; chatting works here.";
    state.chats = (await loadChats()).sort((a, b) => b.updated - a.updated);
    await restoreFolder();
    renderChats();
    openChat(state.chats[0] || null);
  }
  $("invite-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("invite-input").value.trim();
    const match = text.match(/(inv_[A-Za-z0-9_-]+)/);
    if (!match) return showSignIn("Paste the whole invite link.");
    try { await signIn(match[1]); location.reload(); } catch (error) { showSignIn(error.message); }
  });
  $("sign-out").addEventListener("click", async () => {
    await fetch("/guest/session/end", { method: "POST", credentials: "same-origin", headers: { "Content-Type": "application/json" }, body: "{}" });
    showSignIn("Signed out. Your chats stay in this browser.");
  });

  // --- the folder -------------------------------------------------------------
  async function restoreFolder() {
    if (!("showDirectoryPicker" in window)) { $("folder").disabled = true; return; }
    const saved = await store("handles", "readonly", (s) => s.get("folder")).catch(() => null);
    if (saved) setFolder(saved);
  }
  function setFolder(handle) {
    state.folder = handle;
    state.reads.clear();
    $("folder-name").textContent = handle ? handle.name : "Choose a folder";
    $("folder").classList.toggle("set", Boolean(handle));
  }
  $("folder").addEventListener("click", async () => {
    try {
      const handle = await window.showDirectoryPicker({ id: "instinct", mode: "readwrite" });
      await store("handles", "readwrite", (s) => s.put(handle, "folder"));
      setFolder(handle);
    } catch { /* The picker was closed. */ }
  });
  async function folderReady(write) {
    if (!state.folder) return false;
    const mode = { mode: write ? "readwrite" : "read" };
    if ((await state.folder.queryPermission(mode)) === "granted") return true;
    return (await state.folder.requestPermission(mode)) === "granted";
  }

  // --- paths ------------------------------------------------------------------
  function segments(path) {
    if (typeof path !== "string") throw new Error("path must be text");
    const parts = path.replace(/\\/g, "/").split("/").filter((p) => p && p !== ".");
    if (path.startsWith("/") || parts.includes("..")) throw new Error("paths must stay inside the folder");
    if (parts.some((p) => PRIVATE.has(p))) throw new Error("that path is private");
    return parts;
  }
  async function directory(parts, create = false) {
    let handle = state.folder;
    for (const part of parts) handle = await handle.getDirectoryHandle(part, { create });
    return handle;
  }
  async function entry(path) {
    const parts = segments(path);
    if (!parts.length) return { kind: "directory", handle: state.folder, parts };
    const parent = await directory(parts.slice(0, -1));
    const name = parts[parts.length - 1];
    try { return { kind: "file", handle: await parent.getFileHandle(name), parent, name, parts }; } catch {}
    return { kind: "directory", handle: await parent.getDirectoryHandle(name), parent, name, parts };
  }
  async function exists(path) { try { await entry(path); return true; } catch { return false; } }
  async function* walk(handle, prefix) {
    const children = [];
    for await (const [name, child] of handle.entries()) children.push([name, child]);
    children.sort((a, b) => a[0].localeCompare(b[0]));
    for (const [name, child] of children) {
      if (name.startsWith(".") || IGNORED.has(name) || PRIVATE.has(name)) continue;
      const path = prefix ? `${prefix}/${name}` : name;
      if (child.kind === "directory") yield* walk(child, path);
      else yield [path, child];
    }
  }
  async function hash(bytes) {
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  }
  async function readText(fileHandle) {
    const file = await fileHandle.getFile();
    if (file.size > MAX_FILE) throw new Error("the file is over the 2 MB limit");
    const bytes = new Uint8Array(await file.arrayBuffer());
    const text = new TextDecoder("utf-8", { fatal: true }).decode(bytes);
    if (text.includes("\u0000")) throw new Error("binary files are unavailable");
    return { text, sha: await hash(bytes) };
  }
  const lines = (text) => text.split(/\r\n|\n|\r/).slice(0, text.match(/(\r\n|\n|\r)$/) ? -1 : undefined);

  // --- the tools, carried out here --------------------------------------------
  // Same names, arguments and result shapes as the host's own workspace tools.
  const tools = {
    async list_files(args) {
      const target = await entry(args.path || ".");
      const limit = Math.min(Math.max(args.limit || 100, 1), 1000);
      if (target.kind === "file") return { files: [target.parts.join("/")], folders: [], truncated: false };
      const prefix = target.parts.join("/");
      const folders = [];
      for await (const [name, child] of target.handle.entries()) {
        if (child.kind === "directory" && !name.startsWith(".") && !IGNORED.has(name) && !PRIVATE.has(name)) folders.push(`${prefix ? `${prefix}/` : ""}${name}/`);
      }
      const files = [];
      for await (const [path] of walk(target.handle, prefix)) {
        if (files.length === limit) return { files, folders: folders.sort(), truncated: true };
        files.push(path);
      }
      return { files, folders: folders.sort(), truncated: false };
    },
    async read_file(args) {
      const target = await entry(args.path);
      if (target.kind !== "file") throw new Error(`${args.path} is a folder`);
      const { text, sha } = await readText(target.handle);
      state.reads.set(target.parts.join("/"), sha);
      const all = lines(text);
      const startLine = Math.max(args.start_line || 1, 1);
      const count = Math.min(Math.max(args.max_lines || 120, 1), 1000);
      const selected = all.slice(startLine - 1, startLine - 1 + count);
      return {
        path: args.path, sha256: sha, total_lines: all.length, start_line: startLine,
        content: selected.map((line, i) => `${startLine + i}: ${line}`).join("\n"),
        truncated: startLine - 1 + count < all.length,
      };
    },
    async search(args) {
      if (typeof args.query !== "string" || !args.query) throw new Error("query must not be empty");
      const limit = Math.min(Math.max(args.limit || 30, 1), 200);
      let test;
      try {
        const pattern = new RegExp(args.query, args.query === args.query.toLowerCase() ? "i" : "");
        test = (line) => pattern.test(line);
      } catch {
        const needle = args.query === args.query.toLowerCase() ? args.query : null;
        test = (line) => (needle ? line.toLowerCase().includes(needle) : line.includes(args.query));
      }
      const target = await entry(args.path || ".");
      const matches = [];
      const files = target.kind === "file" ? [[target.parts.join("/"), target.handle]] : walk(target.handle, target.parts.join("/"));
      let scanned = 0;
      for await (const [path, handle] of files) {
        const file = await handle.getFile();
        if (file.size > MAX_FILE) continue;
        scanned += file.size;
        if (scanned > 16_000_000) throw new Error("scan byte limit reached; narrow the search path");
        let text;
        try { text = (await readText(handle)).text; } catch { continue; }
        const all = lines(text);
        for (let i = 0; i < all.length; i++) {
          if (!test(all[i])) continue;
          if (matches.length === limit) return { matches, truncated: true };
          matches.push({ path, line: i + 1, content: all[i].slice(0, 2000) });
        }
      }
      return { matches, truncated: false };
    },
    async write_file(args) { return write(args.path, args.content); },
    async replace_text(args) {
      const { text } = await current(args.path);
      if (typeof args.old_text !== "string" || !args.old_text) throw new Error("old_text must not be empty");
      if (text.split(args.old_text).length !== 2) throw new Error("old_text must match exactly once; read a narrower unique section");
      return write(args.path, text.replace(args.old_text, () => args.new_text));
    },
    async edit_lines(args) {
      const { text } = await current(args.path);
      const start = args.start_line, end = args.end_line;
      if (!Number.isInteger(start) || !Number.isInteger(end) || start < 1 || end < start) throw new Error("start_line and end_line must be a valid range");
      const pieces = text.match(/[^\r\n]*(\r\n|\n|\r)|[^\r\n]+$/g) || [];
      if (end > pieces.length) throw new Error(`range exceeds the file's ${pieces.length} existing lines; read_file again`);
      let replacement = args.content;
      if (typeof replacement !== "string") throw new Error("content must be text");
      const ending = (piece) => (piece.match(/(\r\n|\n|\r)$/) || [""])[0];
      if (replacement && !ending(replacement)) replacement += ending(pieces[end - 1]);
      const changed = pieces.slice(0, start - 1).join("") + replacement + pieces.slice(end).join("");
      return { ...(await write(args.path, changed)), start_line: start, end_line: end };
    },
    async make_directory(args) {
      const parts = segments(args.path);
      if (!parts.length) throw new Error("the folder itself cannot be changed");
      await directory(parts, true);
      return { path: parts.join("/"), created: true };
    },
    async move_path(args) {
      const source = await entry(args.source);
      const to = segments(args.destination);
      if (!source.parts.length || !to.length) throw new Error("the folder itself cannot be moved");
      if (await exists(args.destination)) throw new Error("destination already exists");
      if (to.join("/").startsWith(`${source.parts.join("/")}/`)) throw new Error("cannot move a folder into itself");
      await copy(source, to);
      await source.parent.removeEntry(source.name, { recursive: true });
      return { source: source.parts.join("/"), destination: to.join("/") };
    },
    async delete_path(args) {
      const target = await entry(args.path);
      if (!target.parts.length) throw new Error("the folder itself cannot be deleted");
      // Like the host's Trash: deleted things go to .instinct-trash, where
      // the guest can get them back.
      let name = target.name;
      const trash = await state.folder.getDirectoryHandle(".instinct-trash", { create: true });
      try { await (target.kind === "file" ? trash.getFileHandle(name) : trash.getDirectoryHandle(name)); name = `${Date.now()}-${name}`; } catch {}
      await copy(target, [".instinct-trash", name]);
      await target.parent.removeEntry(target.name, { recursive: true });
      return { path: target.parts.join("/"), trashed_to: `.instinct-trash/${name}` };
    },
  };
  async function current(path) {
    const target = await entry(path);
    if (target.kind !== "file") throw new Error("edit requires an existing file");
    const result = await readText(target.handle);
    if (state.reads.get(target.parts.join("/")) !== result.sha) throw new Error("file version changed; read it again before editing");
    return result;
  }
  async function write(path, content) {
    const parts = segments(path);
    if (!parts.length) throw new Error("cannot replace the folder");
    if (typeof content !== "string") throw new Error("content must be text");
    const bytes = new TextEncoder().encode(content);
    if (bytes.length > MAX_FILE) throw new Error("content exceeds the 2 MB write limit");
    const key = parts.join("/");
    if (await exists(key)) {
      const target = await entry(key);
      if (target.kind !== "file") throw new Error("a folder already exists at that path");
      const { sha } = await readText(target.handle);
      if (state.reads.get(key) !== sha) throw new Error("file version changed; read it again before editing");
    }
    const parent = await directory(parts.slice(0, -1), true);
    const handle = await parent.getFileHandle(parts[parts.length - 1], { create: true });
    const stream = await handle.createWritable();
    await stream.write(bytes);
    await stream.close();
    const sha = await hash(bytes);
    state.reads.set(key, sha);
    return { path: key, sha256: sha, bytes: bytes.length };
  }
  async function copy(source, to) {
    const parent = await directory(to.slice(0, -1), true);
    const name = to[to.length - 1];
    if (source.kind === "file") {
      const target = await parent.getFileHandle(name, { create: true });
      const stream = await target.createWritable();
      await stream.write(await (await source.handle.getFile()).arrayBuffer());
      await stream.close();
      return;
    }
    await parent.getDirectoryHandle(name, { create: true });
    for await (const [child, handle] of source.handle.entries()) {
      await copy({ kind: handle.kind, handle }, [...to, child]);
    }
  }

  // --- approvals --------------------------------------------------------------
  let approvals = Promise.resolve();
  function describe(name, args) {
    switch (name) {
      case "write_file": return [`Write ${args.path}`, args.content];
      case "replace_text": return [`Edit ${args.path}`, `Replace:\n${args.old_text}\n\nWith:\n${args.new_text}`];
      case "edit_lines": return [`Edit lines ${args.start_line}–${args.end_line} of ${args.path}`, args.content || "(delete these lines)"];
      case "make_directory": return [`Create folder ${args.path}`, ""];
      case "move_path": return [`Move ${args.source}`, `to ${args.destination}`];
      case "delete_path": return [`Delete ${args.path}`, "It goes to .instinct-trash in your folder."];
      default: return [name, JSON.stringify(args, null, 2)];
    }
  }
  function approve(name, args) {
    const ask = approvals.then(() => new Promise((resolve) => {
      const [title, body] = describe(name, args);
      $("approval-title").textContent = title;
      $("approval-body").textContent = String(body).slice(0, 20000);
      $("approval-body").hidden = !body;
      $("approval").returnValue = "";
      $("approval").showModal();
      $("approval").addEventListener("close", () => resolve($("approval").returnValue === "allow"), { once: true });
    }));
    approvals = ask.catch(() => {});
    return ask;
  }

  // --- chats ------------------------------------------------------------------
  function escape(value) {
    return String(value ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }
  function markdown(text) {
    const blocks = String(text || "").split(/```/);
    return blocks.map((block, i) => {
      if (i % 2) return `<pre><code>${escape(block.replace(/^[^\n]*\n/, ""))}</code></pre>`;
      return block.split(/\n{2,}/).map((para) => {
        const html = escape(para.trim()).replace(/`([^`]+)`/g, "<code>$1</code>").replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
        if (!html) return "";
        const heading = html.match(/^(#{1,3}) (.*)/);
        if (heading) return `<h${heading[1].length + 1}>${heading[2]}</h${heading[1].length + 1}>`;
        if (/^([-*]|\d+\.) /.test(html)) {
          const ordered = /^\d+\./.test(html);
          const items = html.split(/\n(?=(?:[-*]|\d+\.) )/).map((item) => `<li>${item.replace(/^([-*]|\d+\.) /, "")}</li>`).join("");
          return ordered ? `<ol>${items}</ol>` : `<ul>${items}</ul>`;
        }
        return `<p>${html.replace(/\n/g, "<br>")}</p>`;
      }).join("");
    }).join("");
  }
  function renderChats() {
    $("chats").innerHTML = state.chats.map((chat) => `<button class="chat-link" type="button" data-chat="${escape(chat.id)}"${state.chat?.id === chat.id ? ' aria-current="page"' : ""}><span>${escape(chat.title)}</span><span class="remove" data-remove="${escape(chat.id)}" title="Delete chat">×</span></button>`).join("");
  }
  function renderLog() {
    const chat = state.chat;
    $("welcome").hidden = Boolean(chat?.turns.length);
    [...$("log").querySelectorAll(".turn")].forEach((node) => node.remove());
    for (const turn of chat?.turns || []) $("log").append(turnNode(turn));
    $("log").scrollTop = $("log").scrollHeight;
  }
  function turnNode(turn) {
    const node = document.createElement("div");
    node.className = "turn";
    node.innerHTML = `<div class="user"><div class="bubble">${escape(turn.user)}</div></div><div class="steps"></div><div class="live" hidden></div><div class="reply"></div>`;
    for (const step of turn.steps || []) addStep(node, step);
    if (turn.reply !== undefined) setReply(node, turn);
    return node;
  }
  function addStep(node, step) {
    const line = document.createElement("div");
    line.className = `step ${step.state || ""}`;
    const target = step.args?.path || step.args?.source || step.args?.query || "";
    line.innerHTML = `<span class="tool">${escape(step.name)}</span> ${escape(target)}${step.state === "denied" ? " · not allowed" : step.state === "failed" ? ` · ${escape(step.error || "failed")}` : ""}`;
    node.querySelector(".steps").append(line);
  }
  function setReply(node, turn) {
    const reply = node.querySelector(".reply");
    reply.classList.toggle("failed", Boolean(turn.failed));
    reply.innerHTML = markdown(turn.reply);
    node.querySelector(".live").hidden = true;
  }
  function openChat(chat) {
    state.chat = chat;
    renderChats();
    renderLog();
  }
  $("chats").addEventListener("click", async (event) => {
    const remove = event.target.closest("[data-remove]");
    if (remove) {
      event.stopPropagation();
      await dropChat(remove.dataset.remove);
      state.chats = state.chats.filter((c) => c.id !== remove.dataset.remove);
      if (state.chat?.id === remove.dataset.remove) openChat(null); else renderChats();
      return;
    }
    const link = event.target.closest("[data-chat]");
    if (link && !state.run) openChat(state.chats.find((c) => c.id === link.dataset.chat));
  });
  $("new-chat").addEventListener("click", () => { if (!state.run) openChat(null); });

  // --- a turn -----------------------------------------------------------------
  function busy(on, text = "") {
    $("send").disabled = on;
    $("stop").hidden = !on;
    $("status").textContent = text;
  }
  $("message").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); $("composer").requestSubmit(); }
  });
  $("composer").addEventListener("submit", async (event) => {
    event.preventDefault();
    const text = $("message").value.trim();
    if (!text || state.run) return;
    const mode = $("mode").value;
    let offered = [];
    if (state.folder && (await folderReady(mode !== "read"))) {
      offered = mode === "read" ? READING : [...READING, ...MUTATING];
    }
    if (!state.chat) {
      state.chat = { id: crypto.randomUUID(), title: text.slice(0, 60), turns: [], created: Date.now(), updated: Date.now() };
      state.chats.unshift(state.chat);
    }
    const chat = state.chat;
    const history = chat.turns.filter((t) => t.reply !== undefined && !t.failed).flatMap((t) => [{ role: "user", content: t.user }, { role: "assistant", content: t.reply }]);
    const turn = { user: text, steps: [] };
    chat.turns.push(turn);
    chat.updated = Date.now();
    $("message").value = "";
    renderChats();
    $("welcome").hidden = true;
    const node = turnNode(turn);
    $("log").append(node);
    $("log").scrollTop = $("log").scrollHeight;
    busy(true, "Starting…");
    try {
      const { run_id } = await api("/guest/runs", {
        messages: history, task: text, tools: offered, model: $("model").value, effort: $("effort").value,
      });
      state.run = run_id;
      await follow(run_id, node, turn, mode);
    } catch (error) {
      turn.reply = error.message; turn.failed = true; setReply(node, turn);
    }
    state.run = null;
    busy(false);
    await saveChat(chat);
  });
  $("stop").addEventListener("click", () => {
    if (state.run) api(`/guest/runs/${state.run}/cancel`, {}).catch(() => {});
    busy(true, "Stopping…");
  });

  async function follow(runId, node, turn, mode) {
    let after = 0, failures = 0;
    const live = node.querySelector(".live");
    while (true) {
      let page;
      try {
        page = await api(`/guest/runs/${runId}/events?after=${after}&wait=20`);
        failures = 0;
      } catch (error) {
        if (error.message === "signed out" || ++failures > 8) throw error;
        await new Promise((r) => setTimeout(r, 500 * failures));
        continue;
      }
      after = page.next;
      // Show what the model is writing, up to any tool call it starts (the
      // call itself shows up as a step once it runs).
      const raw = page.partial?.root || Object.values(page.partial || {})[0] || "";
      const text = raw.split(/<\|?tool_call|<function=/)[0].trim();
      live.hidden = !text;
      if (text) { live.textContent = text; live.scrollTop = live.scrollHeight; busy(true, "Writing…"); }
      for (const event of page.events) {
        if (event.type === "tool") handleTool(runId, event, node, turn, mode);
        if (event.type === "done") {
          turn.reply = event.result || event.error || (event.status === "cancelled" ? "Stopped." : "No answer came back.");
          turn.failed = event.status !== "completed";
          setReply(node, turn);
          return;
        }
      }
    }
  }
  async function handleTool(runId, event, node, turn, mode) {
    const step = { name: event.name, args: event.args };
    let answer;
    try {
      if (!state.folder || !tools[event.name]) throw new Error("no folder is open");
      if (MUTATING.has(event.name) && mode === "read") throw new Error("the guest allowed reading only");
      if (MUTATING.has(event.name) && mode === "ask") {
        busy(true, "Waiting for you to allow a change…");
        if (!(await approve(event.name, event.args))) { step.state = "denied"; throw new Error("The user declined this change."); }
        busy(true, "Working…");
      }
      answer = { result: JSON.stringify(await tools[event.name](event.args || {})) };
    } catch (error) {
      step.state = step.state || "failed";
      step.error = error.message;
      answer = { error: error.message };
    }
    turn.steps.push(step);
    addStep(node, step);
    await api(`/guest/runs/${runId}/tools/${event.id}`, answer).catch(() => {});
  }

  start();
})();

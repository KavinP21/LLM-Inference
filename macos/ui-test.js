// UI test run inside Instinct.app by macos/test.sh. The app provides sleep,
// until, check and snapshot (see UITest in Instinct.swift).
const $ = (id) => document.getElementById(id);
const api = async (path, body) => {
  const response = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  return response.json();
};
const visible = (element) => element && !element.hidden && element.getClientRects().length > 0;
const press = (element, key, options = {}) => element.dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true, ...options }));

await check("signs in and loads the workspace", async () => {
  await until(() => document.querySelector(".workspace-name")?.textContent.trim());
  await api("/api/live");
  if (visible($("connection-error"))) throw new Error($("connection-error-text").textContent);
});
await snapshot("1-welcome");

await check("header buttons get clicks under the title bar", async () => {
  for (const id of ["toggle-sidebar", "toggle-terminal", "open-settings"]) {
    const box = $(id).getBoundingClientRect();
    const view = await window.webkit.messageHandlers.instinctTest.postMessage({ type: "hit", x: box.x + box.width / 2, y: box.y + box.height / 2 });
    if (!/^WK/.test(view)) throw new Error(`${id} is covered by ${view}`);
  }
});

await check("switches new chats to Work freely", async () => {
  $("open-settings").click();
  await until(() => $("settings").open);
  document.querySelector('input[name="policy"][value="auto"]').click();
  $("close-settings").click();
  await until(() => $("permission-label").textContent === "Work freely");
});

await check("sends a message and shows the reply", async () => {
  $("message").value = "Explain example.py";
  $("message").dispatchEvent(new Event("input", { bubbles: true }));
  press($("message"), "Enter");
  await until(() => document.querySelector(".turn .user-message")?.textContent.includes("example.py"));
  await until(() => document.querySelector(".turn-output .markdown")?.textContent.trim(), 60000);
});
await snapshot("2-reply");

await check("summarizes the chat with /compact", async () => {
  $("message").value = "/compact";
  $("message").dispatchEvent(new Event("input", { bubbles: true }));
  press($("message"), "Enter");
  await until(() => document.querySelector(".chat-summary"), 60000);
  document.querySelector(".chat-summary").open = true;
  await snapshot("2b-compact");
});

await check("opens the model and reasoning picker", async () => {
  if ($("mode-picker").hidden) return;  // Demo mode has no models to pick.
  $("mode-button").click();
  await until(() => visible($("mode-panel")));
  await snapshot("3-picker");
  press($("mode-panel"), "Escape");
  await until(() => !visible($("mode-panel")));
});

await check("runs a sandboxed terminal in the workspace", async () => {
  $("toggle-terminal").click();
  await until(() => document.querySelector("#terminal-host .xterm-rows")?.textContent.includes("workspace"), 20000);
  const input = document.querySelector("#terminal-host .xterm-helper-textarea");
  input.focus();
  input.dispatchEvent(new InputEvent("input", { data: "touch ~/instinct-ui-test; echo done-$((6*7))", inputType: "insertText", bubbles: true }));
  input.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", keyCode: 13, bubbles: true }));
  const rows = () => document.querySelector("#terminal-host .xterm-rows").textContent;
  await until(() => rows().includes("done-42"));
  if ($("terminal-badge").hidden === false && !rows().includes("not permitted")) throw new Error("the sandbox allowed a write to the home folder");
});
await snapshot("4-terminal");
$("close-terminal").click();

await check("rewinds a message back into the box", async () => {
  const turnId = document.querySelector("[data-rewind-turn]").dataset.rewindTurn;
  document.querySelector(`[data-rewind-turn="${turnId}"]`).click();
  await until(() => $("rewind-dialog").open);
  await snapshot("5-rewind");
  $("rewind-confirm").click();
  await until(() => $("message").value === "Explain example.py");
  await until(() => !document.querySelector(".turn .user-message"));
});

await check("opens settings", async () => {
  $("open-settings").click();
  await until(() => $("settings").open);
  await snapshot("6-settings");
  $("close-settings").click();
});

await check("creates and revokes an API key", async () => {
  $("open-settings").click();
  await until(() => $("settings").open);
  $("key-name").value = "UI test";
  $("key-form").requestSubmit();
  await until(() => !$("new-key").hidden && $("new-key-value").textContent.startsWith("sk-instinct-"));
  const key = $("new-key-value").textContent;
  const models = await fetch("/v1/models", { headers: { Authorization: `Bearer ${key}` } });
  if (models.status !== 200) throw new Error(`/v1/models answered ${models.status}`);
  await until(() => document.querySelector("[data-revoke-key]"));
  await snapshot("7-api-keys");
  document.querySelector("[data-revoke-key]").click();
  await until(() => !document.querySelector("[data-revoke-key]"));
  $("close-settings").click();
});

await check("opens another project in place and comes back", async () => {
  $("workspace-name").click();
  await until(() => $("projects").open && document.querySelector("[data-open-project]"));
  await snapshot("8-projects");
  const { projects, current } = await api("/api/projects");
  const first = projects.find((project) => project.id === current);
  await window.instinct.openFolder(first.path.replace(/workspace$/, "workspace-2"));
  await until(() => $("workspace-name").textContent === "workspace-2");
  await until(() => !document.querySelector(".chat-list [data-chat]"));
  $("workspace-name").click();
  await until(() => document.querySelector(`[data-open-project="${first.id}"]`));
  document.querySelector(`[data-open-project="${first.id}"]`).click();
  await until(() => $("workspace-name").textContent === first.title);
});

"""benchmarks/serve_load.py against a stand-in streaming server."""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "serve_load.py"
spec = importlib.util.spec_from_file_location("serve_load", SCRIPT)
serve_load = importlib.util.module_from_spec(spec)
sys.modules["serve_load"] = serve_load  # dataclasses look their module up here
spec.loader.exec_module(serve_load)

TOKEN = "test-token-for-the-fixture"


class Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_POST(self):
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self.send_response(401)
            self.end_headers()
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.lock:
            self.seen.append(body)
        # One "token" per word of the prompt's last message, capped; the whole
        # history before the last message counts as cached.
        prompt = sum(len(m["content"].split()) for m in body["messages"])
        cached = prompt - len(body["messages"][-1]["content"].split())
        words = ["word"] * min(body["max_tokens"], 12)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in words:
            chunk = {"choices": [{"delta": {"content": word + " "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        usage = {
            "prompt_tokens": prompt,
            "completion_tokens": len(words),
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        self.wfile.write(
            f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode()
        )
        self.wfile.write(b"data: [DONE]\n\n")


def test_users_hold_multi_turn_conversations_and_report_cache_hits():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = serve_load.Client(
            f"http://127.0.0.1:{server.server_port}", TOKEN, "m", 10
        )
        level = serve_load.run_level(client, 3, turns=3, words=200, max_tokens=8)
        summary = serve_load.summarize(level)
    finally:
        server.shutdown()
    assert summary["users"] == 3 and summary["turns"] == 9 and not summary["errors"]
    # Each request resent the whole history: system, then user/assistant pairs.
    lengths = sorted(len(body["messages"]) for body in Handler.seen)
    assert lengths == [2] * 3 + [4] * 3 + [6] * 3
    assert all(body["stream"] and body["stream_options"] for body in Handler.seen)
    assert summary["later_turns_cached_fraction"] > 0.9
    assert summary["first_turn_prompt_tokens"] > 200
    assert summary["per_user_decode_tok_per_s"]["p50"] > 0
    assert summary["ttft_later_turns_s"]["p95"] is not None


def test_documents_differ_by_user_and_stay_the_same_per_user():
    assert serve_load.document(1, 300) == serve_load.document(1, 300)
    assert serve_load.document(1, 300) != serve_load.document(2, 300)
    assert len(serve_load.document(3, 300).split()) >= 300

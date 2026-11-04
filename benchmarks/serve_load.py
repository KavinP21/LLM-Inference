"""Concurrent chat users against an OpenAI-compatible server.

Each simulated user holds a conversation of several turns, the way a chat
client does:
- The first message carries the user's own document.
- Every turn resends the whole history and streams the reply.

For each turn the benchmark records time to first token, decode rate, and the
part of the prompt the server reports as cached. From the second turn on,
that cached part should be nearly the whole prompt. Each concurrency level
starts all its users at once.

    INSTINCT_WORKER_TOKEN=... python benchmarks/serve_load.py \\
        --url http://127.0.0.1:8090 --users 1,4,8,16 --output load.json

Only the standard library is used, so it runs wherever the server does. The
bearer token is read from the environment and never printed.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import statistics
import threading
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path

SYSTEM = (
    "You are a careful assistant. Answer from the document the user shares, "
    "quote it when that helps, and say plainly when it doesn't cover something."
)
QUESTIONS = [
    "Summarize the document above in four sentences.",
    "Which part of it is the most surprising, and why?",
    "List three questions the document leaves open.",
    "Rewrite its first paragraph for a ten-year-old.",
]
WORDS = {
    "subject": "the survey team|a junior engineer|the city council|our lab|the "
    "night shift|a visiting professor|the shipping crew|the archive".split("|"),
    "verb": "measured|ignored|rebuilt|questioned|documented|delayed|tested|"
    "funded".split("|"),
    "object": "the river crossing|the old cooling system|a set of field notes|"
    "the budget|the second prototype|a cracked bearing|the vendor's claims|"
    "the evening schedule".split("|"),
    "detail": "after the storm|without telling anyone|for the third time|"
    "under a tight deadline|with borrowed tools|despite the cost|before "
    "sunrise|over two winters".split("|"),
}


def document(user: int, words: int) -> str:
    """A different, deterministic document for each user."""
    rng = random.Random(user)
    sentences, count = [], 0
    while count < words:
        sentence = (
            f"{rng.choice(WORDS['subject']).capitalize()} "
            f"{rng.choice(WORDS['verb'])} {rng.choice(WORDS['object'])} "
            f"{rng.choice(WORDS['detail'])}, and the notes put it at "
            f"{rng.randint(2, 990)} units."
        )
        sentences.append(sentence)
        count += len(sentence.split())
    paragraphs = [" ".join(sentences[i : i + 6]) for i in range(0, len(sentences), 6)]
    return "\n\n".join(paragraphs)


@dataclass
class Turn:
    user: int
    turn: int
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    ttft_s: float | None = None
    decode_tok_per_s: float | None = None
    seconds: float = 0.0
    error: str | None = None


@dataclass
class Level:
    users: int
    seconds: float
    turns: list[Turn] = field(default_factory=list)


class Client:
    def __init__(self, url: str, token: str, model: str, timeout: float) -> None:
        parsed = urllib.parse.urlparse(url)
        self.https = parsed.scheme == "https"
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or (443 if self.https else 80)
        self.token = token
        self.model = model
        self.timeout = timeout

    def _connection(self) -> http.client.HTTPConnection:
        kind = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return kind(self.host, self.port, timeout=self.timeout)

    def chat(self, messages: list[dict], max_tokens: int, seed: int, record: Turn):
        """Stream one reply, filling ``record``; returns the reply text."""
        body = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "seed": seed,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
        )
        connection = self._connection()
        started = time.perf_counter()
        first = last = None
        pieces: list[str] = []
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.token}",
                },
            )
            response = connection.getresponse()
            if response.status != 200:
                record.error = f"HTTP {response.status}: {response.read(300)!r}"
                return ""
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    record.prompt_tokens = usage.get("prompt_tokens", 0)
                    record.completion_tokens = usage.get("completion_tokens", 0)
                    details = usage.get("prompt_tokens_details") or {}
                    record.cached_tokens = details.get("cached_tokens", 0)
                for choice in chunk.get("choices", ()):
                    delta = choice.get("delta") or {}
                    text = delta.get("content") or delta.get("reasoning_content")
                    if text:
                        now = time.perf_counter()
                        first = first or now
                        last = now
                        if delta.get("content"):
                            pieces.append(delta["content"])
        except (OSError, http.client.HTTPException, ValueError) as exc:
            record.error = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()
        record.seconds = round(time.perf_counter() - started, 3)
        if first is not None:
            record.ttft_s = round(first - started, 3)
            if last > first and record.completion_tokens > 1:
                record.decode_tok_per_s = round(
                    (record.completion_tokens - 1) / (last - first), 1
                )
        return "".join(pieces)


def conversation(
    client: Client,
    user: int,
    *,
    turns: int,
    words: int,
    max_tokens: int,
    start: threading.Barrier,
    out: list[Turn],
) -> None:
    messages = [{"role": "system", "content": SYSTEM}]
    start.wait()
    for index in range(turns):
        question = QUESTIONS[index % len(QUESTIONS)]
        if index == 0:
            question = document(user, words) + "\n\n" + question
        messages.append({"role": "user", "content": question})
        record = Turn(user=user, turn=index + 1)
        reply = client.chat(
            messages, max_tokens, seed=user * 100 + index, record=record
        )
        out.append(record)
        if record.error:
            return
        messages.append({"role": "assistant", "content": reply})


def run_level(client: Client, users: int, **options) -> Level:
    records: list[Turn] = []
    start = threading.Barrier(users)
    # Offset user ids per level so no level reuses another's cached documents.
    base = users * 1000
    threads = [
        threading.Thread(
            target=conversation,
            args=(client, base + i),
            kwargs=dict(options, start=start, out=records),
        )
        for i in range(users)
    ]
    began = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return Level(users, time.perf_counter() - began, records)


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def summarize(level: Level) -> dict:
    ok = [t for t in level.turns if not t.error]
    first = [t for t in ok if t.turn == 1]
    later = [t for t in ok if t.turn > 1]
    rates = [t.decode_tok_per_s for t in ok if t.decode_tok_per_s]

    def ttft(group):
        values = [t.ttft_s for t in group if t.ttft_s is not None]
        return {
            "p50": _percentile(values, 0.5),
            "p95": _percentile(values, 0.95),
            "max": max(values, default=None),
        }

    prompt_later = sum(t.prompt_tokens for t in later)
    return {
        "users": level.users,
        "seconds": round(level.seconds, 2),
        "turns": len(level.turns),
        "errors": [t.error for t in level.turns if t.error],
        "aggregate_tok_per_s": round(
            sum(t.completion_tokens for t in ok) / level.seconds, 1
        ),
        "per_user_decode_tok_per_s": {
            "p50": _percentile(rates, 0.5),
            "min": min(rates, default=None),
        },
        "first_turn_prompt_tokens": round(
            statistics.mean([t.prompt_tokens for t in first]), 1
        )
        if first
        else None,
        "ttft_first_turn_s": ttft(first),
        "ttft_later_turns_s": ttft(later),
        "later_turns_cached_fraction": round(
            sum(t.cached_tokens for t in later) / prompt_later, 3
        )
        if prompt_later
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--url", default="http://127.0.0.1:8090")
    parser.add_argument("--token-env", default="INSTINCT_WORKER_TOKEN")
    parser.add_argument("--model", default="", help="sent as the request's model")
    parser.add_argument("--users", default="1,2,4,8", help="concurrency levels")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument(
        "--document-words", type=int, default=1500, help="size of each user's document"
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        parser.error(f"set the server's bearer token in {args.token_env}")
    levels = [int(n) for n in args.users.split(",") if n.strip()]
    if not levels or min(levels) < 1:
        parser.error("--users takes positive integers, e.g. 1,4,8")
    client = Client(args.url, token, args.model, args.timeout)
    options = dict(
        turns=args.turns, words=args.document_words, max_tokens=args.max_tokens
    )
    # One short request first, so loading and compiling don't count.
    warm = Turn(user=-1, turn=0)
    client.chat([{"role": "user", "content": "Say hi."}], 8, 0, warm)
    if warm.error:
        raise SystemExit(f"the server didn't answer: {warm.error}")
    results = []
    for users in levels:
        level = run_level(client, users, **options)
        summary = summarize(level)
        results.append({**summary, "detail": [asdict(t) for t in level.turns]})
        print(json.dumps(summary), flush=True)
    report = {
        "url": args.url,
        "turns": args.turns,
        "document_words": args.document_words,
        "max_tokens": args.max_tokens,
        "levels": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()

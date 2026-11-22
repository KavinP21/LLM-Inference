"""Using another Mac's models: this app as a guest of a host's Instinct.

The guest pastes the invite link the host sent. The app keeps it in a 0600
file in its own folder, and from then on starts with two remote workers, the
host's main and sub-agent models, reached through the host's guest gateway
(``gateway.py``) with the invite as their bearer token. Everything else stays
on this Mac: projects, chats, the journal, the terminal and every command and
file change. Only model calls (the conversation so far and tool results)
travel to the host.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from ..agents.backends.remote import is_loopback

TOKEN_ENV = "INSTINCT_HOST_INVITE"
INVITE = re.compile(r"^inv_[A-Za-z0-9_-]{20,100}$")


def parse_link(link: Any) -> tuple[str, str]:
    """``https://host.ts.net/#invite=inv_...`` -> (base URL, invite)."""
    if not isinstance(link, str) or len(link) > 600:
        raise ValueError("paste the whole invite link")
    parsed = urllib.parse.urlsplit(link.strip())
    invite = urllib.parse.parse_qs(parsed.fragment).get("invite", [""])[0]
    if not INVITE.match(invite):
        raise ValueError("that link has no invite in it")
    if parsed.scheme not in {"https", "http"} or not parsed.hostname:
        raise ValueError("the link must start with https://")
    if parsed.scheme == "http" and not is_loopback(parsed.hostname):
        raise ValueError("a host on another machine must use https")
    return f"{parsed.scheme}://{parsed.netloc}", invite


def health(url: str, invite: str, timeout: float = 10) -> dict[str, Any]:
    """Ask the host's main model about itself, checking the invite."""
    request = urllib.request.Request(
        f"{url}/guest/models/driver/health",
        headers={"Authorization": f"Bearer {invite}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read(1 << 20))
    except urllib.error.HTTPError as exc:
        if exc.code == 401:
            raise ValueError(
                "the host turned that invite down: it may have expired or been revoked"
            ) from None
        raise ValueError(f"the host answered {exc.code}") from None
    except (OSError, ValueError) as exc:
        raise ValueError(f"couldn't reach the host ({type(exc).__name__})") from None


class HostConnection:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, str] | None:
        try:
            saved = json.loads(self.path.read_text())
            url, invite = saved["url"], saved["invite"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        if not isinstance(url, str) or not INVITE.match(str(invite)):
            return None
        return {"url": url, "invite": invite}

    def connect(self, link: Any) -> dict[str, Any]:
        url, invite = parse_link(link)
        identity = health(url, invite)
        if not identity.get("ready"):
            raise ValueError("the host's models aren't loaded yet; try again shortly")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump({"url": url, "invite": invite}, stream)
        os.replace(temporary, self.path)
        return self.status()

    def disconnect(self) -> dict[str, Any]:
        self.path.unlink(missing_ok=True)
        return self.status()

    def status(self) -> dict[str, Any]:
        saved = self.load()
        return {"connected": saved is not None, "host": saved["url"] if saved else None}

    def config(self) -> dict[str, Any] | None:
        """A worker configuration for the saved host, or None.

        Puts the invite in this process's environment, where remote workers
        read their token; it is never written anywhere else.
        """
        saved = self.load()
        if saved is None:
            return None
        os.environ[TOKEN_ENV] = saved["invite"]
        try:
            identity = health(saved["url"], saved["invite"])
            context = int(identity.get("max_model_length") or 65536)
            concurrent = int(identity.get("max_pending") or 2)
        except (ValueError, TypeError):
            # Start anyway; the workers report the problem when they're used.
            context, concurrent = 65536, 2

        def worker(name: str, group: str) -> dict[str, Any]:
            return {
                "type": "remote",
                "name": name,
                "group": group,
                "url": saved["url"],
                "route": group,
                "token_env": TOKEN_ENV,
                "max_concurrency": max(1, concurrent),
                "request_timeout": 900,
                "connect_timeout": 15,
            }

        return {
            "workers": [
                worker("host-main", "driver"),
                worker("host-subagent", "subagent"),
            ],
            "runtime": {
                "driver_group": "driver",
                "subagent_group": "subagent",
                "max_context_tokens": context,
                "max_output_tokens": min(8192, context // 4),
                "max_agents": 6,
                "max_depth": 1,
                "max_concurrent_generations": max(1, concurrent),
                "max_steps_per_agent": 200,
                "max_total_tokens": 50_000_000,
                "deadline_seconds": 14400,
                "generation_timeout_seconds": 900,
                "tool_timeout_seconds": 390,
                "completion_timeout_seconds": 390,
                "max_tool_result_chars": 8000,
            },
        }

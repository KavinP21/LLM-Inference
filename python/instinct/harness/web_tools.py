"""Web search and page fetching for the agent.

Search uses DuckDuckGo's lite HTML page, which needs no account. If the search
page asks for verification, the tool reports that instead of retrying. Pages
are fetched over HTTP(S) from public addresses only, never from this machine
or the local network, and are returned as plain text.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from ..agents.tools import ToolValidationError
from ..common import Toolset, ToolSpec

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) Instinct/0.1"
SEARCH_URL = "https://lite.duckduckgo.com/lite/"
MAX_PAGE_BYTES = 3 << 20
SKIPPED = {"script", "style", "noscript", "svg", "template", "head"}
BLOCKS = {
    "p", "div", "section", "article", "main", "header", "footer", "li", "ul",
    "ol", "tr", "table", "pre", "blockquote", "h1", "h2", "h3", "h4", "h5",
    "h6", "br", "hr", "dt", "dd",
}  # fmt: skip


def _context() -> ssl.SSLContext:
    # python.org builds ship without CA certificates; certifi has them.
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


class _Results(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results: list[dict] = []
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if tag == "a" and "result-link" in classes:
            link = attributes.get("href") or ""
            target = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
            url = target.get("uddg", [link])[0]
            self.results.append({"title": "", "url": url, "snippet": ""})
            self._field = "title"
        elif tag == "td" and "result-snippet" in classes and self.results:
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag in {"a", "td"}:
            self._field = None

    def handle_data(self, data):
        if self._field and self.results:
            result = self.results[-1]
            result[self._field] = " ".join((result[self._field] + " " + data).split())


class _Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.main: list[str] = []  # Text inside <main> or <article>, if any.
        self.title = ""
        self._skip = 0
        self._content = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "title":
            self._in_title = True
        if tag in SKIPPED:
            self._skip += 1
        if tag in {"main", "article"}:
            self._content += 1
        if tag in BLOCKS:
            self._add("\n")
        if tag == "li":
            self._add("- ")

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        if tag in SKIPPED and self._skip:
            self._skip -= 1
        if tag in {"main", "article"} and self._content:
            self._content -= 1
        if tag in BLOCKS:
            self._add("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:
            self._add(data)

    def _add(self, text: str) -> None:
        self.parts.append(text)
        if self._content:
            self.main.append(text)

    def text(self) -> str:
        # Navigation and footers usually sit outside <main>/<article>.
        body = "".join(self.main) if "".join(self.main).strip() else "".join(self.parts)
        lines = (" ".join(line.split()) for line in body.split("\n"))
        text, blank = [], False
        for line in lines:
            if line or not blank:
                text.append(line)
            blank = not line
        return "\n".join(text).strip()


def _check_public(url: str) -> None:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise ToolValidationError("only http and https URLs can be fetched")
    try:
        addresses = {
            info[4][0] for info in socket.getaddrinfo(parts.hostname, parts.port or 443)
        }
    except socket.gaierror as exc:
        raise ToolValidationError(f"could not resolve {parts.hostname}") from exc
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            raise ToolValidationError(
                "only public addresses can be fetched; use run_command for local servers"
            )


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # Redirects are followed by hand so each hop is checked.


class WebTools(Toolset):
    specs = (
        ToolSpec(
            "web_search",
            "Search the web. Returns titles, URLs and snippets; use fetch_url to read a result. Prefer official documentation.",
            {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            "web_search",
        ),
        ToolSpec(
            "fetch_url",
            "Read a web page or text file from a public URL as plain text.",
            {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "max_chars": {"type": "integer", "minimum": 500, "maximum": 40000},
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "fetch_url",
        ),
    )
    replay_safe_tools = frozenset({"web_search", "fetch_url"})

    def __init__(self, *, timeout: float = 15.0):
        self.timeout = timeout
        self._opener = urllib.request.build_opener(
            _NoRedirects, urllib.request.HTTPSHandler(context=_context())
        )

    def _get(self, url: str) -> tuple[str, str, bytes]:
        for _ in range(5):
            _check_public(url)
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept-Language": "en"}
            )
            try:
                response = self._opener.open(request, timeout=self.timeout)
            except urllib.error.HTTPError as error:
                if error.code in {301, 302, 303, 307, 308} and error.headers.get(
                    "Location"
                ):
                    url = urllib.parse.urljoin(url, error.headers["Location"])
                    continue
                raise ToolValidationError(f"HTTP {error.code} from {url}") from error
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                raise ToolValidationError(f"could not fetch {url}: {exc}") from exc
            with response:
                body = response.read(MAX_PAGE_BYTES + 1)
                kind = response.headers.get_content_type()
                return url, kind, body[:MAX_PAGE_BYTES]
        raise ToolValidationError("too many redirects")

    def _search(self, args: dict) -> str:
        if not isinstance(args, dict) or set(args) - {"query", "limit"}:
            raise ToolValidationError("web_search accepts query and limit")
        query, limit = args.get("query"), args.get("limit", 8)
        if not isinstance(query, str) or not query.strip() or len(query) > 400:
            raise ToolValidationError("query must be 1 to 400 characters")
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ToolValidationError("limit must be between 1 and 20")
        _, _, body = self._get(SEARCH_URL + "?" + urllib.parse.urlencode({"q": query}))
        page = body.decode("utf-8", "replace")
        parser = _Results()
        parser.feed(page)
        if not parser.results and "anomaly" in page.lower():
            raise ToolValidationError(
                "the search provider asked for verification; try again later"
            )
        return json.dumps(
            {"query": query, "results": parser.results[:limit]}, ensure_ascii=False
        )

    def _fetch(self, args: dict) -> str:
        if not isinstance(args, dict) or set(args) - {"url", "max_chars"}:
            raise ToolValidationError("fetch_url accepts url and max_chars")
        url, limit = args.get("url"), args.get("max_chars", 12000)
        if not isinstance(url, str) or len(url) > 2048:
            raise ToolValidationError("url must be a string of at most 2048 characters")
        if type(limit) is not int or not 500 <= limit <= 40000:
            raise ToolValidationError("max_chars must be between 500 and 40000")
        final, kind, body = self._get(url)
        text = body.decode("utf-8", "replace")
        title = ""
        if kind in {"text/html", "application/xhtml+xml"}:
            parser = _Text()
            parser.feed(text)
            text, title = parser.text(), " ".join(parser.title.split())
        elif not (
            kind.startswith("text/") or kind in {"application/json", "application/xml"}
        ):
            raise ToolValidationError(f"{kind} content can't be read as text")
        return json.dumps(
            {
                "url": final,
                "title": title,
                "content": text[:limit],
                "truncated": len(text) > limit,
            },
            ensure_ascii=False,
        )

    async def web_search(self, args: dict) -> str:
        return await asyncio.to_thread(self._search, args)

    async def fetch_url(self, args: dict) -> str:
        return await asyncio.to_thread(self._fetch, args)

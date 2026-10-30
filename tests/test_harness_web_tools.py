"""Web tools without the network: result parsing, page text, address checks."""

import pytest
from instinct.agents.tools import ToolValidationError
from instinct.harness.web_tools import _check_public, _Results, _Text

RESULTS = """
<table><tr><td><a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html&amp;rut=x" class='result-link'>asyncio — Asynchronous I/O</a></td></tr>
<tr><td class='result-snippet'>asyncio is a library to write <b>concurrent</b> code.</td></tr></table>
"""


def test_search_results_are_parsed_with_real_urls():
    parser = _Results()
    parser.feed(RESULTS)
    assert parser.results == [
        {
            "title": "asyncio — Asynchronous I/O",
            "url": "https://docs.python.org/3/library/asyncio.html",
            "snippet": "asyncio is a library to write concurrent code.",
        }
    ]


def test_page_text_prefers_the_main_content():
    page = _Text()
    page.feed(
        "<html><head><title>Docs</title><style>x{}</style></head><body>"
        "<nav>Menu Theme</nav><main><h1>Tasks</h1><p>Run <b>things</b>.</p>"
        "<ul><li>one</li></ul></main><footer>Copyright</footer></body></html>"
    )
    assert page.title == "Docs"
    assert page.text() == "Tasks\n\nRun things.\n\n- one"


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1:8000/", "http://localhost/", "ftp://example.com/x"]
)
def test_only_public_http_urls_are_fetched(url):
    with pytest.raises(ToolValidationError):
        _check_public(url)

# SPDX-License-Identifier: Apache-2.0
"""Check that each DOC quote really appears on its page (numbers exact, words mostly present).

A DOC tag means "the page was fetched and the quote read in it". This re-fetches HTML pages with the standard library
(and PDFs when the optional ``pypdf`` package is importable) and checks every quoted number and most quoted words
against the page text. Pages that block scripted fetches (and PDFs without ``pypdf``) come back ``unchecked`` (re-check those by hand or with a browser-capable agent); ``missing`` means the page
was read and the quote is not in it, so the evidence must be fixed or re-tagged. ``reworded`` means every quoted
number is on the page (often in embedded product data) but the wording is not: replace the quote with the page's own
text. Quotes gathered through a summarising fetch tool are often reworded this way.

Example:
    >>> quote_found("Density 2.52 g/cm3 (container)", "Density at 20 C ... container 2.52 g/cm3 flat 2.53")
    True
    >>> quote_found("Density 2.62 g/cm3", "Density at 20 C ... container 2.52 g/cm3")
    False
"""

from __future__ import annotations

import html
import http.client
import io
import logging
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Final, Literal

from alloy.sourcing.fields import yaml_load
from alloy.sourcing.record import Evidence

log = logging.getLogger("alloy.sourcing")

Status = Literal["found", "reworded", "missing", "unchecked"]
STATUSES: Final[tuple[str, ...]] = ("found", "reworded", "missing", "unchecked")
Fetch = Callable[[str], str | None]
WORD_SHARE: Final[float] = 0.8
_NUM = re.compile(r"\d+(?:[.,]\d+)?")
_WORD = re.compile(r"[a-z]{3,}")
_UA: Final[str] = "Mozilla/5.0 (X11; Linux x86_64) alloy-sourcing-verify/0.1"
_MAX_BYTES: Final[int] = 20_000_000
_HEADERS: Final[tuple[dict[str, str], ...]] = (
    {"User-Agent": _UA, "Accept-Language": "en", "Accept": "text/html,application/xhtml+xml,application/pdf,*/*;q=0.8"},
    {"User-Agent": "alloy-sourcing-verify/0.1", "Accept": "*/*"},  # some sites serve a trimmed page to the first set
)
_ATTEMPTS: dict[str, int] = {}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", html.unescape(text)).lower()
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM.findall(text)}


def quote_found(quote: str, page: str) -> bool:
    """True if every number in ``quote`` is in ``page`` and at least 80 % of its words (3+ letters) are."""
    q, p = _norm(quote), _norm(page)
    if not q.strip():
        return False
    if q in p:
        return True
    nums, page_nums = _numbers(q), _numbers(p)
    if not nums <= page_nums:
        return False
    words = set(_WORD.findall(q))
    if not words:
        return bool(nums)
    page_words = set(_WORD.findall(p))
    return len(words & page_words) / len(words) >= WORD_SHARE


class _Text(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("script", "style", "noscript"):
            self._skip += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "noscript") and self._skip:
            self._skip -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def check_quote(quote: str, page: str) -> Status:
    """``found``, ``reworded`` (all numbers on the page, wording not) or ``missing``.

    Example:
        >>> check_quote("Estimated ship weight: 1.04 pounds", '{"weight":1.04,"weight_unit_of_measure":"POUND"}')
        'reworded'
    """
    if quote_found(quote, page):
        return "found"
    nums = _numbers(_norm(quote))
    return "reworded" if nums and nums <= _numbers(_norm(page)) else "missing"


def html_text(raw: str) -> str:
    """Visible text of an HTML page (scripts and styles dropped)."""
    parser = _Text()
    parser.feed(raw)
    return " ".join(parser.parts)


def pdf_text(data: bytes) -> str | None:
    """Text of a PDF via the optional ``pypdf`` package; None if it is not installed or the file does not parse."""
    try:
        import pypdf  # optional: pip install pypdf
    except ImportError:
        return None
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:  # noqa: BLE001 (any parse failure means "unchecked", never a crash)
        log.debug("verify: cannot read PDF: %s", e)
        return None


def fetch_text(url: str, timeout: float = 20.0) -> str | None:
    """Visible text of an HTML URL followed by its raw markup (embedded product data counts), or a PDF's text;
    None for non-HTTP URLs, unreadable PDFs and failures (blocked, timeout)."""
    if not url.startswith(("http://", "https://")):
        return None
    safe = urllib.parse.quote(url, safe=":/?#[]@!$&'()*+,;=%~")  # spaces etc. in hand-copied URLs
    try:
        attempt = _ATTEMPTS[url] = _ATTEMPTS.get(url, -1) + 1  # a re-fetch of the same URL uses the next header set
        req = urllib.request.Request(safe, headers=_HEADERS[attempt % len(_HEADERS)])
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (http(s) only, checked above)
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read(_MAX_BYTES)
    except (urllib.error.URLError, http.client.HTTPException, TimeoutError, OSError, ValueError) as e:
        log.debug("verify: cannot fetch %s: %s", url, e)
        return None
    if "pdf" in ctype or data[:5] == b"%PDF-":
        return pdf_text(data)
    if "html" not in ctype and "text" not in ctype:
        return None
    raw = data.decode("utf-8", errors="replace")
    return html_text(raw) + "\n" + raw


@dataclass(frozen=True)
class Check:
    """One DOC evidence item checked against its page."""

    file: str
    field: str
    url: str
    quote: str
    status: Status


def _doc_evidence(path: Path) -> Iterator[tuple[str, Evidence]]:
    raw = yaml_load(path.read_text(encoding="utf-8")) or {}
    for fname, val in (raw.get("values") or {}).items():
        for ev in val.get("evidence") or ():
            if ev.get("tag") == "DOC":
                yield fname, Evidence(url=str(ev["url"]), quote=str(ev.get("quote", "")), tag="DOC")


def verify_files(paths: list[Path], fetch: Fetch = fetch_text, retry_delay_s: float = 5.0) -> list[Check]:
    """Check every DOC quote in the given cache or record files; each URL is fetched once, and once more (after
    ``retry_delay_s``, with the other header set) when a quote is not found (throttled sites answer bursts with a
    trimmed page)."""
    pages: dict[str, str | None] = {}
    refetched: set[str] = set()
    out: list[Check] = []
    for path in paths:
        for fname, ev in _doc_evidence(path):
            if ev.url not in pages:
                pages[ev.url] = fetch(ev.url)
            page = pages[ev.url]
            status: Status = "unchecked" if page is None else check_quote(ev.quote, page)
            if status == "missing" and ev.url not in refetched:  # one re-fetch per URL: throttled or bot-check pages
                refetched.add(ev.url)
                time.sleep(retry_delay_s)
                pages[ev.url] = page = fetch(ev.url)
                status = "unchecked" if page is None else check_quote(ev.quote, page)
            out.append(Check(str(path), fname, ev.url, ev.quote, status))
    log.info("verify: %d DOC quotes, %d missing, %d reworded", len(out), sum(c.status == "missing" for c in out),
             sum(c.status == "reworded" for c in out))
    return out


def to_json(checks: list[Check]) -> dict[str, object]:
    """Counts by status plus the missing, reworded and unchecked items."""
    counts = {s: sum(c.status == s for c in checks) for s in STATUSES}
    return {"counts": counts, **{s: [asdict(c) for c in checks if c.status == s] for s in STATUSES[1:]}}

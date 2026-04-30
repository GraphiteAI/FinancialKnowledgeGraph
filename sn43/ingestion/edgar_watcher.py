"""
Real-time SEC EDGAR filing watcher.

Polls the public EDGAR "latest filings" endpoint and converts fresh filings
into `EventTask` broadcasts on an `EventBus`. Dedupes on accession number so
a filing is only ever emitted once.

SEC requires a descriptive User-Agent on every request; callers pass one
explicitly (default uses a SN43-branded identifier + contact email from env).

Docs: https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Iterable, Optional
from urllib.parse import urlparse

import httpx

from ingestion.event_bus import EventBus
from sn43.protocol import EventTask, EventType

logger = logging.getLogger("sn43.edgar_watcher")

EDGAR_LATEST_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type={form_type}&company=&dateb=&owner=include&count=40&output=atom"
)
DEFAULT_FORM_TYPES = ("8-K", "10-K", "10-Q", "DEF 14A", "S-1")

# Mapping from SEC form type to our EventType enum
_FORM_TO_EVENT: dict[str, EventType] = {
    "8-K": EventType.EDGAR_8K,
    "10-K": EventType.EDGAR_10K,
    "10-Q": EventType.EDGAR_10Q,
    "DEF 14A": EventType.EDGAR_DEF14A,
    "S-1": EventType.EDGAR_S1,
}


@dataclass
class EdgarFiling:
    """A single filing parsed from the EDGAR atom feed."""
    accession_number: str
    form_type: str
    company_name: str
    cik: str
    filed_at: datetime
    href: str

    def as_event_task(self, deadline_seconds: int = 60) -> EventTask:
        event_type = _FORM_TO_EVENT.get(self.form_type, EventType.EDGAR_OTHER)
        return EventTask(
            event_type=event_type,
            source_url=self.href,
            event_timestamp=self.filed_at,
            deadline=self.filed_at + timedelta(seconds=deadline_seconds),
            dedupe_key=f"edgar:{self.accession_number}",
            subject_cik=self.cik,
        )


# ──────────────────────────── Atom parsing ────────────────────────────

# The EDGAR atom feed has a predictable structure. We avoid an XML lib
# dependency by scraping with regex — robust enough for the <entry> shape
# SEC has shipped for a decade.
_ENTRY_RE = re.compile(r"<entry>(.*?)</entry>", re.DOTALL)
_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.DOTALL)
_LINK_RE = re.compile(r'<link[^>]+href="([^"]+)"')
_UPDATED_RE = re.compile(r"<updated>(.*?)</updated>")
_CATEGORY_RE = re.compile(r'<category[^>]+term="([^"]+)"')
# Title format: "FORM_TYPE - COMPANY NAME (CIK) (Filer)"
_TITLE_PARSE_RE = re.compile(
    r"^(?P<form>[A-Z0-9/\- ]+)\s+-\s+(?P<company>.+?)\s+\((?P<cik>\d{1,10})\)"
)
# Accession numbers appear as the last path segment before -index.htm
_ACCESSION_RE = re.compile(r"/(?P<acc>\d{10}-\d{2}-\d{6})-index\.htm")


def parse_atom_feed(body: str) -> list[EdgarFiling]:
    """Parse the EDGAR atom feed into `EdgarFiling` records."""
    filings: list[EdgarFiling] = []
    for entry in _ENTRY_RE.findall(body):
        title_match = _TITLE_RE.search(entry)
        link_match = _LINK_RE.search(entry)
        updated_match = _UPDATED_RE.search(entry)
        if not (title_match and link_match and updated_match):
            continue

        title = _decode_xml(title_match.group(1).strip())
        href = link_match.group(1).strip()
        updated = updated_match.group(1).strip()

        parsed_title = _TITLE_PARSE_RE.match(title)
        if not parsed_title:
            continue

        form_type = parsed_title.group("form").strip()
        company = parsed_title.group("company").strip()
        cik = parsed_title.group("cik")

        accession_match = _ACCESSION_RE.search(href)
        if not accession_match:
            continue
        accession_number = accession_match.group("acc")

        try:
            filed_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
            if filed_at.tzinfo is None:
                filed_at = filed_at.replace(tzinfo=timezone.utc)
            filed_at = filed_at.astimezone(timezone.utc).replace(tzinfo=None)
        except ValueError:
            continue

        filings.append(
            EdgarFiling(
                accession_number=accession_number,
                form_type=form_type,
                company_name=company,
                cik=cik,
                filed_at=filed_at,
                href=href,
            )
        )
    return filings


def _decode_xml(s: str) -> str:
    return (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&apos;", "'")
    )


# ──────────────────────────── Watcher ────────────────────────────

class EdgarWatcher:
    """Polls EDGAR on an interval and publishes each new filing as an EventTask."""

    def __init__(
        self,
        bus: EventBus,
        user_agent: Optional[str] = None,
        form_types: Iterable[str] = DEFAULT_FORM_TYPES,
        poll_interval_seconds: float = 10.0,
        fetch: Optional[Callable[[str, str], Awaitable[str]]] = None,
    ) -> None:
        self.bus = bus
        self.user_agent = user_agent or _default_user_agent()
        self.form_types = tuple(form_types)
        self.poll_interval = poll_interval_seconds
        self._fetch = fetch or self._default_fetch
        self._stop = asyncio.Event()

    async def run(self) -> None:
        """Poll forever. Use `stop()` to exit."""
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except Exception as e:  # noqa: BLE001
                logger.error(f"poll error: {e}")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    def stop(self) -> None:
        self._stop.set()

    async def poll_once(self) -> list[EventTask]:
        """Poll every configured form type once, publish new filings."""
        published: list[EventTask] = []
        for form_type in self.form_types:
            url = EDGAR_LATEST_URL.format(form_type=form_type.replace(" ", "+"))
            body = await self._fetch(url, self.user_agent)
            for filing in parse_atom_feed(body):
                task = filing.as_event_task()
                if await self.bus.publish(task):
                    published.append(task)
        return published

    async def _default_fetch(self, url: str, user_agent: str) -> str:
        headers = {"User-Agent": user_agent, "Accept": "application/atom+xml"}
        async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text


def _default_user_agent() -> str:
    """SEC requires a descriptive UA with contact info.
    """
    contact = os.environ.get("SN43_EDGAR_CONTACT", "contact@example.com")
    return f"SN43/Phase2 ({contact})"

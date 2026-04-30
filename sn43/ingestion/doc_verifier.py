"""Validator-side document verifier.

Three submission paths, three outcomes:

  A. Whitelisted source (sec.gov etc.): validator fetches URL, hashes,
     compares to miner's claim. Match → `verified=True`. Mismatch or
     fetch fail → not verified.

  B. Non-whitelisted public source (bloomberg.com, etc.): validator
     does NOT fetch — too slow, paywalled, or the source isn't on the
     trusted list. Item enters the graph as `verified=False, source=URL_ONLY`.

  C. Paywalled source: miner provides bytes inline. Validator hashes
     them, confirms they match the miner's claim, stores them as
     evidence. `verified=False, source=PAYWALLED_CONTENT`.

Only path A unlocks `verified=True`. Paths B and C land in the graph as
unverified — useful tips that downstream queries can opt into.

Slashing: if a verified Fact later contradicts an unverified one, the
contributing miners get slashed (see graph_store).
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from typing import Dict, Iterable, Optional, Set, Tuple
from urllib.parse import urlparse

import httpx

logger = logging.getLogger("sn43.doc_verifier")

# SEC EDGAR's published cap is 10 req/sec; 
_SEC_CONCURRENCY = 4
_SEC_MIN_DELAY = 0.15 

# Default whitelist — only these domains get hash-verified by independent fetch.
# Subdomain matching is automatic (host.endswith("." + d)), so bare domain entries
# cover any subdomain. Override via WHITELIST_DOMAINS env var (comma-separated).
#
# Inclusion bar: the source publishes stable, dated articles/filings that don't
# silently mutate post-publication. News corrections are fine — those break the
# hash, which is correct (the bytes have changed, verification fails). Random
# blogs and aggregators are NOT on the whitelist; miners can submit them as
# `url_only` (path B) instead.
_DEFAULT_WHITELIST = {
    # ── Government filings & patents ──
    "sec.gov",            # SEC EDGAR
    "uspto.gov",          # US Patent and Trademark Office
    "patents.google.com", # Google Patents (mirror of USPTO + foreign offices)
    "epo.org",            # European Patent Office
    "wipo.int",           # World Intellectual Property Organization
    "courtlistener.com",  # US federal court records (RECAP)

    # ── Established financial news ──
    "reuters.com",
    "ft.com",
    "bloomberg.com",
    "wsj.com",
    "cnbc.com",
    "marketwatch.com",
    "barrons.com",
    "forbes.com",
    "businessinsider.com",
    "fool.com",           # The Motley Fool
    "investors.com",      # Investor's Business Daily
    "morningstar.com",
    "seekingalpha.com",
    "finance.yahoo.com",  # earnings & filings reposts (note: subdomain-only)

    # ── Wire services & general news ──
    "apnews.com",
    "afp.com",
    "nytimes.com",
    "washingtonpost.com",
    "bbc.com",
    "bbc.co.uk",
    "economist.com",
    "theguardian.com",
    "axios.com",
    "politico.com",
    "npr.org",

    # ── Tech / industry trade press ──
    "techcrunch.com",
    "theverge.com",
    "wired.com",
    "arstechnica.com",
}


def _load_whitelist() -> Set[str]:
    raw = os.getenv("WHITELIST_DOMAINS", "")
    if not raw.strip():
        return set(_DEFAULT_WHITELIST)
    return {d.strip().lower() for d in raw.split(",") if d.strip()}


def _domain_of(url: str) -> str:
    return (urlparse(url).netloc or "").lower()


def is_whitelisted(url: str, whitelist: Optional[Set[str]] = None) -> bool:
    """True if the URL's host is in the whitelist (exact or subdomain match)."""
    wl = whitelist if whitelist is not None else _load_whitelist()
    host = _domain_of(url)
    if not host:
        return False
    if host in wl:
        return True
    return any(host.endswith("." + d) for d in wl)


class DocVerifier:
    """Fetch + hash-verify source documents referenced in a delta.

    One instance is shared across submissions on the validator. Maintains
    a single httpx client (connection reuse) and a semaphore to throttle.
    """

    def __init__(
        self,
        doc_store,
        edgar_contact: Optional[str] = None,
        whitelist: Optional[Set[str]] = None,
    ) -> None:
        self.doc_store = doc_store
        self.whitelist = whitelist if whitelist is not None else _load_whitelist()
        logger.info(f"doc verifier whitelist: {sorted(self.whitelist)}")
        contact = edgar_contact or os.getenv(
            "SN43_EDGAR_CONTACT", "validator@example.com"
        )
        self._client = httpx.AsyncClient(
            timeout=30,
            headers={
                "User-Agent": f"SN43Validator ({contact})",
                "Accept-Encoding": "gzip, deflate",
            },
            follow_redirects=True,
        )
        self._sem = asyncio.Semaphore(_SEC_CONCURRENCY)
        self._last_fetch = 0.0
        self._fetch_lock = asyncio.Lock()
        self._in_flight: Dict[str, asyncio.Future] = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def _throttled_fetch(self, url: str) -> Optional[bytes]:
        """Fetch with concurrency cap + minimum delay; None on failure."""
        async with self._sem:
            async with self._fetch_lock:
                elapsed = time.monotonic() - self._last_fetch
                if elapsed < _SEC_MIN_DELAY:
                    await asyncio.sleep(_SEC_MIN_DELAY - elapsed)
                self._last_fetch = time.monotonic()
            try:
                r = await self._client.get(url)
                r.raise_for_status()
                return r.content
            except Exception as e:
                logger.warning(f"fetch failed for {url}: {e}")
                return None

    async def _fetch_once(self, url: str) -> Optional[bytes]:
        """Fetch each URL at most once even under concurrent requests."""
        existing = self._in_flight.get(url)
        if existing is not None:
            return await existing

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._in_flight[url] = fut
        try:
            content = await self._throttled_fetch(url)
            fut.set_result(content)
            return content
        except Exception as e:  
            fut.set_exception(e)
            raise
        finally:
            self._in_flight.pop(url, None)

    async def verify_anchor(
        self,
        claimed_hash: str,
        url: str,
        inline_bytes: Optional[bytes] = None,
    ) -> Tuple[bool, str]:
        """Verify one anchor.

        Returns (verified, source) where:
          - verified=True only when validator independently fetched a
            whitelisted URL and the bytes hashed correctly
          - source ∈ {whitelisted_fetch, paywalled_content, url_only}

        For non-whitelisted URLs:
          - if inline_bytes is provided AND hashes correctly → store bytes,
            mark `paywalled_content`, verified=False
          - else → mark `url_only`, verified=False, no fetch attempt
        """
        if not url or not claimed_hash:
            return False, "url_only"

        if self.doc_store.has(claimed_hash):
            return True, "whitelisted_fetch"

        if is_whitelisted(url, self.whitelist):
            content = await self._fetch_once(url)
            if content is None:
                return False, "url_only"
            actual = hashlib.sha256(content).hexdigest()
            if actual != claimed_hash:
                logger.warning(
                    f"hash mismatch for {url}: claimed={claimed_hash[:8]}... "
                    f"actual={actual[:8]}... — miner may be fabricating"
                )
                return False, "url_only"
            self.doc_store.put(content)
            logger.info(f"verified doc {claimed_hash[:8]}... from {url}")
            return True, "whitelisted_fetch"

        if inline_bytes is not None:
            actual = hashlib.sha256(inline_bytes).hexdigest()
            if actual != claimed_hash:
                logger.warning(
                    f"paywalled hash mismatch: claimed={claimed_hash[:8]}... "
                    f"actual={actual[:8]}... — rejecting bytes"
                )
                return False, "url_only"
            self.doc_store.put(inline_bytes)
            logger.info(
                f"stored paywalled doc {claimed_hash[:8]}... ({len(inline_bytes)} B)"
            )
            return False, "paywalled_content"

        # Non-whitelisted, URL only.
        logger.info(f"non-whitelisted URL {url} — marking unverified")
        return False, "url_only"

    async def verify_delta(
        self, delta, documents: Optional[Dict[str, bytes]] = None
    ) -> Dict[str, Tuple[bool, str]]:
        """Verify every (hash, url) pair in a delta. `documents` is an optional
        {hash: bytes} map for paywalled sources the miner supplied inline.

        Returns {claimed_hash: (verified, source)}.
        """
        documents = documents or {}
        seen_hashes: Set[str] = set()
        pairs: list[tuple[str, str]] = []
        for item in (*delta.entities, *delta.relationships, *delta.facts):
            chain = getattr(item, "provenance", None)
            if chain is None:
                continue
            for anchor in getattr(chain, "anchors", []) or []:
                h = anchor.source_doc_hash
                if not h or h in seen_hashes:
                    continue
                seen_hashes.add(h)
                pairs.append((h, anchor.source_url))

        if not pairs:
            return {}

        async def _one(h: str, url: str) -> Tuple[str, Tuple[bool, str]]:
            inline = documents.get(h)
            return h, await self.verify_anchor(h, url, inline_bytes=inline)

        results = await asyncio.gather(
            *(_one(h, url) for h, url in pairs), return_exceptions=False
        )
        return dict(results)

"""
Reference miner for the Knowledge Graph subnet.

Miners can improve on this base by:
- Using LLMs (Claude/GPT) for richer entity/relationship extraction
- Adding more data sources (patent databases, court records, news APIs)
- Building sector-specific extraction heuristics
- Adding `ProvenanceChain` anchors with exact page/paragraph references
- Submitting reproducible `CausalEvidence` records (code + data in ArtifactStore)
"""
from __future__ import annotations

import asyncio
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sn43.protocol import (
    ContributeTask,
    Entity,
    EntityType,
    EventTask,
    Fact,
    KnowledgeDelta,
    Relationship,
    RelationshipType,
    VerificationResponse,
    VerifyTask,
)

# SEC EDGAR API base URL (free, public, requires User-Agent header)
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"

# Common supplier/customer keywords in 10-K filings
_SUPPLIER_PATTERNS = [
    r"(?:key|major|primary|principal)\s+supplier[s]?\s+(?:include|are|is)\s+(.+?)[\.\n]",
    r"(?:we|the company)\s+(?:purchase|procure|source)[s]?\s+.+?\s+from\s+(.+?)[\.\n]",
    r"(?:depend|dependent|reliance|rely)\s+(?:on|upon)\s+(.+?)\s+(?:for|as|to)\s+",
]

_CUSTOMER_PATTERNS = [
    r"(?:key|major|primary|principal)\s+customer[s]?\s+(?:include|are|is)\s+(.+?)[\.\n]",
    r"(\w[\w\s,]+?)\s+accounted?\s+for\s+(?:approximately\s+)?(\d+)%\s+of\s+(?:our|total)\s+revenue",
]

_COMPETITOR_PATTERNS = [
    r"(?:we|the company)\s+compete[s]?\s+(?:primarily\s+)?with\s+(.+?)[\.\n]",
    r"(?:key|major|primary|principal)\s+competitor[s]?\s+(?:include|are|is)\s+(.+?)[\.\n]",
]

# Common ticker → company name mappings for entity resolution
_WELL_KNOWN_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "NVDA": "NVIDIA Corporation",
    "TSLA": "Tesla Inc.",
    "META": "Meta Platforms Inc.",
    "TSM": "Taiwan Semiconductor Manufacturing",
    "AVGO": "Broadcom Inc.",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices Inc.",
    "CRM": "Salesforce Inc.",
    "ORCL": "Oracle Corporation",
    "CSCO": "Cisco Systems Inc.",
    "ADBE": "Adobe Inc.",
    "QCOM": "Qualcomm Inc.",
    "TXN": "Texas Instruments Inc.",
    "IBM": "IBM Corporation",
    "NOW": "ServiceNow Inc.",
    "INTU": "Intuit Inc.",
}

_TICKER_SECTORS = {
    "AAPL": "software", "MSFT": "software", "GOOGL": "software",
    "AMZN": "cloud_computing", "NVDA": "semiconductors", "TSM": "semiconductors",
    "AVGO": "semiconductors", "INTC": "semiconductors", "AMD": "semiconductors",
    "QCOM": "semiconductors", "TXN": "semiconductors",
    "META": "software", "CRM": "cloud_computing", "ORCL": "cloud_computing",
    "CSCO": "telecommunications", "ADBE": "software", "IBM": "cloud_computing",
    "NOW": "cloud_computing", "INTU": "software",
    "TSLA": "automotive",
}


class KnowledgeGraphMiner:
    """Reference miner — regex-based extraction from SEC EDGAR filings.

    Stateless per task. Safe to instantiate once and reuse across tasks.
    Swap extraction methods (_extract_entities, _extract_relationships,
    _extract_facts) to plug in LLMs, ML models, or richer heuristics.
    """

    def __init__(self, miner_uid: Optional[int] = None):
        self.miner_uid = miner_uid or int(os.getenv("MINER_UID", "0")) or None
        contact = os.getenv("SN43_EDGAR_CONTACT", "research@example.com")
        self.user_agent = f"SN43Miner ({contact})"

    # ──────────────────── Task handlers ────────────────────

    async def handle_contribute(self, task: ContributeTask) -> KnowledgeDelta:
        """Scheduled task — extract knowledge for the requested sector/tickers."""
        sector = task.sector or "software"
        tickers = task.target_tickers or self._get_tickers_for_sector(sector)

        all_entities: List[Entity] = []
        all_relationships: List[Relationship] = []
        all_facts: List[Fact] = []

        for ticker in tickers[:10]:  # limit to 10 tickers per task
            try:
                filings = await self._scrape_edgar(ticker)
                for filing in filings[:3]:  # limit to 3 filings per ticker
                    text = filing.get("content", "")
                    url = filing.get("url", f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}")
                    filing_date_str = filing.get("date", datetime.utcnow().isoformat())
                    try:
                        filing_date = datetime.fromisoformat(filing_date_str)
                    except (ValueError, TypeError):
                        filing_date = datetime.utcnow()

                    entities = self._extract_entities(text, ticker, url, filing_date, sector)
                    relationships = self._extract_relationships(text, ticker, entities, url, filing_date)
                    facts = self._extract_facts(text, ticker, url, filing_date)

                    all_entities.extend(entities)
                    all_relationships.extend(relationships)
                    all_facts.extend(facts)
            except Exception as e:
                print(f"Error processing {ticker}: {e}", flush=True)
                continue

        # Dedupe by deterministic ID
        unique_entities = _dedupe_by_id(all_entities, "entity_id")
        unique_rels = _dedupe_by_id(all_relationships, "relationship_id")
        unique_facts = _dedupe_by_id(all_facts, "fact_id")

        # Respect task limits
        unique_entities = unique_entities[: task.max_entities]
        unique_rels = unique_rels[: task.max_relationships]
        unique_facts = unique_facts[: task.max_facts]

        return KnowledgeDelta(
            entities=unique_entities,
            relationships=unique_rels,
            facts=unique_facts,
            miner_uid=self.miner_uid,
        )

    async def handle_event(self, task: EventTask) -> KnowledgeDelta:
        """Real-time event (SEC filing drop, earnings call, etc.).

        Default implementation fetches the document and runs the same
        regex extraction. Miners should override this with richer logic
        (LLM extraction, multi-source cross-referencing, etc.).
        """
        ticker = task.subject_ticker or ""
        entities: List[Entity] = []
        relationships: List[Relationship] = []
        facts: List[Fact] = []

        if ticker:
            try:
                text = await self._fetch_url(task.source_url)
                entities = self._extract_entities(
                    text, ticker, task.source_url, task.event_timestamp,
                    _TICKER_SECTORS.get(ticker, "software"),
                )
                relationships = self._extract_relationships(
                    text, ticker, entities, task.source_url, task.event_timestamp,
                )
                facts = self._extract_facts(text, ticker, task.source_url, task.event_timestamp)
            except Exception as e:
                print(f"Event handling failed for {task.dedupe_key}: {e}", flush=True)

        return KnowledgeDelta(
            entities=_dedupe_by_id(entities, "entity_id"),
            relationships=_dedupe_by_id(relationships, "relationship_id"),
            facts=_dedupe_by_id(facts, "fact_id"),
            miner_uid=self.miner_uid,
        )

    async def handle_verify(self, task: VerifyTask) -> VerificationResponse:
        """Check an existing fact/relationship against its source."""
        if task.fact_to_verify is not None:
            fact = task.fact_to_verify
            try:
                content = await self._fetch_url(fact.source_url)
                confirmed = str(fact.value).lower() in content.lower()
                return VerificationResponse(
                    task_id=task.task_id,
                    confirmed=confirmed,
                    confidence=0.8 if confirmed else 0.3,
                    evidence_url=fact.source_url,
                )
            except Exception as e:
                print(f"Verify fetch failed: {e}", flush=True)

        return VerificationResponse(
            task_id=task.task_id, confirmed=False, confidence=0.1,
        )

    # ──────────────────── Data scraping ────────────────────

    async def _fetch_url(self, url: str) -> str:
        import httpx
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, headers={"User-Agent": self.user_agent})
            resp.raise_for_status()
            return resp.text

    async def _scrape_edgar(self, ticker: str) -> List[Dict[str, Any]]:
        """Scrape SEC EDGAR for recent filings for a given ticker."""
        import httpx

        filings = []
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    EDGAR_SEARCH_URL,
                    params={
                        "q": f'"{ticker}" AND formType:"10-K"',
                        "dateRange": "custom",
                        "startdt": "2025-01-01",
                        "enddt": datetime.utcnow().strftime("%Y-%m-%d"),
                    },
                    headers={"User-Agent": self.user_agent},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("hits", {}).get("hits", [])
                    for hit in hits[:3]:
                        source = hit.get("_source", {})
                        filings.append({
                            "content": source.get("file_description", ""),
                            "url": f"https://www.sec.gov/Archives/edgar/data/{source.get('entity_id', '')}/{source.get('file_name', '')}",
                            "date": source.get("file_date", datetime.utcnow().isoformat()),
                            "filing_type": source.get("form_type", "10-K"),
                        })
        except Exception as e:
            print(f"EDGAR scrape failed for {ticker}: {e}", flush=True)

        # If EDGAR fails, fall back to a known-company stub so scheduled
        # tasks still return *something* useful.
        if not filings and ticker in _WELL_KNOWN_COMPANIES:
            filings.append({
                "content": f"{_WELL_KNOWN_COMPANIES[ticker]} ({ticker}) is a publicly traded company.",
                "url": f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K",
                "date": datetime.utcnow().isoformat(),
                "filing_type": "10-K",
            })

        return filings

    # ──────────────────── Entity extraction ────────────────────

    def _extract_entities(
        self,
        text: str,
        ticker: str,
        source_url: str,
        source_date: datetime,
        sector: str,
    ) -> List[Entity]:
        entities: List[Entity] = []

        company_name = _WELL_KNOWN_COMPANIES.get(ticker, ticker)
        company_sector = _TICKER_SECTORS.get(ticker, sector)

        entities.append(Entity(
            entity_id=f"company:{ticker}",
            entity_type=EntityType.COMPANY,
            name=company_name,
            ticker=ticker,
            sector=company_sector,
            source_url=source_url,
            source_date=source_date,
        ))

        for other_ticker, other_name in _WELL_KNOWN_COMPANIES.items():
            if other_ticker == ticker:
                continue
            if other_name.lower() in text.lower() or other_ticker in text:
                entities.append(Entity(
                    entity_id=f"company:{other_ticker}",
                    entity_type=EntityType.COMPANY,
                    name=other_name,
                    ticker=other_ticker,
                    sector=_TICKER_SECTORS.get(other_ticker, ""),
                    source_url=source_url,
                    source_date=source_date,
                ))

        return entities

    def _extract_relationships(
        self,
        text: str,
        ticker: str,
        entities: List[Entity],
        source_url: str,
        source_date: datetime,
    ) -> List[Relationship]:
        relationships: List[Relationship] = []
        text_lower = text.lower()
        primary_id = f"company:{ticker}"

        for entity in entities:
            if entity.entity_id == primary_id:
                continue

            other_name = entity.name.lower()

            # Supplier / dependency
            for pattern in _SUPPLIER_PATTERNS:
                if re.search(pattern, text_lower) and other_name in text_lower:
                    rel_id = f"{primary_id}->{RelationshipType.DEPENDS_ON.value}->{entity.entity_id}"
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_entity_id=primary_id,
                        target_entity_id=entity.entity_id,
                        relationship_type=RelationshipType.DEPENDS_ON,
                        weight=0.5,
                        description=f"{ticker} depends on {entity.ticker or entity.name}",
                        source_url=source_url,
                        source_date=source_date,
                    ))
                    break

            # Competitor
            for pattern in _COMPETITOR_PATTERNS:
                if re.search(pattern, text_lower) and other_name in text_lower:
                    rel_id = f"{primary_id}->{RelationshipType.COMPETES_WITH.value}->{entity.entity_id}"
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_entity_id=primary_id,
                        target_entity_id=entity.entity_id,
                        relationship_type=RelationshipType.COMPETES_WITH,
                        weight=0.5,
                        description=f"{ticker} competes with {entity.ticker or entity.name}",
                        source_url=source_url,
                        source_date=source_date,
                    ))
                    break

            # Customer / revenue-from
            for pattern in _CUSTOMER_PATTERNS:
                match = re.search(pattern, text_lower)
                if match and other_name in text_lower:
                    pct = None
                    if match.lastindex and match.lastindex >= 2:
                        try:
                            pct = int(match.group(2))
                        except (ValueError, IndexError):
                            pass
                    weight = (pct / 100.0) if pct else 0.5
                    rel_id = f"{primary_id}->{RelationshipType.REVENUE_FROM.value}->{entity.entity_id}"
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_entity_id=primary_id,
                        target_entity_id=entity.entity_id,
                        relationship_type=RelationshipType.REVENUE_FROM,
                        weight=min(weight, 1.0),
                        description=f"{ticker} derives revenue from {entity.ticker or entity.name}"
                        + (f" ({pct}%)" if pct else ""),
                        source_url=source_url,
                        source_date=source_date,
                    ))
                    break

        return relationships

    def _extract_facts(
        self,
        text: str,
        ticker: str,
        source_url: str,
        source_date: datetime,
    ) -> List[Fact]:
        facts: List[Fact] = []
        entity_id = f"company:{ticker}"

        revenue_match = re.search(
            r"(?:total\s+)?(?:net\s+)?revenue[s]?\s+(?:was|were|of)\s+\$?([\d,.]+)\s*(billion|million|thousand)?",
            text,
            re.IGNORECASE,
        )
        if revenue_match:
            amount = revenue_match.group(1).replace(",", "")
            unit = revenue_match.group(2) or "USD"
            try:
                facts.append(Fact(
                    fact_id=f"{entity_id}:revenue",
                    entity_id=entity_id,
                    fact_type="revenue",
                    value=float(amount),
                    unit=unit,
                    source_url=source_url,
                    source_date=source_date,
                    valid_from=source_date,
                ))
            except ValueError:
                pass

        employee_match = re.search(
            r"(?:approximately\s+)?([\d,]+)\s+(?:full[- ]time\s+)?employees",
            text,
            re.IGNORECASE,
        )
        if employee_match:
            count = employee_match.group(1).replace(",", "")
            try:
                facts.append(Fact(
                    fact_id=f"{entity_id}:employee_count",
                    entity_id=entity_id,
                    fact_type="employee_count",
                    value=int(count),
                    unit="employees",
                    source_url=source_url,
                    source_date=source_date,
                    valid_from=source_date,
                ))
            except ValueError:
                pass

        return facts

    # ──────────────────── Helpers ────────────────────

    def _get_tickers_for_sector(self, sector: str) -> List[str]:
        tickers = [
            ticker
            for ticker, s in _TICKER_SECTORS.items()
            if s.lower() == sector.lower()
        ]
        if not tickers:
            tickers = list(_WELL_KNOWN_COMPANIES.keys())[:5]
        return tickers


def _dedupe_by_id(items: list, id_attr: str) -> list:
    seen = set()
    out = []
    for item in items:
        key = getattr(item, id_attr)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


# ──────────────────────────── CLI entry point ────────────────────────────

async def _main() -> None:
    """Run a miner that listens to validator event buses.
    """
    from sn43.protocol import EventTask

    miner_uid = int(os.environ.get("MINER_UID", "0"))
    miner = KnowledgeGraphMiner(miner_uid=miner_uid)

    async def handle(task: EventTask):
        return await miner.handle_event(task)

    try:
        import bittensor as bt
    except ImportError as e:
        raise RuntimeError(
            "bittensor not installed"
        ) from e
    from ingestion.multi_validator_client import MultiValidatorClient

    WalletCls = getattr(bt, "Wallet", None) or getattr(bt, "wallet")
    wallet = WalletCls(
        name=os.getenv("BT_WALLET_COLD", "default"),
        hotkey=os.getenv("BT_WALLET_HOT", "default"),
    )
    client = MultiValidatorClient(
        miner_uid=miner_uid,
        netuid=int(os.getenv("NETUID", "43")),
        hotkey=wallet.hotkey,
    )
    print(
        f"Miner UID={miner_uid} hotkey={wallet.hotkey.ss58_address}; "
        f"discovering validators on netuid {client.netuid}",
        flush=True,
    )
    await client.listen_all(handle)


if __name__ == "__main__":
    asyncio.run(_main())

"""
Reference LLM miner — uses the Anthropic API directly to extract entities,
relationships, and facts from SEC filings. Slightly more advanced than the
regex `agent.py`; meant as a starting point that miners can fork and tune.

Pipeline per task:
  1. Receive an EventTask over WebSocket
  2. Fetch the source document (SEC filing, etc.)
  3. Hash + persist the bytes locally so the validator's provenance check
     can resolve the quote you'll attach
  4. Send the document text to Claude with a structured-output prompt
  5. Parse Claude's JSON against a Pydantic schema (skip rows that fail)
  6. Build sn43 protocol objects with proper ProvenanceChains
  7. Sign with your Bittensor hotkey and POST to /submit

This is a reference implementation. To compete, fork this file and:
  - Swap the model (`claude-opus-4-7` for higher quality, slower & pricier)
  - Specialize the EXTRACTION_PROMPT for your sector
  - Cache fetched documents to avoid re-paying API costs on retries
  - Add CausalEvidence (Granger / Event Study) for the 20% reproducibility score
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import httpx
from pydantic import BaseModel, Field, ValidationError

from sn43.protocol import (
    Entity, EntityType, EventTask, ExtractionMethod, Fact,
    KnowledgeDelta, ProvenanceAnchor, ProvenanceChain,
    Relationship, RelationshipType,
)
from storage.doc_store import DocumentStore

logger = logging.getLogger("llm_miner")

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TEXT_CHARS = 200_000  # truncate huge filings to fit context window
EXTRACTION_TIMEOUT = 90.0  # seconds


# ──────────────────────────── Pydantic schema for Claude's output ────────────────────────────

class ExtractedEntity(BaseModel):
    entity_type: str
    key: str
    name: str
    ticker: Optional[str] = None
    sector: Optional[str] = None
    quoted_text: str


class ExtractedRelationship(BaseModel):
    source_entity_type: str
    source_key: str
    target_entity_type: str
    target_key: str
    relationship_type: str
    weight: float = Field(default=0.5, ge=0.0, le=1.0)
    description: str = ""
    quoted_text: str


class ExtractedFact(BaseModel):
    entity_type: str
    entity_key: str
    fact_type: str
    value: Union[float, int, str]
    unit: Optional[str] = None
    quoted_text: str


class Extraction(BaseModel):
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relationships: List[ExtractedRelationship] = Field(default_factory=list)
    facts: List[ExtractedFact] = Field(default_factory=list)


EXTRACTION_PROMPT = """\
You are an extraction agent for SN43, a decentralized financial knowledge
graph of US public companies.

Given the SEC filing text below, extract:
  1. ENTITIES — every public company, person, patent, regulation, or product referenced
  2. RELATIONSHIPS — directed connections between those entities
  3. FACTS — quantitative attributes (revenue, employee count, etc.)

Each item MUST include `quoted_text`: an exact verbatim quote from the source
that supports the claim. If you cannot find a supporting quote, OMIT the item.
Do not fabricate.

Valid entity_type values:
  company, person, patent, product, regulation, event

Valid relationship_type values:
  supplies, competes_with, employs, board_member_of, invested_in,
  subsidiary_of, partners_with, depends_on, litigates_against,
  licensed_from, revenue_from, produces

Entity key conventions (use these EXACT canonical keys):
  - Companies: stock ticker only (AAPL, MSFT, TSM)
  - People: LASTNAME_COMPANY (COOK_AAPL)
  - Patents: patent number (US12345678)
  - Products: PRODUCTNAME_COMPANY (IPHONE_AAPL)
  - Regulations: well-known short codes (EU_DMA, US_SOX, etc.)

CRITICAL OUTPUT RULES:
  - Output exactly ONE JSON object matching the shape below.
  - No markdown fences, no preamble, no trailing prose.
  - Begin with `{` and end with `}`.
  - relationship.source_key and target_key MUST match a key in entities.
  - fact.entity_key MUST match a key in entities.
  - For every product entity, ALSO emit a `produces` edge from parent company.

FACT QUALITY RULES:
  - DO NOT emit dates as standalone facts (report_date, fiscal_quarter_end_date, etc.).
    Those are source metadata — the validator records `merged_at` automatically.
  - For PERIOD-SCOPED financials (revenue, net_income, operating_income, EPS,
    gross_margin, r_and_d_expense), encode the period in fact_type:
        revenue_q1_fy2026, eps_fy2025, net_income_q4_fy2025
    Without a period suffix the next quarter overwrites the previous one.
  - For BALANCE-SHEET items (point-in-time: total_assets, total_liabilities,
    shareholders_equity, cash_and_equivalents, long_term_debt), encode the
    as-of date: `total_assets_at_2025_12_27`.
  - For STATIC attributes (HQ state, incorporation, sector, CEO), emit ONE per
    concept. If headquarters_state == incorporation_state, emit only one.
  - Tax rates and percentages that apply to a period get the same period
    suffix: `effective_tax_rate_q1_fy2026`.

JSON shape:
{
  "entities": [{"entity_type":"company","key":"AAPL","name":"Apple Inc.","ticker":"AAPL","sector":"software","quoted_text":"..."}],
  "relationships": [{"source_entity_type":"company","source_key":"AAPL","target_entity_type":"company","target_key":"TSM","relationship_type":"depends_on","weight":0.9,"description":"...","quoted_text":"..."}],
  "facts": [{"entity_type":"company","entity_key":"AAPL","fact_type":"revenue","value":394.3,"unit":"billion_usd","quoted_text":"..."}]
}
"""


# ──────────────────────────── Extractor ────────────────────────────

async def extract_via_claude(
    text: str,
    subject_ticker: Optional[str],
    api_key: str,
) -> Extraction:
    """Call Claude API, parse JSON, validate against the Pydantic schema."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key)
    truncated = text[:MAX_TEXT_CHARS]
    ticker_hint = f"Filing subject: {subject_ticker}\n\n" if subject_ticker else ""

    msg = await asyncio.wait_for(
        client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=4096,
            messages=[{
                "role": "user",
                "content": (
                    f"{EXTRACTION_PROMPT}\n\n"
                    f"{ticker_hint}=== FILING TEXT ===\n{truncated}"
                ),
            }],
        ),
        timeout=EXTRACTION_TIMEOUT,
    )

    raw = msg.content[0].text.strip()
    # Strip markdown fences if Claude added them despite the rules.
    if raw.startswith("```"):
        # ```json\n{...}\n``` or ```\n{...}\n```
        first_nl = raw.find("\n")
        if first_nl > 0:
            raw = raw[first_nl + 1:]
        if raw.endswith("```"):
            raw = raw[:-3].rstrip()

    try:
        return Extraction.model_validate_json(raw)
    except ValidationError as e:
        logger.error(f"Claude returned unparseable JSON: {e}")
        return Extraction()


# ──────────────────────────── Fetch + store ────────────────────────────

async def fetch_document(url: str, edgar_contact: str) -> tuple[bytes, str]:
    """Fetch URL, return (bytes, decoded text). Sets the SEC-required UA."""
    headers = {"User-Agent": f"SN43Miner ({edgar_contact})"}
    async with httpx.AsyncClient(
        timeout=30, headers=headers, follow_redirects=True,
    ) as client:
        r = await client.get(url)
        r.raise_for_status()
    return r.content, r.content.decode("utf-8", errors="replace")


# ──────────────────────────── Delta builder ────────────────────────────

def build_delta(
    extraction: Extraction,
    task: EventTask,
    doc_hash: str,
    miner_uid: int,
) -> KnowledgeDelta:
    """Convert validated Claude output into sn43 protocol objects."""

    def _anchor(quote: str) -> ProvenanceAnchor:
        return ProvenanceAnchor(
            source_doc_hash=doc_hash,
            source_url=task.source_url,
            page=1, paragraph_start=1, paragraph_end=1,
            quoted_text=quote,
            extraction_method=ExtractionMethod.LLM,
        )

    entities: List[Entity] = []
    for e in extraction.entities:
        try:
            entities.append(Entity(
                entity_id=f"{e.entity_type}:{e.key}",
                entity_type=EntityType(e.entity_type),
                name=e.name,
                ticker=e.ticker,
                sector=e.sector,
                source_url=task.source_url,
                source_date=task.event_timestamp,
                provenance=ProvenanceChain(anchors=[_anchor(e.quoted_text)]),
            ))
        except (ValidationError, ValueError) as ex:
            logger.debug(f"skipping entity {e.key}: {ex}")

    relationships: List[Relationship] = []
    for r in extraction.relationships:
        src_id = f"{r.source_entity_type}:{r.source_key}"
        tgt_id = f"{r.target_entity_type}:{r.target_key}"
        try:
            relationships.append(Relationship(
                relationship_id=f"{src_id}->{r.relationship_type}->{tgt_id}",
                source_entity_id=src_id,
                target_entity_id=tgt_id,
                relationship_type=RelationshipType(r.relationship_type),
                weight=r.weight,
                description=r.description,
                source_url=task.source_url,
                source_date=task.event_timestamp,
                provenance=ProvenanceChain(anchors=[_anchor(r.quoted_text)]),
            ))
        except (ValidationError, ValueError) as ex:
            logger.debug(f"skipping relationship {src_id}->{tgt_id}: {ex}")

    facts: List[Fact] = []
    for f in extraction.facts:
        eid = f"{f.entity_type}:{f.entity_key}"
        try:
            facts.append(Fact(
                fact_id=f"{eid}:{f.fact_type}",
                entity_id=eid,
                fact_type=f.fact_type,
                value=f.value,
                unit=f.unit,
                source_url=task.source_url,
                source_date=task.event_timestamp,
                provenance=ProvenanceChain(anchors=[_anchor(f.quoted_text)]),
            ))
        except (ValidationError, ValueError) as ex:
            logger.debug(f"skipping fact {eid}:{f.fact_type}: {ex}")

    return KnowledgeDelta(
        entities=entities,
        relationships=relationships,
        facts=facts,
        miner_uid=miner_uid,
    )


# ──────────────────────────── Submission ────────────────────────────

async def submit(
    submit_url: str,
    task: EventTask,
    delta: KnowledgeDelta,
    hotkey=None,
) -> dict:
    """POST the delta to the validator. Signs with hotkey if provided."""
    payload = {
        "task_id": task.task_id,
        "delta": json.loads(delta.model_dump_json()),
    }

    headers = {}
    body_kwarg = {"json": payload}
    if hotkey is not None:
        from sn43.auth import canonical_body, sign_payload
        body = canonical_body(payload)
        headers = sign_payload(hotkey, payload)
        headers["Content-Type"] = "application/json"
        body_kwarg = {"content": body}

    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(submit_url, headers=headers, **body_kwarg)
        r.raise_for_status()
        return r.json()


# ──────────────────────────── Main loop ────────────────────────────

class LLMMiner:
    """Self-contained miner: WebSocket loop, Claude extraction, signed submit."""

    def __init__(
        self,
        miner_uid: int,
        anthropic_api_key: str,
        edgar_contact: str,
        doc_store_root: str = "./data/doc_store",
        hotkey=None,
    ) -> None:
        self.miner_uid = miner_uid
        self.api_key = anthropic_api_key
        self.contact = edgar_contact
        self.hotkey = hotkey
        Path(doc_store_root).mkdir(parents=True, exist_ok=True)
        self.doc_store = DocumentStore(root=doc_store_root)

    async def handle_task(self, task: EventTask) -> None:
        try:
            doc_bytes, doc_text = await fetch_document(task.source_url, self.contact)
        except Exception as e: 
            logger.error(f"fetch failed for {task.source_url}: {e}")
            return

        doc_hash = self.doc_store.put(doc_bytes)
        logger.info(
            f"task={task.task_id} fetched {len(doc_bytes)}B, doc={doc_hash[:8]}..."
        )

        try:
            extraction = await extract_via_claude(
                doc_text, task.subject_ticker, self.api_key,
            )
        except Exception as e: 
            logger.error(f"extraction failed: {e}")
            return

        delta = build_delta(extraction, task, doc_hash, self.miner_uid)
        logger.info(
            f"task={task.task_id} extracted "
            f"{len(delta.entities)}e/{len(delta.relationships)}r/{len(delta.facts)}f"
        )
        if delta.is_empty:
            return

        try:
            ack = await submit(self.submit_url, task, delta, hotkey=self.hotkey)
            logger.info(f"task={task.task_id} submitted ack={ack}")
        except Exception as e: 
            logger.error(f"submit failed: {e}")

    async def run(self) -> None:
        import websockets
        logger.info(f"miner UID={self.miner_uid} connecting to {self.events_url}")
        async with websockets.connect(self.events_url) as ws:
            logger.info("connected — awaiting tasks")
            async for raw in ws:
                try:
                    task = EventTask.model_validate_json(raw)
                except Exception as e: 
                    logger.warning(f"bad event: {e}")
                    continue
                asyncio.create_task(self.handle_task(task))


def _require(name: str) -> str:
    v = os.getenv(name)
    if not v:
        print(f"✗ missing env var: {name}", file=sys.stderr)
        sys.exit(1)
    return v


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    miner_uid = int(_require("MINER_UID"))
    api_key = _require("ANTHROPIC_API_KEY")
    contact = os.getenv("SN43_EDGAR_CONTACT", "miner@example.com")
    doc_store_root = os.getenv("DOC_STORE_ROOT", "./data/doc_store")

    try:
        import bittensor as bt
    except ImportError:
        print(
            "✗ bittensor not installed",
            file=sys.stderr,
        )
        sys.exit(1)

    WalletCls = getattr(bt, "Wallet", None) or getattr(bt, "wallet")
    wallet = WalletCls(
        name=os.getenv("BT_WALLET_COLD", "default"),
        hotkey=os.getenv("BT_WALLET_HOT", "default"),
    )
    hotkey = wallet.hotkey
    netuid = int(os.getenv("NETUID", "43"))
    logger.info(
        f"loaded hotkey {hotkey.ss58_address}; "
        f"discovering validators on netuid {netuid}"
    )

    from pathlib import Path as _Path
    _Path(doc_store_root).mkdir(parents=True, exist_ok=True)
    doc_store = DocumentStore(root=doc_store_root)

    async def handle_task(task: EventTask) -> Optional[KnowledgeDelta]:
        try:
            doc_bytes, doc_text = await fetch_document(task.source_url, contact)
        except Exception as e:  
            logger.error(f"fetch failed for {task.source_url}: {e}")
            return None
        doc_hash = doc_store.put(doc_bytes)
        logger.info(f"task={task.task_id} fetched {len(doc_bytes)}B doc={doc_hash[:8]}...")

        try:
            extraction = await extract_via_claude(doc_text, task.subject_ticker, api_key)
        except Exception as e: 
            logger.error(f"extraction failed: {e}")
            return None

        delta = build_delta(extraction, task, doc_hash, miner_uid)
        logger.info(
            f"task={task.task_id} extracted "
            f"{len(delta.entities)}e/{len(delta.relationships)}r/{len(delta.facts)}f"
        )
        return delta if not delta.is_empty else None

    from sn43.ingestion.multi_validator_client import MultiValidatorClient
    client = MultiValidatorClient(
        miner_uid=miner_uid,
        netuid=netuid,
        hotkey=hotkey,
    )
    try:
        asyncio.run(client.listen_all(handle_task))
    except KeyboardInterrupt:
        print("\nshutdown")


if __name__ == "__main__":
    main()

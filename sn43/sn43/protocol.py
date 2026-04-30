"""
Protocol definitions for the Knowledge Graph subnet.

Phase 1 schema: provenance-first, causal-evidence-first.

Graph primitives:
- Entity, Relationship, Fact: graph nodes/edges/attributes
- ProvenanceAnchor, ProvenanceChain: cryptographic chain-of-custody from
  every claim back to the exact location in a source document
- CausalEvidence, CausalMethod, CausalDirection: reproducible causal claims
  attached to relationships (verified in Phase 2 via sandboxed re-execution)

Miner/validator messages:
- KnowledgeDelta: miner submission (entities, relationships, facts)
- ContributeTask, VerifyTask: validator requests
- VerificationResponse: miner answer to a verify request
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────── Enums ────────────────────────────

class EntityType(str, Enum):
    COMPANY = "company"
    PERSON = "person"
    PATENT = "patent"
    PRODUCT = "product"
    REGULATION = "regulation"
    EVENT = "event"


class RelationshipType(str, Enum):
    SUPPLIES = "supplies"
    COMPETES_WITH = "competes_with"
    EMPLOYS = "employs"
    BOARD_MEMBER_OF = "board_member_of"
    INVESTED_IN = "invested_in"
    SUBSIDIARY_OF = "subsidiary_of"
    PARTNERS_WITH = "partners_with"
    DEPENDS_ON = "depends_on"
    LITIGATES_AGAINST = "litigates_against"
    LICENSED_FROM = "licensed_from"
    REVENUE_FROM = "revenue_from"
    PRODUCES = "produces"


class VerificationSource(str, Enum):
    """How a claim's source document was verified.

    - WHITELISTED_FETCH: validator independently fetched the URL from a
      whitelisted domain and the bytes hashed to the miner's claimed hash.
      This is the only source that gives `verified=True`.
    - PAYWALLED_CONTENT: miner provided the bytes inline because the source
      is paywalled / not publicly fetchable. Hash matches what the miner
      claimed but origin can't be independently verified. `verified=False`.
    - URL_ONLY: non-whitelisted public URL. Validator did not fetch.
      Treated as a tip — `verified=False`. Useful for hints that may
      later be backed by a whitelisted source.
    - UNKNOWN: legacy / unset.
    """
    WHITELISTED_FETCH = "whitelisted_fetch"
    PAYWALLED_CONTENT = "paywalled_content"
    URL_ONLY = "url_only"
    UNKNOWN = "unknown"


class CausalMethod(str, Enum):
    """Supported causal-inference methods.

    Each implies a methodology, assumptions, and expected inputs. Validators
    re-execute the miner's code against stored data in Phase 2.
    """
    GRANGER = "granger_causality"
    DID = "difference_in_differences"
    IV = "instrumental_variable"
    SYNTHETIC_CONTROL = "synthetic_control"
    PROPENSITY_MATCHING = "propensity_match"
    DO_CALCULUS = "do_calculus"
    EVENT_STUDY = "event_study"
    REGRESSION_DISCONTINUITY = "rdd"


class CausalDirection(str, Enum):
    SOURCE_TO_TARGET = "s_to_t"
    TARGET_TO_SOURCE = "t_to_s"
    BIDIRECTIONAL = "bidirectional"
    CONFOUNDED = "confounded"  # association exists but not causal


class ExtractionMethod(str, Enum):
    """How a claim was extracted from a source document."""
    MANUAL = "manual"
    REGEX = "regex"
    LLM = "llm"
    HEURISTIC = "heuristic"
    TABULAR_PARSE = "tabular_parse"


class EventType(str, Enum):
    """Real-time events the validator broadcasts to miners."""
    EDGAR_8K = "edgar_8k"
    EDGAR_10K = "edgar_10k"
    EDGAR_10Q = "edgar_10q"
    EDGAR_DEF14A = "edgar_def14a"
    EDGAR_S1 = "edgar_s1"
    EDGAR_OTHER = "edgar_other"
    EARNINGS_CALL = "earnings_call"
    PATENT_GRANT = "patent_grant"


class DiscoveryCategory(str, Enum):
    """What kind of novel claim a miner is submitting unsolicited."""
    HIDDEN_RELATIONSHIP = "hidden_relationship"
    CROSS_DOC_INFERENCE = "cross_doc_inference"
    TEMPORAL_CHANGE = "temporal_change"
    CAUSAL_CLAIM = "causal_claim"
    CORRECTION = "correction"


# ──────────────────────────── Provenance ────────────────────────────

class ProvenanceAnchor(BaseModel):
    """A single anchor pointing to an exact location in a source document.

    The anchor enables validators and customers to verify that the quoted
    text appears at the claimed location in the document identified by
    `source_doc_hash`.
    """

    source_doc_hash: str = Field(
        ...,
        description="sha256 of the source document, resolvable via doc_store",
        min_length=64,
        max_length=64,
    )
    source_url: str = Field(..., min_length=1)
    page: Optional[int] = Field(None, ge=1, description="1-indexed page number")
    paragraph_start: Optional[int] = Field(None, ge=0)
    paragraph_end: Optional[int] = Field(None, ge=0)
    quoted_text: str = Field(..., min_length=1, max_length=4000)
    extraction_method: ExtractionMethod = ExtractionMethod.MANUAL
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_paragraph_range(self) -> "ProvenanceAnchor":
        if self.paragraph_start is not None and self.paragraph_end is not None:
            if self.paragraph_end < self.paragraph_start:
                raise ValueError(
                    "paragraph_end must be >= paragraph_start"
                )
        return self


class ProvenanceChain(BaseModel):
    """An ordered list of anchors backing a single claim.

    Most claims have one anchor (the source paragraph). Claims derived from
    multiple documents (e.g., a supplier relationship cross-referenced across
    two filings) carry multiple anchors in chronological order.
    """

    anchors: List[ProvenanceAnchor] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.anchors) == 0

    @property
    def primary_source_url(self) -> Optional[str]:
        if not self.anchors:
            return None
        return self.anchors[0].source_url


# ──────────────────────────── Causal Evidence ────────────────────────────

class CausalEvidence(BaseModel):
    """Reproducible causal-inference result attached to a relationship edge.

    A single edge may carry multiple CausalEvidence records — e.g., one from
    Granger, another from a synthetic-control study. The graph stores all of
    them and computes consensus at query time.

    Reproducibility:
    - `input_data_hash` identifies the exact input series used
    - `code_hash` identifies the exact inference code
    - `seed` pins any stochastic component
    - validators re-execute and confirm the claimed effect matches
    """

    method: CausalMethod
    direction: CausalDirection
    effect_size: float = Field(..., description="standardized coefficient, ATE, etc.")
    effect_unit: str = Field(..., min_length=1)

    # Evidence strength
    p_value: Optional[float] = Field(None, ge=0.0, le=1.0)
    confidence_interval_low: Optional[float] = None
    confidence_interval_high: Optional[float] = None
    ci_level: float = Field(default=0.95, ge=0.0, le=1.0)
    test_statistic: Optional[float] = None
    n_observations: int = Field(..., ge=1)

    # Period of validity
    data_window_start: datetime
    data_window_end: datetime
    lag_structure: Optional[str] = None

    # Falsifiability
    controls: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    robustness_checks: List[str] = Field(default_factory=list)

    # Reproducibility artifacts (content-addressed)
    input_data_hash: str = Field(..., min_length=64, max_length=64)
    input_data_url: Optional[str] = None
    code_hash: str = Field(..., min_length=64, max_length=64)
    code_url: Optional[str] = None
    seed: Optional[int] = None

    # Provenance
    miner_uid: Optional[int] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def validate_window(self) -> "CausalEvidence":
        if self.data_window_end < self.data_window_start:
            raise ValueError("data_window_end must be >= data_window_start")
        if (
            self.confidence_interval_low is not None
            and self.confidence_interval_high is not None
            and self.confidence_interval_high < self.confidence_interval_low
        ):
            raise ValueError("CI high must be >= CI low")
        return self


# ──────────────────────────── Graph Primitives ────────────────────────────

_ENTITY_ID_RE = re.compile(r"^[a-z_]+:[A-Za-z0-9_.@-]+$")


class Entity(BaseModel):
    """A node in the knowledge graph (company, person, patent, etc.)."""

    entity_id: str = Field(
        ...,
        description="Deterministic ID: '{entity_type}:{key}', e.g. 'company:AAPL'",
    )
    entity_type: EntityType
    name: str = Field(..., min_length=1)
    ticker: Optional[str] = None
    description: Optional[str] = None
    sector: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source_url: str = Field(..., min_length=1)
    source_date: datetime
    provenance: ProvenanceChain = Field(default_factory=ProvenanceChain)
    verified: bool = Field(default=True, description="True if all anchors hash-verified by validator")
    verification_source: VerificationSource = VerificationSource.UNKNOWN
    submitted_by_validator_uid: Optional[int] = Field(
        default=None, description="UID of the validator that pushed this to graph data server"
    )
    merged_at: Optional[datetime] = Field(
        default=None, description="UTC timestamp when graph data server accepted the merge"
    )

    @model_validator(mode="after")
    def validate_entity_id(self) -> "Entity":
        if not _ENTITY_ID_RE.match(self.entity_id):
            raise ValueError(
                f"entity_id must match '{{type}}:{{key}}' pattern, got '{self.entity_id}'"
            )
        expected_prefix = self.entity_type.value + ":"
        if not self.entity_id.startswith(expected_prefix):
            raise ValueError(
                f"entity_id prefix must match entity_type: expected '{expected_prefix}', "
                f"got '{self.entity_id}'"
            )
        return self


class Relationship(BaseModel):
    """An edge in the knowledge graph connecting two entities.

    The `causal_evidence` list carries reproducible causal claims attached to
    this edge. Multiple miners can submit evidence using different methods;
    all are retained and consensus is computed at query time.
    """

    relationship_id: str = Field(
        ...,
        description="Deterministic: '{source_id}->{rel_type}->{target_id}'",
    )
    source_entity_id: str
    target_entity_id: str
    relationship_type: RelationshipType
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: Optional[str] = None
    source_url: str = Field(..., min_length=1)
    source_date: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    provenance: ProvenanceChain = Field(default_factory=ProvenanceChain)
    causal_evidence: List[CausalEvidence] = Field(default_factory=list)
    verified: bool = Field(default=True, description="True if all anchors hash-verified by validator")
    verification_source: VerificationSource = VerificationSource.UNKNOWN
    submitted_by_validator_uid: Optional[int] = Field(default=None)
    merged_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def validate_relationship_id(self) -> "Relationship":
        expected = f"{self.source_entity_id}->{self.relationship_type.value}->{self.target_entity_id}"
        if self.relationship_id != expected:
            raise ValueError(
                f"relationship_id must be '{expected}', got '{self.relationship_id}'"
            )
        if self.source_entity_id == self.target_entity_id:
            raise ValueError("source and target entity must be different")
        return self


class Fact(BaseModel):
    """A specific data point about an entity."""

    fact_id: str = Field(
        ...,
        description="Deterministic: '{entity_id}:{fact_type}'",
    )
    entity_id: str
    fact_type: str = Field(..., min_length=1, description="e.g. 'revenue', 'employee_count'")
    value: Any
    unit: Optional[str] = None
    source_url: str = Field(..., min_length=1)
    source_date: datetime
    valid_from: Optional[datetime] = None
    valid_to: Optional[datetime] = None
    provenance: ProvenanceChain = Field(default_factory=ProvenanceChain)
    verified: bool = Field(default=True, description="True if all anchors hash-verified by validator")
    verification_source: VerificationSource = VerificationSource.UNKNOWN
    superseded: bool = Field(default=False, description="True if a verified fact later contradicted this")
    superseded_by: Optional[str] = Field(default=None, description="fact_id of the verified fact that superseded this")
    submitted_by_validator_uid: Optional[int] = Field(default=None)
    merged_at: Optional[datetime] = Field(default=None)

    @model_validator(mode="after")
    def validate_fact_id(self) -> "Fact":
        expected_prefix = f"{self.entity_id}:{self.fact_type}"
        if not self.fact_id.startswith(expected_prefix):
            raise ValueError(
                f"fact_id must start with '{expected_prefix}', got '{self.fact_id}'"
            )
        return self


# ──────────────────────────── Delta (miner submission) ────────────────────────────

class KnowledgeDelta(BaseModel):
    """A batch of new knowledge submitted by a miner."""

    delta_id: str = Field(default_factory=lambda: uuid4().hex)
    entities: List[Entity] = Field(default_factory=list)
    relationships: List[Relationship] = Field(default_factory=list)
    facts: List[Fact] = Field(default_factory=list)
    miner_uid: Optional[int] = None
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_items(self) -> int:
        return len(self.entities) + len(self.relationships) + len(self.facts)

    @property
    def is_empty(self) -> bool:
        return self.total_items == 0


# ──────────────────────────── Tasks (validator → miner) ────────────────────────────

class ContributeTask(BaseModel):
    """Validator asks a miner to contribute knowledge about a sector/tickers."""

    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_type: Literal["contribute"] = "contribute"
    sector: Optional[str] = None
    target_tickers: Optional[List[str]] = None
    relationship_types: Optional[List[RelationshipType]] = None
    max_entities: int = Field(default=50, ge=1, le=500)
    max_relationships: int = Field(default=100, ge=1, le=1000)
    max_facts: int = Field(default=200, ge=1, le=2000)


class EventTask(BaseModel):
    """Validator broadcasts this to miners the moment a real-world event fires.

    Miners race to extract entities/relationships/facts about the event and
    return a KnowledgeDelta tagged with the same task_id. Submissions get a
    latency bonus that decays exponentially from `event_timestamp`.
    """

    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_type: Literal["event"] = "event"
    event_type: EventType
    source_url: str = Field(..., min_length=1)
    event_timestamp: datetime = Field(
        ..., description="Wall-clock time the event occurred at the source"
    )
    broadcast_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Wall-clock time the validator broadcast this task",
    )
    deadline: datetime = Field(
        ..., description="Submissions after this time score zero latency bonus"
    )
    dedupe_key: str = Field(
        ...,
        min_length=1,
        description="Stable identifier (e.g. EDGAR accession number) so validators and miners skip duplicates",
    )
    # Optional context the validator already extracted (e.g. filer ticker)
    subject_ticker: Optional[str] = None
    subject_cik: Optional[str] = None

    @model_validator(mode="after")
    def validate_deadline(self) -> "EventTask":
        if self.deadline < self.event_timestamp:
            raise ValueError("deadline must be >= event_timestamp")
        return self


class DiscoveryTask(BaseModel):
    """Miner-originated submission envelope (unsolicited, not validator-broadcast).

    Submitted to POST /submit-discovery. Gated behind strict quality
    requirements and a per-miner rate limit. See score_discovery() for the
    scoring rubric.
    """

    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_type: Literal["discovery"] = "discovery"
    miner_uid: int
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    discovery_category: DiscoveryCategory
    claimed_novelty_justification: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="One-sentence reason this claim is worth surfacing.",
    )


class VerifyTask(BaseModel):
    """Validator asks a miner to verify an existing fact or relationship."""

    task_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    task_type: Literal["verify"] = "verify"
    fact_to_verify: Optional[Fact] = None
    relationship_to_verify: Optional[Relationship] = None

    @model_validator(mode="after")
    def at_least_one(self) -> "VerifyTask":
        if self.fact_to_verify is None and self.relationship_to_verify is None:
            raise ValueError("Must provide at least one of fact_to_verify or relationship_to_verify")
        return self


class VerificationResponse(BaseModel):
    """Miner's response to a VerifyTask."""

    task_id: str
    confirmed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_url: Optional[str] = None
    evidence_excerpt: Optional[str] = None
    corrected_value: Optional[Any] = None


# ──────────────────────────── Merge Result ────────────────────────────

class MergeResult(BaseModel):
    """Result of merging a KnowledgeDelta into the graph."""

    new_entities: int = 0
    new_relationships: int = 0
    new_facts: int = 0
    duplicate_entities: int = 0
    duplicate_relationships: int = 0
    duplicate_facts: int = 0

    @property
    def total_new(self) -> int:
        return self.new_entities + self.new_relationships + self.new_facts

    @property
    def total_duplicates(self) -> int:
        return self.duplicate_entities + self.duplicate_relationships + self.duplicate_facts

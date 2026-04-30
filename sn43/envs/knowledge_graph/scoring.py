"""
Scoring functions for knowledge graph contributions.

All scoring is deterministic and structural — no LLM-as-judge, no network
calls against the open internet. Validators use these to score each miner's
KnowledgeDelta.

Weights:
    accuracy         0.15  — structural validity of IDs, URLs, dates
    provenance       0.15  — provenance anchors resolve + quotes match
    reproducibility  0.20  — causal code re-runs and matches claimed effect
    novelty          0.15  — items not already in the graph
    freshness        0.05  — recency of source documents
    latency          0.15  — speed bonus on event-driven submissions
    depth            0.10  — richness of the submission
    coverage         0.05  — underserved sectors
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Dict, Iterable, Optional

from sn43.protocol import CausalEvidence, EventTask, KnowledgeDelta, ProvenanceChain

if TYPE_CHECKING:
    from sandbox.runner import SandboxRunner
    from storage.artifact_store import ArtifactStore
    from storage.doc_store import DocumentStore
    from storage.graph_store import KnowledgeGraphStore


_URL_RE = re.compile(r"^https?://[^\s]+$")
_ENTITY_ID_RE = re.compile(r"^[a-z_]+:[A-Za-z0-9_.@-]+$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


# ──────────────────────────── Accuracy ────────────────────────────

def score_accuracy(delta: KnowledgeDelta) -> float:
    """
    Structural validation of the delta.

    Checks:
    - Entity IDs are well-formed
    - Relationship IDs match the expected pattern
    - Source/target entity IDs are well-formed
    - Source URLs look like valid URLs
    - Source dates are plausible (not in the future, not before 1990)

    Returns 0.0-1.0.
    """
    if delta.is_empty:
        return 0.0

    checks: list[bool] = []
    now = datetime.utcnow()
    min_date = datetime(1990, 1, 1)

    for entity in delta.entities:
        checks.append(bool(_ENTITY_ID_RE.match(entity.entity_id)))
        checks.append(bool(_URL_RE.match(entity.source_url)))
        checks.append(min_date <= entity.source_date <= now)
        checks.append(len(entity.name.strip()) > 0)

    for rel in delta.relationships:
        checks.append(bool(_ENTITY_ID_RE.match(rel.source_entity_id)))
        checks.append(bool(_ENTITY_ID_RE.match(rel.target_entity_id)))
        checks.append(bool(_URL_RE.match(rel.source_url)))
        checks.append(min_date <= rel.source_date <= now)
        expected_id = f"{rel.source_entity_id}->{rel.relationship_type.value}->{rel.target_entity_id}"
        checks.append(rel.relationship_id == expected_id)
        checks.append(rel.source_entity_id != rel.target_entity_id)

    for fact in delta.facts:
        checks.append(bool(_ENTITY_ID_RE.match(fact.entity_id)))
        checks.append(bool(_URL_RE.match(fact.source_url)))
        checks.append(min_date <= fact.source_date <= now)
        checks.append(fact.value is not None)
        checks.append(fact.fact_id.startswith(f"{fact.entity_id}:{fact.fact_type}"))

    if not checks:
        return 0.0
    return sum(checks) / len(checks)


# ──────────────────────────── Provenance ────────────────────────────

def _score_single_chain(
    chain: ProvenanceChain,
    doc_store: Optional["DocumentStore"] = None,
) -> float:
    """
    Score a single provenance chain in [0.0, 1.0].

    Checks (each contributes equally):
    - Chain is non-empty
    - Every anchor has a well-formed sha256 source_doc_hash
    - Every anchor has a valid source_url
    - Every anchor has non-empty quoted_text
    - If doc_store is provided: every quoted_text resolves in its source doc
    """
    if chain.is_empty:
        return 0.0

    anchor_scores: list[float] = []
    for anchor in chain.anchors:
        checks: list[bool] = [
            bool(_SHA256_RE.match(anchor.source_doc_hash)),
            bool(_URL_RE.match(anchor.source_url)),
            len(anchor.quoted_text.strip()) > 0,
        ]
        if doc_store is not None:
            checks.append(
                doc_store.resolve_quote(anchor.source_doc_hash, anchor.quoted_text)
            )
        anchor_scores.append(sum(checks) / len(checks))

    return sum(anchor_scores) / len(anchor_scores)


def score_provenance(
    delta: KnowledgeDelta,
    doc_store: Optional["DocumentStore"] = None,
) -> float:
    """
    Average provenance quality across every item in the delta.

    Items without a provenance chain score 0. Items with a chain score
    based on structural validity of each anchor, plus (if a DocumentStore
    is supplied) whether the quoted text actually resolves in the stored
    source document.

    Returns 0.0-1.0.
    """
    if delta.is_empty:
        return 0.0

    scores: list[float] = []
    for entity in delta.entities:
        scores.append(_score_single_chain(entity.provenance, doc_store))
    for rel in delta.relationships:
        scores.append(_score_single_chain(rel.provenance, doc_store))
    for fact in delta.facts:
        scores.append(_score_single_chain(fact.provenance, doc_store))

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ──────────────────────────── Novelty ────────────────────────────

def score_novelty(delta: KnowledgeDelta, graph: "KnowledgeGraphStore") -> float:
    """Fraction of items in the delta that are new (not already in the graph)."""
    if delta.is_empty:
        return 0.0

    total = 0
    new = 0

    for entity in delta.entities:
        total += 1
        if not graph.has_entity(entity.entity_id):
            new += 1

    for rel in delta.relationships:
        total += 1
        if not graph.has_relationship(rel.relationship_id):
            new += 1

    for fact in delta.facts:
        total += 1
        if not graph.has_fact(fact.fact_id):
            new += 1

    if total == 0:
        return 0.0
    return new / total


# ──────────────────────────── Freshness ────────────────────────────

def score_freshness(delta: KnowledgeDelta) -> float:
    """Average recency score of source documents across all items."""
    if delta.is_empty:
        return 0.0

    now = datetime.utcnow()
    scores: list[float] = []

    all_dates = (
        [e.source_date for e in delta.entities]
        + [r.source_date for r in delta.relationships]
        + [f.source_date for f in delta.facts]
    )

    for source_date in all_dates:
        age = now - source_date
        if age < timedelta(days=7):
            scores.append(1.0)
        elif age < timedelta(days=30):
            scores.append(0.8)
        elif age < timedelta(days=90):
            scores.append(0.5)
        elif age < timedelta(days=365):
            scores.append(0.2)
        else:
            scores.append(0.0)

    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ──────────────────────────── Depth ────────────────────────────

def score_depth(delta: KnowledgeDelta) -> float:
    """
    Richness of the submission.

    Rewards:
    - presence of entities, relationships, facts
    - causal evidence attached to relationships (Phase 1 replaces old
      correlation/causation scalar fields)
    - diversity of relationship types
    - diversity of source URLs
    - presence of temporal validity windows
    """
    if delta.is_empty:
        return 0.0

    score = 0.0

    if delta.entities:
        score += 0.15
    if delta.relationships:
        score += 0.20
    if delta.facts:
        score += 0.15

    if delta.relationships:
        with_causal = sum(1 for r in delta.relationships if r.causal_evidence)
        if with_causal > 0:
            score += 0.20 * (with_causal / len(delta.relationships))

    if delta.relationships:
        unique_types = {r.relationship_type for r in delta.relationships}
        if len(unique_types) >= 3:
            score += 0.10
        elif len(unique_types) >= 2:
            score += 0.05

    all_urls = set()
    for e in delta.entities:
        all_urls.add(e.source_url)
    for r in delta.relationships:
        all_urls.add(r.source_url)
    for f in delta.facts:
        all_urls.add(f.source_url)
    if len(all_urls) > 3:
        score += 0.10
    elif len(all_urls) > 1:
        score += 0.05

    has_temporal = any(
        r.valid_from is not None or r.valid_to is not None
        for r in delta.relationships
    ) or any(
        f.valid_from is not None or f.valid_to is not None
        for f in delta.facts
    )
    if has_temporal:
        score += 0.10

    return min(score, 1.0)


# ──────────────────────────── Coverage ────────────────────────────

def score_coverage_bonus(delta: KnowledgeDelta, graph: "KnowledgeGraphStore") -> float:
    """Bonus for contributing to underserved sectors. Returns 0.0-1.0."""
    if delta.is_empty:
        return 0.0

    underserved = graph.get_underserved_sectors()
    if not underserved:
        return 0.0

    top_underserved = set(s.lower() for s in underserved[:5])

    in_underserved = 0
    total_with_sector = 0
    for entity in delta.entities:
        if entity.sector:
            total_with_sector += 1
            if entity.sector.lower() in top_underserved:
                in_underserved += 1

    if total_with_sector == 0:
        return 0.0
    return in_underserved / total_with_sector


# ──────────────────────────── Latency ────────────────────────────

DEFAULT_LATENCY_HALF_LIFE_SECONDS = 300.0  # 5 min — full credit fades by half
DEFAULT_LATENCY_DEADLINE_SECONDS = 3600.0  # 1 hr — no credit after this


def score_latency(
    event: Optional[EventTask],
    submitted_at: Optional[datetime] = None,
    *,
    half_life_seconds: float = DEFAULT_LATENCY_HALF_LIFE_SECONDS,
    deadline_seconds: float = DEFAULT_LATENCY_DEADLINE_SECONDS,
) -> float:
    """
    Exponential-decay bonus for how fast a miner responded to an event.

    Returns 1.0 if the submission arrives at the moment the event fires,
    decaying to 0.5 at `half_life_seconds`, and hard-capping at 0.0 after
    `deadline_seconds`.

    If no event is attached (e.g. a scheduled `ContributeTask` submission),
    returns 0.0 — latency bonus is event-driven only.
    """
    if event is None:
        return 0.0
    submitted_at = submitted_at or datetime.utcnow()
    elapsed = (submitted_at - event.event_timestamp).total_seconds()
    if elapsed < 0:
        # Submission timestamped before event (clock skew, replay) — treat as on-time
        return 1.0
    if elapsed >= deadline_seconds:
        return 0.0
    return math.pow(0.5, elapsed / half_life_seconds)


# ──────────────────────────── Reproducibility ────────────────────────────

DEFAULT_REPRODUCIBILITY_TOLERANCE = 0.05  # 5% relative tolerance on effect_size


def _effect_matches(claimed: float, reproduced: float, tolerance: float) -> bool:
    """Relative tolerance with absolute floor for small magnitudes."""
    if claimed == 0 and reproduced == 0:
        return True
    denom = max(abs(claimed), 1e-6)
    return abs(reproduced - claimed) / denom <= tolerance


def _iter_causal_evidence(delta: KnowledgeDelta) -> Iterable[CausalEvidence]:
    for rel in delta.relationships:
        for ev in rel.causal_evidence:
            yield ev


def score_reproducibility(
    delta: KnowledgeDelta,
    artifact_store: Optional["ArtifactStore"] = None,
    runner: Optional["SandboxRunner"] = None,
    tolerance: float = DEFAULT_REPRODUCIBILITY_TOLERANCE,
    sample_fraction: float = 1.0,
) -> float:
    """
    Re-execute each CausalEvidence's code against its data and confirm the
    output effect_size matches the claimed value within `tolerance`.

    Returns the fraction of causal evidence records that reproduce.
    Deltas with no causal evidence return 0.0 (no claim → no reward from this
    dimension; accuracy/provenance/depth still score it).

    `sample_fraction` lets validators verify a random subset to control cost.
    """
    evidences = list(_iter_causal_evidence(delta))
    if not evidences:
        return 0.0

    if artifact_store is None or runner is None:
        # Infrastructure missing — treat every claim as unverified.
        return 0.0

    if 0.0 < sample_fraction < 1.0:
        import random
        sample_size = max(1, int(len(evidences) * sample_fraction))
        evidences = random.sample(evidences, sample_size)

    verified = 0
    for ev in evidences:
        code = artifact_store.get_code(ev.code_hash)
        data = artifact_store.get_data(ev.input_data_hash)
        if code is None or data is None:
            continue
        result = runner.run(code, data)
        if not result.ok:
            continue
        reproduced = result.output.get("effect_size")
        if not isinstance(reproduced, (int, float)):
            continue
        if _effect_matches(ev.effect_size, float(reproduced), tolerance):
            verified += 1

    return verified / len(evidences)


# ──────────────────────────── Discovery ────────────────────────────

DISCOVERY_ACCURACY_FLOOR = 0.95
DISCOVERY_PROVENANCE_FLOOR = 0.9
DISCOVERY_NOVELTY_FLOOR = 0.6
DISCOVERY_SCORE_CAP = 0.7


def score_discovery(
    delta: KnowledgeDelta,
    graph: "KnowledgeGraphStore",
    doc_store: Optional["DocumentStore"] = None,
    artifact_store: Optional["ArtifactStore"] = None,
    runner: Optional["SandboxRunner"] = None,
) -> float:
    """
    Score an unsolicited (miner-originated) submission.

    Unlike event- or schedule-driven submissions, discovery submissions must
    pass hard gates on accuracy, provenance, and novelty before any other
    dimension is even computed. If any gate fails, returns 0.0 and the
    submission is hard-rejected by the validator.

    The final score is capped at DISCOVERY_SCORE_CAP (0.7) so that task-driven
    mining stays the more-rewarded path and discovery is a supplement, not the
    core flow.
    """
    accuracy = score_accuracy(delta)
    if accuracy < DISCOVERY_ACCURACY_FLOOR:
        return 0.0

    provenance = score_provenance(delta, doc_store)
    if provenance < DISCOVERY_PROVENANCE_FLOOR:
        return 0.0

    novelty = score_novelty(delta, graph)
    if novelty < DISCOVERY_NOVELTY_FLOOR:
        return 0.0

    reproducibility = score_reproducibility(delta, artifact_store, runner)
    depth = score_depth(delta)

    total = (
        accuracy * 0.15
        + provenance * 0.25   # heavier — provenance is the anti-spam gate
        + reproducibility * 0.25
        + novelty * 0.20
        + depth * 0.15
    )
    return min(DISCOVERY_SCORE_CAP, total)


# ──────────────────────────── Total ────────────────────────────

def _verification_penalties(delta: KnowledgeDelta) -> Dict[str, float]:
    """Multipliers applied to per-dimension scores based on the delta's
    verification mix.

    Rules:
      - `url_only` items (non-whitelisted public URL, no fetch) ⇒
            accuracy contribution = 0
      - `paywalled_content` items (miner provided bytes, no independent
        verification) ⇒
            accuracy contribution = 0
            reproducibility contribution = 0
            provenance contribution capped at 0.5

      - novelty, depth, freshness, latency, coverage_bonus are unaffected.
        An unverified item still moves the graph forward and earns
        recency / latency credit normally.

    Returns a dict of {dimension: multiplier} in [0.0, 1.0].
    """
    items = list(delta.entities) + list(delta.relationships) + list(delta.facts)
    n = len(items)
    if n == 0:
        return {"accuracy": 1.0, "provenance": 1.0, "reproducibility": 1.0}

    n_verified = 0
    n_url_only = 0
    n_paywalled = 0
    for item in items:
        if getattr(item, "verified", True):
            n_verified += 1
            continue
        src = getattr(item, "verification_source", None)
        src_value = getattr(src, "value", src)  # enum or string
        if src_value == "paywalled_content":
            n_paywalled += 1
        else:
            # url_only or unknown — treat conservatively as url_only
            n_url_only += 1

    # accuracy: only verified items count toward accuracy
    accuracy_mult = n_verified / n

    # reproducibility: paywalled items contribute zero (no independent
    # source check possible). url_only items just don't typically carry
    # causal evidence, so they're a non-issue here.
    reproducibility_mult = (n - n_paywalled) / n

    # provenance: paywalled items capped at 0.5 (questionable origin)
    # url_only and verified items contribute fully.
    provenance_mult = (n_verified + n_url_only + 0.5 * n_paywalled) / n

    return {
        "accuracy": accuracy_mult,
        "provenance": provenance_mult,
        "reproducibility": reproducibility_mult,
    }


def compute_total_score(
    delta: KnowledgeDelta,
    graph: "KnowledgeGraphStore",
    doc_store: Optional["DocumentStore"] = None,
    *,
    event: Optional[EventTask] = None,
    artifact_store: Optional["ArtifactStore"] = None,
    runner: Optional["SandboxRunner"] = None,
    sample_fraction: float = 1.0,
) -> Dict[str, float]:
    """
    Compute weighted total score for a miner's knowledge delta.

    Per-dimension scores are
    further adjusted by `_verification_penalties` based on the mix of
    verified / url_only / paywalled items in the delta — non-public
    sources can't earn full accuracy/provenance/reproducibility credit
    even if structurally valid.
    """
    accuracy = score_accuracy(delta)
    provenance = score_provenance(delta, doc_store)
    reproducibility = score_reproducibility(
        delta, artifact_store, runner, sample_fraction=sample_fraction
    )
    novelty = score_novelty(delta, graph)
    freshness = score_freshness(delta)
    latency = score_latency(event, delta.submitted_at)
    depth = score_depth(delta)
    coverage = score_coverage_bonus(delta, graph)

    penalties = _verification_penalties(delta)
    accuracy *= penalties["accuracy"]
    provenance *= penalties["provenance"]
    reproducibility *= penalties["reproducibility"]

    total = (
        accuracy * 0.15
        + provenance * 0.15
        + reproducibility * 0.20
        + novelty * 0.15
        + freshness * 0.05
        + latency * 0.15
        + depth * 0.10
        + coverage * 0.05
    )

    return {
        "accuracy": round(accuracy, 4),
        "provenance": round(provenance, 4),
        "reproducibility": round(reproducibility, 4),
        "novelty": round(novelty, 4),
        "freshness": round(freshness, 4),
        "latency": round(latency, 4),
        "depth": round(depth, 4),
        "coverage_bonus": round(coverage, 4),
        "total": round(total, 4),
    }

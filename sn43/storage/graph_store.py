"""
Knowledge graph storage layer.

Uses NetworkX MultiDiGraph in memory with JSON file persistence.
Designed to be hosted by validators — miners contribute deltas,
validators merge them into this shared graph.
"""
from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import networkx as nx

from sn43.protocol import (
    Entity,
    EntityType,
    Fact,
    KnowledgeDelta,
    MergeResult,
    Relationship,
)

logger = logging.getLogger("sn43.graph_store")


def _values_disagree(stored, new, tolerance: float) -> bool:
    """True if `stored` and `new` are meaningfully different.

    Numbers: relative difference exceeds `tolerance` (e.g. 0.05 = 5%).
    Anything else: strict !=. Used to decide whether a verified fact
    contradicts an unverified one — small numeric drift (rounding,
    units) shouldn't trigger slashing.
    """
    try:
        a = float(stored)
        b = float(new)
        if a == 0 and b == 0:
            return False
        denom = max(abs(a), abs(b))
        return abs(a - b) / denom > tolerance
    except (TypeError, ValueError):
        return stored != new


class KnowledgeGraphStore:
    """
    In-memory knowledge graph with JSON persistence.

    Entities are graph nodes, relationships are graph edges,
    facts are stored in a separate dict keyed by fact_id.
    """

    def __init__(self, persist_path: str = "data/knowledge_graph.json"):
        self.persist_path = persist_path
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.facts: Dict[str, Dict[str, Any]] = {}
        # Track which miner contributed what
        self.contributions: Dict[str, List[int]] = defaultdict(list)  # item_id -> [miner_uids]
        self._load()

    # ──────────────────── Persistence ────────────────────

    # Fields removed in Phase 1 — strip from legacy persisted data on load.
    _DEPRECATED_RELATIONSHIP_FIELDS = frozenset(
        {"correlation_score", "causation_score"}
    )

    @classmethod
    def _strip_deprecated(cls, edge_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            k: v for k, v in edge_data.items()
            if k not in cls._DEPRECATED_RELATIONSHIP_FIELDS
        }

    def _load(self) -> None:
        if not os.path.exists(self.persist_path):
            logger.info(f"No existing graph at {self.persist_path}, starting fresh")
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            # Restore entities
            for eid, edata in data.get("entities", {}).items():
                self.entities[eid] = edata
                self.graph.add_node(eid, **edata)
            # Restore relationships (edges)
            for edge in data.get("relationships", []):
                clean = self._strip_deprecated(edge)
                self.graph.add_edge(
                    clean["source"],
                    clean["target"],
                    key=clean["relationship_id"],
                    **{k: v for k, v in clean.items() if k not in ("source", "target")},
                )
            # Restore facts
            self.facts = data.get("facts", {})
            # Restore contributions
            self.contributions = defaultdict(list, data.get("contributions", {}))
            logger.info(
                f"Loaded graph: {self.entity_count()} entities, "
                f"{self.relationship_count()} relationships, "
                f"{self.fact_count()} facts"
            )
        except Exception as e:
            logger.error(f"Failed to load graph from {self.persist_path}: {e}")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
        # Serialize entities
        entities = dict(self.entities)
        # Serialize relationships
        relationships = []
        for src, tgt, key, data in self.graph.edges(keys=True, data=True):
            edge_data = {"source": src, "target": tgt, "relationship_id": key}
            edge_data.update(data)
            relationships.append(edge_data)
        payload = {
            "entities": entities,
            "relationships": relationships,
            "facts": self.facts,
            "contributions": dict(self.contributions),
            "saved_at": datetime.utcnow().isoformat(),
        }
        with open(self.persist_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"Saved graph to {self.persist_path}")

    # ──────────────────── Add operations ────────────────────

    def add_entity(self, entity: Entity, miner_uid: int = -1) -> bool:
        """Add entity. Returns True if new, False if duplicate."""
        if entity.entity_id in self.entities:
            return False
        edata = entity.model_dump(mode="json")
        self.entities[entity.entity_id] = edata
        self.graph.add_node(entity.entity_id, **edata)
        self.contributions[entity.entity_id].append(miner_uid)
        return True

    def add_relationship(self, rel: Relationship, miner_uid: int = -1) -> bool:
        """Add relationship — or append new causal evidence to an existing one.

        See sn43_owner/storage/graph_store.py for the full rationale.
        """
        src, tgt, key = rel.source_entity_id, rel.target_entity_id, rel.relationship_id

        if not self.graph.has_edge(src, tgt, key=key):
            rdata = rel.model_dump(mode="json")
            self.graph.add_edge(src, tgt, key=key, **rdata)
            self.contributions[key].append(miner_uid)
            return True

        # Existing edge — append any unseen causal evidence.
        edge_data = self.graph[src][tgt][key]
        existing = list(edge_data.get("causal_evidence", []))
        existing_keys = {
            (e.get("method"), e.get("code_hash"), e.get("input_data_hash"))
            for e in existing
        }

        appended = 0
        for ev in rel.causal_evidence:
            ev_dict = ev.model_dump(mode="json")
            ek = (ev_dict.get("method"), ev_dict.get("code_hash"), ev_dict.get("input_data_hash"))
            if ek in existing_keys:
                continue
            existing.append(ev_dict)
            existing_keys.add(ek)
            appended += 1

        if appended == 0:
            return False

        edge_data["causal_evidence"] = existing
        self.contributions[key].append(miner_uid)
        logger.info(
            f"appended {appended} causal evidence record(s) to {key} "
            f"(now {len(existing)} total) from miner_uid={miner_uid}"
        )
        return True

    def add_fact(self, fact: Fact, miner_uid: int = -1) -> bool:
        """Add fact. Returns True if new, False if duplicate."""
        if fact.fact_id in self.facts:
            return False
        self.facts[fact.fact_id] = fact.model_dump(mode="json")
        self.contributions[fact.fact_id].append(miner_uid)
        return True

    def merge_delta(self, delta: KnowledgeDelta, miner_uid: int = -1) -> MergeResult:
        """Merge a KnowledgeDelta into the graph. Returns counts of new vs duplicate."""
        result = MergeResult()
        for entity in delta.entities:
            if self.add_entity(entity, miner_uid):
                result.new_entities += 1
            else:
                result.duplicate_entities += 1
        for rel in delta.relationships:
            if self.add_relationship(rel, miner_uid):
                result.new_relationships += 1
            else:
                result.duplicate_relationships += 1
        for fact in delta.facts:
            if self.add_fact(fact, miner_uid):
                result.new_facts += 1
            else:
                result.duplicate_facts += 1
        return result

    # ──────────────────── Contradiction detection ────────────────────

    def find_contradicting_unverified_facts(
        self, new_fact: Fact, tolerance: float = 0.05
    ) -> List[Dict]:
        """See sn43_owner/storage/graph_store.py for full docstring."""
        contradictions: List[Dict] = []
        new_value = new_fact.value
        for fact_id, stored in self.facts.items():
            if fact_id == new_fact.fact_id:
                continue
            if stored.get("entity_id") != new_fact.entity_id:
                continue
            if stored.get("fact_type") != new_fact.fact_type:
                continue
            if stored.get("verified", True):
                continue
            if stored.get("superseded", False):
                continue
            stored_value = stored.get("value")
            if _values_disagree(stored_value, new_value, tolerance):
                contradictions.append({
                    "fact_id": fact_id,
                    "contributors": list(self.contributions.get(fact_id, [])),
                    "stored_value": stored_value,
                })
        return contradictions

    def mark_superseded(self, fact_id: str, by_fact_id: str) -> bool:
        if fact_id not in self.facts:
            return False
        self.facts[fact_id]["superseded"] = True
        self.facts[fact_id]["superseded_by"] = by_fact_id
        return True

    # ──────────────────── Query operations ────────────────────

    def has_entity(self, entity_id: str) -> bool:
        return entity_id in self.entities

    def has_relationship(self, relationship_id: str) -> bool:
        for _, _, key in self.graph.edges(keys=True):
            if key == relationship_id:
                return True
        return False

    def has_fact(self, fact_id: str) -> bool:
        return fact_id in self.facts

    def get_entity(self, entity_id: str) -> Optional[Dict[str, Any]]:
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Dict[str, Any]]:
        return [
            e for e in self.entities.values()
            if e.get("entity_type") == entity_type.value
        ]

    def get_entities_by_sector(self, sector: str) -> List[Dict[str, Any]]:
        return [
            e for e in self.entities.values()
            if e.get("sector", "").lower() == sector.lower()
        ]

    def get_relationships_for(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get all relationships where entity is source or target."""
        rels = []
        # Outgoing
        if self.graph.has_node(entity_id):
            for _, tgt, key, data in self.graph.edges(entity_id, keys=True, data=True):
                rels.append(data)
            # Incoming
            for src, _, key, data in self.graph.in_edges(entity_id, keys=True, data=True):
                rels.append(data)
        return rels

    def get_facts_for(self, entity_id: str) -> List[Dict[str, Any]]:
        return [
            f for f in self.facts.values()
            if f.get("entity_id") == entity_id
        ]

    # ──────────────────── Stats & coverage ────────────────────

    def entity_count(self) -> int:
        return len(self.entities)

    def relationship_count(self) -> int:
        return self.graph.number_of_edges()

    def fact_count(self) -> int:
        return len(self.facts)

    def get_coverage_stats(self) -> Dict[str, Any]:
        """High-level stats about graph coverage."""
        type_counts: Dict[str, int] = defaultdict(int)
        sector_counts: Dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            type_counts[e.get("entity_type", "unknown")] += 1
            sector = e.get("sector")
            if sector:
                sector_counts[sector] += 1

        rel_type_counts: Dict[str, int] = defaultdict(int)
        for _, _, _, data in self.graph.edges(keys=True, data=True):
            rel_type_counts[data.get("relationship_type", "unknown")] += 1

        return {
            "total_entities": self.entity_count(),
            "total_relationships": self.relationship_count(),
            "total_facts": self.fact_count(),
            "entities_by_type": dict(type_counts),
            "entities_by_sector": dict(sector_counts),
            "relationships_by_type": dict(rel_type_counts),
        }

    def get_underserved_sectors(self) -> List[str]:
        """Return sectors sorted by least coverage (fewest entities)."""
        from envs.utils import SECTORS

        sector_counts: Dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            sector = e.get("sector")
            if sector:
                sector_counts[sector.lower()] += 1

        scored = []
        for sector in SECTORS:
            count = sector_counts.get(sector.lower(), 0)
            scored.append((sector, count))
        scored.sort(key=lambda x: x[1])
        return [s[0] for s in scored]

    def get_random_fact_for_verification(self) -> Optional[Fact]:
        """Pick a random fact from the graph for verification tasks."""
        if not self.facts:
            return None
        fact_data = random.choice(list(self.facts.values()))
        return Fact(**fact_data)

    def get_random_relationship_for_verification(self) -> Optional[Relationship]:
        """Pick a random relationship from the graph for verification tasks."""
        edges = list(self.graph.edges(keys=True, data=True))
        if not edges:
            return None
        _, _, _, data = random.choice(edges)
        return Relationship(**self._strip_deprecated(data))

    def snapshot(self) -> Dict[str, Any]:
        """Serializable summary for miners to query (not the full graph)."""
        return {
            "stats": self.get_coverage_stats(),
            "underserved_sectors": self.get_underserved_sectors()[:5],
            "sample_entities": list(self.entities.keys())[:50],
        }

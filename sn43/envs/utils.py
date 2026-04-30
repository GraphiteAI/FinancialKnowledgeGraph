"""
Task type definitions and configuration for the Knowledge Graph subnet.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List, Optional
from uuid import uuid4

from sn43.protocol import ContributeTask, Fact, RelationshipType, VerifyTask

if TYPE_CHECKING:
    from storage.graph_store import KnowledgeGraphStore


class TaskType(str, Enum):
    CONTRIBUTE = "contribute"
    VERIFY = "verify"


SECTORS = [
    "semiconductors",
    "software",
    "cloud_computing",
    "pharmaceuticals",
    "biotechnology",
    "financial_services",
    "energy",
    "automotive",
    "retail",
    "telecommunications",
    "defense",
    "healthcare",
    "real_estate",
    "media_entertainment",
    "agriculture",
]


@dataclass
class TaskConfig:
    """Configuration for a task type."""

    task_type: TaskType
    selection_weight: float  # probability weight for choosing this task type
    timeout_seconds: float   # max time for miner to respond


TASK_CONFIGS = {
    TaskType.CONTRIBUTE: TaskConfig(
        task_type=TaskType.CONTRIBUTE,
        selection_weight=0.7,
        timeout_seconds=300.0,
    ),
    TaskType.VERIFY: TaskConfig(
        task_type=TaskType.VERIFY,
        selection_weight=0.3,
        timeout_seconds=60.0,
    ),
}


def generate_contribute_task(graph: "KnowledgeGraphStore") -> ContributeTask:
    """
    Generate a ContributeTask targeting underserved areas of the graph.

    Picks the least-covered sector and optionally suggests relationship types.
    """
    underserved = graph.get_underserved_sectors()
    sector = underserved[0] if underserved else random.choice(SECTORS)

    rel_types = random.sample(
        list(RelationshipType),
        k=min(4, len(RelationshipType)),
    )

    return ContributeTask(
        task_id=uuid4().hex[:12],
        sector=sector,
        target_tickers=None,
        relationship_types=rel_types,
        max_entities=50,
        max_relationships=100,
        max_facts=200,
    )


def generate_verify_task(
    graph: "KnowledgeGraphStore",
) -> Optional[VerifyTask]:
    """
    Generate a VerifyTask by picking a random existing fact from the graph.

    Returns None if the graph has no facts to verify.
    """
    fact = graph.get_random_fact_for_verification()
    if fact is None:
        return None

    return VerifyTask(
        task_id=uuid4().hex[:12],
        fact_to_verify=fact,
    )


def select_task_type() -> TaskType:
    """Randomly select a task type based on configured weights."""
    types = list(TASK_CONFIGS.keys())
    weights = [TASK_CONFIGS[t].selection_weight for t in types]
    return random.choices(types, weights=weights, k=1)[0]

"""
Content-addressed artifact storage for code + input data.

Phase 2 reproducibility verification works like this:
1. Miner computes a causal claim against input series X using code C
2. Miner stores X and C in the artifact store, obtains hashes H_x and H_c
3. Miner submits CausalEvidence with input_data_hash=H_x, code_hash=H_c
4. Validator fetches X and C by hash, re-executes C against X in a sandbox
5. Validator confirms the claimed effect matches the reproduced result

This module provides step 2 and the fetch half of step 4. Sandboxed
execution (step 4's runner) lives in sn43/sandbox/ and ships in Phase 2.
"""
from __future__ import annotations

import hashlib
import logging
import os
from enum import Enum
from typing import Optional

logger = logging.getLogger("sn43.artifact_store")


class ArtifactKind(str, Enum):
    CODE = "code"
    DATA = "data"


def compute_artifact_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


class ArtifactStore:
    """Content-addressed store for causal-evidence code + input data.

    Layout: {root}/{kind}/{hash[:2]}/{hash[2:]}
    """

    def __init__(self, root: str = "data/artifact_store"):
        self.root = root
        for kind in ArtifactKind:
            os.makedirs(os.path.join(self.root, kind.value), exist_ok=True)

    def _path(self, kind: ArtifactKind, artifact_hash: str) -> str:
        if len(artifact_hash) != 64:
            raise ValueError(
                f"artifact_hash must be 64-char sha256 hex, got {len(artifact_hash)}"
            )
        return os.path.join(
            self.root, kind.value, artifact_hash[:2], artifact_hash[2:]
        )

    def put(self, kind: ArtifactKind, content: bytes) -> str:
        """Store artifact, return its sha256 hash."""
        artifact_hash = compute_artifact_hash(content)
        path = self._path(kind, artifact_hash)
        if os.path.exists(path):
            return artifact_hash
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        logger.info(
            f"Stored {kind.value} artifact {artifact_hash[:8]}... "
            f"({len(content)} bytes)"
        )
        return artifact_hash

    def get(self, kind: ArtifactKind, artifact_hash: str) -> Optional[bytes]:
        path = self._path(kind, artifact_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def has(self, kind: ArtifactKind, artifact_hash: str) -> bool:
        return os.path.exists(self._path(kind, artifact_hash))

    def put_code(self, content: bytes) -> str:
        return self.put(ArtifactKind.CODE, content)

    def put_data(self, content: bytes) -> str:
        return self.put(ArtifactKind.DATA, content)

    def get_code(self, artifact_hash: str) -> Optional[bytes]:
        return self.get(ArtifactKind.CODE, artifact_hash)

    def get_data(self, artifact_hash: str) -> Optional[bytes]:
        return self.get(ArtifactKind.DATA, artifact_hash)

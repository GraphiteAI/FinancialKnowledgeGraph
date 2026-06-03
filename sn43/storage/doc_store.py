"""
Content-addressed document storage.

Stores source documents (SEC filings, earnings transcripts, patents) keyed
by sha256 hash. Enables validators to resolve a ProvenanceAnchor's
`source_doc_hash` back to the original bytes to verify quoted text.

v1 uses the local filesystem. An IPFS or S3 backend can swap in behind the
same interface without touching callers.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

logger = logging.getLogger("sn43.doc_store")


def compute_doc_hash(content: bytes) -> str:
    """sha256 of raw bytes, hex-encoded."""
    return hashlib.sha256(content).hexdigest()


class DocumentStore:
    """Content-addressed store for source documents.

    Layout: {root}/{hash[:2]}/{hash[2:]}
    (git-style sharding to avoid fat directories)
    """

    def __init__(self, root: str = "data/doc_store"):
        self.root = root
        os.makedirs(self.root, exist_ok=True)

    def _path(self, doc_hash: str) -> str:
        if len(doc_hash) != 64:
            raise ValueError(f"doc_hash must be 64-char sha256 hex, got {len(doc_hash)}")
        return os.path.join(self.root, doc_hash[:2], doc_hash[2:])

    def put(self, content: bytes) -> str:
        """Store content, return its sha256 hash."""
        doc_hash = compute_doc_hash(content)
        path = self._path(doc_hash)
        if os.path.exists(path):
            return doc_hash
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)
        logger.info(f"Stored doc {doc_hash[:8]}... ({len(content)} bytes)")
        return doc_hash

    def get(self, doc_hash: str) -> Optional[bytes]:
        """Fetch content by hash. Returns None if not present."""
        path = self._path(doc_hash)
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return f.read()

    def has(self, doc_hash: str) -> bool:
        return os.path.exists(self._path(doc_hash))

    def verify(self, doc_hash: str, content: bytes) -> bool:
        """Check that content hashes to the claimed doc_hash."""
        return compute_doc_hash(content) == doc_hash

    def resolve_quote(
        self,
        doc_hash: str,
        quoted_text: str,
    ) -> bool:
        """Return True if quoted_text appears in the stored document.

        Used by score_provenance() to verify that an anchor's quote matches
        the source. Does a lenient containment check (whitespace-normalized).
        """
        content = self.get(doc_hash)
        if content is None:
            return False
        try:
            from sn43.pdf_utils import bytes_to_text
            text = bytes_to_text(content)
        except Exception:
            return False
        return _normalized_contains(text, quoted_text)


def _normalized_contains(haystack: str, needle: str) -> bool:
    """Whitespace-insensitive substring check."""
    h = " ".join(haystack.split())
    n = " ".join(needle.split())
    if not n:
        return False
    return n in h

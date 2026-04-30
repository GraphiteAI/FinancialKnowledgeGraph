"""Hotkey-based request signing for miner submissions.

Miners sign every submission with their Bittensor wallet hotkey. Validators
verify the signature against the on-chain hotkey for the claimed UID, blocking
forgery, replay, and tampering.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import deque
from typing import Callable, Dict, Optional

logger = logging.getLogger("sn43.auth")

SIG_VERSION = "v1"
TIMESTAMP_TOLERANCE_SECONDS = 60


class SignatureError(Exception):
    """Raised when signature verification fails. Map to HTTP 401."""


def canonical_body(payload: Dict) -> bytes:
    """Deterministic JSON for signing — sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _message(timestamp: str, nonce: str, body: bytes) -> bytes:
    body_hash = hashlib.sha256(body).hexdigest()
    return f"{SIG_VERSION}.{timestamp}.{nonce}.{body_hash}".encode("utf-8")


def sign_payload(hotkey, payload: Dict) -> Dict[str, str]:
    """Return headers a miner attaches to a signed submission.

    `hotkey` is a bittensor `Keypair` (i.e. `wallet.hotkey`).
    """
    ts = str(int(time.time()))
    nonce = uuid.uuid4().hex
    body = canonical_body(payload)
    sig = hotkey.sign(_message(ts, nonce, body)).hex()
    return {
        "X-Signature-Version": SIG_VERSION,
        "X-Timestamp": ts,
        "X-Nonce": nonce,
        "X-Hotkey": hotkey.ss58_address,
        "X-Signature": sig,
    }


class NonceCache:
    """Per-hotkey LRU of recent nonces. Rejects replays within the timestamp
    tolerance window — combined with timestamp checks, an attacker can't
    re-send a captured submission."""

    def __init__(self, max_per_key: int = 10_000) -> None:
        self._seen: Dict[str, deque] = {}
        self._sets: Dict[str, set] = {}
        self.max_per_key = max_per_key

    def check_and_add(self, hotkey: str, nonce: str) -> bool:
        s = self._sets.setdefault(hotkey, set())
        if nonce in s:
            return False
        d = self._seen.setdefault(hotkey, deque(maxlen=self.max_per_key))
        if len(d) == self.max_per_key:
            s.discard(d[0])  # evicted-on-append
        d.append(nonce)
        s.add(nonce)
        return True


def verify_request(
    headers: Dict[str, str],
    raw_body: bytes,
    claimed_uid: int,
    get_hotkey_for_uid: Callable[[int], Optional[str]],
    nonce_cache: NonceCache,
) -> str:
    """Verify a signed submission. Returns the hotkey ss58 on success.

    Raises SignatureError on any failure (caller maps to HTTP 401).
    """
    h = {k.lower(): v for k, v in headers.items()}
    try:
        version = h["x-signature-version"]
        ts = h["x-timestamp"]
        nonce = h["x-nonce"]
        hotkey = h["x-hotkey"]
        signature_hex = h["x-signature"]
    except KeyError as e:
        raise SignatureError(f"missing header: {e.args[0]}")

    if version != SIG_VERSION:
        raise SignatureError(f"unsupported signature version {version!r}")

    try:
        skew = abs(time.time() - int(ts))
    except ValueError:
        raise SignatureError("invalid X-Timestamp")
    if skew > TIMESTAMP_TOLERANCE_SECONDS:
        raise SignatureError(f"timestamp skew {skew:.0f}s > {TIMESTAMP_TOLERANCE_SECONDS}s")

    expected = get_hotkey_for_uid(claimed_uid)
    if not expected:
        raise SignatureError(f"uid {claimed_uid} not in metagraph")
    if expected != hotkey:
        raise SignatureError(f"hotkey {hotkey!r} does not match uid {claimed_uid}")

    if not nonce_cache.check_and_add(hotkey, nonce):
        raise SignatureError("nonce reused (replay)")

    try:
        from bittensor import Keypair  # type: ignore
    except ImportError as e:
        raise SignatureError(f"bittensor not installed on validator: {e}")

    try:
        kp = Keypair(ss58_address=hotkey)
        if not kp.verify(_message(ts, nonce, raw_body), bytes.fromhex(signature_hex)):
            raise SignatureError("signature does not match payload")
    except SignatureError:
        raise
    except Exception as e:
        raise SignatureError(f"verify error: {e}")

    return hotkey

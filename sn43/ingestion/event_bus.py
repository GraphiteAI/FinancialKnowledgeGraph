"""
Event bus: validator broadcasts EventTasks to connected miners in real time.

Architecture:
- Validator runs an `EventBusServer` (FastAPI + WebSocket) and `publish()`es
  tasks as JSON over `/events`
- Miners connect via `MinerClient`, receive tasks in an async loop, and POST
  submissions back to `/submit`
- Validators dedupe on `EventTask.dedupe_key` so a filing observed by two
  watchers only broadcasts once

The `EventBus` class is the in-process abstraction used by both sides; the
server wraps it with WebSocket fan-out, and tests exercise it directly
without going through the network.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable, Optional, Set

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

from sn43.auth import NonceCache, SignatureError, verify_request
from sn43.protocol import (
    DiscoveryCategory,
    EventTask,
    KnowledgeDelta,
)

logger = logging.getLogger("sn43.event_bus")


# ──────────────────────────── In-process bus ────────────────────────────

class EventBus:
    """Async pub/sub with two-tier dedupe by task.dedupe_key.

    Two states per dedupe_key:
      - PENDING: published, no miner has submitted yet. Lost on restart so
                 unanswered tasks are re-published when the validator comes
                 back up (no miner work was wasted on them yet anyway).
      - CONFIRMED: at least one miner submitted for this task. Persisted to
                 disk; survives restarts. Re-publishing would just spam
                 miners with already-extracted filings.

    The bus instance is shared between the EDGAR watcher (publisher) and
    the WebSocket server (consumer). The server fans out to connected miners.
    """

    def __init__(
        self,
        dedupe_cache_size: int = 10_000,
        persist_path: Optional[str] = None,
    ) -> None:
        self._subscribers: Set["asyncio.Queue[EventTask]"] = set()
        # FIFO eviction for confirmed cache via OrderedDict insertion order.
        from collections import OrderedDict
        self._confirmed: "OrderedDict[str, None]" = OrderedDict()
        self._pending: Set[str] = set()
        self._max_size = dedupe_cache_size
        self._lock = asyncio.Lock()
        self._persist_path = persist_path
        if persist_path:
            self._load_confirmed()

    def _load_confirmed(self) -> None:
        from pathlib import Path
        path = Path(self._persist_path)
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
            keys = data.get("confirmed", [])
            # Newest at the end so eviction (popitem(last=False)) removes
            # the oldest entries first. Trim to max_size on load too.
            for k in keys[-self._max_size:]:
                self._confirmed[k] = None
            logger.info(
                f"loaded {len(self._confirmed)} confirmed dedupe entries from {path}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to load dedupe cache from {path}: {e}")

    def _save_confirmed(self) -> None:
        if not self._persist_path:
            return
        try:
            from pathlib import Path
            path = Path(self._persist_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps({"confirmed": list(self._confirmed)}))
            tmp.replace(path)  # atomic on POSIX
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to save dedupe cache: {e}")

    async def publish(self, task: EventTask) -> bool:
        """Fan out a task to all subscribers. Returns False if deduped.

        Skips publishing if the dedupe_key is either already confirmed
        (some miner responded in a past run) or currently pending (we
        already published it and are awaiting a response).
        """
        async with self._lock:
            if task.dedupe_key in self._confirmed:
                logger.debug(f"Deduped (confirmed) event {task.dedupe_key}")
                return False
            if task.dedupe_key in self._pending:
                logger.debug(f"Deduped (pending) event {task.dedupe_key}")
                return False
            self._pending.add(task.dedupe_key)

        delivered = 0
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(task)
                delivered += 1
            except asyncio.QueueFull:
                logger.warning("subscriber queue full, dropping task")
        logger.info(
            f"Published {task.event_type.value} ({task.dedupe_key}) "
            f"to {delivered} subscribers"
        )
        return True

    async def confirm(self, dedupe_key: str) -> None:
        """Mark a task as having received at least one miner submission.

        Promotes from pending → confirmed; persists to disk so a restart
        won't re-publish this task. Called by the validator's submission
        scoring callback once a real submission arrives.
        """
        async with self._lock:
            self._pending.discard(dedupe_key)
            if dedupe_key in self._confirmed:
                # Touch to refresh insertion order (LRU within FIFO).
                self._confirmed.move_to_end(dedupe_key)
            else:
                self._confirmed[dedupe_key] = None
                while len(self._confirmed) > self._max_size:
                    self._confirmed.popitem(last=False)
        self._save_confirmed()

    @asynccontextmanager
    async def subscribe(
        self, maxsize: int = 256
    ) -> AsyncIterator["asyncio.Queue[EventTask]"]:
        queue: "asyncio.Queue[EventTask]" = asyncio.Queue(maxsize=maxsize)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def dedupe_cache_len(self) -> int:
        return len(self._confirmed) + len(self._pending)


# ──────────────────────────── WebSocket server ────────────────────────────

@dataclass
class SubmissionRecord:
    """One KnowledgeDelta a miner posted in response to an EventTask."""
    task_id: str
    miner_uid: Optional[int]
    delta: KnowledgeDelta
    received_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    inline_documents: dict = field(default_factory=dict)
    inline_code: dict = field(default_factory=dict)   # {hash: bytes}
    inline_data: dict = field(default_factory=dict)   # {hash: bytes}


@dataclass
class DiscoverySubmissionRecord:
    """An unsolicited KnowledgeDelta a miner posted without a validator task."""
    miner_uid: int
    category: DiscoveryCategory
    justification: str
    delta: KnowledgeDelta
    received_at: float = field(default_factory=lambda: asyncio.get_event_loop().time())


class DiscoveryRateLimiter:
    """In-memory per-miner-per-day counter for unsolicited submissions.

    Resets at UTC midnight each day. Resets on validator restart — this is a
    v1 soft limit; add persistence or a token-bucket when spam becomes real.
    """

    def __init__(self, max_per_day: int = 10) -> None:
        self.max_per_day = max_per_day
        self._counts: dict[tuple[int, str], int] = {}

    @staticmethod
    def _today_key() -> str:
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d")

    def check_and_increment(self, miner_uid: int) -> bool:
        key = (miner_uid, self._today_key())
        n = self._counts.get(key, 0)
        if n >= self.max_per_day:
            return False
        self._counts[key] = n + 1
        return True

    def remaining(self, miner_uid: int) -> int:
        key = (miner_uid, self._today_key())
        return max(0, self.max_per_day - self._counts.get(key, 0))


class EventBusServer:
    """FastAPI app that fans an EventBus out over WebSockets and accepts
    miner submissions over HTTP POST.

    Bring-your-own bus so validators can publish from any coroutine.
    """

    def __init__(
        self,
        bus: Optional[EventBus] = None,
        discovery_rate_limiter: Optional[DiscoveryRateLimiter] = None,
        require_signatures: bool = False,
        get_hotkey_for_uid: Optional[Callable[[int], Optional[str]]] = None,
        doc_store=None,
    ) -> None:
        self.bus = bus or EventBus()
        self.app = FastAPI(title="SN43 Event Bus")
        self.submissions: list[SubmissionRecord] = []
        self.discoveries: list[DiscoverySubmissionRecord] = []
        self._submission_callbacks: list[Callable[[SubmissionRecord], None]] = []
        self._discovery_callbacks: list[
            Callable[[DiscoverySubmissionRecord], None]
        ] = []
        self.rate_limiter = discovery_rate_limiter or DiscoveryRateLimiter()
        self.require_signatures = require_signatures
        self.get_hotkey_for_uid = get_hotkey_for_uid
        self._nonce_cache = NonceCache() if require_signatures else None
        self.doc_store = doc_store
        if require_signatures and get_hotkey_for_uid is None:
            raise ValueError(
                "require_signatures=True needs a get_hotkey_for_uid callback"
            )
        self._wire_routes()

    def on_submission(self, callback: Callable[[SubmissionRecord], None]) -> None:
        """Register a handler called every time a miner submits a delta."""
        self._submission_callbacks.append(callback)

    def on_discovery(
        self, callback: Callable[[DiscoverySubmissionRecord], None]
    ) -> None:
        """Register a handler called every time a miner submits an unsolicited delta."""
        self._discovery_callbacks.append(callback)

    def _wire_routes(self) -> None:
        app = self.app
        bus = self.bus

        @app.get("/healthz")
        async def healthz() -> dict:
            return {
                "ok": True,
                "subscribers": bus.subscriber_count,
                "dedupe_cache": bus.dedupe_cache_len,
            }

        @app.websocket("/events")
        async def events_ws(ws: WebSocket) -> None:
            await ws.accept()
            async with bus.subscribe() as queue:
                try:
                    while True:
                        task = await queue.get()
                        await ws.send_text(task.model_dump_json())
                except WebSocketDisconnect:
                    logger.info("miner disconnected from /events")
                except Exception as e:
                    logger.error(f"ws error: {e}")

        @app.post("/submit")
        async def submit(request: Request) -> dict:
            import base64

            raw_body = await request.body()
            try:
                payload = json.loads(raw_body)
                task_id = payload["task_id"]
                delta = KnowledgeDelta(**payload["delta"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                raise HTTPException(400, f"bad submission: {e}")

            if self.require_signatures:
                if delta.miner_uid is None:
                    raise HTTPException(401, "miner_uid required when signatures enforced")
                try:
                    verify_request(
                        dict(request.headers),
                        raw_body,
                        int(delta.miner_uid),
                        self.get_hotkey_for_uid,
                        self._nonce_cache,
                    )
                except SignatureError as e:
                    logger.warning(f"signature rejected uid={delta.miner_uid}: {e}")
                    raise HTTPException(401, f"signature verification failed: {e}")

            inline_documents: dict = {}
            for h, b64 in (payload.get("documents") or {}).items():
                try:
                    inline_documents[h] = base64.b64decode(b64)
                except Exception as e:
                    logger.warning(f"bad base64 for paywalled doc {h[:8]}...: {e}")

            inline_code: dict = {}
            inline_data: dict = {}
            artifacts = payload.get("artifacts") or {}
            for h, b64 in (artifacts.get("code") or {}).items():
                try:
                    inline_code[h] = base64.b64decode(b64)
                except Exception as e:
                    logger.warning(f"bad base64 for code {h[:8]}...: {e}")
            for h, b64 in (artifacts.get("data") or {}).items():
                try:
                    inline_data[h] = base64.b64decode(b64)
                except Exception as e:
                    logger.warning(f"bad base64 for data {h[:8]}...: {e}")

            record = SubmissionRecord(
                task_id=task_id,
                miner_uid=delta.miner_uid,
                delta=delta,
                inline_documents=inline_documents,
                inline_code=inline_code,
                inline_data=inline_data,
            )
            self.submissions.append(record)
            for cb in self._submission_callbacks:
                try:
                    cb(record)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"submission callback failed: {e}")
            return {"ok": True, "received": delta.total_items}

        @app.post("/submit-discovery")
        async def submit_discovery(payload: dict) -> dict:
            """Unsolicited miner submission — not tied to a validator task.

            Scored with stricter gates (see score_discovery). Rate-limited
            per miner_uid per UTC day.
            """
            try:
                delta = KnowledgeDelta(**payload["delta"])
                category = DiscoveryCategory(payload["discovery_category"])
                justification = str(payload["claimed_novelty_justification"])
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(400, f"bad discovery submission: {e}")

            if delta.miner_uid is None:
                raise HTTPException(400, "delta.miner_uid is required")

            if not self.rate_limiter.check_and_increment(delta.miner_uid):
                raise HTTPException(
                    429,
                    f"discovery rate limit exceeded ({self.rate_limiter.max_per_day}/day)",
                )

            record = DiscoverySubmissionRecord(
                miner_uid=delta.miner_uid,
                category=category,
                justification=justification,
                delta=delta,
            )
            self.discoveries.append(record)
            for cb in self._discovery_callbacks:
                try:
                    cb(record)
                except Exception as e:  # noqa: BLE001
                    logger.warning(f"discovery callback failed: {e}")
            return {
                "ok": True,
                "received": delta.total_items,
                "remaining_today": self.rate_limiter.remaining(delta.miner_uid),
            }


# ──────────────────────────── Miner client ────────────────────────────

class MinerClient:
    """Minimal client: receives EventTasks over WebSocket, submits deltas.

    Intended for reference miners and tests. Production miners will swap
    the transport but keep the same task handling pattern.
    """

    def __init__(
        self,
        ws_url: str,
        submit_url: str,
        miner_uid: Optional[int] = None,
    ) -> None:
        self.ws_url = ws_url
        self.submit_url = submit_url
        self.miner_uid = miner_uid

    async def listen(
        self,
        handler: Callable[[EventTask], "asyncio.Future | KnowledgeDelta | None"],
        max_tasks: Optional[int] = None,
    ) -> None:
        """Receive events and hand each to `handler`. If handler returns a
        KnowledgeDelta, submit it. `max_tasks` stops after N events (for tests).
        """
        import websockets  # lazy import so tests without websockets still work
        import httpx

        count = 0
        async with websockets.connect(self.ws_url) as ws:
            async for msg in ws:
                task = EventTask.model_validate_json(msg)
                result = handler(task)
                if asyncio.iscoroutine(result):
                    result = await result
                if isinstance(result, KnowledgeDelta):
                    async with httpx.AsyncClient() as client:
                        await client.post(
                            self.submit_url,
                            json={
                                "task_id": task.task_id,
                                "delta": json.loads(result.model_dump_json()),
                            },
                            timeout=10.0,
                        )
                count += 1
                if max_tasks is not None and count >= max_tasks:
                    break

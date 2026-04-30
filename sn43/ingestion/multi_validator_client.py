"""
Multi-validator client for SN43 miners.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Set, Union

import httpx

from sn43.protocol import EventTask, KnowledgeDelta

logger = logging.getLogger("sn43.multi_validator")

# Refresh the validator list every N seconds — handles registration churn.
DEFAULT_REFRESH_SECONDS = 600

# Filter out validators with stake below this floor (anti-spam).
DEFAULT_MIN_STAKE = 10000.0


@dataclass
class ValidatorEndpoint:
    """A discovered validator endpoint."""
    uid: int
    hotkey: str
    ip: str
    port: int

    @property
    def ws_url(self) -> str:
        return f"ws://{self.ip}:{self.port}/events"

    @property
    def submit_url(self) -> str:
        return f"http://{self.ip}:{self.port}/submit"

    def __hash__(self) -> int:
        return hash((self.uid, self.hotkey, self.ip, self.port))


# Handler signature: (task) -> KnowledgeDelta or None or coroutine of either.
TaskHandler = Callable[
    [EventTask],
    Union[None, KnowledgeDelta, Awaitable[Union[None, KnowledgeDelta]]],
]


class MultiValidatorClient:
    """Multi-validator listener + fan-out submitter for SN43 miners."""

    def __init__(
        self,
        miner_uid: int,
        netuid: int = 43,
        subtensor=None,
        hotkey=None,
        min_validator_stake: float = DEFAULT_MIN_STAKE,
        refresh_seconds: int = DEFAULT_REFRESH_SECONDS,
        submit_timeout: float = 30.0,
    ) -> None:
        if subtensor is None:
            try:
                import bittensor as bt
            except ImportError as e:
                raise RuntimeError(
                    "bittensor not installed; required for MultiValidatorClient. "
                ) from e
            SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
            subtensor = SubtensorCls()
        self.subtensor = subtensor
        self.netuid = netuid
        self.miner_uid = miner_uid
        self.hotkey = hotkey
        self.min_stake = min_validator_stake
        self.refresh_seconds = refresh_seconds
        self.submit_timeout = submit_timeout

        self._seen_events: Set[str] = set()
        self._seen_lock = asyncio.Lock()

        self._endpoints: List[ValidatorEndpoint] = []
        self._endpoints_lock = asyncio.Lock()

    # ──────────────────────────── Discovery ────────────────────────────

    def _subnet_owner_uid(self, metagraph) -> Optional[int]:
        """Return the UID of the subnet owner's hotkey (or None).
        """
        try:
            owner_hotkey = self.subtensor.get_subnet_owner_hotkey(self.netuid)
            if owner_hotkey and owner_hotkey in metagraph.hotkeys:
                return metagraph.hotkeys.index(owner_hotkey)
        except Exception:
            pass
        owner_hotkey = getattr(metagraph, "owner_hotkey", None)
        if owner_hotkey and owner_hotkey in metagraph.hotkeys:
            return metagraph.hotkeys.index(owner_hotkey)
        return None

    def discover_validators(self) -> List[ValidatorEndpoint]:
        """Read the metagraph, return validators that are serving an axon.
        """
        try:
            metagraph = self.subtensor.metagraph(self.netuid)
        except Exception as e:
            logger.warning(f"metagraph fetch failed: {e}")
            return []

        owner_uid = self._subnet_owner_uid(metagraph)

        extra_uids: Set[int] = set()
        import os as _os
        for raw in _os.getenv("EXTRA_VALIDATOR_UIDS", "").split(","):
            raw = raw.strip()
            if raw:
                try:
                    extra_uids.add(int(raw))
                except ValueError:
                    pass

        endpoints: List[ValidatorEndpoint] = []
        for uid in range(len(metagraph.uids)):
            try:
                has_permit = bool(metagraph.validator_permit[uid])
                is_owner = owner_uid is not None and uid == owner_uid
                is_extra = uid in extra_uids
                if not (has_permit or is_owner or is_extra):
                    continue
                if not (is_owner or is_extra):
                    stake = float(metagraph.S[uid])
                    if stake < self.min_stake:
                        continue
                axon = metagraph.axons[uid]
                if axon is None or axon.ip in ("", "0.0.0.0") or int(axon.port) == 0:
                    continue
                endpoints.append(ValidatorEndpoint(
                    uid=int(uid),
                    hotkey=str(metagraph.hotkeys[uid]),
                    ip=str(axon.ip),
                    port=int(axon.port),
                ))
            except (IndexError, AttributeError, ValueError) as e:
                logger.debug(f"skipping uid {uid}: {e}")
        return endpoints

    async def _refresh_endpoints(self) -> None:
        """Background loop: refresh validator list every refresh_seconds."""
        while True:
            await asyncio.sleep(self.refresh_seconds)
            new = self.discover_validators()
            async with self._endpoints_lock:
                old_uids = {e.uid for e in self._endpoints}
                new_uids = {e.uid for e in new}
                added = new_uids - old_uids
                removed = old_uids - new_uids
                if added or removed:
                    logger.info(
                        f"validator roster change: +{sorted(added)} -{sorted(removed)}"
                    )
                self._endpoints = new

    # ──────────────────────────── Listening ────────────────────────────

    async def listen_all(self, handler: TaskHandler) -> None:
        """Connect to every validator's WebSocket, fan out submissions.

        Runs forever until cancelled. Individual WebSocket failures are
        logged + reconnected; one bad validator doesn't take down the
        whole client.
        """
        self._endpoints = self.discover_validators()
        if not self._endpoints:
            logger.warning(
                f"no validators found on netuid {self.netuid}. "
                "Is anyone running validator+axon publish? "
            )
            return

        logger.info(
            f"connecting to {len(self._endpoints)} validators: "
            + ", ".join(f"uid={e.uid}@{e.ip}:{e.port}" for e in self._endpoints)
        )

        refresh_task = asyncio.create_task(self._refresh_endpoints())

        try:
            listeners = [
                asyncio.create_task(self._listen_one(ep, handler))
                for ep in self._endpoints
            ]
            await asyncio.gather(*listeners, return_exceptions=True)
        finally:
            refresh_task.cancel()

    async def _listen_one(self, ep: ValidatorEndpoint, handler: TaskHandler) -> None:
        """Persistent WebSocket loop for one validator.
        """
        import websockets

        QUICK_RETRIES = [1, 5, 30]
        LONG_RETRY = 3600  # 1 hour

        attempt = 0
        while True:
            try:
                async with websockets.connect(ep.ws_url) as ws:
                    logger.info(f"connected to validator uid={ep.uid} @ {ep.ws_url}")
                    attempt = 0 
                    async for raw in ws:
                        await self._handle_event(raw, handler)
            except asyncio.CancelledError:
                raise
            except Exception as e:  
                wait = (
                    QUICK_RETRIES[attempt]
                    if attempt < len(QUICK_RETRIES)
                    else LONG_RETRY
                )
                attempt += 1
                stage = (
                    f"quick retry {attempt}/{len(QUICK_RETRIES)}"
                    if attempt <= len(QUICK_RETRIES)
                    else "hourly retry"
                )
                logger.warning(
                    f"validator uid={ep.uid} ({ep.ws_url}): {e}. "
                    f"{stage} — reconnecting in {wait}s"
                )
                await asyncio.sleep(wait)

    async def _handle_event(self, raw: str, handler: TaskHandler) -> None:
        """Parse, dedupe, dispatch to handler, fan out resulting delta."""
        try:
            task = EventTask.model_validate_json(raw)
        except Exception as e:  
            logger.debug(f"bad event JSON: {e}")
            return

        async with self._seen_lock:
            if task.dedupe_key in self._seen_events:
                return
            self._seen_events.add(task.dedupe_key)

        try:
            result = handler(task)
            if asyncio.iscoroutine(result):
                result = await result
        except Exception as e:  
            logger.error(f"handler raised on task {task.task_id}: {e}")
            return

        if not isinstance(result, KnowledgeDelta) or result.is_empty:
            return

        async with self._endpoints_lock:
            targets = list(self._endpoints)
        await self._fanout_submit(task, result, targets)

    # ──────────────────────────── Fan-out submit ────────────────────────────

    async def _fanout_submit(
        self,
        task: EventTask,
        delta: KnowledgeDelta,
        targets: List[ValidatorEndpoint],
    ) -> None:
        """Submit the same delta to every validator's /submit. Parallel."""
        results = await asyncio.gather(
            *(self._submit_one(task, delta, ep) for ep in targets),
            return_exceptions=True,
        )
        ok = sum(1 for r in results if r is True)
        fails = [
            (ep.uid, r) for ep, r in zip(targets, results)
            if not (r is True)
        ]
        logger.info(
            f"task={task.task_id} fan-out submit: ok={ok}/{len(targets)}"
            + (f" failures={fails[:3]}" if fails else "")
        )

    async def _submit_one(
        self,
        task: EventTask,
        delta: KnowledgeDelta,
        ep: ValidatorEndpoint,
    ) -> bool:
        """POST one signed submission. Returns True on 2xx."""
        payload = {
            "task_id": task.task_id,
            "delta": json.loads(delta.model_dump_json()),
        }

        headers = {}
        body_kwarg = {"json": payload}
        if self.hotkey is not None:
            from sn43.auth import canonical_body, sign_payload
            body = canonical_body(payload)
            headers = sign_payload(self.hotkey, payload)
            headers["Content-Type"] = "application/json"
            body_kwarg = {"content": body}

        try:
            async with httpx.AsyncClient(timeout=self.submit_timeout) as client:
                r = await client.post(ep.submit_url, headers=headers, **body_kwarg)
                r.raise_for_status()
                return True
        except Exception as e:  
            logger.debug(f"submit to uid={ep.uid} failed: {e}")
            return False

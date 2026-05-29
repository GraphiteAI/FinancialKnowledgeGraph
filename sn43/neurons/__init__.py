"""
Validator for the SN43 Knowledge Graph subnet.

What lives in this module:
- CLI setup + Bittensor wallet/subtensor helpers
- MinerPerformance + RankStability (score aggregation + convergence)
- ValidatorService — composes EventBus, EventBusServer, EdgarWatcher, scoring,
  and weight-setting into one async daemon
- `validator` CLI command — the single entry point operators run
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence

import click
import httpx
import numpy as np
from dotenv import load_dotenv

load_dotenv()

try:
    import bittensor as bt
except ImportError:
    bt = None

from envs.knowledge_graph.scoring import compute_total_score
from ingestion.edgar_watcher import EdgarWatcher
from ingestion.event_bus import EventBus, EventBusServer, SubmissionRecord
from sandbox.runner import SandboxRunner
from sn43.protocol import EventTask, KnowledgeDelta
from storage.artifact_store import ArtifactStore
from storage.doc_store import DocumentStore
from storage.graph_store import KnowledgeGraphStore


logger = logging.getLogger("neurons")

# ──────────────────────────── Config ────────────────────────────

NETUID = 43
CENTRAL_SERVER_URL = os.getenv("CENTRAL_SERVER_URL", "http://localhost:8000")
EVENT_BUS_HOST = os.getenv("EVENT_BUS_HOST", "0.0.0.0")
EVENT_BUS_PORT = int(os.getenv("EVENT_BUS_PORT", "8765"))

MERGE_THRESHOLD = 0.3
SET_WEIGHTS_PERIOD = 3600   # re-submit weights every hour
EMISSION_PERCENT = 20        # share of emissions allocated to this set of miners

BURN_UIDS = [
    int(u.strip()) for u in os.getenv("BURN_UIDS", "20,74,102,155,177").split(",")
    if u.strip()
]

HEARTBEAT = time.monotonic()


# ──────────────────── MinerPerformance + RankStability ────────────────────

@dataclass
class MinerPerformance:
    uid: int
    scores: List[float] = field(default_factory=list)
    last_seen: float = 0.0
    slash_multiplier: float = 1.0

    @property
    def run_count(self) -> int:
        return len(self.scores)

    @property
    def mean_score(self) -> float:
        """Reward signal published to chain.

        Combines quality (mean per-submission score) with quantity
        (sqrt of submission count) so a miner with 100 high-quality
        submissions outranks a miner with 1 — but with diminishing
        returns so spam doesn't dominate.

        Final formula:  mean(scores) * sqrt(N) * slash_multiplier

        Example:
          - Miner A: 100 scores averaging 0.55 → 0.55 * 10 = 5.5
          - Miner B: 1 score of 0.55           → 0.55 * 1  = 0.55
          - A : B reward ratio = 91% : 9% (after chain normalization)

        Magnitude is irrelevant on chain — `set_weights_once` rescales
        the whole vector to sum to EMISSION_PERCENT/100. Only relative
        ordering matters.
        """
        if not self.scores:
            return 0.0
        return float(np.sum(self.scores)) * self.slash_multiplier

    def add_score(self, score: float) -> None:
        self.scores.append(score)
        self.last_seen = time.time()

    def slash(self, factor: float = 0.5, reason: str = "") -> None:
        """Multiply the slash_multiplier by `factor`. Compounds.

        At factor=0.5: first slash halves rewards, second quarters them, etc.
        Slashes are in-memory only (lost on restart, re-derived from chain).
        """
        before = self.slash_multiplier
        self.slash_multiplier *= factor
        logger.warning(
            f"slashed miner uid={self.uid}: multiplier {before:.3f} → "
            f"{self.slash_multiplier:.3f} ({reason})"
        )


def normalized_kendall_distance(rank_a: Sequence[int], rank_b: Sequence[int]) -> float:
    n = len(rank_a)
    pos_b = {m: i for i, m in enumerate(rank_b)}
    total = n * (n - 1) // 2
    if total == 0:
        return 0.0
    miners = list(rank_a)
    discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            if pos_b.get(miners[i], 0) > pos_b.get(miners[j], 0):
                discord += 1
    return discord / total


class RankStability:
    def __init__(self, uids: List[int], min_window: int = 5):
        self.uids = list(uids)
        self.n = len(self.uids)
        self.uid_to_idx = {uid: idx for idx, uid in enumerate(self.uids)}
        self.min_window = min_window
        self.history: List[List[int]] = []

    def add_ranking(self, ranking: Sequence[int]):
        self.history.append(list(ranking))

    def per_miner_std(self):
        if len(self.history) < 2:
            return np.full(self.n, np.inf)
        T = len(self.history)
        pos = np.zeros((T, self.n), dtype=float)
        for t, ranking in enumerate(self.history):
            for rank_pos, uid in enumerate(ranking):
                if uid in self.uid_to_idx:
                    pos[t, self.uid_to_idx[uid]] = rank_pos
        return np.std(pos, axis=0, ddof=0)

    def has_converged(
        self,
        mean_std_threshold: float = 2.0,
        max_std_threshold: float = 6.0,
        kendall_threshold: float = 0.02,
    ) -> bool:
        if len(self.history) < self.min_window:
            return False
        stds = self.per_miner_std()
        if np.mean(stds) >= mean_std_threshold or np.max(stds) >= max_std_threshold:
            return False
        dists = [
            normalized_kendall_distance(self.history[i], self.history[i + 1])
            for i in range(len(self.history) - 1)
        ]
        if np.mean(dists) >= kendall_threshold:
            return False
        return True


# ──────────────────── Subtensor helpers ────────────────────

_SUBTENSOR = None


async def get_subtensor():
    global _SUBTENSOR
    if _SUBTENSOR is None:
        if bt is None:
            raise RuntimeError("bittensor not installed")
        AsyncSubtensorCls = (
            getattr(bt, "AsyncSubtensor", None)
            or getattr(bt, "async_subtensor")
        )
        _SUBTENSOR = AsyncSubtensorCls(
            os.getenv("SUBTENSOR_ENDPOINT", "wss://entrypoint-finney.opentensor.ai:443")
        )
        try:
            await _SUBTENSOR.initialize()
        except Exception as e:
            logger.error(f"Failed to initialize subtensor: {e}")
            raise
    return _SUBTENSOR

def _api_url(path: str) -> str:
    return f"{CENTRAL_SERVER_URL}/api/v1{path}"


async def _post_signed_to_central(
    path: str,
    body: Dict[str, Any],
    *,
    hotkey,
    miner_uid: int,
    timeout: float = 120.0,
) -> "httpx.Response":
    """POST to central server, signed with the validator's hotkey.
    """
    if hotkey is not None:
        from sn43.auth import canonical_body, sign_payload
        body_bytes = canonical_body(body)
        headers = sign_payload(hotkey, body)
        headers["Content-Type"] = "application/json"
        headers["X-Validator-UID"] = str(miner_uid)  # validator's own UID
        kwargs = {"content": body_bytes}

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(_api_url(path), headers=headers, **kwargs)
        resp.raise_for_status()
        return resp


async def _get_signed_from_central(
    path: str,
    *,
    hotkey,
    validator_uid: int,
    timeout: float = 30.0,
) -> "httpx.Response":
    if hotkey is not None:
        from sn43.auth import canonical_body, sign_payload
        payload: dict = {}
        headers = sign_payload(hotkey, payload)
        headers["X-Validator-UID"] = str(validator_uid)
        headers["Content-Type"] = "application/json"
        body = canonical_body(payload)

    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.get(_api_url(path), headers=headers, content=body)
        resp.raise_for_status()
        return resp


# ──────────────────────────── ValidatorService ────────────────────────────

class ValidatorService:
    """Composition root for the validator daemon.

    Owns the event bus, watcher, scoring stack, and per-miner score tracking.
    `run()` is the async entry point that starts every subsystem.
    """

    def __init__(
        self,
        *,
        graph_persist_path: str = "data/knowledge_graph.json",
        doc_store_path: str = "data/doc_store",
        artifact_store_path: str = "data/artifact_store",
        require_signatures: bool = False,
        get_hotkey_for_uid: Optional[Callable[[int], Optional[str]]] = None,
        validator_hotkey=None,  # bittensor Keypair, used to sign requests to central
        validator_uid: Optional[int] = None,
    ) -> None:
        self.graph = KnowledgeGraphStore(persist_path=graph_persist_path)
        self.doc_store = DocumentStore(root=doc_store_path)
        self.artifact_store = ArtifactStore(root=artifact_store_path)
        self.runner = SandboxRunner()
        self.validator_hotkey = validator_hotkey
        self.validator_uid = validator_uid

        self.bus = EventBus(
            persist_path=os.getenv("DEDUPE_CACHE_PATH", "data/dedupe_cache.json"),
        )
        self.server = EventBusServer(
            bus=self.bus,
            require_signatures=require_signatures,
            get_hotkey_for_uid=get_hotkey_for_uid,
        )
        self.watcher = EdgarWatcher(bus=self.bus)
        from ingestion.doc_verifier import DocVerifier
        self.doc_verifier = DocVerifier(
            doc_store=self.doc_store,
            edgar_contact=os.getenv("SN43_EDGAR_CONTACT"),
        )

        self._tasks_in_flight: Dict[str, EventTask] = {}
        self.miners: Dict[int, MinerPerformance] = {}

        self.server.on_submission(self._on_submission)

    # ─────── Event loop plumbing ───────

    async def run(self) -> None:
        """Start every subsystem and block until cancelled."""
        tasks = [
            asyncio.create_task(self._publish_loop()),
            asyncio.create_task(self.watcher.run()),
            asyncio.create_task(self._serve_http()),
            asyncio.create_task(self._heartbeat_loop()),
        ]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for t in tasks:
                t.cancel()
            raise

    async def _serve_http(self) -> None:
        """Run the EventBusServer FastAPI app under uvicorn."""
        import uvicorn
        config = uvicorn.Config(
            self.server.app,
            host=EVENT_BUS_HOST,
            port=EVENT_BUS_PORT,
            log_level="info",
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def _publish_loop(self) -> None:
        """
        Wrap the EventBus publish with task tracking so we can look up the
        original EventTask when a miner submits against a task_id.

        We subscribe to our own bus so every task passing through is cached.
        """
        async with self.bus.subscribe() as queue:
            while True:
                task = await queue.get()
                self._tasks_in_flight[task.task_id] = task
                # Bound the cache
                if len(self._tasks_in_flight) > 10_000:
                    # Drop the oldest 5000
                    keys = list(self._tasks_in_flight.keys())
                    for k in keys[:5000]:
                        self._tasks_in_flight.pop(k, None)

    async def _heartbeat_loop(self) -> None:
        global HEARTBEAT
        while True:
            HEARTBEAT = time.monotonic()
            await asyncio.sleep(30)

    # ─────── Submission handling ───────

    def _on_submission(self, record: SubmissionRecord) -> None:
        """Fires every time a miner POSTs a delta. Schedules async scoring."""
        asyncio.create_task(self._score_submission(record))

    async def _score_submission(self, record: SubmissionRecord) -> None:
        try:
            event_for_confirm = self._tasks_in_flight.get(record.task_id)
            if event_for_confirm is not None:
                await self.bus.confirm(event_for_confirm.dedupe_key)

            for h, content in record.inline_code.items():
                actual = self.artifact_store.put_code(content)
                if actual != h:
                    logger.warning(
                        f"code hash mismatch: claimed={h[:8]}... actual={actual[:8]}..."
                    )
            for h, content in record.inline_data.items():
                actual = self.artifact_store.put_data(content)
                if actual != h:
                    logger.warning(
                        f"data hash mismatch: claimed={h[:8]}... actual={actual[:8]}..."
                    )

            verifications = await self.doc_verifier.verify_delta(
                record.delta, documents=record.inline_documents
            )
            self._mark_items_verified(record.delta, verifications)

            verified_count = sum(1 for ok, _ in verifications.values() if ok)
            total_anchors = len(verifications)
            if total_anchors and verified_count < total_anchors:
                kinds = [k for _, k in verifications.values()]
                logger.info(
                    f"task={record.task_id}: {verified_count}/{total_anchors} verified; "
                    f"sources={dict((k, kinds.count(k)) for k in set(kinds))}"
                )

            event = self._tasks_in_flight.get(record.task_id)
            scores = compute_total_score(
                record.delta,
                graph=self.graph,
                doc_store=self.doc_store,
                event=event,
                artifact_store=self.artifact_store,
                runner=self.runner,
            )
            total = scores["total"]

            merged = False
            if total >= MERGE_THRESHOLD:
                self._apply_contradiction_slashes(record.delta)

                await self._merge_via_central_server(record.delta, verifications)
                self.graph.merge_delta(
                    record.delta, miner_uid=record.miner_uid or -1
                )
                merged = True

            miner_uid = record.miner_uid or -1
            perf = self.miners.setdefault(miner_uid, MinerPerformance(uid=miner_uid))
            perf.add_score(total)

            logger.info(
                f"scored miner={miner_uid} task={record.task_id} "
                f"total={total:.3f} merged={merged} "
                f"components={{" + ", ".join(
                    f"{k}={v:.2f}" for k, v in scores.items() if k != 'total'
                ) + "}}"
            )
        except Exception as e:
            logger.error(f"scoring failed for task {record.task_id}: {e}")
            traceback.print_exc()

    def _apply_contradiction_slashes(self, delta: KnowledgeDelta) -> None:
        """For every verified Fact in the delta, slash miners whose unverified
        facts it contradicts. Marks the contradicted facts as superseded so
        they don't trigger again.

        Slashing factor: 0.5 per offense (compounds). Same miner contributing
        multiple offending facts in the same delta gets slashed once per
        unique contradiction (the loop iterates over distinct fact_ids).
        """
        for new_fact in delta.facts:
            if not new_fact.verified:
                continue
            contradictions = self.graph.find_contradicting_unverified_facts(new_fact)
            for c in contradictions:
                stale_id = c["fact_id"]
                self.graph.mark_superseded(stale_id, new_fact.fact_id)
                # Slash every miner who contributed the contradicted fact.
                for uid in set(c["contributors"]):
                    if uid < 0:
                        continue
                    perf = self.miners.setdefault(uid, MinerPerformance(uid=uid))
                    perf.slash(
                        0.5,
                        reason=f"verified fact {new_fact.fact_id} contradicted "
                               f"unverified {stale_id} (was {c['stored_value']!r}, "
                               f"now {new_fact.value!r})",
                    )

    def _mark_items_verified(
        self,
        delta: KnowledgeDelta,
        verifications: Dict[str, tuple],
    ) -> None:
        """Mutate delta items in place: set `verified` and `verification_source`.

        An item is `verified=True` only when EVERY anchor's hash was
        independently fetch-verified by the validator. Any anchor that's
        paywalled-content or url-only drops the whole item to unverified,
        with verification_source set to whichever anchor was weakest:
            url_only > paywalled_content > whitelisted_fetch  (worst → best)
        """
        from sn43.protocol import VerificationSource

        worst_priority = {"url_only": 2, "paywalled_content": 1, "whitelisted_fetch": 0}

        def _classify(item) -> None:
            chain = getattr(item, "provenance", None)
            anchors = getattr(chain, "anchors", []) if chain else []
            if not anchors:
                item.verified = False
                item.verification_source = VerificationSource.UNKNOWN
                return
            all_verified = True
            worst = "whitelisted_fetch"
            for a in anchors:
                ok, src = verifications.get(a.source_doc_hash, (False, "url_only"))
                if not ok:
                    all_verified = False
                if worst_priority[src] > worst_priority[worst]:
                    worst = src
            item.verified = all_verified
            item.verification_source = VerificationSource(worst)

        for ent in delta.entities:
            _classify(ent)
        for rel in delta.relationships:
            _classify(rel)
        for fact in delta.facts:
            _classify(fact)

    async def _merge_via_central_server(
        self,
        delta: KnowledgeDelta,
        verifications: Optional[Dict[str, tuple]] = None,
    ) -> None:
        """Push the accepted delta to the central server's canonical graph.

        Bundles every verified source-doc AND every causal-evidence artifact
        (code + input data) the validator has on disk so the central server
        can store the bytes as cryptographic proof. This is what makes the
        central server's graph independently auditable — it holds both the
        claim AND the source document AND the executable code that produced
        any causal evidence.
        """
        import base64

        # Pull verified doc bytes from local store, base64-encode for transport.
        # `verifications` is now {hash: (ok, source_kind)}.
        documents: Dict[str, str] = {}
        if verifications:
            for h, v in verifications.items():
                ok = v[0] if isinstance(v, tuple) else v
                if not ok:
                    continue
                content = self.doc_store.get(h)
                if content is None:
                    continue
                documents[h] = base64.b64encode(content).decode("ascii")

        # Forward causal-evidence artifacts referenced by this delta.
        code_payload: Dict[str, str] = {}
        data_payload: Dict[str, str] = {}
        for rel in delta.relationships:
            for ev in (rel.causal_evidence or []):
                if ev.code_hash and ev.code_hash not in code_payload:
                    blob = self.artifact_store.get_code(ev.code_hash)
                    if blob is not None:
                        code_payload[ev.code_hash] = base64.b64encode(blob).decode("ascii")
                if ev.input_data_hash and ev.input_data_hash not in data_payload:
                    blob = self.artifact_store.get_data(ev.input_data_hash)
                    if blob is not None:
                        data_payload[ev.input_data_hash] = base64.b64encode(blob).decode("ascii")

        body = {
            "delta": delta.model_dump(mode="json"),
            "documents": documents,
            "artifacts": {"code": code_payload, "data": data_payload},
        }

        try:
            await _post_signed_to_central(
                "/deltas/merge", body,
                hotkey=self.validator_hotkey,
                miner_uid=(self.validator_uid if self.validator_uid is not None else 0),
            )
        except Exception as e:
            logger.warning(f"central server merge failed: {e}")

    def restore_scores_from_central(self, subtensor=None) -> bool:
        try:
            import httpx
            url = _api_url("/miners/contributions")
            body = b""
            if self.validator_hotkey is not None and self.validator_uid is not None:
                from sn43.auth import canonical_body, sign_payload
                payload: dict = {}
                headers = sign_payload(self.validator_hotkey, payload)
                headers["X-Validator-UID"] = str(self.validator_uid)
                headers["Content-Type"] = "application/json"
                # Body MUST match what we signed — empty body would make the
                # server-side body-hash differ from ours and fail verification.
                body = canonical_body(payload)
            with httpx.Client(timeout=15) as client:
                r = client.get(url, headers=headers, content=body)
                r.raise_for_status()
                data = r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning(f"central restore failed: {e}")
            return False

        # Try to fetch published weights from chain to back-calculate quality.
        chain_weights: Dict[int, float] = {}
        if subtensor is not None and self.validator_hotkey is not None:
            try:
                metagraph = subtensor.metagraph(NETUID)
                my_uid = metagraph.hotkeys.index(self.validator_hotkey.ss58_address)
                row = [float(w) for w in metagraph.W[my_uid]]
                for uid, w in enumerate(row):
                    if w > 0:
                        chain_weights[uid] = w
            except (ValueError, IndexError, AttributeError) as e:
                logger.info(
                    f"central restore: chain weights unavailable ({e}); "
                    "falling back to synthetic mean for back-calculation"
                )

        FALLBACK_MEAN = 0.5  # used only if chain weight is missing for a UID
        restored = 0
        nonzero_q = 0
        for uid_str, count in counts.items():
            try:
                uid = int(uid_str)
                n = max(1, int(count))
            except (TypeError, ValueError):
                continue

            if uid in BURN_UIDS:
                continue

            w = chain_weights.get(uid)
            if w is not None:
                back_mean = w / (n ** 0.5)
                back_mean = max(0.0, min(1.0, back_mean))
            else:
                back_mean = FALLBACK_MEAN

            if back_mean > 0:
                nonzero_q += 1

            perf = MinerPerformance(uid=uid)
            perf.scores = [back_mean] * n
            perf.last_seen = time.time()
            self.miners[uid] = perf
            restored += 1

        top_n = max(int(c) for c in counts.values())
        if chain_weights:
            logger.info(
                f"central restore: seeded {restored} miners with real counts + "
                f"back-derived means from chain weights "
                f"({nonzero_q} non-zero quality, top_n={top_n})"
            )
        else:
            logger.info(
                f"central restore: seeded {restored} miners with real counts "
                f"(synthetic mean=0.5 — chain weights unavailable; top_n={top_n})"
            )
        return True

    def restore_scores_from_metagraph(self, subtensor, own_hotkey_ss58: Optional[str] = None) -> None:
        """Seed self.miners from this validator's previously-set chain weights.

        Called once at startup. Reconstructs synthetic score histories that 
        preserve the pre-restart relative weight ratios.

        Algorithm:
          - mean_score formula is `mean(scores) * sqrt(N) * slash`
          - To preserve `weight_i / weight_j = chain_weight_i / chain_weight_j`,
            we need `sqrt(N_i) / sqrt(N_j) = w_i / w_j`, i.e. N ∝ w².
          - Set every restored miner's mean to a synthetic 1.0 and pick
            N proportional to chain_weight²; normalize so the top-weight
            miner gets `BASELINE_N` synthetic submissions.
        """

        SYNTHETIC_MEAN = 0.5
        BASELINE_N = 10  

        try:
            metagraph = subtensor.metagraph(NETUID)
            own_hotkey = own_hotkey_ss58 or getattr(
                metagraph, "_self_hotkey", None
            ) or getattr(self.validator_hotkey, "ss58_address", None)
            if own_hotkey is None:
                logger.info("metagraph score restore: own hotkey not bound; skipping")
                return

            try:
                my_uid = metagraph.hotkeys.index(own_hotkey)
            except ValueError:
                logger.info("metagraph score restore: own hotkey not registered yet")
                return

            try:
                weights_row = [float(w) for w in metagraph.W[my_uid]]
            except (IndexError, AttributeError):
                logger.info("metagraph score restore: W matrix unavailable")
                return

            max_w = max(weights_row) if weights_row else 0.0
            if max_w <= 0:
                logger.info("metagraph score restore: no positive weights to restore")
                return

            restored = 0
            for uid, w in enumerate(weights_row):
                if w <= 0:
                    continue
                n = max(1, int(round(((w / max_w) ** 2) * BASELINE_N)))
                perf = MinerPerformance(uid=uid)
                perf.scores = [SYNTHETIC_MEAN] * n
                perf.last_seen = time.time()
                self.miners[uid] = perf
                restored += 1

            logger.info(
                f"metagraph score restore: seeded {restored} miners "
                f"(synthetic mean=1.0, baseline_n={BASELINE_N}, "
                f"N ∝ chain_weight²)"
            )
        except Exception as e:
            logger.warning(f"metagraph score restore failed: {e}")

    # ─────── Weight setting ───────

    async def set_weights_once(self, wallet: Any) -> None:
        """Compute weights and submit to chain.
        """
        try:
            sub = await get_subtensor()
            metagraph = await sub.metagraph(NETUID)
            uids = [int(uid) for uid in metagraph.uids]
            miner_share = EMISSION_PERCENT / 100   

            weights: List[float] = []
            for uid in uids:
                perf = self.miners.get(uid)
                weights.append(perf.mean_score if perf else 0.0)

            total = sum(weights)
            mode_log = "own_scores"
            if total > 0:
                scale = miner_share / total
                weights = [w * scale for w in weights]
            else:
                try:
                    consensus = [float(x) for x in metagraph.incentive]
                    csum = sum(consensus)
                    if csum > 0:
                        scale = miner_share / csum
                        weights = [c * scale for c in consensus]
                        mode_log = "metagraph_consensus"
                    else:
                        weights = [0.0] * len(uids)
                        mode_log = "no_data"
                except (AttributeError, TypeError):
                    weights = [0.0] * len(uids)
                    mode_log = "no_data"

            uid_to_idx = {uid: i for i, uid in enumerate(uids)}
            for burn_uid in BURN_UIDS:
                if burn_uid in uid_to_idx:
                    weights[uid_to_idx[burn_uid]] = 0.0

            miners_sum = sum(weights)
            burn_share = 1.0 - miners_sum 

            selected_burn_uid = await _select_burn_uid(sub)
            if selected_burn_uid in uid_to_idx:
                weights[uid_to_idx[selected_burn_uid]] = burn_share
                logger.info(
                    f"burn rotation: assigning {burn_share:.4f} to UID "
                    f"{selected_burn_uid} (miners_sum={miners_sum:.4f}, mode={mode_log})"
                )
            else:
                logger.warning(
                    f"selected burn UID {selected_burn_uid} not on metagraph; "
                    f"{burn_share:.4f} of weight stays unassigned this cycle"
                )

            await sub.set_weights(
                wallet=wallet,
                netuid=NETUID,
                weights=weights,
                uids=uids,
                wait_for_inclusion=False,
                wait_for_finalization=False,
            )
            logger.info(
                f"set_weights submitted for {len(uids)} uids "
                f"(mode={mode_log}, miners={miners_sum:.4f}, "
                f"burn[uid={selected_burn_uid}]={burn_share:.4f})"
            )
        except Exception as e:
            logger.error(f"set_weights failed: {e}")


def _selector(posts: List[int], bucket: int) -> int:
    if len(posts) != 4:
        raise ValueError(f"selector needs exactly 4 posts, got {len(posts)}")
    s = f"{posts[0]}|{posts[1]}|{posts[2]}|{posts[3]}|{bucket}"
    h = hashlib.sha256(s.encode()).hexdigest()
    idx = int(h, 16) % len(BURN_UIDS)
    return BURN_UIDS[idx]


BLOCKS_PER_BUCKET = 300


async def _current_bucket(sub) -> int:
    block = None
    try:
        get_block = getattr(sub, "get_current_block", None)
        if get_block is not None:
            res = get_block()
            if hasattr(res, "__await__"):
                block = await res
            else:
                block = res
    except Exception as e:  # noqa: BLE001
        logger.warning(f"get_current_block failed ({e}); falling back to wall clock")
    if not isinstance(block, int):
        # Last resort — at least logs the issue so operators notice.
        logger.warning(
            "burn rotation falling back to wall-clock bucket; "
            "validators with drifted clocks will diverge"
        )
        return int(time.time() // 3600)
    return block // BLOCKS_PER_BUCKET


async def _pull_post(sub, uid: int) -> Optional[int]:
    try:
        commit = await sub.get_revealed_commitment(netuid=NETUID, uid=uid)
        if not commit:
            return None
        latest = commit[-1]
        # commit format: (block, payload)
        payload = latest[1] if isinstance(latest, (list, tuple)) and len(latest) >= 2 else latest
        return int(str(payload).strip())
    except Exception as e:  # noqa: BLE001
        logger.debug(f"pull_post UID {uid}: {e}")
        return None


async def _select_burn_uid(sub) -> int:
    bucket = await _current_bucket(sub)

    posts: List[int] = []
    for uid in BURN_UIDS:
        v = await _pull_post(sub, uid)
        if v is not None:
            posts.append(v)
        if len(posts) >= 4:
            break
    if len(posts) >= 4:
        try:
            return _selector(posts[:4], bucket)
        except ValueError:
            pass

    h = hashlib.sha256(f"sn43-burn-fallback|{bucket}".encode()).hexdigest()
    idx = int(h, 16) % len(BURN_UIDS)
    logger.info(
        f"burn rotation: fewer than 4 commitments, using bucket-only fallback "
        f"({len(posts)} posts found, bucket={bucket})"
    )
    return BURN_UIDS[idx]


async def _watchdog(timeout: int = 600) -> None:
    while True:
        await asyncio.sleep(60)
        if time.monotonic() - HEARTBEAT > timeout:
            logger.error(f"Watchdog: no heartbeat for {timeout}s, exiting")
            os._exit(1)


# ──────────────────────────── CLI ────────────────────────────

@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"], case_sensitive=False,
    ),
    default=None,
)
def cli(log_level: Optional[str]) -> None:
    global NETUID
    level_name = (log_level or os.getenv("LOG_LEVEL") or "INFO").upper()
    NETUID = int(os.getenv("NETUID", NETUID))
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )



@cli.command("validator")
def validator_cmd() -> None:
    """Run the validator daemon: event bus + EDGAR watcher + scoring + weights.
    """
    wallet = None
    get_hotkey_for_uid: Optional[Callable[[int], Optional[str]]] = None

    def require_env(name: str) -> str:
        value = os.getenv(name)
        if not value:
            raise RuntimeError(f"Missing: {name}")
        return value

    coldkey = require_env("BT_WALLET_COLD")
    hotkey = require_env("BT_WALLET_HOT")
    if bt is None:
        raise RuntimeError(
            "bittensor not installed"
        )
    
    WalletCls = getattr(bt, "Wallet", None) or getattr(bt, "wallet")
    wallet = WalletCls(name=coldkey, hotkey=hotkey)

    SubtensorCls = getattr(bt, "Subtensor", None) or getattr(bt, "subtensor")
    subtensor = SubtensorCls(network=os.getenv("BT_NETWORK", "finney"))

    def _hotkey_for(uid: int) -> Optional[str]:
        try:
            metagraph = subtensor.metagraph(NETUID)
            return metagraph.hotkeys[uid]
        except (IndexError, KeyError, AttributeError):
            return None

    get_hotkey_for_uid = _hotkey_for

    own_validator_uid: Optional[int] = None
    own_validator_hotkey = None
    if wallet is not None:
        try:
            metagraph = subtensor.metagraph(NETUID)
            own_validator_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
            own_validator_hotkey = wallet.hotkey
            logger.info(
                f"validator self-discovered: uid={own_validator_uid} "
                f"hotkey={own_validator_hotkey.ss58_address}"
            )
        except (ValueError, AttributeError) as e:
            logger.warning(
                f"could not locate own UID on metagraph ({e}); "
                "central calls will fall back to API-key auth"
            )

    service = ValidatorService(
        require_signatures=True,
        get_hotkey_for_uid=get_hotkey_for_uid,
        validator_hotkey=own_validator_hotkey,
        validator_uid=own_validator_uid,
    )

    if wallet is not None:
        if not service.restore_scores_from_central(subtensor=subtensor):
            try:
                service.restore_scores_from_metagraph(
                    subtensor,
                    own_hotkey_ss58=wallet.hotkey.ss58_address,
                )
            except Exception as e:
                logger.warning(f"could not restore scores from metagraph: {e}")

    if wallet is not None:
        try:
            external_ip = os.getenv("EXTERNAL_IP", "").strip()
            if not external_ip:
                try:
                    external_ip = bt.utils.networking.get_external_ip()
                except (AttributeError, Exception):
                    import urllib.request
                    external_ip = urllib.request.urlopen(
                        "https://api.ipify.org", timeout=5
                    ).read().decode().strip()
            AxonCls = getattr(bt, "Axon", None) or getattr(bt, "axon")
            axon = AxonCls(wallet=wallet, port=EVENT_BUS_PORT, ip=external_ip)
            ok = axon.serve(netuid=NETUID, subtensor=subtensor)
            logger.info(
                f"published axon to chain: {external_ip}:{EVENT_BUS_PORT} "
                f"(success={ok})"
            )
        except Exception as e:
            logger.warning(
                f"could not publish axon to chain: {e}. "
                "Miners won't discover this validator — set EXTERNAL_IP "
                "env var or check network connectivity to subtensor."
            )

    async def _weight_loop() -> None:
        await asyncio.sleep(15)
        await service.set_weights_once(wallet)
        while True:
            await asyncio.sleep(SET_WEIGHTS_PERIOD)
            await service.set_weights_once(wallet)

    async def _main() -> None:
        mode = f"NETUID {NETUID}"
        sig_mode = "ENFORCED"
        logger.info(
            f"Starting validator: event bus on {EVENT_BUS_HOST}:{EVENT_BUS_PORT}, "
            f"central server {CENTRAL_SERVER_URL}, mode={mode}, signatures={sig_mode}"
        )
        tasks = [service.run(), _watchdog(timeout=600)]
        tasks.append(_weight_loop())
        await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(_main())

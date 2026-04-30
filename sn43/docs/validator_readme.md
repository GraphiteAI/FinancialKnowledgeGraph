# SN43 Validator Guide

How to run the SN43 validator daemon.

## The short version

The validator is a single async process (`python -m neurons validator`). It:

1. Polls SEC EDGAR for fresh filings (every 10 seconds)
2. Broadcasts each new filing as an `EventTask` on a WebSocket event bus
3. Accepts `KnowledgeDelta` submissions from miners over HTTP — verifies hotkey signatures against the metagraph
4. **Independently fetches each provenance URL on the whitelist**, hashes the bytes, and confirms the hash matches what the miner claimed
5. Scores each submission on 8 dimensions, including re-running miner causal code in a sandbox
6. Pushes accepted deltas (plus verified source bytes + causal artifacts) to the graph data server's canonical graph
7. Slashes miners 50% when a verified Fact contradicts an unverified Fact they contributed
8. Sets weights on the Bittensor chain every hour based on per-miner mean scores

---

## Running

### With Docker (recommended)

```bash
cd sn43
cp .env.example .env
# Edit .env: set BT_WALLET_COLD, BT_WALLET_HOT, SN43_EDGAR_CONTACT

docker compose up -d
docker compose logs -f sn43-validator
```

The compose file exposes port `8765` — miners connect here over WebSocket.

### Without Docker

```bash
cd sn43
pip install -e ".[dev]"
cp .env.example .env  # edit it

# IMPORTANT: validators publish their event-bus port to chain on startup.
# Set EXTERNAL_IP to your machine's public IP (or let bittensor auto-detect).
export EXTERNAL_IP=YOUR_PUBLIC_IP

# Open the event-bus port so miners can reach your WebSocket /events and
# POST /submit endpoints. Default port is 8765.
sudo ufw allow 8765/tcp
sudo ufw reload

python -m neurons validator
```

The validator publishes an axon record `(hotkey, ip, port)` to the chain on startup via `serve_axon`. This is how miners discover you — they read the metagraph, find every UID with `validator_permit=True` and a serving axon, and open a WebSocket per validator. **Without firewall access on the published port, miners will see you on chain but fail to connect — you'll never receive submissions.**

You should see this in the startup log:

```
[INFO] neurons: published axon to chain: 203.0.113.5:8765 (success=True)
```

Verify reachability from an external machine:

```bash
curl http://203.0.113.5:8765/healthz
# → {"ok":true,"subscribers":N,"dedupe_cache":M}
```

If `success=False` in the log, miners won't find you. Common causes:
- `EXTERNAL_IP` is wrong (set to a private LAN IP)
- The wallet isn't registered on this subnet
- Your subtensor connection is unstable — retry on next restart

If `success=True` but `curl` from outside times out:
- Firewall blocking 8765 — `sudo ufw allow 8765/tcp` or check cloud provider security groups
- Validator binding to `127.0.0.1` instead of `0.0.0.0` — set `EVENT_BUS_HOST=0.0.0.0`

### Behind a reverse proxy (nginx/Caddy)

If you proxy your event bus through a domain (e.g. `wss://my-validator.example.com/events`), keep 8765 closed to the public — only open 443. Set `EVENT_BUS_PORT=443` (or whatever public port your proxy listens on) so the published axon record matches what miners actually need to dial. Internally uvicorn can still bind 8765 on `127.0.0.1`; only the **published** port matters for discovery.

See [`output/nginx_https_setup.md`](../../output/nginx_https_setup.md) for a complete proxy + Let's Encrypt setup.

### Causal-evidence dependencies

The validator's sandbox re-executes miner causal code (`score_reproducibility`). For Granger and Event Study evidence to score, install the same scientific Python stack the miner uses:

```bash
pip install pandas statsmodels numpy
```

Without these, sandbox runs will fail with `ModuleNotFoundError` and reproducibility scores 0 across the board.

---

## What the validator owns

All composed inside `ValidatorService` ([neurons/__init__.py](../neurons/__init__.py)):

| Subsystem | Job |
|---|---|
| `EventBus` | In-process async pub/sub with dedupe by accession number |
| `EventBusServer` | FastAPI app: `/events` WebSocket (broadcast to miners), `/submit` HTTP (miner submissions), `/healthz` |
| `EdgarWatcher` | Polls SEC EDGAR atom feed, parses new filings, publishes `EventTask`s |
| `DocVerifier` | Independently fetches each provenance URL on the whitelist, hashes, verifies match — what stops miners from fabricating provenance |
| `SandboxRunner` | Runs miner causal-inference code in a hardened subprocess to verify reproducibility |
| `DocumentStore` | Content-addressed source documents (sha256) — used by provenance scoring |
| `ArtifactStore` | Content-addressed code + data blobs — used by reproducibility scoring |
| `KnowledgeGraphStore` | Local copy of the graph (canonical lives on graph data server) — also tracks superseded facts for slashing |
| Signature verification | `sn43.auth.verify_request` against the live metagraph; replay-protected via timestamp + nonce LRU |
| `MinerPerformance` | Per-miner score history + slash multiplier; restored from chain weights on startup |
| Scoring stack | `compute_total_score()` across 8 dimensions |
| Weight loop | Every hour, `sub.set_weights()` on the Bittensor chain |

---

## Auth to graph data server (hotkey-signed, no API key)

The validator authenticates to the graph data server using its **Bittensor hotkey**, not an API key. Every outbound request to graph data server is signed with `wallet.hotkey.sign(...)`; graph data server re-checks the signature against the metagraph for the claimed UID, requires `validator_permit=True`, enforces a 60s timestamp + per-hotkey nonce LRU for replay protection.

Required headers (auto-added by `_post_signed_to_central` / `_get_signed_from_central`):
- `X-Validator-UID: <your-uid>`
- `X-Hotkey: <ss58>`
- `X-Signature: <hex sig>`
- `X-Signature-Version: v1`
- `X-Timestamp: <unix seconds>`
- `X-Nonce: <unique per request>`

**Per-item attribution:** every entity, relationship, and fact merged through `/deltas/merge` gets stamped with `submitted_by_validator_uid` and `merged_at` (UTC) on the canonical graph. If a validator's hotkey is later compromised, the subnet owner can call `POST /api/v1/admin/validator/revoke` to remove every item that validator submitted (optionally filtered by timestamp).

---

## How scoring works

For every miner submission arriving at `/submit`:

1. **Verify the hotkey signature** — body sha256 + timestamp + nonce, checked against the metagraph for the claimed UID. Stale or replayed requests rejected with HTTP 401.
2. **Persist any inline causal artifacts** the miner uploaded (code + data) into the local `ArtifactStore`.
3. **Verify every provenance URL**: for each unique `(hash, url)` in the delta's anchors,
   - if URL is on the whitelist (`sec.gov`, news, patents, etc.): fetch independently, hash, compare. Match → store, mark anchor as `whitelisted_fetch`. Mismatch → log, skip.
   - if not whitelisted but miner sent inline bytes: hash the bytes, store as `paywalled_content`.
   - if not whitelisted and no inline bytes: record the URL claim only, mark as `url_only`.
4. **Tag every item** in the delta with `verified=True/False` based on whether ALL its anchors verified via path A.
5. **Look up the original `EventTask`** by `task_id` (cached from the bus publish loop).
6. **Run all 8 scoring dimensions**:
   - **accuracy** (15%) — IDs, URLs, dates structurally valid
   - **provenance** (15%) — quoted text resolves in stored source documents
   - **reproducibility** (20%) — causal code re-runs in sandbox, output matches claimed effect within 5%
   - **novelty** (15%) — fraction of items not already in graph
   - **freshness** (5%) — source document recency
   - **latency** (15%) — exponential decay from event timestamp (5-min half-life)
   - **depth** (10%) — richness of the delta
   - **coverage** (5%) — underserved-sector bonus
7. **Slashing check**: for every verified Fact in the delta, scan the local graph for unverified facts that match `(entity_id, fact_type)` but disagree on `value` (>5% relative diff for numerics; strict for strings). Any matches → mark superseded, halve the contributing miner's `slash_multiplier`.
8. **If `total >= 0.3`**, POST `{delta, documents, artifacts}` to graph data server `/deltas/merge`.
9. **Append score** to `MinerPerformance[miner_uid].scores`. Final per-miner weight is computed by:

   ```
   weight = mean(scores) * sqrt(N) * slash_multiplier
   ```

   where `N` is the miner's submission count. The `sqrt(N)` factor balances quality and quantity:
   - Pure mean would let a 1-submission miner tie a 100-submission miner with equal quality
   - Pure sum would reward spam
   - `mean × sqrt(N)` rewards sustained high-quality contribution with diminishing returns. A 100-submission miner outranks a 1-submission miner ~10×, not 100×.

**Cadence:** the validator publishes weights ~15 seconds after startup (so the chain immediately sees a fresh commitment from you, not stale ones from before the restart) and then every `SET_WEIGHTS_PERIOD = 3600` seconds (1 hour) afterward.

## Tuning knobs

Environment variables with sensible defaults:

| Var | Default | What it controls |
|---|---|---|
| `EVENT_BUS_HOST` / `EVENT_BUS_PORT` | `0.0.0.0` / `8765` | Bus bind address; the port gets published to chain so miners find this validator |
| `EXTERNAL_IP` | auto-detect | Public IP that miners use to reach this validator. Set explicitly when behind NAT or on a cloud VM with a separate egress IP |
| `NETUID` | `43` | Subnet ID on Bittensor |
| `BT_NETWORK` | `finney` | Subtensor network (`finney`, `test`, `local`) |
| `BT_WALLET_COLD` / `BT_WALLET_HOT` | required (mainnet) | Validator's coldkey + hotkey wallet names |
| `SN43_EDGAR_CONTACT` | `validator@example.com` | Used in EDGAR User-Agent; **change this in production** |
| `WHITELIST_DOMAINS` | SEC, USPTO, major news/wire/finance domains | Comma-separated list — only these get hash-verified by independent fetch |
| `DEDUPE_CACHE_PATH` | `data/dedupe_cache.json` | Where the bus persists the "confirmed" dedupe cache (tasks with at least one miner response). Survives restarts so old filings aren't re-broadcast. |
| `LOG_LEVEL` | `INFO` | `DEBUG` shows every signature check + URL fetch |

Inside `neurons/__init__.py`:
- `MERGE_THRESHOLD = 0.3` — deltas below this are rejected
- `SET_WEIGHTS_PERIOD = 3600` — weight-setting interval in seconds
- `EMISSION_PERCENT = 5` — share of emissions allocated to miners (rest goes to selected burn UID)

---

## Operational notes

**SEC EDGAR rate limits.** The watcher polls per-form-type (5 defaults) at ~0.5 req/s, well under SEC's 10 req/s cap. The `DocVerifier` adds independent fetches when miners cite SEC URLs — those are throttled to ≤4 concurrent + 150ms between requests, deduped by URL across in-flight requests, and cached by hash.

**Sandbox cost.** Every causal claim triggers a sandbox re-execution. Tune `score_reproducibility(..., sample_fraction=0.1)` in scoring.py to sample 10% per epoch if validator load gets heavy.

**WebSocket capacity.** The bus fans out to all connected miners. Default subscriber queue size is 256 tasks; slow consumers get dropped with a warning rather than blocking the publisher.

**Weight setting failures.** If `sub.set_weights()` fails (network or chain issue), the error is logged and the next hour's cycle retries. No tight retry loop — the chain catches up.

**Whitelist drift.** Adding a domain to the whitelist doesn't retroactively re-verify items already in the graph. Existing items keep their original `verification_source`. To re-verify history, walk unverified items and have a miner resubmit them.

---

## Health check

```bash
curl http://localhost:8765/healthz
# {"ok": true, "subscribers": 3, "dedupe_cache": 127}
```

- `subscribers` — how many miners are currently connected
- `dedupe_cache` — how many recent EDGAR events are in the dedupe set

If `subscribers == 0`, no miners are connected — check networking / firewall between validator and miners.

---
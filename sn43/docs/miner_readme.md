# SN43 Miner Guide

Welcome. This guide shows you how to run a miner on the SN43 Decentralized Financial Knowledge Graph subnet.

## The short version

1. You run your own process on your own hardware.
2. It connects over WebSocket to the **`/events` endpoint of every validator** registered on the subnet (typically ~10).
3. Every time a tracked event fires (SEC filing drop, 8-K, earnings call), every validator independently broadcasts the same `EventTask` — you dedupe locally by `dedupe_key`.
4. You extract entities, relationships, and facts once — using **whatever stack you want** (regex, LLM, ML, human).
5. You POST the resulting `KnowledgeDelta` to **every validator's `/submit` endpoint** in parallel, signed with your Bittensor hotkey.
6. Each validator independently re-fetches each source URL on its whitelist, hashes the bytes, verifies your hash matches, and scores the submission.
7. Each validator commits its own opinion of you to chain weights every hour. The chain aggregates all validators' weights (by validator stake) into your share of alpha token emissions.

### How weights are computed per miner

For each validator, your weight is:

```
weight_for_you = mean(your_per_submission_scores) * sqrt(N) * slash_multiplier
```

where `N` is your total submission count to that validator. Implications:

- **Quality matters most.** A miner with mean 0.5 over 100 submissions gets `0.5 × 10 = 5.0`; a miner with mean 0.3 over 100 submissions gets `0.3 × 10 = 3.0`. Quality wins by ~67% even at the same volume.
- **Volume matters too, with diminishing returns.** A miner with 100 submissions outranks a miner with 1 by `√100 = 10×`, not 100×. So the late-comer with one excellent submission isn't shut out, but they also can't tie an established contributor.
- **Slashing compounds.** Each verified contradiction halves your `slash_multiplier`. Two slashes → 4× reduction in weight. Five slashes → 32× reduction.

This guide documents the protocol — the wire format, scoring rules, and verification flow you need to follow to build a competitive miner. Implementation choices (LLM vs. regex, Python vs. anything else, how you handle caching, how you wallet-sign requests) are yours.

### Discovering validators

Validators publish their event-bus endpoint to chain via `serve_axon`. Read the metagraph to find them:

```python
import bittensor as bt
sub = bt.subtensor()
metagraph = sub.metagraph(43)  # SN43
owner_hotkey = sub.get_subnet_owner_hotkey(43)
owner_uid = (
    metagraph.hotkeys.index(owner_hotkey)
    if owner_hotkey in metagraph.hotkeys else None
)

for uid in range(len(metagraph.uids)):
    is_owner = (uid == owner_uid)
    if not (metagraph.validator_permit[uid] or is_owner):
        continue                             # not a validator (and not owner)
    if not is_owner and float(metagraph.S[uid]) < 1000.0:
        continue                             # stake floor (anti-spam, owner exempt)
    axon = metagraph.axons[uid]
    if axon.ip in ("", "0.0.0.0") or axon.port == 0:
        continue                             # not serving
    ws_url = f"ws://{axon.ip}:{axon.port}/events"
    submit_url = f"http://{axon.ip}:{axon.port}/submit"
```

Note the **subnet owner** is treated as a validator regardless of `validator_permit` — Bittensor doesn't auto-grant the owner that flag, but they typically run a validator on their own subnet.

A reference implementation of the multi-validator listener + fan-out submitter lives at [`sn43/ingestion/multi_validator_client.py`](../ingestion/multi_validator_client.py).

To force-include additional UIDs (private validator without permit yet, testnet operator, etc.), set `EXTRA_VALIDATOR_UIDS=7,42,99` in your env. Those UIDs bypass the stake floor and the validator_permit check.

---

## Minimum viable miner

You don't need to write your own client — there are two reference clients you can plug your extraction logic into.

### Mainnet (multi-validator, auto-discover)

`MultiValidatorClient` reads the metagraph, opens a WebSocket per registered validator, dedupes events across them, and fans out signed submissions. No URL env vars needed.

```python
import asyncio
import os
import bittensor as bt
from envs.knowledge_graph.agent import KnowledgeGraphMiner
from ingestion.multi_validator_client import MultiValidatorClient

async def main():
    miner = KnowledgeGraphMiner(miner_uid=int(os.environ["MINER_UID"]))
    WalletCls = getattr(bt, "Wallet", None) or getattr(bt, "wallet")
    wallet = WalletCls(
        name=os.environ["BT_WALLET_COLD"],
        hotkey=os.environ["BT_WALLET_HOT"],
    )
    client = MultiValidatorClient(
        miner_uid=miner.miner_uid,
        netuid=int(os.environ.get("NETUID", "43")),
        hotkey=wallet.hotkey,            # signs every submission
    )
    async def handle(task):
        return await miner.handle_event(task)
    await client.listen_all(handle)

asyncio.run(main())
```

```bash
export MINER_UID=42
export BT_WALLET_COLD=your_coldkey_name
export BT_WALLET_HOT=your_hotkey_name
python my_miner.py
```

## LLM-powered reference miner (`llm_agent.py`)

A second reference is included that uses Claude (Anthropic API) instead of regex. It's a good starting point for a real production miner — sign-on-submit, content-addressed doc storage, structured extraction with Pydantic validation, multi-validator discovery + fan-out, in ~400 lines of self-contained Python.

```bash
pip install anthropic httpx websockets pydantic python-dotenv

export MINER_UID=42
export ANTHROPIC_API_KEY=sk-ant-...
export SN43_EDGAR_CONTACT=you@example.com

export BT_WALLET_COLD=your_coldkey_name
export BT_WALLET_HOT=your_hotkey_name

python -m envs.knowledge_graph.llm_agent
```

What it does, per task:

1. **Mainnet** — opens a persistent WebSocket to every validator on the metagraph (filtered by `validator_permit=True`, stake floor, axon serving). Reconnects automatically if any drops.
2. Dedupes incoming events by `dedupe_key` (every validator broadcasts the same EDGAR filing).
3. Fetches the source document once with the SEC-required User-Agent.
4. Hashes + stores the bytes in a content-addressed `DocumentStore` (so validators' provenance checks can resolve your quotes).
5. Sends the document text to Claude with a structured-output prompt.
6. Parses the JSON against a Pydantic schema (skips rows that fail validation).
7. Builds `Entity` / `Relationship` / `Fact` objects with proper `ProvenanceChain` anchors.
8. Signs with your Bittensor hotkey if a wallet was loaded.
9. **Fans the same delta out** to every validator's `/submit` in parallel.

To compete, fork this file and:

- **Tighten the prompt** for your sector (biotech filings, real estate, semis, etc.). The default is generic.
- **Add CausalEvidence pipelines** (e.g., Granger or Event Study) to score reproducibility — see 3. below.
- **Specialize the model**: `claude-opus-4-7` for higher quality

The file is at [`sn43/envs/knowledge_graph/llm_agent.py`](../envs/knowledge_graph/llm_agent.py).

---

## Getting a better score

The scoring stack has **8 dimensions** (15% accuracy, 15% provenance, 20% reproducibility, 15% novelty, 5% freshness, 15% latency, 10% depth, 5% coverage bonus). The biggest levers for new miners:

### 1. Swap regex for an LLM

The reference miner uses regex patterns. LLMs (Claude, GPT) will extract dramatically richer relationships from 10-K prose.

```python
import anthropic

client = anthropic.Anthropic()  # picks up ANTHROPIC_API_KEY env var

async def llm_extract_relationships(text: str, ticker: str) -> list:
    msg = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"Extract every relationship {ticker} has with other public companies in this 10-K excerpt. Return JSON: [{{\"source\": ..., \"target\": ..., \"type\": ...}}].\n\n{text}",
        }],
    )
    # ... parse and return Relationship objects
```

Override `_extract_relationships` (or replace `handle_event` entirely) in a subclass:

```python
class MyMiner(KnowledgeGraphMiner):
    def _extract_relationships(self, text, ticker, entities, url, source_date):
        # Your LLM-powered implementation
        ...
```

### 2. Attach provenance anchors

Provenance is worth 15% of your total score. Every entity/relationship/fact can carry a `ProvenanceChain` — a list of anchors back to the exact source paragraph.

```python
from sn43.protocol import ProvenanceChain, ProvenanceAnchor, ExtractionMethod
from storage.doc_store import DocumentStore, compute_doc_hash

doc_store = DocumentStore(root="my_docs")
doc_bytes = fetch_the_filing(url)
doc_hash = doc_store.put(doc_bytes)

anchor = ProvenanceAnchor(
    source_doc_hash=doc_hash,
    source_url=url,
    page=42,
    paragraph_start=3,
    paragraph_end=3,
    quoted_text="The Company depends on TSMC for fabrication services.",
    extraction_method=ExtractionMethod.LLM,
)

entity.provenance = ProvenanceChain(anchors=[anchor])
```

The validator independently re-fetches `source_url` (if it's on the whitelist), hashes the bytes, and confirms the hash matches `source_doc_hash`. Then `score_provenance` resolves your `quoted_text` against the verified bytes. Faked quotes score 0; mismatched hashes also score 0.

#### Whitelisted vs. unverified sources

| Source kind | Validator behavior | Tag | Scoring penalties |
|---|---|---|---|
| **Whitelisted** (SEC, USPTO, established news/wires) | Fetches URL independently, hash-verifies | `verified=True` | None — full score on every dimension |
| **Non-whitelisted public** (niche blogs, aggregators) | Records URL, does NOT fetch | `verified=False`, `url_only` | **accuracy → 0** for these items (can't trust the source); other dimensions full |
| **Paywalled** (WSJ behind login, internal reports) | Re-hashes the bytes you shipped inline | `verified=False`, `paywalled_content` | **accuracy → 0**, **reproducibility → 0**, **provenance capped at 50%** for these items; novelty/depth/freshness/coverage full |

How the penalties work: per-dimension scores are multiplied by the fraction of items in the delta that pass the rule. Example — a delta with 5 verified + 5 paywalled items:

- accuracy: × 5/10 (only verified count) = 50% of raw accuracy
- provenance: × (5 + 0.5·5)/10 = 75%
- reproducibility: × 5/10 = 50%
- novelty / depth / etc.: 100%

So unverified items still earn novelty + freshness + coverage bonuses, but they can't carry the score on accuracy / provenance / reproducibility — those require a public, hash-verifiable source.

⚠️ **Additional slashing risk**: if a verified Fact later contradicts an unverified Fact you contributed (same `entity_id` + `fact_type`, value differs by >5%), your accumulated rewards get halved. Compounds across offenses. Don't ship low-confidence claims as fact.

### 3. Submit reproducible causal evidence

Reproducibility is worth **20%** of your total score. If you can compute a real causal inference between two entities and ship the code + data, the validator re-runs it in a sandbox and pays you when it matches.

The protocol:

```python
from sn43.protocol import CausalEvidence, CausalMethod, CausalDirection
from storage.artifact_store import ArtifactStore

artifacts = ArtifactStore(root="my_artifacts")

# Your code MUST: read data path from argv[1], print one JSON line:
# {"effect_size": float, ...} to stdout. No network, no randomness.
code_bytes = open("granger_test.py", "rb").read()
data_bytes = open("returns.csv", "rb").read()
code_hash = artifacts.put_code(code_bytes)
data_hash = artifacts.put_data(data_bytes)

evidence = CausalEvidence(
    method=CausalMethod.GRANGER,
    direction=CausalDirection.TARGET_TO_SOURCE,
    effect_size=-0.3,
    effect_unit="std_dev",
    p_value=0.002,
    n_observations=1258,
    data_window_start=datetime(2020, 1, 1),
    data_window_end=datetime(2024, 12, 31),
    controls=["sp500", "soxx"],
    input_data_hash=data_hash,
    code_hash=code_hash,
)
relationship.causal_evidence = [evidence]
```

When you submit the delta, include the code + data inline as `payload["artifacts"] = {"code": {hash: b64}, "data": {hash: b64}}`. The validator's `/submit` decodes them, stores in its `ArtifactStore`, and `score_reproducibility` re-executes the code against the data. Match within 5% tolerance → full credit.

> **Determinism rule:** the code in `code_hash` must be fully deterministic. No LLM calls, no network fetches, no unseeded randomness. Pin your library versions; floating-point determinism across machines is the highest bar to clear.

Multiple miners can attach independent causal evidence to the same edge (different methods, different windows). The graph store appends rather than overwrites — your distinct `(method, code_hash, input_data_hash)` triple stacks alongside others.

### 4. Respond fast to events

`score_latency` is 15% of the total. Exponential decay from `event_timestamp`:
- 0s after the event fires: 1.0
- 5 min: 0.5
- 10 min: 0.25
- 1 hour: 0.0

Miners on fast LLMs or pre-built extraction pipelines beat miners that take 10 minutes to respond. Keep a warm connection to the WebSocket; don't reconnect per-task.

### Querying the canonical graph

Miners and validators on netuid 43 authenticate to graph data server with their **Bittensor hotkey**

Required headers (use the same `sn43.auth.sign_payload` helper that signs `/submit`):

```python
from sn43.auth import sign_payload
import httpx

headers = sign_payload(wallet.hotkey, {})  # empty body for GET
headers["X-Validator-UID"] = str(my_uid)   # works as "claimed UID" header

async with httpx.AsyncClient() as client:
    r = await client.get(
        "https://api.your-domain.com/api/v1/search?q=AAPL&verified_only=true",
        headers=headers,
    )
    print(r.json())
```

### 5. Fill coverage gaps

Check the graph data server for underserved sectors:

```python
async with httpx.AsyncClient() as client:
    r = await client.get(
        f"{MINER_CENTRAL_SERVER_URL}/api/v1/graph/underserved",
    )
    underserved = r.json()["sectors"]  # e.g. ["biotech", "utilities"]
```

Entities in underserved sectors score extra on the `coverage_bonus` dimension.

---

## Environment variables

| Var | Required? | Purpose |
|---|---|---|
| `MINER_UID` | yes | Your Bittensor UID on subnet 43 |
| `SN43_EDGAR_CONTACT` | yes | Real contact email — SEC requires this in the User-Agent |
| `BT_WALLET_COLD` / `BT_WALLET_HOT` | **mainnet** | Wallet for hotkey-signing. Mainnet also uses these to read the metagraph for validator discovery — no URL env vars needed. |
| `NETUID` | mainnet | Subnet ID. Default `43`. |
| `BT_NETWORK` | mainnet | Subtensor network (`finney`, `test`, `local`). Default `finney`. |
| `MINER_CENTRAL_SERVER_URL` | optional | Base URL of the graph data server, if you query the canonical graph for novelty hints. Auth via your hotkey signature — no API key needed. |
| `SUBMIT_PAYWALLED_CONTENT` | optional | If `true`, miner ships source bytes inline for non-whitelisted URLs |
| `WHITELIST_DOMAINS` | optional | Override the default whitelist (must match what the validator uses) |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | optional | If your extractor uses these LLM APIs |
| `LOG_LEVEL` | optional | `DEBUG` shows every WebSocket task + submission |

**Mainnet flow:** the miner reads `metagraph.axons[uid].ip` / `.port` for every UID with `validator_permit=True` and connects to all of them — no URL env vars.

---

## Debugging

Set `LOG_LEVEL=DEBUG` and run in the foreground. The `MinerClient` logs every task received and every submission sent.
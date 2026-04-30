# SN43 — Decentralized Financial Knowledge Graph

A Bittensor subnet where miners build and maintain a knowledge graph of US-listed companies. The graph maps relationships — supply chains, competition, executive networks, revenue dependencies, patent overlaps — and attaches **reproducible causal claims** to those relationships, with **cryptographic provenance** back to source documents.

## What makes this different

1. **Every claim cites its source.** Entities, relationships, and facts carry a `ProvenanceChain` — anchors back to the exact paragraph of a source document, verifiable by sha256.
2. **Validators independently fetch sources.** Miners can't fabricate provenance: the validator pulls the URL itself, hashes the bytes, and matches them against the miner's claim. Mismatches don't score.
3. **Causal claims are reproducible.** Miners ship the code + data they used (e.g. a Granger test or event study); validators re-run them in a sandbox and confirm the effect size matches within tolerance.
4. **Real-time ingestion.** SEC filings become tasks within seconds of publication. Faster submissions earn more via `score_latency`.
5. **Miners are self-hosted.** No agent-pulling, no Docker-per-miner. Miners run their own Python process (or anything else) and connect to the validator over WebSocket.
6. **Hotkey-signed submissions.** Each miner submission is signed with their Bittensor hotkey; validators verify against the metagraph, blocking forgery and replay.

## Architecture

```
        ┌─────────────────────────────────────┐
        │     Graph Data Server               │
        │  canonical graph + customer APIs    │
        └────────────▲────────────────────────┘
                     │ /deltas/merge
                     │
        ┌────────────┴─────────────────────┐
        │   Validator Daemon (this repo)   │
        │                                  │
        │  • EdgarWatcher → EventBus       │
        │  • /events WebSocket             │
        │  • /submit HTTP                  │
        │  • Scoring (8 dims, incl.        │
        │    sandboxed reproducibility)    │
        │  • Bittensor weight setting      │
        └──────▲───────────────────────▲───┘
               │                       │
               │ WebSocket /events     │ HTTP POST /submit
               │                       │
     ┌─────────┴──────────┐   ┌────────┴─────────┐
     │  Miner (self-host) │   │ Miner (self-host)│   ... N miners
     │  any stack         │   │ any stack        │
     │  (LLM/regex/ML)    │   │ (LLM/regex/ML)   │
     └─────────┬──────────┘   └────────┬─────────┘
               │                        │
               └────────────┬───────────┘
                            │
                ┌───────────▼────────────┐
                │ SEC EDGAR / Patents /  │
                │ Court records / etc.   │
                └────────────────────────┘
```

## How it works

1. **Validator** watches SEC EDGAR in real time. On every new filing, it broadcasts an `EventTask` over a WebSocket bus.
2. **Miners** (each running their own process) receive the task, extract entities/relationships/facts — using regex, LLMs, or anything else — and POST a `KnowledgeDelta` back to the validator's `/submit` endpoint.
3. **Validator** independently fetches every source URL on its whitelist, hashes the bytes, and verifies the hash matches what the miner claimed. Unverified or non-whitelisted sources still land in the graph but are tagged accordingly (see below).
4. **Validator** scores the submission on 8 dimensions (accuracy, provenance, reproducibility, novelty, freshness, latency, depth, coverage). For causal claims, it re-executes the miner's code in a sandbox.
5. **High-scoring deltas** are pushed to the graph data server and merged into the canonical graph along with the verified source bytes (cryptographic proof).
6. **Miners earn alpha tokens** proportional to their mean score. Weights are set on the Bittensor chain every hour.

## Verified vs. unverified submissions

Not every useful claim has a publicly fetchable source. To support news scoops, paywalled research, and analyst tips while still preventing abuse, every submission falls into one of three buckets:

| Path | Source | Miner sends | Validator does | `verified` flag |
|---|---|---|---|---|
| **A. Whitelisted** | sec.gov, etc. | hash + URL + quote | independently fetches URL, hashes, compares | ✅ `True` |
| **B. Non-whitelisted public** | Bloomberg, blogs, niche sites | hash + URL + quote | records the URL claim, no fetch | ❌ `False`, tag = `url_only` |
| **C. Paywalled** | WSJ behind login, internal reports | hash + URL + quote + **bytes inline** | re-hashes the bytes to confirm match, stores them as evidence | ❌ `False`, tag = `paywalled_content` |

**Default whitelist:** `sec.gov`, `www.sec.gov`, `data.sec.gov`. Override with `WHITELIST_DOMAINS=sec.gov,reuters.com,...` in the validator's `.env`.

Items merged with `verified=False` enter the canonical graph but are visually distinguished (dashed edges in the graph viewer, "Verified only" filter to hide them entirely). They still earn miners the **same scoring rewards** — provenance, novelty, depth, etc. apply normally.

### Slashing rule (applies to Facts only)

When a *verified* `Fact` enters the graph and contradicts an *unverified* `Fact` already there (same `entity_id` + `fact_type`, value differs by >5%), the validator:

1. Marks the unverified fact as `superseded` (with a pointer to the verified fact that beat it)
2. Halves the contributing miner's accumulated rewards (`slash_multiplier *= 0.5`, compounds across offenses)

In short: **non-public claims are welcome, but you eat the cost when a public source proves you wrong.** Don't ship low-confidence speculation as fact.

Slashing applies only to Facts (numeric, easy to compare). Relationships and Entities are not slashed — those are too contextual to disprove cleanly.

### How to submit unverified claims (miner side)

You don't need to do anything special for path B (non-whitelisted public URL) — just submit normally with a non-whitelisted `source_url` and the validator will tag it.

For path C (paywalled), the miner ships the source bytes inline alongside the delta for any non-whitelisted URL. See [docs/miner_readme.md](docs/miner_readme.md) for the wire format.

### Score restoration on validator restart

Validators read their previously-set weights from the chain on startup and seed `MinerPerformance.scores` from there — so your standing isn't reset every time the validator process restarts. 

## Quick start

### Run a validator

```bash
cp .env.example .env
# Edit: BT_WALLET_COLD, BT_WALLET_HOT, SN43_EDGAR_CONTACT,

pip install -e ".[dev]"
python -m neurons validator
# Event bus listens on :8765 by default — miners connect here
```

Or with Docker:

```bash
docker compose up -d
docker compose logs -f sn43-validator
```

### Run a miner

You write your own — that's the whole point. See [docs/miner_readme.md](docs/miner_readme.md) for the protocol details (wire format, scoring, signing).

Two reference miners are included:

```bash
# Regex-based (no API key, no LLM, low scores):
python -m envs.knowledge_graph.agent

# LLM-based (uses your Anthropic API key — better scores, real costs):
export ANTHROPIC_API_KEY=sk-ant-...
python -m envs.knowledge_graph.llm_agent
```

The `llm_agent.py` reference is a good starting point for production: it handles WebSocket reconnects, content-addressed doc storage, hotkey-signed submissions, and structured-output extraction in ~350 lines. Fork it, tune the prompt for your sector, add `CausalEvidence` pipelines for the 20% reproducibility score, and ship.

## Project structure

```
sn43/
├── sn43/
│   ├── __init__.py           # Thin package init (re-exports protocol)
│   ├── protocol.py           # Entity, Relationship, Fact, KnowledgeDelta,
│   │                          # ProvenanceChain, CausalEvidence, EventTask,
│   │                          # VerificationSource, …
│   └── auth.py               # Hotkey-based signing + verification of
│                              # miner submissions (NonceCache, sign_payload,
│                              # verify_request)
├── neurons/
│   └── __init__.py           # ValidatorService (event bus + watcher +
│                              # scoring + slashing + Bittensor weight setting)
├── envs/
│   ├── utils.py              # Sectors + task generation helpers
│   └── knowledge_graph/
│       ├── agent.py          # Reference regex miner (lightweight)
│       ├── llm_agent.py      # Reference Claude-API miner (production starting point)
│       └── scoring.py        # 8-dimension scoring + total
├── ingestion/
│   ├── edgar_watcher.py      # SEC EDGAR atom-feed poller
│   ├── event_bus.py          # EventBus + FastAPI /events WS + /submit HTTP
│   │                          # + MinerClient (single-validator reference)
│   ├── multi_validator_client.py   # MultiValidatorClient: discover + fan-out
│   │                          # to all validators on the metagraph
│   └── doc_verifier.py       # Validator-side URL fetcher: pulls each
│                              # provenance URL, hashes, verifies match
├── sandbox/
│   └── runner.py             # Subprocess sandbox with CPU/memory rlimits
├── storage/
│   ├── graph_store.py        # NetworkX graph + JSON persistence + slash/
│   │                          # contradiction detection
│   ├── doc_store.py          # Content-addressed source documents (sha256)
│   ├── artifact_store.py     # Content-addressed code + data blobs (sha256)
│   └── miner_cache.py        # SQLite cache for miners' local data
└── docs/
    ├── miner_readme.md       # How to run a miner
    └── validator_readme.md   # How to run the validator
```

## Scoring

| Dimension | Weight | Measures |
|---|---|---|
| accuracy | 15% | Structural validity of IDs, URLs, dates |
| provenance | 15% | Source anchors resolve + quoted text matches in the stored document |
| **reproducibility** | **20%** | Miner causal code re-runs in sandbox and matches the claimed effect (within 5%) |
| novelty | 15% | Items not already in the graph |
| freshness | 5% | Recency of source documents |
| **latency** | **15%** | Exponential decay from event timestamp (5-min half-life) |
| depth | 10% | Richness: relationships, facts, causal evidence, temporal windows |
| coverage | 5% | Bonus for underserved sectors |

See [docs/miner_readme.md](docs/miner_readme.md) for how to score higher on each.

## Further reading

- [docs/miner_readme.md](docs/miner_readme.md) — step-by-step miner guide
- [docs/validator_readme.md](docs/validator_readme.md) — validator operations

## License

This codebase is published as a reference implementation for SN43 operators.

# Cairn — the Git of autonomous intelligence

[![CI](https://github.com/Atomic-God/Cairn/actions/workflows/ci.yml/badge.svg)](https://github.com/Atomic-God/Cairn/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/cairn-ai?color=blue)](https://pypi.org/project/cairn-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Zero Dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)](pyproject.toml)

> A *cairn* is a simple, unyielding stack of stones left by travellers to mark
> an indisputable path across untracked wilderness. You do not argue with a
> cairn; you trust it, follow it, and build upon it.

**Cairn is the universal, content-addressed state graph and deterministic
replay protocol for AI agent executions.** Cryptographically verifiable
execution history and deterministic replay against recorded inputs — turning
non-deterministic agent runs into auditable, forkable infrastructure you can
actually reason about.

Today's agents pass mutable JSON dicts across HTTP endpoints. When an agent
fails at step 42 of a 60-step workflow, you cannot reproduce it, you cannot
audit it, you cannot fork it without re-paying for the earlier steps, and you
cannot prove what context the model had when it made an irreversible tool
call. Cairn replaces that sand with a Merkle DAG:

- **Content-addressed cognitive states** — every prompt, inference, tool call
  and observation is hashed with BLAKE3 into an immutable node:
  `NodeId = BLAKE3(ParentHash ‖ ActorId ‖ ContextRoot ‖ OpPayload ‖ Seed)`.
- **Time-travel debugging** — `cairn checkout` to any node, change the prompt,
  branch forward. Upstream steps are never recomputed.
- **Deterministic replay** — re-execute any trace offline with zero API calls;
  the harness verifies the re-execution was bit-identical against the
  recorded inputs, or fails loudly on any divergence.
- **Cryptographic audit trails** — every node is Ed25519-signed; `cairn verify`
  re-derives every id and signature. Tampering is mathematically detectable.
- **Zero-copy structural sharing** — identical context is stored once, however
  many runs reference it.

Zero dependencies. No Docker, no database, no cloud account — just a `.cairn/`
directory next to your code, like `.git/`.

---

## Try it in 60 seconds (no API keys, no network)

```bash
git clone https://github.com/Atomic-God/Cairn.git && cd Cairn
python examples/mock_agent.py
```

A five-step SRE agent diagnoses a mock outage, fails on step 4, **rolls back
to step 3, branches, and fixes it with an alternative rule** — then proves
the fix replays deterministically with bit-identical node ids. All offline,
in about a second.

## Quickstart

```bash
pip install cairn-ai
cd my-agent-project
cairn init
```

Instrument your agent (one line to wrap your loop):

```python
from cairn import Session

with Session.create(".", actor="planner") as s:
    client = s.client(llm=my_llm_call, tools={"search": search, "calc": calc})

    client.system("You are a helpful planner.")
    client.user("Plan a trip to Kyoto.")
    plan = client.infer({"messages": messages}, model="acme-1", seed=1234)
    hits = client.tool("search", {"q": "Kyoto temples"})
```

Every step is now an immutable, signed node in `.cairn/`. Inspect it:

```bash
cairn log --oneline
# a3f9c1… [     system] seq=0 planner
# 7bd201… [       user] seq=1 planner
# 9e44aa… [  inference] seq=2 planner
# c81d33… [  tool_call] seq=3 planner
# f00d12… [tool_result] seq=4 planner

cairn show 9e44aa --context   # the exact context the model saw
cairn verify                  # cryptographic audit: OK
```

The agent took a wrong turn at step 3? Don't restart — time-travel:

```bash
cairn checkout 7bd201 -b fix-prompt
# ... re-run your agent script against the "fix-prompt" branch;
# steps 0-1 are reused by reference, never recomputed.
```

Reproduce a failure offline, with zero token spend:

```bash
cairn replay <head> --agent myagent:run --branch repro
# Replayed 5 nodes onto branch 'repro'
# Node ids identical to recording: YES — deterministic replay proven
```

## 1-Line Framework Integrations

Cairn integrates directly with modern LLM SDKs and agent frameworks without adding mandatory dependencies.

### OpenAI (Sync, Async & Streaming)

```python
from openai import OpenAI
from cairn import Session
from cairn.integrations.openai import wrap_openai, wrap_tool

session = Session.create("./my-agent", actor="researcher")
client = wrap_openai(OpenAI(), session=session)

# Prompts, model parameters, seed, tool calls, and streaming responses are
# automatically captured as immutable nodes in the Merkle DAG:
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Analyze quarterly filings"}],
    seed=42,
)

@wrap_tool(session=session)
def search_sec_filings(ticker: str):
    return {"filing_date": "2026-10-01", "revenue": "$12.4B"}
```

### LangChain & LangGraph

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from cairn import Session
from cairn.integrations.langchain import CairnCallbackHandler

session = Session.create("./my-agent", actor="auditor")
handler = CairnCallbackHandler(session)

# Attach handler to any LangChain ChatModel, Chain, or AgentExecutor:
llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])
executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])
result = executor.invoke({"input": "Verify ledger entries"})
```

### Anthropic Claude

```python
from anthropic import Anthropic
from cairn import Session
from cairn.integrations.anthropic import wrap_anthropic

session = Session.create("./my-agent", actor="analyst")
client = wrap_anthropic(Anthropic(), session=session)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Extract KPIs"}],
)
```

## How it works

```
┌────────────────────────────────────────────────────────┐
│                     AI AGENT RUNTIME                   │
│           (Prompts, Tool Invocations, Inferences)      │
└───────────────────────────┬────────────────────────────┘
                            │
                    AsyncAgentClient
                            │
          ┌─────────────────┴─────────────────┐
          ▼                                   ▼
┌───────────────────┐               ┌───────────────────┐
│  Execution Graph  │               │   Payload Store   │
│  (BLAKE3 DAG)     │               │ (zlib Compressed) │
└─────────┬─────────┘               └─────────┬─────────┘
          │                                   │
          └─────────────────┬─────────────────┘
                            │
                   .cairn/ Repository
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
  [ Offline Replay ]  [ Lineage Verify ]  [ Time-Travel ]
  (Socket Netguard)   (Merkle Traversal)  (Zero-Copy Fork)
```

### The state graph

An agent execution is decomposed into **nodes**, each an append-only,
content-addressed commit:

| Field | Role |
|---|---|
| `parents` | Hashes of parent nodes (forms the DAG) |
| `actor` | Which agent/identity committed the step |
| `kind` | `system` `user` `inference` `tool_call` `tool_result` `observation` `note` |
| `payload` | The operation itself (canonical JSON) |
| `context_root` | BLAKE3 of the full context snapshot at this step |
| `seed` | The entropy seed for this step (recorded, not hidden) |
| `seq` / `wall_time` | Logical clock + wall clock (signed, not hashed) |
| `signature` | Ed25519 over `(node_id, seq, wall_time)` |

The node id commits to *everything the step depended on*. Change one byte of
an ancestor and every downstream id changes — exactly like Git, but for
cognition instead of code.

Objects live in `.cairn/objects/` addressed by their hash, so ten thousand
parallel runs sharing 95% of their context store that context **once**.

### Deterministic replay, honestly

`cairn.replay.run()` re-executes your agent function with a `ReplayClient`
that intercepts `infer()` and `tool()`:

1. The agent's **request is fingerprinted** and compared to the recorded
   request. Any divergence → `ReplayDivergenceError`. The replay proves
   identical behavior, or it fails — there is no silent drift.
2. The recorded **response is served** — no network, no tokens, no flakes.
3. Each replayed step is committed to a fresh branch. If actor, requests and
   seeds match, **every node id is bit-identical** to the original: anyone
   holding the bundle can independently verify that the re-execution was
   consistent with the recorded trace.

Unless you opt out (`allow_network=True` / `CAIRN_REPLAY_NETGUARD=0`), replay
runs inside a **network guard** that blocks all socket I/O — so a successful
replay is proof your agent made no live calls. Pair it with `CAIRN_REPLAY=1`
in your agent code to select recorded backends, and with adapter stubs for
`requests`/`httpx` to serve recorded HTTP payloads deterministically.

Cross-model re-basing falls out naturally: pass a different `model=` during
replay and the harness records the swap — the trace forks legitimately with
new ids from that step on.

### Async agents and streaming

```python
from cairn.aio import AsyncAgentClient

async with Session.create(".", actor="streamer") as s:
    client = AsyncAgentClient(s, llm=my_async_llm, tools={"calc": my_async_tool})
    reply = await client.infer({"q": 1}, model="acme-1", seed=7)
    # Stream an SSE response; chunks pass through untouched, the assembled
    # message is committed exactly once at the end (never on error).
    async for chunk in client.astream(sse_chunks(), model="acme-1",
                                      request={...}, seed=7):
        print(chunk, end="", flush=True)
```

The sync `AgentClient` has a matching `client.stream()` for synchronous
iterators. Stream recorders accept OpenAI-style
`choices[0].delta.content` chunks out of the box, plus plain strings/bytes.

### Durability

Refs and objects are written **atomically** (process-unique temp file,
`fsync`, `os.replace`), and every ref mutation is guarded by a
**cross-platform file lock** (`.cairn/LOCK`: `flock` on POSIX, `msvcrt` on
Windows) — parallel processes and threads can commit against one repository
without interleaving. Ref names are normalised to POSIX form internally, so
traces are portable across operating systems.

### Audit

```bash
cairn verify            # all branches
cairn verify main exp   # specific refs
```

Verification re-derives every node id from its preimage, checks every Ed25519
signature, confirms every parent link resolves and every context blob is
intact. It needs no trusted third party — the math is the auditor.

### Portability

```bash
cairn export <ref> -o trace.car   # everything reachable from <ref>
cairn import trace.car ./elsewhere
```

A `.car` bundle is a gzipped tar of content-addressed objects plus branch
heads. Attach it to a bug report, publish it for independent verification, or
fork someone's trace without re-running a single step.

## What Cairn Does Not Guarantee

* **External Truth:** Cairn proves that an external API or tool returned an exact payload at an exact timestamp. It does not verify whether the third-party system provided factually correct data.
* **Stochastic Elimination:** Cairn does not force inherently random LLM weights to behave deterministically during unseeded live runs. Replay is deterministic because it verifies against recorded model and tool payloads.
* **Corporate Authorization:** Ed25519 signatures verify node authorship and cryptographic provenance; they do not automatically enforce business-level organizational policy.
* **Actor-name authentication:** the `actor` field is a self-asserted label — the Ed25519 *key* is the identity. Cairn records every (key, actor-name) pair in `.cairn/identities.json` and `cairn verify` flags keys seen under multiple names, but it is not a PKI.

> ⚠️ **Data Privacy Notice:** Cairn records raw operational payloads (system prompts, tool inputs, and tool outputs) into `.cairn/objects/`. If your agents handle confidential data, database credentials, or PII, be aware that these will be stored in your local `.cairn` directory. Always audit stored payloads before exporting or sharing state graphs. Configurable payload redaction filters are slated for a future release.

## Python API

```python
from cairn import Session, Repo, replay_run, verify, export_bundle, import_bundle

# Low-level recording (you make the real calls yourself)
with Session.create(".", actor="worker-3") as s:
    s.system("...")
    s.user("...")
    s.inference(request, response, model="acme-1", seed=42)
    s.tool_call("calc", {"expr": "2+2"})
    s.tool_result("calc", {"value": 4})
    s.observation("sensor-1", {"temp": 21.5})
    s.note("operator annotation (not part of model context)")

# Branching / time travel
s.checkout(some_node_id, new_branch="experiment")

# Deterministic replay
def agent(client):
    client.system("...")
    return client.infer({"q": 1}, model="acme-1", seed=42)

report = replay_run(repo, head, agent)
assert report.ids_match  # proven deterministic

# Audit
report = verify(repo)
assert report.ok
```

### The client contract

Write your agent against the small client interface once — it works for both
recording and replay:

| Method | Recording | Replay |
|---|---|---|
| `system/user/note(text)` | commits the step | verifies text matches, re-commits |
| `infer(request, model, seed)` | calls your `llm`, records I/O | verifies request fingerprint, returns recorded response |
| `tool(name, arguments)` | calls your tool, records I/O | verifies name + arguments, returns recorded result |
| `observation(source, data)` | records the observation | returns recorded data (or verifies provided data) |

## CLI reference

| Command | Purpose |
|---|---|
| `cairn init [path]` | Create a `.cairn` repository |
| `cairn commit -m "msg" [--kind note\|observation]` | Record a note/observation node |
| `cairn status` | HEAD, branch, object counts |
| `cairn log [--oneline] [-n N] [ref]` | Node history, newest first |
| `cairn show <ref> [--context]` | Full node, optionally with context snapshot |
| `cairn diff <a> <b>` | Diff two nodes' preimage fields |
| `cairn checkout <ref> [-b branch]` | Time-travel; `-b` branches from a node |
| `cairn branch [-l] [-d name] [name [at]]` | Branch management |
| `cairn verify [refs...]` | Cryptographic audit (incl. orphans, identity notes) |
| `cairn reflog [branch]` | Append-only ref history for a branch |
| `cairn replay <ref> --check` | Dry-run: is this trace replayable? |
| `cairn replay <ref> --agent mod:fn [--branch N] [--start ID]` | Deterministic re-execution |
| `cairn export <ref> -o file.car` | Portable bundle |
| `cairn import file.car [dest] [--force]` | Import a bundle |
| `cairn key [--armor] [--rebind ACTOR]` | This machine's Ed25519 identity |

## Design notes

- **Zero hard dependencies.** BLAKE3 and Ed25519 are implemented in pure
  Python inside this repo (validated against the official BLAKE3 vectors and
  the RFC 8032 vectors). Two optional accelerators exist, both
  byte-identical to the pure implementations: the `blake3` PyPI package
  (`pip install cairn-ai[fast]`) for hashing, and the PyO3 native engine
  in `rust/` (BLAKE3 + Ed25519) for the full hot path. The pure backend is
  always the correctness reference.
- **Native engine.** `rust/` builds `_cairn_native` via PyO3
  (`cargo build --release --features extension-module`, or
  `maturin develop --manifest-path rust/Cargo.toml`); drop the resulting
  `.so`/`.pyd` next to `cairn/_native.py` and it is picked up automatically.
  `cargo test` and `cargo clippy` cover the Rust side. CI builds it and runs
  the whole Python suite against it.
- **Identity.** Your Ed25519 key lives at `~/.config/cairn/identity.key`
  (0600). Override with `CAIRN_KEY_FILE`. Nodes are signed by whoever commits
  them; verification needs only the public key embedded in the node.
- **Determinism boundaries.** Replay is deterministic *given* the recorded
  inputs, tool responses, and seeds. Wall-clock time and signatures are
  explicitly excluded from node ids (they're signed metadata instead), so a
  replay at a different time by a different key still proves identical
  *execution*.
- **Performance.** Node commit overhead is dominated by one Ed25519 sign
  and BLAKE3 hashing of the *delta* — contexts are stored as a chain of
  per-commit deltas, so commit cost stays constant no matter how long the
  chain grows. The optional native engine (`rust/`, a PyO3
  extension) brings this to ~0.6ms/commit end-to-end — the crypto itself
  runs in ~55µs (sign) and ~0.5µs (hash), well under 200µs; the remainder
  is durable I/O (`fsync`, atomic ref writes). The pure-Python fallback
  (~48ms/commit) remains the correctness reference and keeps
  `pip install cairn-ai` dependency-free. Blobs larger than
  `CAIRN_MAX_BLOB_BYTES` (default 8 MiB) are rejected with a clear error
  instead of stalling the pure-Python hasher.
- **Concurrency.** Commits to the same branch are linearizable: concurrent
  writers rebase onto the fresh branch tip under the repo lock, so every
  committed node stays reachable — no silent orphans. Every ref movement is
  recorded in an append-only reflog (`.cairn/logs/refs/heads/<branch>`,
  `cairn reflog`), so history rewrites are always detectable.
- **Specification.** [SPEC.md](SPEC.md) defines the object envelope
  (`CAIR` magic `43 41 49 52`, protocol version `01`), canonical JSON,
  node identity, signing, refs, and replay — everything an independent
  implementation needs to read and verify a Cairn repository.
- **Install.** `pip install cairn-ai`, or run `install.sh` for a
  user-local install (`~/.cairn/venv` + `~/.local/bin/cairn`). CI runs the
  full suite on Linux, macOS, and Windows across Python 3.10–3.13
  (`.github/workflows/ci.yml`).

## Repository layout

```
cairn/            # the package
  integrations/   # 1-line adapters: OpenAI, LangChain, Anthropic, CrewAI
  _blake3_pure.py # pure-Python BLAKE3 (official-vector validated)
  _native.py      # optional native accelerator loader (zero-dep fallback)
  hash.py         # hash layer: native → fast → pure backends
  signing.py      # Ed25519 (RFC 8032 validated); native-accelerated
  keys.py         # identity management
  codec.py        # canonical JSON / compression primitives
  node.py         # node model, NodeId, signatures
  store.py        # .cairn content-addressed object store + refs
  session.py      # Session + AgentClient (instrumentation)
  aio.py          # AsyncAgentClient + sync/async streaming interceptors
  replay.py       # deterministic replay harness + offline network guard
  verify.py       # cryptographic audit
  export.py       # .car bundles
  cli.py          # command-line interface
rust/             # optional PyO3 native engine (BLAKE3 + Ed25519)
  src/lib.rs      # extension module `_cairn_native`
  src/tests.rs    # Rust unit tests (cargo test)
SPEC.md           # storage & signing specification
install.sh        # user-local installer (zero dependencies)
.github/workflows/ci.yml  # Linux/macOS/Windows × Python 3.10–3.13
tests/            # full suite incl. official crypto vectors
examples/         # worked examples (incl. zero-key mock_agent.py demo)
```

## Roadmap

- **Phase 2** — team collaboration (signed fetch/push), hosted trace
  clusters, policy firewalls over the graph, SOC2/HIPAA attestation reports.
- **Phase 3** — cross-organization signed state proofs for machine-to-machine
  settlement.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Bug reports with a `cairn export`
bundle attached are pure joy.

## License

MIT — see [LICENSE](LICENSE).

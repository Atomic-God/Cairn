# Changelog

All notable changes to this project will be documented in this file.
The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [0.2.0] - 2026-10-17

Hardening release: every gap found by the thousand-agent stress audit is
closed, and the implementation now matches its stated claims.

### Fixed

* **Same-branch write races no longer orphan commits.** Commits are
  linearizable: under the repo lock each commit re-reads the branch tip and
  rebases its new messages onto it, so every committed node stays reachable.
  Previously, N concurrent writers to one branch left all but one writer's
  chain orphaned and invisible to `log`/`verify`/`replay`.
* **CLI ref arguments accept branch names.** `cairn log/show/replay/export
  main` used to fail with `invalid object id: 'main'`; a new
  `Repo.resolve_ref()` resolves branch → head, hex id, or unique prefix.
* **SPEC.md `ensure_ascii`** corrected to `True` (the code's actual
  behaviour); an independent implementation following the old text would have
  hashed non-ASCII payloads differently.

### Added

* **Delta-chain context snapshots.** Each commit stores only the messages
  appended since the previous root instead of re-snapshotting the full
  history — commit cost is O(new messages), constant for long-running
  agents. Repositories written by ≤ 0.1.x (bare-list snapshots) still read
  correctly. This makes the documented "commit overhead is dominated by one
  Ed25519 sign" claim actually true.
* **Append-only reflog** (`.cairn/logs/refs/heads/<branch>`) recording every
  ref movement; history rewrites via `set_head` are now detectable
  (`cairn reflog <branch>`).
* **Identity registry** (`.cairn/identities.json`) recording every
  (signing-key, actor-name) pair; `cairn verify` surfaces keys seen under
  multiple actor names, and `cairn key [--rebind]` manages bindings. Actor
  names remain self-asserted labels — the Ed25519 key is the identity.
* **Orphan detection**: `Repo.find_orphans()` and `cairn verify` now report
  nodes reachable from no ref instead of silently ignoring them.
* **`ReplayClient.stream()` / `astream()`**: recorded streaming inferences
  replay as chunks, so streaming agents replay without code changes.
* **`Session(..., create_branch=True)`**: open-or-create branch semantics
  for multi-agent orchestration.
* **Blob size guard** (`CAIRN_MAX_BLOB_BYTES`, default 8 MiB) with a clear
  error instead of a silent multi-minute pure-Python hash stall.
* Non-finite floats (NaN/Infinity) are rejected by canonical JSON encoding,
  enforcing the SPEC.
* **Key-file loading no longer strips bytes.** `load_or_create` used
  `.strip()` on the raw 32-byte secret, silently corrupting ~4.6% of
  generated keys (any key starting/ending in an ASCII whitespace byte)
  on their next load.

## [0.1.0] - Initial Public Release

### Added

* Content-addressed execution nodes using BLAKE3 hashing.
* Ed25519 cryptographic signatures for state provenance.
* Dual-engine architecture: native PyO3 core with pure-Python fallback.
* Storage engine with atomic rename operations, fsync durability, and cross-platform lockfiles (`flock`/`msvcrt`).
* Branch-pinning support for concurrent multi-agent executions.
* Async client (`AsyncAgentClient`) with zero-overhead streaming chunk accumulation.
* Offline deterministic replay harness with socket-level network guard.
* CLI tools: `init`, `commit`, `log`, `checkout`, `branch`, `verify`, and `replay`.
* Zero-key incident response demonstration script.

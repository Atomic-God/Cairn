# Contributing to Cairn

Thanks for helping build the Git of autonomous intelligence. A few notes:

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest tests/ -q
```

Cairn has **zero hard dependencies** — keep it that way. `pip install -e
".[fast]"` adds the optional Rust BLAKE3 backend for speed; the pure-Python
backend in `cairn/_blake3_pure.py` is the correctness reference and must keep
passing `tests/test_hash.py` against the official vectors.

## What "production level" means here

- **Determinism first.** Anything that goes into a node id must be canonical.
  If you touch `node.compute_node_id`, `codec.canonical`, or the store
  layout, the full suite must stay green — and think hard about migration.
- **Crypto is load-bearing.** `signing.py` and `_blake3_pure.py` are validated
  against official vectors (`tests/vectors/`). Don't "optimize" them without
  re-running those tests.
- **No silent behavior change.** Replay divergence must always raise. If you
  add a node kind, update `replay.py`, `cli.py show/diff`, and the docs.
- **Tests are the spec.** Every new feature ships with tests that fail
  without the feature. The suite currently covers hash vectors (107),
  Ed25519 vectors (20), node/store/session/replay/verify/export, and CLI
  end-to-end.

## Pull requests

- Keep the public API (`cairn/__init__.py`) stable within Phase 1.
- Update `README.md` when you change user-visible behavior.
- Bug reports are most useful with a `cairn export` bundle attached:
  `cairn export <ref> -o bug.car`.

## Security

Found a cryptographic or key-management issue? Please report it privately
via GitHub Private Vulnerability Reporting —
https://github.com/Atomic-God/Cairn/security/advisories/new — so we can
fix it before disclosure. See [SECURITY.md](SECURITY.md) for the full
policy and threat model.

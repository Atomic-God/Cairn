# Cairn Storage & Signing Specification

Version 1.0 — 3 October 2026.

This document defines the on-disk format, the canonical serialisation, the
identity scheme, and the signing scheme of Cairn (`cairn-core`), so that any
independent implementation can read and verify a Cairn repository byte-for-byte.

## 1. Hash function

All content addressing uses **BLAKE3** (256-bit output, hex-encoded as
64 lowercase characters). The reference implementation ships a pure-Python
BLAKE3 validated against the official test vectors, and uses the native
`blake3` package when installed. Three domain separators are defined:

| Constant      | Value (bytes)          | Used for            |
|---------------|------------------------|---------------------|
| `DOMAIN_NODE` | `"cairn-node-v1\x00"`  | node identity       |
| `DOMAIN_BLOB` | `"cairn-blob-v1\x00"`  | context blob identity |
| `DOMAIN_SIG`  | `"cairn-sig-v1\x00"`   | signature base      |

## 2. Canonical JSON

All hashed and signed JSON structures use canonical JSON:

- UTF-8 encoding, no byte-order mark.
- `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)`.
- `None` → `null`; no trailing whitespace; no duplicate keys.
- Floats MUST NOT appear in hashed structures. All numeric schema fields
  are integers; wall-clock times are ISO-8601 strings with explicit offsets.

Two implementations following these rules produce byte-identical preimages
for the same logical content, so node ids and signatures interoperate.

## 3. Node identity

A node is one recorded agent step. Its id is the BLAKE3 hex digest of a
domain-separated, length-prefixed preimage:

```
preimage = DOMAIN_NODE
         ‖ u16be(len(parents)) ‖ parents[0] ‖ … ‖ parents[n-1]   (raw 32-byte ids)
         ‖ u16be(len(actor_utf8)) ‖ actor_utf8
         ‖ u16be(len(kind_utf8)) ‖ kind_utf8
         ‖ context_root                                        (raw 32 bytes)
         ‖ u32be(len(payload_canon)) ‖ payload_canon           (canonical JSON)
         ‖ u64be(seed)
```

`u16be`/`u32be`/`u64be` are unsigned big-endian integers of the given width.
`seed` must fit in uint64. Parents are ordered; `actor` and `kind` are UTF-8.

The node id is therefore bound to: its parents, its author, its kind, its
full context root, its payload, and its seed. Rewiring a parent changes the
id — the DAG is tamper-evident by construction.

## 4. Signing

Each node carries an Ed25519 signature. The signed bytes are:

```
signing_base = DOMAIN_SIG ‖ node_id ‖ u64be(seq) ‖ wall_time_utf8
```

where `node_id` is the raw 32-byte id, `seq` is the per-branch sequence
number, and `wall_time` is the ISO-8601 timestamp string. The signature
(hex, 128 chars) is made with the actor's Ed25519 secret key; `pubkey`
(hex, 64 chars) is the corresponding public key stored on the node.

Verification of a node is three independent checks:

1. Recompute the id from the preimage (§3) — content addressing.
2. Verify the Ed25519 signature over `signing_base` — authorship.
3. Envelope magic and version (§5) — container integrity.

The signature binds the id *plus* the branch sequence number and wall time,
so a node cannot be replayed onto a different branch position undetected.

## 5. Object envelope

Every stored object is a file under `.cairn/objects/<first-2-hex>/<rest>/`
whose bytes are the **zlib-compressed** canonical JSON of an envelope:

```json
{
  "magic": "CAIR",
  "v": 1,
  "t": "node" | "blob",
  "node": { … } | "data": "<base64>"
}
```

- `magic` — the ASCII bytes `43 41 49 52` (`"CAIR"`), recorded as a JSON
  field so the envelope stays human-inspectable once decompressed.
- `v` — the protocol version, currently `1` (written `01` in fixed-width
  contexts). A reader MUST reject objects whose magic is absent/different
  or whose version it does not understand.
- `t: "node"` — the `node` field holds the node document (§6).
- `t: "blob"` — the `data` field holds base64 context bytes; its address is
  `BLAKE3(DOMAIN_BLOB ‖ data)`.

Objects are addressed by their BLAKE3 identity, so identical content is
stored exactly once however many sessions reference it — zero-copy
structural sharing.

## 6. Node document

```json
{
  "id": "<64 hex chars>",
  "parents": ["<node id>", …],
  "actor": "string",
  "kind": "system|user|inference|tool_call|tool_result|observation|note",
  "payload": { … },
  "context_root": "<64 hex chars>",
  "seed": 123 | null,
  "seq": 0,
  "wall_time": "2026-10-03T12:00:00+00:00",
  "pubkey": "<64 hex chars>",
  "signature": "<128 hex chars>"
}
```

Inner payloads per kind:

- `system` — `{"content": str}`: the system prompt.
- `user` — `{"content": str}`: user input.
- `inference` — `{"model": str, "request": {...}, "response": {...},
  "streamed"?: true}`: one model call; `response.streamed` is true when the
  node was assembled from a stream (§8).
- `tool_call` — `{"tool": str, "arguments": {...}, "call_id": str}`.
- `tool_result` — `{"tool": str, "result": {...}, "call_id": str, "ok": bool}`.
- `observation` — tool / environment observation payload.
- `note` — `{"content": str}`: free-form annotation (ignored by replay).

## 7. Context blobs

A context blob captures the conversation / execution state at commit time
(messages, tool specs, model, usage). Contexts are stored as a **chain of
deltas**, newest first: each commit stores only the messages appended since
the previous root, so commit cost is O(len(new)) rather than O(len(history)).

A delta blob is the canonical JSON object

```json
{"messages": [<new messages>], "prev": "<parent root hex or null>"}
```

and `context_root` on a node is `BLAKE3(DOMAIN_BLOB ‖ canonical_json(delta))`,
exactly the blob id. `repo.get_context(root)` reconstructs the full message
list by walking `prev` links. Blobs written by Cairn ≤ 0.1.x hold a bare
message list (a full snapshot); readers MUST treat a top-level JSON array as
a chain of length one, so old repositories keep reading correctly.

Checkout restores the session context from this chain.

## 8. Refs

Refs live under `.cairn/refs/` with `HEAD` at `.cairn/HEAD`:

- `.cairn/HEAD` — UTF-8 text, either `ref: refs/heads/<name>` or a bare
  node id (detached).
- `.cairn/refs/heads/<name>` — UTF-8 hex node id + newline.
- `.cairn/config.json` — `{"version": 1, "hash": "blake3", …}`.

Ref names are logical POSIX paths: `/` separators, no `\`, no `..`, no
redundant separators or `.` segments. Implementations on Windows MUST map
these logical names to host-native paths and back.

All ref and object writes are atomic: write to a process-unique sibling
temp file (`.tmp.<pid>.<random>`), flush, `fsync`, then `os.replace` onto
the final name, so readers see the old or the new version, never a torn
file. A `.cairn/LOCK` file guards ref mutation across processes —
`flock(LOCK_EX)` on POSIX, byte-range locking via `msvcrt` on Windows —
held for the duration of each ref update (commit, checkout, branch
create/delete). Stale temp files are cleaned on open.

Every ref update also appends one line to the branch's **reflog** at
`.cairn/logs/refs/heads/<name>`:

```
<old-id|64 zeros> <new-id> <unix-timestamp> <actor>
```

The reflog is append-only and is never rewritten — not even by history
rewrites — so a `set_head` rewind is always detectable afterwards
(`cairn reflog <branch>`).

`.cairn/identities.json` records every `(signing-key, actor-name)` pair ever
seen: `{"pubkeys": {<hex>: {"actor": ..., "actors_seen": [...],
"first_seen": ...}}}`. Actor names are self-asserted labels; the Ed25519 key
is the identity. Verifiers SHOULD surface keys seen under multiple actor
names.

## 9. Streaming

A streamed inference is committed exactly once, when the stream is
exhausted (or a done-sentinel such as `[DONE]` is observed). If the stream
raises, nothing is committed. Chunks pass through to the consumer
unmodified; the assembled text is stored in `response.content` with
`response.streamed == true`. Async streams (`AsyncStreamRecorder`) follow
the same contract over async iterators.

## 10. Replay

A replay re-executes an agent function against a recorded trace instead of
live backends. The harness serves the recorded request/response pairs in
order and asserts:

- kind sequence matches the trace,
- request payloads are identical (canonical JSON equality),
- model and seed match where recorded.

If all match, the re-executed node ids equal the recorded ids, proving the
run was deterministic. Unless explicitly opted out (`allow_network=True` or
`CAIRN_REPLAY_NETGUARD=0`), replay runs inside a network guard that blocks
all socket I/O, so a successful replay is proof the agent made no live
calls. `CAIRN_REPLAY=1` selects recorded backends when an agent supports
both live and replayed modes.

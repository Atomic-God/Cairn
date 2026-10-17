"""The Cairn node: the immutable, content-addressed unit of agent execution.

Node identity is defined exactly as:

    NodeId = BLAKE3( ParentHash || ActorId || ContextRoot || OpPayload || Seed )

with a domain separator and explicit length prefixes (see
:func:`compute_node_id`). Every node is signed with the author's Ed25519 key;
the signature binds the node id, the logical sequence number, and the wall
clock timestamp.

Node kinds (``kind`` field) used by :mod:`cairn.session`:

- ``system``      – system prompt injection
- ``user``        – user message
- ``inference``   – one model generation (request + response + model id)
- ``tool_call``   – outbound tool invocation (name + arguments)
- ``tool_result`` – tool / environment observation
- ``note``        – free-form annotation (does not affect replay)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from . import codec, hash as hashmod, signing

DOMAIN_NODE = b"cairn-node-v1\x00"
DOMAIN_SIG = b"cairn-sig-v1\x00"
DOMAIN_BLOB = b"cairn-blob-v1\x00"

KINDS = (
    "system",
    "user",
    "inference",
    "tool_call",
    "tool_result",
    "observation",
    "note",
)


def compute_node_id(
    parents: list[str],
    actor: str,
    kind: str,
    payload: Any,
    context_root: str,
    seed: int,
) -> str:
    """Compute the canonical node id (hex) from the node preimage."""
    for p in parents:
        _require_hex(p, 64, "parent id")
    _require_hex(context_root, 64, "context root")
    actor_b = actor.encode("utf-8")
    kind_b = kind.encode("utf-8")
    payload_b = codec.canonical(payload)
    if not (0 <= seed < 2**64):
        raise ValueError("seed must fit in uint64")
    preimage = b"".join(
        [
            DOMAIN_NODE,
            codec.u16be(len(parents)),
            b"".join(bytes.fromhex(p) for p in parents),
            codec.u16be(len(actor_b)),
            actor_b,
            codec.u16be(len(kind_b)),
            kind_b,
            bytes.fromhex(context_root),
            codec.u32be(len(payload_b)),
            payload_b,
            codec.u64be(seed),
        ]
    )
    return hashmod.hexdigest(preimage)


def encode_context_delta(prev_root: str | None,
                         new_messages: list) -> bytes:
    """Canonical bytes of one context delta.

    A context is stored as a chain of deltas, newest first: each delta holds
    only the messages appended since its parent, plus a pointer to the
    parent's root (``None`` for the empty context). Appending is O(len(new))
    instead of O(len(full history)), which keeps commit cost constant no
    matter how long the chain grows.
    """
    return codec.canonical({"prev": prev_root, "messages": list(new_messages)})


def compute_context_root(prev_root: str | None, new_messages: list) -> str:
    """Content address of the context chain after appending ``new_messages``.

    This is exactly the blob id under which the delta is stored, so
    ``repo.get_context(node.context_root)`` always reconstructs the full
    message list by walking the chain.
    """
    return compute_blob_id(encode_context_delta(prev_root, new_messages))


def compute_blob_id(data: bytes) -> str:
    return hashmod.hexdigest(DOMAIN_BLOB + bytes(data))


def signing_bytes(node_id: str, seq: int, wall_time: str) -> bytes:
    _require_hex(node_id, 64, "node id")
    if not (0 <= seq < 2**64):
        raise ValueError("seq must fit in uint64")
    return b"".join(
        [DOMAIN_SIG, bytes.fromhex(node_id), codec.u64be(seq),
         wall_time.encode("utf-8")]
    )


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_hex(value: str, length: int, what: str) -> None:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{what} must be {length} hex chars")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{what} is not valid hex") from exc


@dataclass(frozen=True)
class Node:
    """An immutable state transition in the agent execution DAG."""

    id: str
    parents: tuple[str, ...]
    actor: str
    kind: str
    payload: Any
    context_root: str
    seed: int
    seq: int
    wall_time: str
    pubkey: str
    signature: str

    def verify_id(self) -> None:
        """Recompute the node id from the preimage; raise if it differs."""
        recomputed = compute_node_id(
            list(self.parents), self.actor, self.kind,
            self.payload, self.context_root, self.seed,
        )
        if recomputed != self.id:
            raise ValueError(
                f"node id mismatch: stored {self.id[:16]}… != "
                f"recomputed {recomputed[:16]}…"
            )

    def verify_signature(self) -> None:
        """Verify the Ed25519 signature; raise on failure."""
        msg = signing_bytes(self.id, self.seq, self.wall_time)
        signing.verify(
            bytes.fromhex(self.signature), msg, bytes.fromhex(self.pubkey)
        )

    def short(self) -> str:
        return self.id[:12]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "parents": list(self.parents),
            "actor": self.actor,
            "kind": self.kind,
            "payload": self.payload,
            "context_root": self.context_root,
            "seed": self.seed,
            "seq": self.seq,
            "wall_time": self.wall_time,
            "pubkey": self.pubkey,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Node":
        try:
            node = cls(
                id=d["id"],
                parents=tuple(d["parents"]),
                actor=d["actor"],
                kind=d["kind"],
                payload=d["payload"],
                context_root=d["context_root"],
                seed=int(d["seed"]),
                seq=int(d["seq"]),
                wall_time=d["wall_time"],
                pubkey=d["pubkey"],
                signature=d["signature"],
            )
        except KeyError as exc:
            raise ValueError(f"node dict missing field: {exc}") from exc
        node.verify_id()
        return node


def create_node(
    *,
    parents: list[str],
    actor: str,
    kind: str,
    payload: Any,
    context_root: str,
    seed: int,
    seq: int,
    signer: signing.Signer,
    wall_time: str | None = None,
) -> Node:
    """Create, hash, and sign a new node.

    ``signer`` is a :class:`cairn.signing.Signer` holding the author's cached
    identity — reuse one per session; do not construct one per commit.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown node kind: {kind!r} (expected one of {KINDS})")
    if not actor:
        raise ValueError("actor must be non-empty")
    node_id = compute_node_id(parents, actor, kind, payload, context_root, seed)
    wt = wall_time or utcnow_iso()
    sig = signer.sign(signing_bytes(node_id, seq, wt)).hex()
    node = Node(
        id=node_id,
        parents=tuple(parents),
        actor=actor,
        kind=kind,
        payload=payload,
        context_root=context_root,
        seed=seed,
        seq=seq,
        wall_time=wt,
        pubkey=signer.public_key.hex(),
        signature=sig,
    )
    # Defensive: the freshly computed id must match (cheap BLAKE3 check).
    # Signature validity is covered by tests, not re-proven per commit.
    node.verify_id()
    return node

"""Node identity, canonicalization, and signature tests."""

import pytest

from cairn import node as nodemod
from cairn import signing
from cairn.node import (
    Node,
    compute_blob_id,
    compute_context_root,
    compute_node_id,
    create_node,
)


@pytest.fixture()
def keypair():
    return signing.generate_keypair()


@pytest.fixture()
def signer(keypair):
    from cairn.signing import Signer
    return Signer(keypair[0])


def _ctx():
    return compute_context_root(None, [{"role": "system", "content": "hi"}])


def test_node_id_is_deterministic(signer):
    kwargs = dict(
        parents=[], actor="agent-1", kind="user",
        payload={"text": "hello"}, context_root=_ctx(), seed=42, seq=0,
        signer=signer, wall_time="2026-01-01T00:00:00+00:00",
    )
    a = create_node(**kwargs)
    b = create_node(**kwargs)
    assert a.id == b.id
    assert a.signature == b.signature


def test_node_id_changes_with_each_preimage_field(signer):
    base = dict(
        parents=[], actor="agent-1", kind="user",
        payload={"text": "hello"}, context_root=_ctx(), seed=42, seq=0,
        signer=signer, wall_time="2026-01-01T00:00:00+00:00",
    )
    base_id = create_node(**base).id
    variants = [
        {**base, "actor": "agent-2"},
        {**base, "kind": "note"},
        {**base, "payload": {"text": "hellp"}},
        {**base, "context_root": compute_context_root(None, [])},
        {**base, "seed": 43},
        {**base, "parents": ["ab" * 32]},
    ]
    ids = {create_node(**v).id for v in variants}
    assert len(ids) == len(variants)
    assert base_id not in ids


def test_parent_order_matters(signer):
    p1, p2 = "aa" * 32, "bb" * 32
    a = compute_node_id([p1, p2], "a", "note", {}, _ctx(), 0)
    b = compute_node_id([p2, p1], "a", "note", {}, _ctx(), 0)
    assert a != b


def test_signature_binds_seq_and_wall_time(signer):
    n1 = create_node(
        parents=[], actor="a", kind="note", payload={}, context_root=_ctx(),
        seed=0, seq=0, signer=signer, wall_time="2026-01-01T00:00:00+00:00",
    )
    n2 = create_node(
        parents=[], actor="a", kind="note", payload={}, context_root=_ctx(),
        seed=0, seq=1, signer=signer, wall_time="2026-01-01T00:00:00+00:00",
    )
    # Same node id (seq/wall_time are not in the preimage) but different sigs.
    assert n1.id == n2.id
    assert n1.signature != n2.signature
    n1.verify_signature()
    n2.verify_signature()


def test_tampered_node_detected(signer):
    n = create_node(
        parents=[], actor="a", kind="note", payload={"x": 1},
        context_root=_ctx(), seed=0, seq=0, signer=signer,
    )
    # Rewriting the payload while keeping the original id must fail id check.
    with pytest.raises(ValueError):
        Node.from_dict({**n.to_dict(), "payload": {"x": 2}, "id": n.id})


def test_from_dict_roundtrip(signer):
    n = create_node(
        parents=[], actor="a", kind="note", payload={"x": [1, 2, {"y": None}]},
        context_root=_ctx(), seed=7, seq=3, signer=signer,
    )
    m = Node.from_dict(n.to_dict())
    assert m == n


def test_context_root_is_stable():
    ctx = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    assert compute_context_root(None, ctx) == compute_context_root(None, ctx)
    assert compute_context_root(None, ctx) != compute_context_root(None, ctx + [{"x": 1}])


def test_blob_id_properties():
    assert compute_blob_id(b"abc") == compute_blob_id(b"abc")
    assert compute_blob_id(b"abc") != compute_blob_id(b"abd")
    assert len(compute_blob_id(b"")) == 64


def test_rejects_bad_kinds(signer):
    with pytest.raises(ValueError):
        create_node(
            parents=[], actor="a", kind="bogus", payload={},
            context_root=_ctx(), seed=0, seq=0, signer=signer,
        )


def test_unicode_payload_canonical(signer):
    n = create_node(
        parents=[], actor="agent", kind="user", payload={"text": "héllo 🌍"},
        context_root=_ctx(), seed=0, seq=0, signer=signer,
    )
    assert Node.from_dict(n.to_dict()).id == n.id

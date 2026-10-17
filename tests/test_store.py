"""Object store tests: content addressing, refs, traversal, persistence."""

import os

import pytest

from cairn import signing
from cairn.node import compute_context_root, create_node
from cairn.store import Repo, RepoError


@pytest.fixture()
def repo(tmp_path):
    return Repo.init(str(tmp_path / "proj"))


@pytest.fixture()
def keypair():
    return signing.generate_keypair()


@pytest.fixture()
def signer(keypair):
    from cairn.signing import Signer
    return Signer(keypair[0])


def _node(signer, parents=(), seq=0, payload=None, kind="note"):
    return create_node(
        parents=list(parents), actor="agent", kind=kind,
        payload=payload or {"n": seq},
        context_root=compute_context_root(None, []),
        seed=0, seq=seq, signer=signer,
        wall_time="2026-01-01T00:00:00+00:00",
    )


def test_init_creates_layout(repo):
    assert os.path.isdir(os.path.join(repo.dotdir, "objects"))
    assert os.path.isdir(os.path.join(repo.dotdir, "refs", "heads"))
    assert repo.get_head() is None
    assert repo.current_branch() == "main"


def test_init_twice_fails(tmp_path):
    Repo.init(str(tmp_path))
    with pytest.raises(RepoError):
        Repo.init(str(tmp_path))


def test_open_missing_fails(tmp_path):
    with pytest.raises(RepoError):
        Repo.open(str(tmp_path / "nope"))


def test_put_get_node_roundtrip(repo, signer):
    n = _node(signer)
    oid = repo.put_node(n)
    assert oid == n.id
    assert repo.get_node(oid) == n


def test_content_addressed_dedup(repo, signer):
    n = _node(signer)
    repo.put_node(n)
    repo.put_node(n)  # second write is a no-op
    assert repo.count_objects() == 1


def test_get_missing_object_fails(repo):
    with pytest.raises(RepoError):
        repo.get_node("ab" * 32)


def test_prefix_resolution(repo, signer):
    n = _node(signer)
    repo.put_node(n)
    assert repo.get_node(n.id[:12]).id == n.id
    with pytest.raises(RepoError):
        repo.get_node("zzzz")


def test_blob_roundtrip_and_verify(repo):
    data = b'{"hello": "world"}'
    oid = repo.put_blob(data)
    assert repo.get_blob(oid) == data
    # Corrupt the stored blob -> verification must fail on read.
    path = repo._object_path(oid)
    with open(path, "wb") as f:
        f.write(b"garbage")
    with pytest.raises(Exception):
        repo.get_blob(oid)


def test_blob_dedup(repo):
    oid1 = repo.put_blob(b"same")
    oid2 = repo.put_blob(b"same")
    assert oid1 == oid2
    assert repo.count_objects() == 1


def test_branch_workflow(repo, signer):
    n1 = _node(signer, seq=0)
    repo.put_node(n1)
    repo.set_head(n1.id)
    assert repo.get_head() == n1.id

    repo.create_branch("feature", at=n1.id)
    assert repo.get_branch("feature") == n1.id
    assert sorted(repo.list_branches()) == ["feature", "main"]

    n2 = _node(signer, parents=[n1.id], seq=1)
    repo.put_node(n2)
    repo.checkout("feature")
    repo.set_head(n2.id)
    assert repo.get_branch("feature") == n2.id
    assert repo.get_branch("main") == n1.id


def test_checkout_detached(repo, signer):
    n1 = _node(signer, seq=0)
    repo.put_node(n1)
    repo.set_head(n1.id)
    repo.checkout(n1.id)
    assert repo.current_branch() is None
    assert repo.get_head() == n1.id


def test_checkout_new_branch(repo, signer):
    n1 = _node(signer, seq=0)
    repo.put_node(n1)
    repo.set_head(n1.id)
    repo.checkout(n1.id, new_branch="exp")
    assert repo.current_branch() == "exp"
    assert repo.get_branch("exp") == n1.id


def test_walk_order_and_merges(repo, signer):
    a = _node(signer, seq=0)
    b = _node(signer, parents=[a.id], seq=1)
    c = _node(signer, parents=[a.id], seq=1, payload={"n": "c"})
    m = _node(signer, parents=[b.id, c.id], seq=2)
    for n in (a, b, c, m):
        repo.put_node(n)
    order = [n.id for n in repo.walk(m.id)]
    assert order[0] == m.id
    assert set(order) == {a.id, b.id, c.id, m.id}
    # Invariant: every node's parents appear after the node itself.
    pos = {oid: i for i, oid in enumerate(order)}
    for nid, parents in ((m.id, (b.id, c.id)), (b.id, (a.id,)), (c.id, (a.id,))):
        for p in parents:
            assert pos[p] > pos[nid]


def test_find_walks_up(tmp_path, repo):
    sub = tmp_path / "proj" / "a" / "b"
    sub.mkdir(parents=True)
    found = Repo.find(str(sub))
    assert found.dotdir == repo.dotdir


def test_persistence_across_open(tmp_path, signer):
    path = str(tmp_path / "proj")
    r1 = Repo.init(path)
    n = _node(signer)
    r1.put_node(n)
    r1.set_head(n.id)
    r2 = Repo.open(path)
    assert r2.get_head() == n.id
    assert r2.get_node(n.id) == n

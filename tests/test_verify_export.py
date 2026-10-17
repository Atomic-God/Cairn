"""Audit verification and bundle export/import tests."""

import os

import pytest

from cairn import export as exportmod
from cairn import signing as signingmod
from cairn.verify import verify as verify_repo
from cairn.session import Session
from cairn.store import Repo, RepoError


@pytest.fixture()
def key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))


@pytest.fixture()
def keypair():
    return signingmod.generate_keypair()


def _repo_with_trace(path, keypair, n=4):
    sk, _ = keypair
    s = Session.create(path, actor="auditor", secret_key=sk)
    s.system("sys", seed=1)
    for i in range(n):
        s.user(f"msg {i}", seed=10 + i)
    head = s.head
    repo = s.repo
    s.close()
    return repo, head


def test_verify_clean_repo(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p"), keypair)
    report = verify_repo(repo)
    assert report.ok, report.errors
    assert report.nodes_checked == 5
    assert report.blobs_checked == 5


def test_verify_detects_tampered_object(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p"), keypair)
    node = repo.get_node(head)
    # Corrupt the stored object bytes (simulate disk tampering).
    path = repo._object_path(head)
    with open(path, "r+b") as f:
        data = bytearray(f.read())
        data[len(data) // 2] ^= 0xFF
        f.seek(0)
        f.write(data)
    report = verify_repo(repo)
    assert not report.ok
    assert any("cannot load" in e or "id mismatch" in e for e in report.errors)


def test_verify_detects_missing_parent(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p"), keypair)
    node = repo.get_node(head)
    parent = node.parents[0]
    os.remove(repo._object_path(parent))
    report = verify_repo(repo)
    assert not report.ok
    assert any("missing parent" in e or "cannot load" in e for e in report.errors)


def test_verify_detects_forged_signature(tmp_path, keypair, key_env):
    from cairn import codec
    from cairn.node import Node

    repo, head = _repo_with_trace(str(tmp_path / "p"), keypair)
    node = repo.get_node(head)
    # Rewrite the node with a garbage signature but a correct id
    # (force the write past the store's content-addressed dedup).
    forged = Node.from_dict({**node.to_dict(), "signature": "00" * 64})
    path = repo._object_path(forged.id)
    os.remove(path)
    repo._write_object(forged.id, {"t": "node", "node": forged.to_dict()})
    report = verify_repo(repo)
    assert not report.ok
    assert any("bad signature" in e for e in report.errors)


def test_export_import_roundtrip(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p1"), keypair)
    bundle = str(tmp_path / "trace.car")
    manifest = exportmod.export_bundle(repo, head, bundle)
    assert os.path.exists(bundle)
    assert manifest["objects"] >= 10  # 5 nodes + 5 context blobs

    repo2 = exportmod.import_bundle(str(tmp_path / "p2"), bundle)
    assert repo2.get_branch("main") == head
    assert repo2.get_node(head) == repo.get_node(head)
    # Imported repo verifies cleanly.
    report = verify_repo(repo2)
    assert report.ok, report.errors


def test_export_partial_ref(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p1"), keypair, n=6)
    mid = list(repo.walk(head))[3]  # 4th newest
    bundle = str(tmp_path / "partial.car")
    manifest = exportmod.export_bundle(repo, mid.id, bundle)
    repo2 = exportmod.import_bundle(str(tmp_path / "p2"), bundle)
    # Only the ancestors of mid are present.
    assert repo2.has(mid.id)
    assert not repo2.has(head)
    assert verify_repo(repo2, refs=["main"]).ok


def test_import_refuses_to_clobber(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p1"), keypair)
    bundle = str(tmp_path / "trace.car")
    exportmod.export_bundle(repo, head, bundle)

    dest = str(tmp_path / "p2")
    sk, _ = keypair
    s = Session.create(dest, actor="other", secret_key=sk)
    s.system("different", seed=99)
    s.close()

    with pytest.raises(RepoError):
        exportmod.import_bundle(dest, bundle)
    # force=True overwrites
    repo2 = exportmod.import_bundle(dest, bundle, force=True)
    assert repo2.get_branch("main") == head


def test_read_manifest(tmp_path, keypair, key_env):
    repo, head = _repo_with_trace(str(tmp_path / "p1"), keypair)
    bundle = str(tmp_path / "trace.car")
    exportmod.export_bundle(repo, head, bundle)
    m = exportmod.read_manifest(bundle)
    assert m["exported_ref"] == head
    assert m["heads"]["main"] == head

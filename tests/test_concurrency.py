"""Concurrency, atomicity, and path-normalization tests."""

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from cairn import signing as signingmod
from cairn import verify as verify_repo
from cairn.session import Session
from cairn.store import Repo, RepoError, normalize_ref


@pytest.fixture()
def key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))


@pytest.fixture()
def signer():
    sk, _ = signingmod.generate_keypair()
    return signingmod.Signer(sk)


def test_concurrent_commits_shared_session(tmp_path, signer, key_env):
    """Threads sharing one Session must form a single linear chain."""
    s = Session.create(str(tmp_path / "p"), actor="swarm", secret_key=signer.secret_key)
    s.system("sys", seed=0)

    def work(tid):
        for i in range(10):
            s.note(f"t{tid}-{i}", seed=tid * 100 + i)

    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(work, range(8)))

    nodes = list(s.log())
    assert len(nodes) == 81
    # Linear chain: each node's single parent is the previous node.
    seen = set()
    for node in nodes:
        assert node.id not in seen
        seen.add(node.id)
    for child, parent in zip(nodes, nodes[1:]):
        assert list(child.parents) == [parent.id]
    assert verify_repo(s.repo).ok
    s.close()


def test_concurrent_commits_separate_branches(tmp_path, signer, key_env):
    """Separate sessions/branches committing concurrently stay consistent."""
    path = str(tmp_path / "p")
    s0 = Session.create(path, actor="a0", secret_key=signer.secret_key)
    s0.system("sys", seed=0)
    base = s0.head
    s0.close()

    def work(tid):
        repo = Repo.open(path)
        s = Session(repo, actor=f"a{tid}", secret_key=signer.secret_key)
        s.checkout(base, new_branch=f"w{tid}")
        for i in range(5):
            s.note(f"t{tid}-{i}", seed=i)
        s.close()
        return s.head

    with ThreadPoolExecutor(max_workers=4) as ex:
        heads = list(ex.map(work, range(4)))

    repo = Repo.open(path)
    assert verify_repo(repo).ok
    for tid, head in enumerate(heads):
        assert repo.get_branch(f"w{tid}") == head


def test_no_tmp_files_left_behind(tmp_path, signer, key_env):
    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=signer.secret_key)
    for i in range(5):
        s.note(f"n{i}", seed=i)
    s.close()
    leftovers = []
    for root, _, files in os.walk(os.path.join(str(tmp_path / "p"), ".cairn")):
        leftovers += [f for f in files if ".tmp." in f]
    assert leftovers == []


def test_lock_file_created(tmp_path, signer, key_env):
    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=signer.secret_key)
    s.note("x", seed=1)
    s.close()
    assert os.path.exists(os.path.join(str(tmp_path / "p"), ".cairn", "LOCK"))


def test_envelope_magic_validated(tmp_path, signer, key_env):
    from cairn import codec

    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=signer.secret_key)
    n = s.note("x", seed=1)
    s.close()
    repo = s.repo
    # Rewrite the object with a bad magic field.
    env = {"magic": "EVIL", "v": 1, "t": "node", "node": n.to_dict()}
    path = repo._object_path(n.id)
    os.remove(path)
    with open(path, "wb") as f:
        f.write(codec.compress(codec.canonical(env)))
    with pytest.raises(RepoError, match="bad magic"):
        repo.get_node(n.id)


def test_normalize_ref():
    assert normalize_ref("main") == "main"
    assert normalize_ref("a\\b") == "a/b"
    assert normalize_ref("//x/./y//") == "x/y"
    assert normalize_ref("replay/2024") == "replay/2024"
    with pytest.raises(RepoError):
        normalize_ref("../escape")
    with pytest.raises(RepoError):
        normalize_ref("")
    with pytest.raises(RepoError):
        normalize_ref("a/../b")


def test_branch_backslash_normalized(tmp_path, signer, key_env):
    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=signer.secret_key)
    s.note("x", seed=1)
    s.branch("team\\exp")
    assert "team/exp" in s.repo.list_branches()
    s.checkout("team\\exp")
    assert s.repo.current_branch() == "team/exp"
    s.close()

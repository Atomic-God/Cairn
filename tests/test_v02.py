"""v0.2.0 regression tests: delta contexts, linearizable commits, reflog,
identity registry, orphans, replay streaming, branch auto-create."""

import asyncio
import json
import multiprocessing as mp
import os

import pytest

from cairn import codec
from cairn.node import compute_context_root, encode_context_delta
from cairn.session import Session
from cairn.store import BlobTooLargeError, Repo, RepoError
from cairn.verify import verify as verify_fn
from cairn import replay as replay_mod


@pytest.fixture()
def keypair():
    from cairn import signing
    return signing.generate_keypair()


@pytest.fixture()
def key_env(tmp_path, monkeypatch, keypair):
    sk, _ = keypair
    p = tmp_path / "k.key"
    p.write_bytes(sk)
    monkeypatch.setenv("CAIRN_KEY_FILE", str(p))
    return keypair


def _session(path, keypair, **kw):
    sk, _ = keypair
    return Session.create(path, secret_key=sk, **kw)


# --- delta-chain contexts -------------------------------------------------

def test_delta_chain_reconstructs(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "r"), keypair)
    s.system("sys", seed=1)
    s.user("hello", seed=2)
    n = s.inference({"q": 1}, {"a": 2}, model="m", seed=3)
    s.close()
    repo = Repo.find(str(tmp_path / "r"))
    msgs = repo.get_context(n.context_root)
    assert [m["kind"] for m in msgs] == ["system", "user", "inference"]
    assert msgs[1]["text"] == "hello"
    # Each commit stored only its own delta, not the full history.
    assert len(repo.get_context(n.context_root)) == 3


def test_delta_chain_is_incremental(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "r"), keypair)
    n1 = s.system("a", seed=1)
    n2 = s.user("b", seed=2)
    s.close()
    repo = Repo.find(str(tmp_path / "r"))
    d2 = codec.canonical_loads(repo.get_blob(n2.context_root))
    assert d2["prev"] == n1.context_root
    assert [m["text"] for m in d2["messages"]] == ["b"]
    # root chaining is deterministic
    assert n2.context_root == compute_context_root(n1.context_root,
                                                    d2["messages"])


def test_legacy_full_snapshot_blob_still_reads(tmp_path, keypair, key_env):
    # v0.1.x stored a bare message list as the context blob.
    s = _session(str(tmp_path / "r"), keypair)
    repo = s.repo
    legacy = [{"kind": "system", "text": "old"}]
    oid = repo.put_blob(codec.canonical(legacy))
    s.close()
    assert repo.get_context(oid) == legacy


def test_notes_reuse_context_root(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "r"), keypair)
    n1 = s.system("sys", seed=1)
    n2 = s.note("annotation", seed=2)
    assert n2.context_root == n1.context_root
    s.close()


# --- linearizable same-branch commits --------------------------------------

def _race_worker(args):
    path, sk_hex, i = args
    from cairn import signing
    sk = bytes.fromhex(sk_hex)
    try:
        with Session.open(path, actor=f"w{i}", secret_key=sk) as s:
            for j in range(5):
                s.note(f"{i}/{j}")
                s.user(f"msg {i}/{j}")
        return True
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def test_same_branch_writes_linearize_no_orphans(tmp_path, keypair, key_env):
    sk, _ = keypair
    path = str(tmp_path / "r")
    with _session(path, keypair) as s:
        s.system("genesis", seed=0)
    nproc, per = 8, 5
    with mp.Pool(nproc) as pool:
        res = pool.map(_race_worker,
                       [(path, sk.hex(), i) for i in range(nproc)])
    assert all(r is True for r in res), res[:3]
    repo = Repo.find(path)
    chain = list(repo.walk("main"))
    # 1 genesis + nproc*per*2 (note+user each) — every commit reachable.
    assert len(chain) == 1 + nproc * per * 2
    assert repo.find_orphans() == []
    rep = verify_fn(repo)
    assert rep.ok and rep.nodes_checked == len(chain)
    # seq numbers are dense and ordered along the chain
    seqs = sorted(n.seq for n in chain)
    assert seqs == list(range(len(chain)))


# --- reflog -----------------------------------------------------------------

def test_reflog_records_commits_and_rewind(tmp_path, keypair, key_env):
    path = str(tmp_path / "r")
    with _session(path, keypair, actor="a") as s:
        s.system("one", seed=1)
        mid = s.head
        s.system("two", seed=2)
    repo = Repo.find(path)
    log = repo.read_reflog("main")
    assert len(log) == 2
    assert log[0]["old"] == "0" * 64  # branch creation
    assert log[0]["actor"] == "a"
    # Silent history rewrite leaves a reflog trail.
    repo.set_head(mid)
    log = repo.read_reflog("main")
    assert len(log) == 3
    assert log[-1]["new"] == mid


# --- identity registry ------------------------------------------------------

def test_identity_registry_tracks_pairs(tmp_path, keypair, key_env):
    from cairn import signing
    sk1, _ = keypair
    sk2, _ = signing.generate_keypair()
    path = str(tmp_path / "r")
    with Session.create(path, actor="alice", secret_key=sk1) as s:
        s.system("hi", seed=1)
    with Session.open(path, actor="alice", secret_key=sk2) as s:
        s.system("hi again", seed=2)
    # Same key, different actor name: recorded, not blocked.
    with Session.open(path, actor="mallory?", secret_key=sk1) as s:
        s.note("suspicious", seed=3)
    from cairn.signing import Signer
    pub1 = Signer(sk1).public_key.hex()
    repo = Repo.find(path)
    reg = repo.list_identities()
    assert reg[pub1]["actor"] == "mallory?"
    assert set(reg[pub1]["actors_seen"]) == {"alice", "mallory?"}
    rep = verify_fn(repo)
    assert rep.ok
    assert any("mallory" in n or "alice" in n for n in rep.identity_notes)


# --- orphans ------------------------------------------------------------------

def test_find_orphans(tmp_path, keypair, key_env):
    from cairn.node import create_node
    from cairn import signing as signingmod
    sk, _ = keypair
    signer = signingmod.Signer(sk)
    path = str(tmp_path / "r")
    with _session(path, keypair) as s:
        s.system("genesis", seed=1)
    repo = Repo.find(path)
    root = repo.put_context(None, [])
    orphan = create_node(parents=[], actor="ghost", kind="note",
                         payload={"text": "unreachable"}, context_root=root,
                         seed=9, seq=99, signer=signer)
    repo.put_node(orphan)
    assert repo.find_orphans() == [orphan.id]
    rep = verify_fn(repo)
    assert rep.ok  # orphans are not corruption
    assert rep.orphans == [orphan.id]


# --- replay streaming ---------------------------------------------------------

def test_replay_stream(tmp_path, keypair, key_env):
    from cairn import replay_run
    path = str(tmp_path / "r")
    chunks = ["hel", "lo ", "world"]

    def llm(request, model=None, seed=None):
        return {"content": "hello world", "streamed": True}

    with _session(path, keypair) as s:
        client = s.client()
        client.system("sys")
        rec = client.stream(iter(chunks), model="m",
                            request={"q": 1}, seed=7)
        list(rec)  # exhaust -> commits
        head = s.head
        repo = s.repo

    def agent(client):
        client.system("sys")
        seen = list(client.stream({"q": 1}, model="m", seed=7))
        assert "".join(seen) == "hello world"
        return "ok"

    report = replay_mod.run(repo, head, agent, branch="rs")
    assert report.ok and report.ids_match
    assert report.result == "ok"


def test_replay_astream(tmp_path, keypair, key_env):
    from cairn import replay_run
    path = str(tmp_path / "r")
    with _session(path, keypair) as s:
        client = s.client()
        client.system("sys")
        rec = client.stream(iter(["ab", "cd"]), model="m",
                            request={"q": 1}, seed=7)
        list(rec)
        head = s.head
        repo = s.repo

    async def agent(client):
        client.system("sys")
        parts = []
        async for ch in client.astream({"q": 1}, model="m", seed=7):
            parts.append(ch)
        assert "".join(parts) == "abcd"

    trace = replay_mod.linear_trace(repo, head)
    session = Session(repo, branch="rsa", secret_key=keypair[0],
                      create_branch=True)
    rclient = replay_mod.ReplayClient(session, trace)
    asyncio.run(agent(rclient))
    rclient.assert_exhausted()


# --- branch auto-create --------------------------------------------------------

def test_session_create_branch_opt_in(tmp_path, keypair, key_env):
    path = str(tmp_path / "r")
    with _session(path, keypair) as s:
        s.system("genesis", seed=1)
    # Without opt-in: clear error.
    with pytest.raises(ValueError, match="does not exist"):
        Session.open(path, branch="feature", secret_key=keypair[0])
    # With opt-in: branch created at the current head.
    with Session.open(path, branch="feature", secret_key=keypair[0],
                      create_branch=True) as s:
        s.note("on feature", seed=2)
    repo = Repo.find(path)
    assert "feature" in repo.list_branches()


# --- codec / blob guards ---------------------------------------------------------

def test_nonfinite_floats_rejected():
    with pytest.raises(ValueError):
        codec.canonical({"v": float("nan")})
    with pytest.raises(ValueError):
        codec.canonical({"v": float("inf")})
    # finite floats still fine
    assert codec.canonical({"v": 0.5}) == b'{"v":0.5}'


def test_blob_size_guard(tmp_path, keypair, key_env, monkeypatch):
    monkeypatch.setenv("CAIRN_MAX_BLOB_BYTES", "1024")
    s = _session(str(tmp_path / "r"), keypair)
    with pytest.raises(BlobTooLargeError, match="CAIRN_MAX_BLOB_BYTES"):
        s.repo.put_blob(b"x" * 2048)
    s.close()


def test_key_file_roundtrip_never_strips_bytes(tmp_path, monkeypatch):
    """Regression: ~4.6% of random keys were mangled by .strip() on load."""
    from cairn import keys, signing
    # Force keys whose first/last bytes are ASCII whitespace.
    bad = [b" " + b"\x01" * 30 + b"\n",
           b"\t" + b"\x02" * 30 + b"\r",
           b"\x0b" + b"\x03" * 30 + b"\x0c"]
    for i, sk in enumerate(bad):
        p = tmp_path / f"k{i}.key"
        p.write_bytes(sk)
        loaded, _ = keys.load_or_create(str(p))
        assert loaded == sk
    # And 500 random keys round-trip through write+load.
    for i in range(500):
        sk, _ = signing.generate_keypair()
        p = tmp_path / f"r{i}.key"
        p.write_bytes(sk)
        loaded, pk = keys.load_or_create(str(p))
        assert loaded == sk
        assert pk == signing.Signer(sk).public_key

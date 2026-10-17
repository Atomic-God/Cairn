"""Session tests: recording, context tracking, branching, persistence."""

import os

import pytest

from cairn import signing
from cairn.session import Session
from cairn.store import Repo


@pytest.fixture()
def key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))


@pytest.fixture()
def keypair():
    return signing.generate_keypair()


def _session(path, keypair, **kw):
    sk, _ = keypair
    return Session.create(path, actor="tester", secret_key=sk, **kw)


def test_record_linear_trace(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    s.system("sys", seed=1)
    s.user("hi", seed=2)
    s.inference({"messages": []}, {"text": "yo"}, model="m", seed=3)
    s.tool_call("calc", {"expr": "1+1"}, seed=4)
    s.tool_result("calc", 2, seed=5)
    s.note("done", seed=6)
    assert s.head is not None
    kinds = [n.kind for n in s.log()]
    assert kinds == ["note", "tool_result", "tool_call",
                     "inference", "user", "system"]
    s.close()


def test_parents_link_correctly(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    a = s.system("a", seed=1)
    b = s.user("b", seed=2)
    assert b.parents == (a.id,)
    assert a.parents == ()


def test_seq_increments(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    seqs = [s.system("x", seed=i).seq for i in range(4)]
    assert seqs == [0, 1, 2, 3]


def test_context_root_evolves_and_restores(tmp_path, keypair, key_env):
    path = str(tmp_path / "p")
    s = _session(path, keypair)
    s.system("sys", seed=1)
    mid = s.user("hello", seed=2)
    s.close()

    s2 = Session.open(path, actor="tester", secret_key=keypair[0])
    assert s2.head == mid.id
    n = s2.inference({"m": []}, {"t": "hi"}, model="m", seed=3)
    # Context includes the two earlier messages plus this inference.
    assert len(s2.messages) == 3
    assert s2.messages[0]["text"] == "sys"
    s2.close()
    # Context chain stored and reconstructable via the node's context_root.
    repo = Repo.find(path)
    messages = repo.get_context(n.context_root)
    assert any("hello" in str(m) for m in messages)


def test_notes_do_not_pollute_context(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    s.system("sys", seed=1)
    before = list(s.messages)
    s.note("a private annotation", seed=2)
    assert s.messages == before
    s.close()


def test_branch_and_checkout(tmp_path, keypair, key_env):
    path = str(tmp_path / "p")
    s = _session(path, keypair)
    a = s.system("a", seed=1)
    b = s.user("b", seed=2)
    s.branch("exp")
    c = s.user("c-main", seed=3)

    s.checkout("exp")
    assert s.head == b.id
    d = s.user("d-exp", seed=4)
    assert d.parents == (b.id,)

    s.checkout("main")
    assert s.head == c.id
    s.close()

    repo = Repo.find(path)
    assert repo.get_branch("main") == c.id
    assert repo.get_branch("exp") == d.id


def test_checkout_old_node_time_travel(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    a = s.system("a", seed=1)
    s.user("wrong turn", seed=2)
    # Go back to `a`, branch, and redo — upstream nodes are never recomputed.
    s.checkout(a.id, new_branch="fix")
    fixed = s.user("right turn", seed=3)
    assert fixed.parents == (a.id,)
    assert [n.kind for n in s.log()] == ["user", "system"]
    s.close()


def test_recording_client_end_to_end(tmp_path, keypair, key_env):
    calls = []

    def fake_llm(request, model=None, seed=None):
        calls.append(("llm", request))
        return {"text": "4"}

    def fake_calc(args):
        calls.append(("tool", args))
        return {"value": 4}

    s = _session(str(tmp_path / "p"), keypair)
    c = s.client(llm=fake_llm, tools={"calc": fake_calc})
    c.system("You are a calculator.")
    c.user("What is 2+2?")
    resp = c.infer({"messages": ["hi"]}, model="fake-1", seed=99)
    assert resp == {"text": "4"}
    out = c.tool("calc", {"expr": "2+2"})
    assert out == {"value": 4}
    assert calls == [("llm", {"messages": ["hi"]}), ("tool", {"expr": "2+2"})]
    kinds = [n.kind for n in s.log()]
    assert kinds == ["tool_result", "tool_call", "inference", "user", "system"]
    # tool_call and tool_result share a call_id
    nodes = list(s.log())
    assert nodes[0].payload["call_id"] == nodes[1].payload["call_id"]
    s.close()


def test_tool_failure_recorded(tmp_path, keypair, key_env):
    def boom(args):
        raise RuntimeError("kaput")

    s = _session(str(tmp_path / "p"), keypair)
    c = s.client(llm=lambda *a, **k: {}, tools={"bad": boom})
    with pytest.raises(RuntimeError):
        c.tool("bad", {})
    nodes = list(s.log())
    assert nodes[0].kind == "tool_result"
    assert nodes[0].payload["ok"] is False
    assert "kaput" in nodes[0].payload["result"]["error"]
    s.close()


def test_client_requires_backends(tmp_path, keypair, key_env):
    s = _session(str(tmp_path / "p"), keypair)
    c = s.client()
    with pytest.raises(RuntimeError):
        c.infer({}, model="m")
    with pytest.raises(KeyError):
        c.tool("nope", {})
    s.close()


def test_deterministic_ids_across_repos(tmp_path, keypair, key_env):
    """Same ops + same seeds => same node ids, even in a fresh repo."""
    def build(path):
        s = _session(path, keypair)
        s.system("s", seed=1)
        s.inference({"q": 1}, {"a": 2}, model="m", seed=5)
        head = s.head
        s.close()
        return head

    h1 = build(str(tmp_path / "p1"))
    h2 = build(str(tmp_path / "p2"))
    assert h1 == h2

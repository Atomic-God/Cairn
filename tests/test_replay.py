"""Replay tests: determinism proofs, divergence detection, re-basing."""

import pytest

from cairn import replay
from cairn.replay import (
    ReplayClient,
    ReplayDivergenceError,
    ReplayError,
    ReplayExhaustedError,
    ReplayIncompleteError,
    linear_trace,
)
from cairn.session import Session
from cairn import signing as signingmod


@pytest.fixture()
def key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))


@pytest.fixture()
def keypair():
    return signingmod.generate_keypair()


def _record(path, keypair):
    """Record a canonical agent trace; return (repo, head)."""
    sk, _ = keypair
    s = Session.create(path, actor="agent-1", secret_key=sk)
    c = s.client(
        llm=lambda req, model=None, seed=None: {"text": f"echo:{req['q']}"},
        tools={"calc": lambda args: {"value": eval(args["expr"])}},
    )
    c.system("You are a calculator.", )
    c.user("What is 2+2?")
    c.infer({"q": "2+2?"}, model="fake-1", seed=11)
    c.tool("calc", {"expr": "2+2"})
    c.note("solved")
    head = s.head
    repo = s.repo
    s.close()
    return repo, head


def _agent(client):
    client.system("You are a calculator.")
    client.user("What is 2+2?")
    r = client.infer({"q": "2+2?"}, model="fake-1", seed=11)
    assert r == {"text": "echo:2+2?"}
    v = client.tool("calc", {"expr": "2+2"})
    assert v == {"value": 4}
    client.note("solved")
    return "ok-result"


def test_replay_proves_determinism(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)
    report = replay.run(repo, head, _agent, secret_key=keypair[0])
    assert report.ok
    assert report.ids_match
    assert report.nodes_replayed == 6
    assert report.result == "ok-result"
    # The replay branch exists and its head matches the original head id.
    assert repo.get_branch(report.branch) == head


def test_replay_detects_request_divergence(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)

    def bad_agent(client):
        client.system("You are a calculator.")
        client.user("What is 2+2?")
        client.infer({"q": "3+3?"}, model="fake-1", seed=11)  # different!

    with pytest.raises(ReplayDivergenceError):
        replay.run(repo, head, bad_agent, secret_key=keypair[0])


def test_replay_detects_tool_arg_divergence(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)

    def bad_agent(client):
        client.system("You are a calculator.")
        client.user("What is 2+2?")
        client.infer({"q": "2+2?"}, model="fake-1", seed=11)
        client.tool("calc", {"expr": "3*3"})  # different!

    with pytest.raises(ReplayDivergenceError):
        replay.run(repo, head, bad_agent, secret_key=keypair[0])


def test_replay_detects_wrong_order(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)

    def bad_agent(client):
        client.user("What is 2+2?")  # skipped system
        client.system("You are a calculator.")

    with pytest.raises(ReplayDivergenceError):
        replay.run(repo, head, bad_agent, secret_key=keypair[0])


def test_replay_detects_extra_calls(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)

    def greedy(client):
        _agent(client)
        client.note("one more")  # beyond the trace

    with pytest.raises(ReplayExhaustedError):
        replay.run(repo, head, greedy, secret_key=keypair[0])


def test_replay_detects_incomplete(tmp_path, keypair, key_env):
    repo, head = _record(str(tmp_path / "p"), keypair)

    def lazy(client):
        client.system("You are a calculator.")

    with pytest.raises(ReplayIncompleteError):
        replay.run(repo, head, lazy, secret_key=keypair[0])


def test_cross_model_rebase(tmp_path, keypair, key_env):
    """Same requests, different model => legitimate fork with new ids."""
    repo, head = _record(str(tmp_path / "p"), keypair)

    def rebased(client):
        client.system("You are a calculator.")
        client.user("What is 2+2?")
        # New model, same request fingerprint -> allowed, recorded as rebase.
        client.infer({"q": "2+2?"}, model="local-weights-2", seed=11)
        client.tool("calc", {"expr": "2+2"})
        client.note("solved")

    report = replay.run(repo, head, rebased, secret_key=keypair[0])
    assert not report.ids_match
    assert report.rebased_models == ["fake-1 -> local-weights-2"]
    assert not report.ok  # not a bit-identical replay, by design


def test_replay_from_mid_trace_checkpoint(tmp_path, keypair, key_env):
    """Replay can start from a mid-trace node (context restored from blob)."""
    sk, _ = keypair
    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=sk)
    s.system("sys", seed=1)
    mid = s.user("checkpoint here", seed=2)
    s.inference({"q": "x"}, {"a": "y"}, model="m", seed=3)
    head = s.head
    repo = s.repo
    s.close()

    def tail_agent(client):
        r = client.infer({"q": "x"}, model="m", seed=3)
        assert r == {"a": "y"}
        return "tail-done"

    report = replay.run(repo, head, tail_agent, secret_key=sk,
                        branch="replay/tail", start=mid.id)
    # Replaying only the tail: ids of replayed nodes must match the originals.
    assert report.nodes_replayed == 1
    assert report.ids_match
    # Context was restored: the replayed inference saw the earlier messages.
    assert len(Session(repo, branch="replay/tail", secret_key=sk).messages) == 3


def test_linear_trace_rejects_merges(tmp_path, keypair, key_env):
    sk, _ = keypair
    s = Session.create(str(tmp_path / "p"), actor="a", secret_key=sk)
    a = s.system("a", seed=1)
    s.checkout(a.id, new_branch="b1")
    s.user("x", seed=2)
    s.checkout("main")
    s.user("y", seed=3)
    # Manually craft a merge node with two parents.
    from cairn.node import create_node, compute_context_root
    m = create_node(
        parents=[s.repo.get_branch("b1"), s.repo.get_branch("main")],
        actor="a", kind="note", payload={"text": "merge"},
        context_root=compute_context_root(None, []), seed=9, seq=9,
        signer=signingmod.Signer(sk),
    )
    s.repo.put_node(m)
    with pytest.raises(ReplayError):
        linear_trace(s.repo, m.id)
    s.close()


def test_fingerprint_stability():
    assert (replay.fingerprint_request({"b": 1, "a": [1, 2]})
            == replay.fingerprint_request({"a": [1, 2], "b": 1}))


def test_network_guard_blocks_sockets():
    import socket
    from cairn.replay import network_guard, ReplayNetworkBlockedError
    with network_guard():
        with pytest.raises(ReplayNetworkBlockedError):
            socket.create_connection(("example.com", 80), timeout=1)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with pytest.raises(ReplayNetworkBlockedError):
                s.connect(("example.com", 80))
        finally:
            s.close()
    # Restored afterwards: unconnected socket creation still works.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()


def test_replay_blocks_network_by_default(tmp_path, keypair, key_env):
    import socket
    from cairn.replay import ReplayNetworkBlockedError
    repo, head = _record(str(tmp_path / "p"), keypair)

    def sneaky(client):
        client.system("You are a calculator.")
        # Trying to phone home mid-replay must fail loudly.
        socket.create_connection(("example.com", 80), timeout=1)

    with pytest.raises(ReplayNetworkBlockedError):
        replay.run(repo, head, sneaky, secret_key=keypair[0])


def test_replay_allow_network_opt_out(tmp_path, keypair, key_env, monkeypatch):
    import socket
    repo, head = _record(str(tmp_path / "p"), keypair)

    real_create = socket.create_connection
    calls = []
    def fake_create(*a, **k):
        calls.append(a)
        raise OSError("no network in test")
    monkeypatch.setattr(socket, "create_connection", fake_create)

    def agent(client):
        _agent(client)
        try:
            socket.create_connection(("example.com", 80), timeout=1)
        except OSError:
            pass
        return "done"

    report = replay.run(repo, head, agent, secret_key=keypair[0],
                        allow_network=True)
    assert report.ok
    assert calls, "expected the agent's socket call to go through unguarded"

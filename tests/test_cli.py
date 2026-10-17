"""End-to-end CLI tests: drive the real `cairn` entry point in a sandbox."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cairn import signing as signingmod
from cairn.session import Session

CLI = [sys.executable, "-m", "cairn"]
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


@pytest.fixture()
def workdir(tmp_path, monkeypatch):
    d = tmp_path / "work"
    d.mkdir()
    monkeypatch.chdir(d)
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))
    return d


def _run(*argv, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in [PROJECT_ROOT, env.get("PYTHONPATH", "")] if p]
    )
    return subprocess.run(
        CLI + list(argv), capture_output=True, text=True, cwd=cwd, env=env
    )


def _record_trace(path, key):
    sk, _ = key
    s = Session.create(path, actor="cli-agent", secret_key=sk)
    s.system("sys", seed=1)
    s.user("hello", seed=2)
    s.inference({"q": 1}, {"a": 2}, model="m", seed=3)
    head = s.head
    s.close()
    return head


def test_cli_init_and_status(workdir):
    r = _run("init")
    assert r.returncode == 0, r.stderr
    assert "Initialized empty cairn repository" in r.stdout
    r = _run("status")
    assert r.returncode == 0, r.stderr
    assert "On branch main" in r.stdout
    # init twice fails without --exist-ok
    r = _run("init")
    assert r.returncode == 1


def test_cli_log_show_diff(workdir):
    key = signingmod.generate_keypair()
    _run("init", cwd=str(workdir))
    head = _record_trace(str(workdir), key)

    r = _run("log", "--oneline")
    assert r.returncode == 0, r.stderr
    lines = r.stdout.strip().splitlines()
    assert len(lines) == 3
    assert "[  inference]" in lines[0]

    r = _run("show", head)
    assert r.returncode == 0, r.stderr
    doc = json.loads(r.stdout)
    assert doc["kind"] == "inference"

    r = _run("show", head, "--context")
    assert r.returncode == 0, r.stderr
    assert "context" in r.stdout

    r = _run("diff", head, head)
    assert r.returncode == 0, r.stderr
    assert "identical preimage fields" in r.stdout


def test_cli_branch_and_checkout(workdir):
    key = signingmod.generate_keypair()
    _run("init", cwd=str(workdir))
    head = _record_trace(str(workdir), key)

    r = _run("branch", "exp")
    assert r.returncode == 0, r.stderr
    r = _run("branch", "-l")
    assert "* main" in r.stdout and "  exp" in r.stdout

    r = _run("checkout", head[:12])
    assert r.returncode == 0, r.stderr
    assert "Detached HEAD" in r.stdout
    r = _run("checkout", "main")
    assert r.returncode == 0, r.stderr
    assert "Switched to branch 'main'" in r.stdout

    r = _run("branch", "-d", "exp")
    assert r.returncode == 0, r.stderr


def test_cli_verify(workdir):
    key = signingmod.generate_keypair()
    _run("init", cwd=str(workdir))
    _record_trace(str(workdir), key)
    r = _run("verify")
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cli_replay_check_and_run(workdir, tmp_path):
    key = signingmod.generate_keypair()
    _run("init", cwd=str(workdir))
    head = _record_trace(str(workdir), key)

    r = _run("replay", head, "--check")
    assert r.returncode == 0, r.stderr
    assert "replayable" in r.stdout

    # Write a replay agent module and run it through the CLI.
    agent_src = '''
def agent(client):
    client.system("sys")
    client.user("hello")
    assert client.infer({"q": 1}, model="m", seed=3) == {"a": 2}
    return "done"
'''
    (workdir / "myagent.py").write_text(agent_src)
    r = _run("replay", head, "--agent", "myagent:agent", "--branch", "cli-replay")
    assert r.returncode == 0, r.stderr
    assert "deterministic replay proven" in r.stdout
    assert "Agent returned: 'done'" in r.stdout


def test_cli_export_import(workdir, tmp_path):
    key = signingmod.generate_keypair()
    _run("init", cwd=str(workdir))
    head = _record_trace(str(workdir), key)

    bundle = str(tmp_path / "t.car")
    r = _run("export", head, "-o", bundle)
    assert r.returncode == 0, r.stderr
    assert os.path.exists(bundle)

    dest = tmp_path / "imported"
    dest.mkdir()
    r = _run("import", bundle, str(dest))
    assert r.returncode == 0, r.stderr
    r = _run("verify", cwd=str(dest))
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


def test_cli_key(workdir):
    _run("init", cwd=str(workdir))
    r = _run("key")
    assert r.returncode == 0, r.stderr
    assert "Fingerprint:" in r.stdout
    assert "Public key:" in r.stdout


def test_cli_no_repo_errors_cleanly(workdir):
    (workdir / "empty").mkdir()
    r = _run("status", cwd=str(workdir / "empty"))
    assert r.returncode == 1
    assert "not inside a cairn repository" in r.stderr


def test_mock_agent_demo(tmp_path, monkeypatch):
    """The zero-key demo runs offline and produces the expected DAG shape."""
    import importlib.util
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))
    spec = importlib.util.spec_from_file_location(
        "mock_agent", os.path.join(PROJECT_ROOT, "examples", "mock_agent.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.main(workdir=str(tmp_path / "demo"))
    assert summary["p99_ms"] == 180
    assert summary["elapsed"] < 30  # offline demo must be fast
    assert summary["main_head"] != summary["alt_head"]


def test_incident_response_demo(tmp_path, monkeypatch):
    """The flagship incident-response scenario runs offline and replays <2s."""
    import importlib.util
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))
    spec = importlib.util.spec_from_file_location(
        "incident_response",
        os.path.join(PROJECT_ROOT, "examples", "incident_response.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    summary = mod.main(workdir=str(tmp_path / "incident"))
    assert summary["p99_ms"] == 120
    assert summary["replay_seconds"] < 2.0
    assert summary["main_head"] != summary["safe_head"]


def test_cli_commit(workdir):
    """`cairn commit` records note/observation nodes from the shell."""
    r = _run("init")
    assert r.returncode == 0, r.stderr
    r = _run("commit", "-m", "hello", "--actor", "cli-bot")
    assert r.returncode == 0, r.stderr
    assert "committed" in r.stdout and "[note]" in r.stdout
    r = _run("commit", "-m", "p99 ok", "--kind", "observation",
             "--source", "monitor")
    assert r.returncode == 0, r.stderr
    assert "[observation]" in r.stdout
    r = _run("log", "--oneline")
    assert r.returncode == 0, r.stderr
    assert "note" in r.stdout and "observation" in r.stdout
    r = _run("verify")
    assert r.returncode == 0, r.stderr


def test_cli_refs_accept_branch_names(workdir):
    """log/show/replay/export resolve branch names, not just hex ids."""
    r = _run("init")
    assert r.returncode == 0, r.stderr
    r = _run("commit", "-m", "hello")
    assert r.returncode == 0, r.stderr
    for cmd in (["log", "main", "--oneline"], ["show", "main"],
                ["replay", "main", "--check"]):
        r = _run(*cmd)
        assert r.returncode == 0, (cmd, r.stderr)
    r = _run("export", "main", "-o", str(workdir / "t.car"))
    assert r.returncode == 0, r.stderr
    r = _run("log", "no-such-branch")
    assert r.returncode != 0

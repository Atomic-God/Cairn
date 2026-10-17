"""Async client and streaming accumulator tests (stdlib asyncio only)."""

import asyncio

import pytest

from cairn import signing as signingmod
from cairn.aio import (
    AsyncAgentClient,
    AsyncStreamRecorder,
    StreamRecorder,
    default_extract_text,
    default_is_done,
)
from cairn.session import Session


@pytest.fixture()
def key_env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAIRN_KEY_FILE", str(tmp_path / "test.key"))


@pytest.fixture()
def signer():
    sk, _ = signingmod.generate_keypair()
    return signingmod.Signer(sk)


def _session(path, signer):
    return Session.create(path, actor="aio", secret_key=signer.secret_key)


# ---------------------------------------------------------------- helpers

async def fake_async_llm(request, model=None, seed=None):
    await asyncio.sleep(0)
    return {"content": f"async:{request['q']}", "model": model}


async def fake_async_tool(arguments):
    await asyncio.sleep(0)
    return {"v": arguments["x"] * 2}


def openai_style_chunks(words):
    for w in words:
        yield {"choices": [{"delta": {"content": w}, "finish_reason": None}]}
    yield {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    yield "[DONE]"


async def async_openai_style_chunks(words):
    for w in words:
        await asyncio.sleep(0)
        yield {"choices": [{"delta": {"content": w}, "finish_reason": None}]}
    yield "[DONE]"


# ---------------------------------------------------------------- tests

def test_async_client_end_to_end(tmp_path, signer, key_env):
    async def main():
        async with _session(str(tmp_path / "p"), signer) as s:
            c = AsyncAgentClient(
                s, llm=fake_async_llm, tools={"double": fake_async_tool}
            )
            c.system("sys")
            c.user("hi")
            r = await c.infer({"q": "?"}, model="m", seed=1)
            assert r == {"content": "async:?", "model": "m"}
            v = await c.tool("double", {"x": 21})
            assert v == {"v": 42}
            return s.head

    head = asyncio.run(main())
    assert head is not None


def test_async_client_accepts_sync_backends(tmp_path, signer, key_env):
    async def main():
        s = _session(str(tmp_path / "p"), signer)
        c = AsyncAgentClient(
            s,
            llm=lambda req, model=None, seed=None: {"content": "sync-llm"},
            tools={"t": lambda args: {"ok": True}},
        )
        r = await c.infer({"q": 1}, model="m", seed=2)
        assert r == {"content": "sync-llm"}
        v = await c.tool("t", {})
        assert v == {"ok": True}
        kinds = [n.kind for n in s.log()]
        assert kinds == ["tool_result", "tool_call", "inference"]
        s.close()

    asyncio.run(main())


def test_async_tool_failure_recorded(tmp_path, signer, key_env):
    async def boom(args):
        raise RuntimeError("async kaput")

    async def main():
        s = _session(str(tmp_path / "p"), signer)
        c = AsyncAgentClient(s, tools={"bad": boom})
        with pytest.raises(RuntimeError):
            await c.tool("bad", {})
        node = list(s.log())[0]
        assert node.kind == "tool_result" and node.payload["ok"] is False
        s.close()

    asyncio.run(main())


def test_stream_recorder_assembles_and_passes_through(tmp_path, signer, key_env):
    s = _session(str(tmp_path / "p"), signer)
    chunks = openai_style_chunks(["Hel", "lo", " world"])
    rec = StreamRecorder(s, chunks, model="m",
                         request={"messages": [{"role": "user", "content": "hi"}]},
                         seed=7)
    seen = list(rec)  # chunks pass through untouched, incl. finish_reason chunk
    assert seen == [
        {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]},
        {"choices": [{"delta": {"content": " world"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "stop"}]},
    ]
    assert rec.text == "Hello world"
    assert rec.node is not None
    assert rec.node.kind == "inference"
    assert rec.node.payload["response"]["streamed"] is True
    assert rec.node.payload["response"]["content"] == "Hello world"
    s.close()


def test_stream_recorder_via_client(tmp_path, signer, key_env):
    s = _session(str(tmp_path / "p"), signer)
    from cairn.session import AgentClient

    c = AgentClient(s)
    out = "".join(
        default_extract_text(ch)
        for ch in c.stream(openai_style_chunks(["a", "b"]), model="m",
                           request={"q": 1}, seed=3)
    )
    assert out == "ab"
    assert [n.kind for n in s.log()] == ["inference"]
    s.close()


def test_async_stream_recorder(tmp_path, signer, key_env):
    async def main():
        s = _session(str(tmp_path / "p"), signer)
        c = AsyncAgentClient(s)
        rec = c.astream(async_openai_style_chunks(["x", "y", "z"]),
                        model="m", request={"q": 2}, seed=9)
        seen = []
        async for chunk in rec:
            seen.append(default_extract_text(chunk))
        assert "".join(seen) == "xyz"
        assert rec.text == "xyz"
        assert rec.node.payload["response"]["streamed"] is True
        s.close()

    asyncio.run(main())


def test_stream_error_commits_nothing(tmp_path, signer, key_env):
    def bad_stream():
        yield {"choices": [{"delta": {"content": "partial"}}]}
        raise RuntimeError("connection reset")

    s = _session(str(tmp_path / "p"), signer)
    rec = StreamRecorder(s, bad_stream(), model="m", request={}, seed=1)
    with pytest.raises(RuntimeError):
        list(rec)
    assert rec.node is None
    assert list(s.log()) == []
    s.close()


def test_extract_text_variants():
    assert default_extract_text("hi") == "hi"
    assert default_extract_text(b"hi") == "hi"
    assert default_extract_text({"content": "c"}) == "c"
    assert default_extract_text(
        {"choices": [{"delta": {"content": "d"}}]}) == "d"
    assert default_extract_text(None) == ""
    assert default_is_done("[DONE]") is True
    assert default_is_done({"done": True}) is True
    assert default_is_done({"choices": [{"delta": {"content": "x"}}]}) is False

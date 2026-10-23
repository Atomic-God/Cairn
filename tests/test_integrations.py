"""Unit tests for Cairn framework integrations (OpenAI, LangChain, Anthropic, CrewAI)."""

import asyncio
from types import SimpleNamespace
import pytest

from cairn import Session, signing
from cairn.integrations.openai import wrap_openai, wrap_tool
from cairn.integrations.langchain import CairnCallbackHandler
from cairn.integrations.anthropic import wrap_anthropic
from cairn.integrations.crewai import CairnCrewCallback


@pytest.fixture()
def keypair():
    return signing.generate_keypair()


def _session(path, keypair, **kw):
    sk, _ = keypair
    return Session.create(path, actor="tester", secret_key=sk, **kw)


# ---------------------------------------------------------------------------
# Mock classes for testing without external SDK dependencies
# ---------------------------------------------------------------------------

class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class MockToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.function = MockFunction(name, arguments)


class MockMessage:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class MockChoice:
    def __init__(self, message, finish_reason="stop"):
        self.message = message
        self.finish_reason = finish_reason


class MockUsage:
    def __init__(self, prompt=10, completion=20):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion


class MockChatCompletion:
    def __init__(self, content, tool_calls=None, prompt=10, completion=20):
        self.choices = [MockChoice(MockMessage(content, tool_calls))]
        self.usage = MockUsage(prompt, completion)


class MockChunkDelta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class MockChunkChoice:
    def __init__(self, delta):
        self.delta = delta


class MockChunk:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [MockChunkChoice(MockChunkDelta(content, tool_calls))]


class MockCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return [MockChunk("Hello "), MockChunk("world!")]
        return MockChatCompletion("Hello world!", prompt=15, completion=25)


class MockAsyncCompletions:
    def __init__(self):
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def _stream():
                for c in [MockChunk("Async "), MockChunk("stream!")]:
                    yield c
            return _stream()
        return MockChatCompletion("Async response", prompt=30, completion=40)


class MockOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=MockCompletions())


class MockAsyncOpenAIClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=MockAsyncCompletions())


# ---------------------------------------------------------------------------
# OpenAI Tests
# ---------------------------------------------------------------------------

def test_openai_sync_completion(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    raw_client = MockOpenAIClient()
    client = wrap_openai(raw_client, session=s)

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Hello"}],
        seed=123,
    )

    assert resp.choices[0].message.content == "Hello world!"
    assert s.head is not None
    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["model"] == "gpt-4o"
    assert node.payload["response"]["content"] == "Hello world!"
    assert node.payload["usage"]["total_tokens"] == 40


def test_openai_sync_streaming(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    raw_client = MockOpenAIClient()
    client = wrap_openai(raw_client, session=s)

    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Stream me"}],
        stream=True,
    )

    chunks = list(stream)
    assert len(chunks) == 2
    assert s.head is not None
    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["response"]["content"] == "Hello world!"
    assert node.payload["response"]["streamed"] is True


def test_openai_async_completion(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    raw_client = MockAsyncOpenAIClient()
    client = wrap_openai(raw_client, session=s)

    async def _run():
        return await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Async test"}],
        )

    resp = asyncio.run(_run())
    assert resp.choices[0].message.content == "Async response"
    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["model"] == "gpt-4o-mini"
    assert node.payload["usage"]["total_tokens"] == 70


def test_openai_async_streaming(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    raw_client = MockAsyncOpenAIClient()
    client = wrap_openai(raw_client, session=s)

    async def _run():
        stream = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Async stream"}],
            stream=True,
        )
        collected = []
        async for chunk in stream:
            collected.append(chunk)
        return collected

    chunks = asyncio.run(_run())
    assert len(chunks) == 2
    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["response"]["content"] == "Async stream!"
    assert node.payload["response"]["streamed"] is True


def test_wrap_tool_sync_success_and_error(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)

    @wrap_tool(session=s, name="calculator")
    def calc(a, b):
        if b == 0:
            raise ZeroDivisionError("division by zero")
        return a / b

    res = calc(10, 2)
    assert res == 5.0

    nodes = list(s.log(limit=2))
    assert nodes[0].kind == "tool_result"
    assert nodes[0].payload["ok"] is True
    assert nodes[0].payload["result"] == 5.0
    assert nodes[1].kind == "tool_call"
    assert nodes[1].payload["arguments"]["a"] == 10

    with pytest.raises(ZeroDivisionError):
        calc(10, 0)

    err_nodes = list(s.log(limit=2))
    assert err_nodes[0].kind == "tool_result"
    assert err_nodes[0].payload["ok"] is False
    assert "division by zero" in err_nodes[0].payload["result"]["error"]


def test_wrap_tool_async(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)

    @wrap_tool(session=s, name="async_fetch")
    async def fetch_data(key: str):
        await asyncio.sleep(0.001)
        return {"data": f"value_for_{key}"}

    res = asyncio.run(fetch_data("alpha"))
    assert res == {"data": "value_for_alpha"}

    nodes = list(s.log(limit=2))
    assert nodes[0].kind == "tool_result"
    assert nodes[0].payload["result"] == {"data": "value_for_alpha"}
    assert nodes[1].kind == "tool_call"


# ---------------------------------------------------------------------------
# LangChain Tests
# ---------------------------------------------------------------------------

def test_langchain_callback_chat_and_tools(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    handler = CairnCallbackHandler(s)

    # 1. Simulate Chat Model Start + End
    run_id = "run-123"
    handler.on_chat_model_start(
        serialized={"name": "ChatOpenAI"},
        messages=[[SimpleNamespace(type="human", content="Summarize docs")]],
        run_id=run_id,
        metadata={"ls_model_name": "gpt-4o"},
    )

    generation = SimpleNamespace(
        text="Summary here",
        message=SimpleNamespace(content="Summary here", tool_calls=None),
    )
    llm_result = SimpleNamespace(
        generations=[[generation]],
        llm_output={"token_usage": {"prompt_tokens": 12, "completion_tokens": 18, "total_tokens": 30}},
    )
    handler.on_llm_end(llm_result, run_id=run_id)

    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["model"] == "gpt-4o"
    assert node.payload["response"]["content"] == "Summary here"
    assert node.payload["usage"]["total_tokens"] == 30

    # 2. Simulate Tool Start + End
    tool_run_id = "tool-456"
    handler.on_tool_start(
        serialized={"name": "search_db"},
        input_str='{"query": "revenue 2026"}',
        run_id=tool_run_id,
    )
    call_node = s.repo.get_node(s.head)
    assert call_node.kind == "tool_call"
    assert call_node.payload["name"] == "search_db"
    assert call_node.payload["arguments"] == {"query": "revenue 2026"}

    handler.on_tool_end({"results": ["Q1: $10M", "Q2: $12M"]}, run_id=tool_run_id)
    res_node = s.repo.get_node(s.head)
    assert res_node.kind == "tool_result"
    assert res_node.payload["ok"] is True
    assert len(res_node.payload["result"]["results"]) == 2

    # 3. Simulate Tool Error
    tool_err_id = "tool-789"
    handler.on_tool_start(
        serialized={"name": "db_write"},
        input_str="raw_unparsed_string",
        run_id=tool_err_id,
    )
    handler.on_tool_error(RuntimeError("disk full"), run_id=tool_err_id)
    err_node = s.repo.get_node(s.head)
    assert err_node.kind == "tool_result"
    assert err_node.payload["ok"] is False
    assert "disk full" in err_node.payload["result"]["error"]


# ---------------------------------------------------------------------------
# Anthropic Tests
# ---------------------------------------------------------------------------

class MockTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class MockToolUseBlock:
    def __init__(self, id, name, input_dict):
        self.type = "tool_use"
        self.id = id
        self.name = name
        self.input = input_dict


class MockAnthropicResponse:
    def __init__(self, blocks, input_tokens=50, output_tokens=25):
        self.content = blocks
        self.stop_reason = "end_turn"
        self.usage = SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens)


class MockAnthropicMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        blocks = [
            MockTextBlock("Thinking... "),
            MockToolUseBlock("call-99", "check_weather", {"city": "Tokyo"}),
        ]
        return MockAnthropicResponse(blocks)


class MockAnthropicClient:
    def __init__(self):
        self.messages = MockAnthropicMessages()


def test_anthropic_sync_completion(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    raw_client = MockAnthropicClient()
    client = wrap_anthropic(raw_client, session=s)

    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        system="Be concise",
    )

    assert len(resp.content) == 2
    node = s.repo.get_node(s.head)
    assert node.kind == "inference"
    assert node.payload["model"] == "claude-3-5-sonnet-20241022"
    assert node.payload["response"]["content"] == "Thinking... "
    assert node.payload["response"]["tool_calls"][0]["name"] == "check_weather"
    assert node.payload["usage"]["total_tokens"] == 75


# ---------------------------------------------------------------------------
# CrewAI Tests
# ---------------------------------------------------------------------------

def test_crewai_callback(tmp_path, keypair):
    s = _session(str(tmp_path / "repo"), keypair)
    cb = CairnCrewCallback(s)

    # Step callback
    step = SimpleNamespace(agent="researcher", text="Searching arxiv")
    cb.on_step(step)

    step_node = s.repo.get_node(s.head)
    assert step_node.kind == "observation"
    assert "researcher" in step_node.payload["source"]

    # Task callback
    task = SimpleNamespace(description="Draft quarterly financial report", raw="Revenue grew 22%")
    cb.on_task(task)

    task_node = s.repo.get_node(s.head)
    assert task_node.kind == "tool_result"
    assert task_node.payload["result"]["output"] == "Revenue grew 22%"

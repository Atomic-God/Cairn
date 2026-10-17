"""Async and streaming instrumentation for Cairn.

Modern agent loops are async and streaming. This module provides:

- :class:`AsyncAgentClient` — a drop-in async counterpart to
  :class:`cairn.session.AgentClient`. It awaits your async (or sync) LLM and
  tool callables, then records the completed step exactly like the sync
  client. Recording itself stays synchronous: it is local, fast (~50ms), and
  never touches the network.
- :class:`StreamRecorder` / :class:`AsyncStreamRecorder` — wrap a token
  stream (e.g. OpenAI SSE chunks). Chunks are yielded downstream untouched
  (zero perceived delay); when the stream completes, the fully assembled
  message is committed as a single ``inference`` node. Cairn commits *per
  message*, never per token, so streaming adds no per-token overhead.

``Session`` also supports ``async with`` directly.
"""

from __future__ import annotations

import inspect
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, Iterable, Iterator

from .session import Session


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


class AsyncAgentClient:
    """Async record-and-execute client.

    Mirrors :class:`cairn.session.AgentClient`; ``llm`` and ``tools`` may be
    async callables (sync callables are also accepted).
    """

    def __init__(
        self,
        session: Session,
        *,
        llm: Callable[..., Any] | None = None,
        tools: Dict[str, Callable[..., Any]] | None = None,
    ):
        self.session = session
        self._llm = llm
        self._tools = dict(tools or {})

    # -- record-only steps (no I/O) ------------------------------------
    def system(self, text: str):
        return self.session.system(text)

    def user(self, text: str):
        return self.session.user(text)

    def note(self, text: str):
        return self.session.note(text)

    def observation(self, source: str, data: Any):
        return self.session.observation(source, data)

    # -- world-involving steps ------------------------------------------
    async def infer(
        self,
        request: dict,
        model: str = "unknown",
        *,
        seed: int | None = None,
        usage: dict | None = None,
    ) -> Any:
        if self._llm is None:
            raise RuntimeError("no LLM backend configured for recording client")
        response = await _maybe_await(self._llm(request, model=model, seed=seed))
        self.session.inference(request, response, model, seed=seed, usage=usage)
        return response

    async def tool(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        import secrets as _secrets

        call_id = _secrets.token_hex(8)
        self.session.tool_call(name, arguments, call_id=call_id)
        try:
            result = await _maybe_await(self._tools[name](arguments))
        except Exception as exc:
            self.session.tool_result(name, {"error": str(exc)},
                                     call_id=call_id, ok=False)
            raise
        self.session.tool_result(name, result, call_id=call_id, ok=True)
        return result

    def stream(
        self,
        chunks: Iterable,
        *,
        model: str,
        request: dict,
        seed: int | None = None,
        **kwargs: Any,
    ) -> "StreamRecorder":
        """Wrap a sync chunk iterator; see :class:`StreamRecorder`."""
        return StreamRecorder(self.session, chunks, model=model,
                              request=request, seed=seed, **kwargs)

    def astream(
        self,
        chunks: AsyncIterator,
        *,
        model: str,
        request: dict,
        seed: int | None = None,
        **kwargs: Any,
    ) -> "AsyncStreamRecorder":
        """Wrap an async chunk iterator; see :class:`AsyncStreamRecorder`."""
        return AsyncStreamRecorder(self.session, chunks, model=model,
                                   request=request, seed=seed, **kwargs)


# ---------------------------------------------------------------------------
# Chunk helpers
# ---------------------------------------------------------------------------

def default_extract_text(chunk: Any) -> str:
    """Pull display text out of a stream chunk.

    Handles OpenAI-style ``{"choices": [{"delta": {"content": ...}}]}``
    dicts, plain strings/bytes, and ``{"content": ...}`` dicts.
    """
    if chunk is None:
        return ""
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, bytes):
        return chunk.decode("utf-8", "replace")
    if isinstance(chunk, dict):
        choices = chunk.get("choices")
        if choices:
            delta = choices[0].get("delta", {})
            return delta.get("content") or ""
        for key in ("content", "text", "delta"):
            value = chunk.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, dict):
                inner = value.get("content")
                if isinstance(inner, str):
                    return inner
        return ""
    return str(chunk)


def default_is_done(chunk: Any) -> bool:
    """Detect end-of-stream control chunks.

    Note: a chunk carrying ``finish_reason`` may still carry trailing text,
    so it is treated as content — the stream ends when the iterator is
    exhausted (or on an explicit ``[DONE]`` / ``{"done": true}`` signal).
    """
    if chunk == "[DONE]":
        return True
    if isinstance(chunk, dict) and chunk.get("done") is True:
        return True
    return False


class StreamRecorder:
    """Zero-copy streaming interceptor (synchronous).

    Wraps ``chunks`` (any iterable of SSE/token chunks), yields each chunk
    downstream unchanged, and — when the iterator is exhausted — commits one
    ``inference`` node with the fully assembled message. If the stream raises,
    nothing is committed and the exception propagates.
    """

    def __init__(
        self,
        session: Session,
        chunks: Iterable,
        *,
        model: str,
        request: dict,
        seed: int | None = None,
        usage: dict | None = None,
        extract_text: Callable[[Any], str] | None = None,
        is_done: Callable[[Any], bool] | None = None,
    ):
        self.session = session
        self._chunks = chunks
        self._model = model
        self._request = request
        self._seed = seed
        self._usage = usage
        self._extract = extract_text or default_extract_text
        self._is_done = is_done or default_is_done
        self.node = None

    def __iter__(self) -> Iterator:
        parts: list[str] = []
        for chunk in self._chunks:
            if self._is_done(chunk):
                break
            parts.append(self._extract(chunk))
            yield chunk
        text = "".join(parts)
        payload: Dict[str, Any] = {"content": text, "model": self._model,
                                   "streamed": True}
        self.node = self.session.inference(
            self._request, payload, self._model,
            seed=self._seed, usage=self._usage,
        )

    @property
    def text(self) -> str | None:
        """Assembled text, available after the stream is exhausted."""
        if self.node is None:
            return None
        response = self.node.payload.get("response") or {}
        return response.get("content")


class AsyncStreamRecorder:
    """Zero-copy streaming interceptor (asynchronous). See StreamRecorder."""

    def __init__(
        self,
        session: Session,
        chunks: AsyncIterator,
        *,
        model: str,
        request: dict,
        seed: int | None = None,
        usage: dict | None = None,
        extract_text: Callable[[Any], str] | None = None,
        is_done: Callable[[Any], bool] | None = None,
    ):
        self.session = session
        self._chunks = chunks
        self._model = model
        self._request = request
        self._seed = seed
        self._usage = usage
        self._extract = extract_text or default_extract_text
        self._is_done = is_done or default_is_done
        self.node = None

    def __aiter__(self) -> AsyncIterator:
        return self._gen()

    async def _gen(self) -> AsyncIterator:
        parts: list[str] = []
        async for chunk in self._chunks:
            if self._is_done(chunk):
                break
            parts.append(self._extract(chunk))
            yield chunk
        text = "".join(parts)
        payload: Dict[str, Any] = {"content": text, "model": self._model,
                                   "streamed": True}
        self.node = self.session.inference(
            self._request, payload, self._model,
            seed=self._seed, usage=self._usage,
        )

    @property
    def text(self) -> str | None:
        """Assembled text, available after the stream is exhausted."""
        if self.node is None:
            return None
        response = self.node.payload.get("response") or {}
        return response.get("content")

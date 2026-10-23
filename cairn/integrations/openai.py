"""OpenAI SDK integration for Cairn.

Provides zero-friction, one-line instrumentation for OpenAI clients:
- :func:`wrap_openai` — wrap sync ``OpenAI`` or async ``AsyncOpenAI`` clients
- :func:`wrap_tool` — decorator / wrapper for agent tools that records tool calls & results

Usage::

    from openai import OpenAI
    from cairn import Session
    from cairn.integrations.openai import wrap_openai, wrap_tool

    session = Session.create("./agent-repo", actor="researcher")
    client = wrap_openai(OpenAI(), session=session)

    # 1. Normal chat completion (sync or async)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Analyze Q3 revenue"}],
        seed=42,
    )
    # The prompt, inference, model, seed, and token usage are now an immutable,
    # signed node in the Cairn Merkle DAG.

    # 2. Streaming is automatically intercepted and committed at end-of-stream:
    stream = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Stream poem"}],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

    # 3. Tool execution recording:
    @wrap_tool(session=session)
    def fetch_metrics(quarter: str):
        return {"revenue": 12500000}
"""

from __future__ import annotations

import functools
import inspect
import secrets
from typing import Any, Callable, Dict, Optional, Sequence, Union

from ..session import Session


def _extract_response_dict(resp: Any) -> Dict[str, Any]:
    """Normalize OpenAI ChatCompletion response into canonical dict."""
    if hasattr(resp, "model_dump"):
        dump = resp.model_dump()
        choices = dump.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        return {
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
        }
    if isinstance(resp, dict):
        choices = resp.get("choices", [])
        msg = choices[0].get("message", {}) if choices else {}
        return {
            "content": msg.get("content"),
            "tool_calls": msg.get("tool_calls"),
            "finish_reason": choices[0].get("finish_reason") if choices else None,
        }
    # Fallback duck-typing
    content = None
    tool_calls = None
    finish_reason = None
    choices = getattr(resp, "choices", None)
    if choices and len(choices) > 0:
        c0 = choices[0]
        finish_reason = getattr(c0, "finish_reason", None)
        msg = getattr(c0, "message", None)
        if msg:
            content = getattr(msg, "content", None)
            raw_tc = getattr(msg, "tool_calls", None)
            if raw_tc:
                tool_calls = []
                for tc in raw_tc:
                    fn = getattr(tc, "function", None)
                    tool_calls.append({
                        "id": getattr(tc, "id", None),
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    })
    return {"content": content, "tool_calls": tool_calls, "finish_reason": finish_reason}


def _extract_usage(resp: Any) -> Optional[Dict[str, int]]:
    """Extract token usage dict from completion response."""
    usage = getattr(resp, "usage", None)
    if usage is None and isinstance(resp, dict):
        usage = resp.get("usage")
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


class _StreamInterceptor:
    """Synchronous stream wrapper that commits an inference node upon completion."""

    def __init__(self, stream: Any, session: Session, request: dict, model: str, seed: Optional[int]):
        self._stream = stream
        self._session = session
        self._request = request
        self._model = model
        self._seed = seed
        self._assembled_content = []
        self._tool_calls_map = {}
        self._usage = None

    def __iter__(self):
        try:
            for chunk in self._stream:
                self._record_chunk(chunk)
                yield chunk
        finally:
            self._commit_assembled()

    def _record_chunk(self, chunk: Any):
        # Accumulate usage if present in chunk
        usage = getattr(chunk, "usage", None)
        if usage:
            self._usage = _extract_usage(chunk)
        choices = getattr(chunk, "choices", None)
        if not choices and isinstance(chunk, dict):
            choices = chunk.get("choices")
        if choices and len(choices) > 0:
            c0 = choices[0]
            delta = getattr(c0, "delta", None)
            if delta is None and isinstance(c0, dict):
                delta = c0.get("delta")
            if delta:
                content = getattr(delta, "content", None)
                if content is None and isinstance(delta, dict):
                    content = delta.get("content")
                if content:
                    self._assembled_content.append(content)
                tc_list = getattr(delta, "tool_calls", None)
                if tc_list is None and isinstance(delta, dict):
                    tc_list = delta.get("tool_calls")
                if tc_list:
                    for tc in tc_list:
                        idx = getattr(tc, "index", 0)
                        if isinstance(tc, dict):
                            idx = tc.get("index", 0)
                        if idx not in self._tool_calls_map:
                            self._tool_calls_map[idx] = {"id": "", "name": "", "arguments": ""}
                        entry = self._tool_calls_map[idx]
                        tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
                        if tc_id:
                            entry["id"] = tc_id
                        fn = getattr(tc, "function", None) or (tc.get("function") if isinstance(tc, dict) else None)
                        if fn:
                            name = getattr(fn, "name", None) or (fn.get("name") if isinstance(fn, dict) else None)
                            args = getattr(fn, "arguments", None) or (fn.get("arguments") if isinstance(fn, dict) else None)
                            if name:
                                entry["name"] += name
                            if args:
                                entry["arguments"] += args

    def _commit_assembled(self):
        content = "".join(self._assembled_content) or None
        tool_calls = list(self._tool_calls_map.values()) if self._tool_calls_map else None
        response_data = {
            "content": content,
            "tool_calls": tool_calls,
            "streamed": True,
        }
        self._session.inference(
            request=self._request,
            response=response_data,
            model=self._model,
            seed=self._seed,
            usage=self._usage,
        )


class _AsyncStreamInterceptor:
    """Asynchronous stream wrapper that commits an inference node upon completion."""

    def __init__(self, stream: Any, session: Session, request: dict, model: str, seed: Optional[int]):
        self._stream = stream
        self._session = session
        self._request = request
        self._model = model
        self._seed = seed
        self._sync_interceptor = _StreamInterceptor(stream, session, request, model, seed)

    async def __aiter__(self):
        try:
            async for chunk in self._stream:
                self._sync_interceptor._record_chunk(chunk)
                yield chunk
        finally:
            self._sync_interceptor._commit_assembled()


class _CompletionsWrapper:
    """Wraps client.chat.completions to record calls into Cairn."""

    def __init__(self, completions_target: Any, session: Session):
        self._target = completions_target
        self._session = session

    def create(self, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        seed = kwargs.get("seed")
        messages = kwargs.get("messages", [])
        stream = kwargs.get("stream", False)

        request_payload = {
            "messages": messages,
            "model": model,
        }
        for extra in ("temperature", "tools", "tool_choice", "response_format", "seed"):
            if extra in kwargs:
                request_payload[extra] = kwargs[extra]

        # Call real target
        result = self._target.create(*args, **kwargs)

        if inspect.isawaitable(result):
            # Async client
            async def _async_runner():
                resolved = await result
                if stream:
                    return _AsyncStreamInterceptor(resolved, self._session, request_payload, model, seed)
                # Non-streaming async response
                resp_dict = _extract_response_dict(resolved)
                usage = _extract_usage(resolved)
                self._session.inference(
                    request=request_payload,
                    response=resp_dict,
                    model=model,
                    seed=seed,
                    usage=usage,
                )
                return resolved

            return _async_runner()

        # Sync client
        if stream:
            return _StreamInterceptor(result, self._session, request_payload, model, seed)

        resp_dict = _extract_response_dict(result)
        usage = _extract_usage(result)
        self._session.inference(
            request=request_payload,
            response=resp_dict,
            model=model,
            seed=seed,
            usage=usage,
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class _ChatWrapper:
    def __init__(self, chat_target: Any, session: Session):
        self._target = chat_target
        self._session = session
        self.completions = _CompletionsWrapper(chat_target.completions, session)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class CairnOpenAIProxy:
    """Transparent proxy around an OpenAI or AsyncOpenAI client instance."""

    def __init__(self, client: Any, session: Session):
        self._client = client
        self._session = session
        self.chat = _ChatWrapper(client.chat, session)

    @property
    def cairn_session(self) -> Session:
        """The underlying Cairn session."""
        return self._session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(client: Any, session: Session) -> CairnOpenAIProxy:
    """Wrap an OpenAI or AsyncOpenAI client to automatically record to Cairn.

    Parameters:
        client: An instance of ``openai.OpenAI`` or ``openai.AsyncOpenAI``
        session: The active :class:`cairn.session.Session`

    Returns:
        A proxy object with identical API to the original client.
    """
    return CairnOpenAIProxy(client, session)


def wrap_tool(
    func: Optional[Callable] = None,
    *,
    session: Optional[Session] = None,
    name: Optional[str] = None,
) -> Any:
    """Decorator or wrapper to record agent tool execution in Cairn.

    Can be used as:
        @wrap_tool(session=session)
        def my_tool(x: int):
            ...

    Or:
        recorded_tool = wrap_tool(my_tool, session=session)
    """
    if session is None and callable(func) and not inspect.isfunction(func):
        raise ValueError("session must be provided to wrap_tool")

    def decorator(fn: Callable) -> Callable:
        tool_name = name or getattr(fn, "__name__", "unnamed_tool")

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_id = secrets.token_hex(8)
                arguments = kwargs.copy()
                if args:
                    arguments["_args"] = list(args)
                session.tool_call(tool_name, arguments, call_id=call_id)
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    session.tool_result(tool_name, {"error": str(exc)}, call_id=call_id, ok=False)
                    raise
                session.tool_result(tool_name, result, call_id=call_id, ok=True)
                return result

            return async_wrapper
        else:
            @functools.wraps(fn)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                call_id = secrets.token_hex(8)
                arguments = kwargs.copy()
                if args:
                    arguments["_args"] = list(args)
                session.tool_call(tool_name, arguments, call_id=call_id)
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    session.tool_result(tool_name, {"error": str(exc)}, call_id=call_id, ok=False)
                    raise
                session.tool_result(tool_name, result, call_id=call_id, ok=True)
                return result

            return sync_wrapper

    if func is not None:
        return decorator(func)
    return decorator

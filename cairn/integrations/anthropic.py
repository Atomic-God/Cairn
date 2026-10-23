"""Anthropic Claude SDK integration for Cairn.

Provides zero-friction, one-line instrumentation for Anthropic clients::

    from anthropic import Anthropic
    from cairn import Session
    from cairn.integrations.anthropic import wrap_anthropic

    session = Session.create("./agent-repo", actor="claude-agent")
    client = wrap_anthropic(Anthropic(), session=session)

    # 1. Normal message creation
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[{"role": "user", "content": "Analyze system logs"}],
    )

    # Every message, tool request, and token count is committed to Cairn.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from ..session import Session


def _extract_anthropic_response(resp: Any) -> Dict[str, Any]:
    content_blocks = []
    tool_calls = []

    raw_content = getattr(resp, "content", None)
    if isinstance(raw_content, list):
        for block in raw_content:
            b_type = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if b_type == "text":
                text = getattr(block, "text", "") or (block.get("text", "") if isinstance(block, dict) else "")
                content_blocks.append(text)
            elif b_type == "tool_use":
                t_id = getattr(block, "id", None) or (block.get("id") if isinstance(block, dict) else None)
                t_name = getattr(block, "name", None) or (block.get("name") if isinstance(block, dict) else None)
                t_input = getattr(block, "input", None) or (block.get("input") if isinstance(block, dict) else None)
                tool_calls.append({"id": t_id, "name": t_name, "arguments": t_input})

    return {
        "content": "".join(content_blocks) if content_blocks else None,
        "tool_calls": tool_calls if tool_calls else None,
        "stop_reason": getattr(resp, "stop_reason", None),
    }


def _extract_anthropic_usage(resp: Any) -> Optional[Dict[str, int]]:
    usage = getattr(resp, "usage", None)
    if not usage:
        return None
    input_tokens = getattr(usage, "input_tokens", 0)
    output_tokens = getattr(usage, "output_tokens", 0)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


class _MessagesWrapper:
    def __init__(self, target_messages: Any, session: Session):
        self._target = target_messages
        self._session = session

    def create(self, *args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages", [])
        system = kwargs.get("system")
        tools = kwargs.get("tools")

        request_payload = {
            "messages": messages,
            "model": model,
        }
        if system:
            request_payload["system"] = system
        if tools:
            request_payload["tools"] = tools

        result = self._target.create(*args, **kwargs)

        if inspect.isawaitable(result):
            async def _async_runner():
                resolved = await result
                resp_dict = _extract_anthropic_response(resolved)
                usage = _extract_anthropic_usage(resolved)
                self._session.inference(
                    request=request_payload,
                    response=resp_dict,
                    model=model,
                    usage=usage,
                )
                return resolved

            return _async_runner()

        resp_dict = _extract_anthropic_response(result)
        usage = _extract_anthropic_usage(result)
        self._session.inference(
            request=request_payload,
            response=resp_dict,
            model=model,
            usage=usage,
        )
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)


class CairnAnthropicProxy:
    """Transparent proxy around an Anthropic client instance."""

    def __init__(self, client: Any, session: Session):
        self._client = client
        self._session = session
        self.messages = _MessagesWrapper(client.messages, session)

    @property
    def cairn_session(self) -> Session:
        return self._session

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_anthropic(client: Any, session: Session) -> CairnAnthropicProxy:
    """Wrap an Anthropic or AsyncAnthropic client to automatically record to Cairn."""
    return CairnAnthropicProxy(client, session)

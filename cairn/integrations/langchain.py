"""LangChain and LangGraph integration for Cairn.

Provides zero-overhead, cryptographic state graph recording for any
LangChain or LangGraph execution using a standard callback handler::

    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from cairn import Session
    from cairn.integrations.langchain import CairnCallbackHandler

    session = Session.create("./agent-repo", actor="langchain-agent")
    handler = CairnCallbackHandler(session)

    # 1. Use directly with ChatModel / LLM:
    llm = ChatOpenAI(model="gpt-4o", callbacks=[handler])

    # 2. Or pass to AgentExecutor / LCEL chain invocation:
    agent_executor.invoke({"input": "What is 42 * 1337?"}, config={"callbacks": [handler]})

    # Every model call, prompt, tool execution, and output is now
    # content-addressed and replayable in Cairn.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from ..session import Session

# Try importing BaseCallbackHandler from langchain_core, fall back gracefully
try:
    from langchain_core.callbacks.base import BaseCallbackHandler
except ImportError:
    class BaseCallbackHandler:  # type: ignore
        """Fallback stub when langchain-core is not installed."""
        pass


class CairnCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that records executions into a Cairn Session."""

    def __init__(self, session: Session, *, actor: Optional[str] = None):
        super().__init__()
        self.session = session
        self.actor = actor or session.actor
        self._current_run_map: Dict[str, Dict[str, Any]] = {}
        self._tool_run_map: Dict[str, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Capture the incoming chat prompt and model parameters."""
        model_name = "unknown"
        if metadata and "ls_model_name" in metadata:
            model_name = metadata["ls_model_name"]
        elif kwargs.get("invocation_params"):
            model_name = kwargs["invocation_params"].get("model_name") or kwargs["invocation_params"].get("model") or "unknown"

        # Serialize messages
        flat_messages = []
        for msg_group in messages:
            for m in msg_group:
                role = getattr(m, "type", "unknown")
                if role == "human":
                    role = "user"
                elif role == "ai":
                    role = "assistant"
                content = getattr(m, "content", str(m))
                flat_messages.append({"role": role, "content": content})

        self._current_run_map[str(run_id)] = {
            "model": model_name,
            "request": {
                "messages": flat_messages,
                "invocation_params": kwargs.get("invocation_params", {}),
            },
        }

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Capture standard completion prompts."""
        model_name = "unknown"
        if kwargs.get("invocation_params"):
            model_name = kwargs["invocation_params"].get("model_name") or "unknown"

        self._current_run_map[str(run_id)] = {
            "model": model_name,
            "request": {"prompts": prompts},
        }

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Commit the LLM inference node to the Cairn DAG."""
        run_info = self._current_run_map.pop(str(run_id), {})
        model_name = run_info.get("model", "unknown")
        request = run_info.get("request", {})

        # Extract output text and tool calls
        generations = getattr(response, "generations", [])
        content = None
        tool_calls = None
        if generations and len(generations) > 0 and len(generations[0]) > 0:
            gen = generations[0][0]
            content = getattr(gen, "text", None)
            msg = getattr(gen, "message", None)
            if msg:
                content = getattr(msg, "content", content)
                raw_tc = getattr(msg, "tool_calls", None)
                if raw_tc:
                    tool_calls = raw_tc

        # Token usage
        usage = None
        llm_output = getattr(response, "llm_output", None)
        if isinstance(llm_output, dict):
            token_usage = llm_output.get("token_usage")
            if isinstance(token_usage, dict):
                usage = {
                    "prompt_tokens": token_usage.get("prompt_tokens", 0),
                    "completion_tokens": token_usage.get("completion_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                }

        self.session.inference(
            request=request,
            response={"content": content, "tool_calls": tool_calls},
            model=model_name,
            usage=usage,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Record failed inference."""
        run_info = self._current_run_map.pop(str(run_id), {})
        self.session.inference(
            request=run_info.get("request", {}),
            response={"error": str(error), "failed": True},
            model=run_info.get("model", "unknown"),
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool invocation."""
        tool_name = serialized.get("name", "unknown_tool")
        call_id = str(run_id)

        # Parse arguments if JSON, else keep as string or dict
        args: Any = input_str
        if isinstance(input_str, str):
            try:
                args = json.loads(input_str)
            except Exception:
                args = {"input": input_str}
        elif not isinstance(args, dict):
            args = {"input": args}

        self._tool_run_map[call_id] = {"name": tool_name, "args": args}
        self.session.tool_call(tool_name, args, call_id=call_id)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool success result."""
        call_id = str(run_id)
        info = self._tool_run_map.pop(call_id, {})
        tool_name = info.get("name", "tool")

        result = output
        if not isinstance(result, (dict, list, str, int, float, bool, type(None))):
            result = str(output)

        self.session.tool_result(tool_name, result, call_id=call_id, ok=True)

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: Any,
        parent_run_id: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Capture tool failure."""
        call_id = str(run_id)
        info = self._tool_run_map.pop(call_id, {})
        tool_name = info.get("name", "tool")

        self.session.tool_result(
            tool_name,
            {"error": str(error)},
            call_id=call_id,
            ok=False,
        )

"""Instrumenting OpenAI agents with Cairn in one line.

Demonstrates:
1. `wrap_openai(client, session)`: Intercepts completions & streaming, records
   prompt, response, model, seed, and token counts to the Merkle DAG.
2. `@wrap_tool(session=...)`: Intercepts tool calls and tool results with
   zero boilerplate.
3. Cryptographic audit: `cairn verify` verifies the trace mathematically.

Usage:
    pip install openai cairn-vcs
    export OPENAI_API_KEY="sk-..."
    python examples/openai_wrapper.py
"""

from __future__ import annotations

import json
import os
import sys

from cairn import Session
from cairn.integrations.openai import wrap_openai, wrap_tool

try:
    from openai import OpenAI
except ImportError:
    print("This example requires the `openai` package: pip install openai")
    sys.exit(0)


def run_agent():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY to run against real OpenAI API.")
        print("Showing pattern structure below:")

    # 1. Open or create a Cairn session
    with Session.create("./agent-cairn-repo", actor="support-specialist") as s:

        # 2. Wrap tools so every invocation is logged as a signed node
        @wrap_tool(session=s, name="lookup_order")
        def lookup_order(order_id: str) -> dict:
            """Look up customer order status."""
            return {
                "order_id": order_id,
                "status": "shipped",
                "estimated_delivery": "Tomorrow by 5 PM",
            }

        # 3. One-line client instrumentation
        client = wrap_openai(OpenAI(api_key=api_key or "sk-mock"), session=s)

        tools_spec = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_order",
                    "description": "Look up order tracking and delivery status",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order_id": {"type": "string"},
                        },
                        "required": ["order_id"],
                    },
                },
            }
        ]

        messages = [
            {"role": "system", "content": "You are a customer support agent. Be helpful and concise."},
            {"role": "user", "content": "Where is my order ORD-9921?"},
        ]

        if not api_key:
            # Simulated demonstration
            s.system("You are a customer support agent. Be helpful and concise.")
            s.user("Where is my order ORD-9921?")
            lookup_order("ORD-9921")
            print("Recorded sample run to ./agent-cairn-repo!")
            print("Run `cairn log` to view the Merkle DAG.")
            return

        # Real OpenAI execution:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools_spec,
            seed=42,
        )

        msg = resp.choices[0].message
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments)
                if fn_name == "lookup_order":
                    tool_res = lookup_order(**args)
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_res),
                    })

            # Final answer
            final_resp = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                seed=42,
            )
            print("Agent reply:", final_resp.choices[0].message.content)


if __name__ == "__main__":
    run_agent()

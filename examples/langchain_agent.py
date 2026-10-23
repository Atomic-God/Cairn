"""LangChain agent execution instrumented with Cairn in one line.

Demonstrates:
1. `CairnCallbackHandler(session)`: Passed to any LangChain Runnable, ChatModel,
   or AgentExecutor.
2. Automatically captures prompts, model inferences, tool calls, and outputs
   as signed nodes in the Cairn DAG.

Usage:
    pip install langchain langchain-openai cairn-ai
    export OPENAI_API_KEY="sk-..."
    python examples/langchain_agent.py
"""

from __future__ import annotations

import os
import sys

from cairn import Session
from cairn.integrations.langchain import CairnCallbackHandler

try:
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_tool_calling_agent, AgentExecutor
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    print("This example requires langchain: pip install langchain langchain-openai")
    sys.exit(0)


@tool
def calculate_vat(amount: float, country_code: str) -> float:
    """Calculate VAT rate for an invoice."""
    rates = {"US": 0.0, "UK": 0.20, "DE": 0.19, "IN": 0.18}
    return round(amount * rates.get(country_code.upper(), 0.15), 2)


def run_langchain():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Note: Set OPENAI_API_KEY to run live against OpenAI.")
        return

    # 1. Initialize Cairn session
    with Session.create("./langchain-cairn-repo", actor="billing-agent") as session:
        # 2. Attach Cairn callback handler
        handler = CairnCallbackHandler(session)

        llm = ChatOpenAI(model="gpt-4o", temperature=0, callbacks=[handler])
        tools = [calculate_vat]

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an automated corporate billing auditor."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, tools, prompt)
        executor = AgentExecutor(agent=agent, tools=tools, callbacks=[handler])

        # 3. Execute agent
        result = executor.invoke({
            "input": "Calculate the VAT for an invoice of 5000 EUR in Germany."
        })

        print("LangChain Result:", result["output"])
        print("\nTrace recorded! Run:")
        print("  cairn log --oneline")
        print("  cairn verify")


if __name__ == "__main__":
    run_langchain()

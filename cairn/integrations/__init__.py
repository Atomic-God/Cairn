"""Cairn Framework Integrations.

Provides zero-friction, one-line instrumentation for the most popular
AI agent frameworks and LLM SDKs:

- :mod:`cairn.integrations.openai` — wrap sync & async OpenAI clients, stream interceptor, tool decorator
- :mod:`cairn.integrations.langchain` — CairnCallbackHandler for LangChain, LangGraph, and LCEL
- :mod:`cairn.integrations.anthropic` — wrap sync & async Anthropic Claude clients
- :mod:`cairn.integrations.crewai` — step & task callbacks for CrewAI agents

All integrations maintain Cairn's zero-dependency core: third-party framework
packages are imported lazily and only required if that integration is used.
"""

from .anthropic import wrap_anthropic
from .crewai import CairnCrewCallback
from .langchain import CairnCallbackHandler
from .openai import wrap_openai, wrap_tool

__all__ = [
    "CairnCallbackHandler",
    "CairnCrewCallback",
    "wrap_anthropic",
    "wrap_openai",
    "wrap_tool",
]

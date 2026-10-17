"""Cairn: the content-addressed state graph for AI agent executions.

Every agent run becomes an append-only DAG of immutable, signed,
content-addressed nodes — time-travel debugging, deterministic replay, and
cryptographic audit trails, with zero dependencies.
"""

from .aio import AsyncAgentClient, AsyncStreamRecorder, StreamRecorder
from .export import export_bundle, import_bundle
from .node import Node, compute_node_id
from .replay import (
    ReplayClient,
    ReplayError,
    ReplayNetworkBlockedError,
    ReplayReport,
    network_guard,
    run as replay_run,
)
from .session import AgentClient, Session
from .store import Repo
from .verify import verify

__version__ = "0.2.0"

__all__ = [
    "AgentClient",
    "AsyncAgentClient",
    "AsyncStreamRecorder",
    "Node",
    "Repo",
    "ReplayClient",
    "ReplayError",
    "ReplayNetworkBlockedError",
    "ReplayReport",
    "Session",
    "StreamRecorder",
    "compute_node_id",
    "export_bundle",
    "import_bundle",
    "network_guard",
    "replay_run",
    "verify",
]

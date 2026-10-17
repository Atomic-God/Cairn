"""Deterministic replay: re-execute an agent from its recorded trace.

The replay protocol is what turns a Cairn trace from a log into *infrastructure*:

1. The recorded trace is a linear chain of signed, content-addressed nodes.
2. :func:`run` re-executes a user-supplied agent function against that trace.
   Every outbound call the agent makes — model inference, tool invocation —
   is intercepted: instead of touching the network, the harness feeds back the
   exact payload recorded during the original run.
3. Before returning a recorded payload, the harness fingerprints the agent's
   *request* and compares it to the recorded request. Any divergence raises
   :class:`ReplayDivergenceError` — the replay proves the agent behaved
   identically, or it fails loudly.
4. Replayed nodes are committed to a fresh branch. When the replay used the
   same actor, model set, and seeds, every replayed node id is **bit-identical**
   to the original — a mathematical proof of deterministic re-execution.

Cross-model re-basing (the spec's headline feature) works by passing a
different ``model`` to ``client.infer`` during replay: the request fingerprint
must still match, but the response comes from the recorded trace, so the new
model continues from the exact recorded state. The rebased branch then
diverges legitimately, with its own node ids.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Iterator

from . import codec, hash as hashmod
from .node import Node
from .session import Session
from .store import Repo

REQUEST_DOMAIN = b"cairn-request-v1\x00"


class ReplayError(Exception):
    """Base class for replay failures."""


class ReplayDivergenceError(ReplayError):
    """The agent's behavior diverged from the recorded trace."""


class ReplayExhaustedError(ReplayError):
    """The agent made more calls than the recorded trace contains."""


class ReplayIncompleteError(ReplayError):
    """The agent finished without consuming the whole trace."""


class ReplayNetworkBlockedError(ReplayError):
    """Agent code attempted network I/O during a guarded (offline) replay."""


@contextmanager
def network_guard() -> Iterator[None]:
    """Block all outbound socket connections for the duration of the block.

    Zero-dependency offline enforcement for deterministic replay: patches
    ``socket.socket.connect`` and ``socket.create_connection`` so any attempt
    at live network I/O raises :class:`ReplayNetworkBlockedError` instead of
    silently hitting the network (and spending API credits).

    This is process-global while active; use it around single-threaded
    replay runs (which is what :func:`run` does by default).
    """
    import socket

    orig_connect = socket.socket.connect
    orig_create_connection = socket.create_connection

    def _blocked_connect(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise ReplayNetworkBlockedError(
            "network access is blocked during deterministic replay; route "
            "all I/O through client.infer()/client.tool() so the harness "
            "can serve recorded payloads"
        )

    def _blocked_create_connection(*args: Any, **kwargs: Any) -> Any:
        raise ReplayNetworkBlockedError(
            "network access is blocked during deterministic replay; route "
            "all I/O through client.infer()/client.tool() so the harness "
            "can serve recorded payloads"
        )

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    socket.create_connection = _blocked_create_connection  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = orig_connect  # type: ignore[method-assign]
        socket.create_connection = orig_create_connection


def fingerprint_request(obj: Any) -> str:
    """Stable fingerprint of an outbound request (model/tool input)."""
    return hashmod.hexdigest(REQUEST_DOMAIN + codec.canonical(obj))


def linear_trace(repo: Repo, ref: str) -> list[Node]:
    """Return the trace oldest-first; require it to be a linear chain."""
    nodes = list(repo.walk(ref))
    nodes.reverse()
    if not nodes:
        raise ReplayError(f"no nodes reachable from {ref!r}")
    for prev, cur in zip(nodes, nodes[1:]):
        if list(cur.parents) != [prev.id]:
            raise ReplayError(
                "cannot replay a non-linear trace "
                f"(node {cur.short()} has parents "
                f"{[p[:8] for p in cur.parents]}); "
                "replay requires a linear chain — branch the trace first"
            )
    return nodes


@dataclass
class ReplayReport:
    ref: str
    branch: str
    nodes_replayed: int
    ids_match: bool
    rebased_models: list[str] = field(default_factory=list)
    result: Any = None

    @property
    def ok(self) -> bool:
        return self.ids_match and not self.rebased_models


class ReplayClient:
    """Drop-in replacement for :class:`cairn.session.AgentClient`.

    Intercepts outbound calls and serves the recorded payloads, verifying
    that the agent's requests are byte-identical to the originals.
    """

    def __init__(self, session: Session, trace: list[Node]):
        self.session = session
        self._trace = list(trace)
        self._pos = 0
        self._replayed_ids: list[str] = []
        self.rebased_models: list[str] = []

    # -- internals ----------------------------------------------------
    def _expect(self, kind: str) -> Node:
        if self._pos >= len(self._trace):
            raise ReplayExhaustedError(
                f"agent attempted a {kind!r} call beyond the end of the "
                f"recorded trace ({len(self._trace)} nodes)"
            )
        node = self._trace[self._pos]
        if node.kind != kind:
            raise ReplayDivergenceError(
                f"trace expects {node.kind!r} at step {self._pos} "
                f"({node.short()}), agent attempted {kind!r}"
            )
        return node

    def _advance(self, node: Node, payload: dict | None = None) -> Node:
        replayed = self.session._commit(
            node.kind,
            payload if payload is not None else node.payload,
            seed=node.seed,
            seq=node.seq,
            in_context=(node.kind != "note"),
        )
        self._replayed_ids.append(replayed.id)
        self._pos += 1
        return replayed

    def assert_exhausted(self) -> None:
        if self._pos < len(self._trace):
            remaining = self._trace[self._pos]
            raise ReplayIncompleteError(
                f"agent finished after {self._pos} of {len(self._trace)} "
                f"recorded steps (next: {remaining.kind!r} {remaining.short()})"
            )

    @property
    def replayed_ids(self) -> list[str]:
        return list(self._replayed_ids)

    # -- client API ---------------------------------------------------
    def system(self, text: str) -> Node:
        node = self._expect("system")
        if node.payload.get("text") != text:
            raise ReplayDivergenceError("system prompt differs from recording")
        return self._advance(node)

    def user(self, text: str) -> Node:
        node = self._expect("user")
        if node.payload.get("text") != text:
            raise ReplayDivergenceError("user message differs from recording")
        return self._advance(node)

    def note(self, text: str) -> Node:
        node = self._expect("note")
        if node.payload.get("text") != text:
            raise ReplayDivergenceError("note differs from recording")
        return self._advance(node)

    def observation(self, source: str, data: Any = None) -> Any:
        node = self._expect("observation")
        if node.payload.get("source") != source:
            raise ReplayDivergenceError("observation source differs from recording")
        recorded = node.payload.get("data")
        if data is not None and codec.canonical(data) != codec.canonical(recorded):
            raise ReplayDivergenceError("observation data differs from recording")
        self._advance(node)
        return recorded

    def infer(
        self,
        request: dict,
        model: str = "unknown",
        *,
        seed: int | None = None,
        usage: dict | None = None,
    ) -> Any:
        node = self._expect("inference")
        recorded_request = node.payload.get("request")
        if fingerprint_request(request) != fingerprint_request(recorded_request):
            raise ReplayDivergenceError(
                f"inference request diverged at step {self._pos} "
                f"({node.short()}); replay is deterministic only if the "
                "agent asks for exactly what was recorded"
            )
        recorded_model = node.payload.get("model")
        payload_override = None
        if model != recorded_model:
            # Cross-model re-basing: legitimate, but it forks the identity —
            # the replayed node records the new model, so its id differs.
            self.rebased_models.append(f"{recorded_model} -> {model}")
            payload_override = {**node.payload, "model": model}
        self._advance(node, payload=payload_override)
        return node.payload.get("response")

    def tool(self, name: str, arguments: dict) -> Any:
        call_node = self._expect("tool_call")
        if call_node.payload.get("name") != name:
            raise ReplayDivergenceError(
                f"tool name diverged: agent called {name!r}, "
                f"trace has {call_node.payload.get('name')!r}"
            )
        recorded_args = call_node.payload.get("arguments")
        if fingerprint_request(arguments) != fingerprint_request(recorded_args):
            raise ReplayDivergenceError(
                f"tool arguments diverged for {name!r} at step {self._pos}"
            )
        self._advance(call_node)
        result_node = self._expect("tool_result")
        self._advance(result_node)
        return result_node.payload.get("result")

    def stream(
        self,
        request: dict,
        *,
        model: str = "unknown",
        seed: int | None = None,
        chunk_chars: int = 64,
    ) -> Iterator[str]:
        """Replay a recorded streaming inference as chunks.

        The recorded response text is yielded in ``chunk_chars`` slices and
        the step advances exactly as :meth:`infer` would, so an agent written
        against ``client.stream(...)`` replays without code changes.
        Returns the full response dict after exhaustion (via ``send``/the
        generator's ``value``); most callers just iterate the chunks.
        """
        node = self._expect("inference")
        recorded_request = node.payload.get("request")
        if fingerprint_request(request) != fingerprint_request(recorded_request):
            raise ReplayDivergenceError(
                f"inference request diverged at step {self._pos} "
                f"({node.short()}); replay is deterministic only if the "
                "agent asks for exactly what was recorded"
            )
        recorded_model = node.payload.get("model")
        payload_override = None
        if model != recorded_model:
            self.rebased_models.append(f"{recorded_model} -> {model}")
            payload_override = {**node.payload, "model": model}
        response = node.payload.get("response") or {}
        text = response.get("content", "") if isinstance(response, dict) else ""
        for i in range(0, len(text), chunk_chars):
            yield text[i:i + chunk_chars]
        self._advance(node, payload=payload_override)
        return response

    async def astream(
        self,
        request: dict,
        *,
        model: str = "unknown",
        seed: int | None = None,
        chunk_chars: int = 64,
    ):
        """Async variant of :meth:`stream` for async streaming agents."""
        node = self._expect("inference")
        recorded_request = node.payload.get("request")
        if fingerprint_request(request) != fingerprint_request(recorded_request):
            raise ReplayDivergenceError(
                f"inference request diverged at step {self._pos} "
                f"({node.short()})"
            )
        recorded_model = node.payload.get("model")
        payload_override = None
        if model != recorded_model:
            self.rebased_models.append(f"{recorded_model} -> {model}")
            payload_override = {**node.payload, "model": model}
        response = node.payload.get("response") or {}
        text = response.get("content", "") if isinstance(response, dict) else ""
        for i in range(0, len(text), chunk_chars):
            yield text[i:i + chunk_chars]
        self._advance(node, payload=payload_override)


def _replay_branch_name() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"replay/{ts}"


def run(
    repo: Repo,
    ref: str,
    agent_fn: Callable[[ReplayClient], Any],
    *,
    branch: str | None = None,
    actor: str | None = None,
    key_path: str | None = None,
    secret_key: bytes | None = None,
    start: str | None = None,
    allow_network: bool = False,
) -> ReplayReport:
    """Re-execute ``agent_fn`` deterministically against the trace at ``ref``.

    If ``start`` is given (a node id within the trace), only the nodes *after*
    it are replayed; the execution context is restored from ``start``'s
    context blob. This lets you skip a trusted prefix and re-verify just the
    tail of a long trace.

    Unless ``allow_network`` is set (or ``CAIRN_REPLAY_NETGUARD=0``), the
    agent runs inside :func:`network_guard`: any attempt at live socket I/O
    raises :class:`ReplayNetworkBlockedError`, proving the replay was truly
    offline.

    Returns a :class:`ReplayReport`. Raises :class:`ReplayError` subclasses
    on any divergence, so a returned report with ``ids_match`` is a proof of
    deterministic re-execution.
    """
    trace = linear_trace(repo, ref)
    if start is not None:
        start = repo._resolve_prefix(start)
        ids = [n.id for n in trace]
        if start not in ids:
            raise ReplayError(f"start node {start[:12]}… is not in the trace")
        trace = trace[ids.index(start) + 1:]
        if not trace:
            raise ReplayError("start is the head of the trace: nothing to replay")
    branch = branch or _replay_branch_name()

    start_id = trace[0].parents[0] if trace[0].parents else None
    if start_id is not None:
        repo.create_branch(branch, at=start_id)
    else:
        # Unborn branch: replay starts at genesis.
        repo.set_branch(branch, "")

    session = Session(
        repo,
        actor=actor if actor is not None else trace[0].actor,
        branch=branch,
        key_path=key_path,
        secret_key=secret_key,
    )
    client = ReplayClient(session, trace)
    use_guard = not allow_network and os.environ.get("CAIRN_REPLAY_NETGUARD", "1") != "0"
    if use_guard:
        with network_guard():
            result = agent_fn(client)
    else:
        result = agent_fn(client)
    client.assert_exhausted()
    session.close()

    original_ids = [n.id for n in trace]
    ids_match = client.replayed_ids == original_ids
    return ReplayReport(
        ref=repo._resolve_prefix(ref),
        branch=branch,
        nodes_replayed=len(trace),
        ids_match=ids_match,
        rebased_models=client.rebased_models,
        result=result,
    )

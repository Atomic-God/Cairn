"""The Session: one-line instrumentation for agent executions.

A :class:`Session` wraps an agent's execution loop. Every significant event —
system prompt, user message, model inference, tool call, tool result, or
environment observation — is appended to the content-addressed DAG as an
immutable, signed node::

    from cairn import Session

    with Session.create("./my-project", actor="planner") as s:
        s.system("You are a helpful planner.")
        s.user("Plan a trip to Kyoto.")
        reply = my_llm(messages)                       # your own call
        s.inference({"messages": messages}, {"text": reply},
                    model="acme-1", seed=1234)
        s.tool_call("search", {"q": "Kyoto temples"})
        hits = search("Kyoto temples")                  # your own call
        s.tool_result("search", {"hits": hits})

Or use the unified client, which performs the real calls *and* records them::

    with Session.create("./my-project", actor="planner") as s:
        client = s.client(llm=my_llm, tools={"search": search})
        client.system("You are a helpful planner.")
        reply = client.infer({"messages": messages}, model="acme-1", seed=1234)
        hits = client.tool("search", {"q": "Kyoto temples"})

The same ``client`` interface is what :mod:`cairn.replay` uses to
deterministically re-execute the trace without touching the network.
"""

from __future__ import annotations

import os
import secrets
import threading
from typing import Any, Callable

from . import codec, keys, signing
from .node import Node, compute_context_root, create_node
from .store import Repo, RepoError, normalize_ref

DEFAULT_ACTOR = "agent"


class Session:
    """An instrumented agent execution bound to a repository branch."""

    def __init__(
        self,
        repo: Repo,
        *,
        actor: str = DEFAULT_ACTOR,
        branch: str | None = None,
        key_path: str | None = None,
        secret_key: bytes | None = None,
        create_branch: bool = False,
    ):
        self.repo = repo
        self.actor = actor or DEFAULT_ACTOR
        if branch is not None:
            branch = normalize_ref(branch)
            if branch not in repo.list_branches():
                # Allow selecting the unborn main branch; anything else must
                # exist — unless the caller asked for auto-creation.
                from .store import MAIN_BRANCH

                if branch != MAIN_BRANCH or repo.get_branch(branch) is not None:
                    if create_branch:
                        at = repo.get_head()
                        if at is not None:
                            repo.create_branch(branch, at=at)
                            repo.checkout(branch)
                        else:
                            # Unborn branch: point HEAD at it; the first
                            # commit creates the ref.
                            with repo.locked():
                                repo._write_text(
                                    repo.head_path,
                                    f"ref: refs/heads/{branch}\n")
                    else:
                        raise ValueError(
                            f"branch {branch!r} does not exist "
                            f"(pass create_branch=True to create it)")
                else:
                    repo.checkout(branch)
            else:
                repo.checkout(branch)
            pinned: str | None = branch
        else:
            # Best-effort pin to whatever HEAD points at; the common case is
            # single-threaded construction, and checkout() re-pins exactly.
            pinned = repo.current_branch()
        if secret_key is not None:
            if len(secret_key) != 32:
                raise ValueError("secret_key must be 32 bytes")
            self._signer = signing.Signer(bytes(secret_key))
        else:
            sk, _ = keys.load_or_create(key_path)
            self._signer = signing.Signer(sk)
        self._commit_lock = threading.Lock()
        # Branch pinning: a Session always advances the branch it was opened
        # on (or checked out to), never "whatever HEAD points at right now".
        # This is what makes concurrent sessions on different branches safe:
        # the shared HEAD file may move under us, but our commits still land
        # on our own branch ref. One writer per branch keeps history linear;
        # concurrent writers to the *same* branch fork (both chains persist,
        # content-addressed) and the last committer wins the ref.
        self._branch: str | None = pinned
        self._sync_to_head()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        path: str,
        *,
        actor: str = DEFAULT_ACTOR,
        branch: str | None = None,
        key_path: str | None = None,
        secret_key: bytes | None = None,
        create_branch: bool = False,
    ) -> "Session":
        """``cairn init`` + open a session in one call."""
        repo = Repo.init(os.path.abspath(path), exist_ok=True)
        return cls(repo, actor=actor, branch=branch,
                   key_path=key_path, secret_key=secret_key,
                   create_branch=create_branch)

    @classmethod
    def open(
        cls,
        path: str,
        *,
        actor: str = DEFAULT_ACTOR,
        branch: str | None = None,
        key_path: str | None = None,
        secret_key: bytes | None = None,
        create_branch: bool = False,
    ) -> "Session":
        repo = Repo.find(path)
        return cls(repo, actor=actor, branch=branch,
                   key_path=key_path, secret_key=secret_key,
                   create_branch=create_branch)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    async def __aenter__(self) -> "Session":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.close()

    @property
    def signer(self) -> signing.Signer:
        """The cached signing identity for this session."""
        return self._signer

    def close(self) -> None:
        """Sessions write through on every commit; this is a no-op hook."""

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _sync_to_head(self) -> None:
        # Branch-aware: read OUR branch's ref, never the global HEAD file
        # (which concurrent sessions may have moved). Detached sessions read
        # the raw HEAD content.
        if self._branch is not None:
            self._head = self.repo.get_branch(self._branch)
        else:
            raw = self.repo._read_text(self.repo.head_path).strip()
            self._head = None if raw.startswith("ref:") else (raw or None)
        if self._head is not None:
            head_node = self.repo.get_node(self._head)
            self._seq = head_node.seq + 1
            self._context_root = head_node.context_root
            try:
                self._messages: list[dict] = self.repo.get_context(
                    head_node.context_root)
            except Exception:
                self._messages = []
        else:
            self._seq = 0
            self._messages = []
            self._context_root = None
        self._synced_len = len(self._messages)

    @property
    def head(self) -> str | None:
        return self._head

    @property
    def messages(self) -> list[dict]:
        return list(self._messages)

    # ------------------------------------------------------------------
    # Core commit machinery
    # ------------------------------------------------------------------
    def _commit(
        self,
        kind: str,
        payload: dict,
        *,
        seed: int | None = None,
        seq: int | None = None,
        in_context: bool = True,
    ) -> Node:
        # The mutex makes a shared Session safe for threaded agents: commits
        # serialize into one linear chain instead of forking. The repo lock
        # inside additionally serializes against other processes.
        with self._commit_lock:
            return self._commit_locked(kind, payload, seed=seed, seq=seq,
                                       in_context=in_context)

    def _commit_locked(
        self,
        kind: str,
        payload: dict,
        *,
        seed: int | None = None,
        seq: int | None = None,
        in_context: bool = True,
    ) -> Node:
        if seed is None:
            seed = secrets.randbits(64)
        new_messages = [{"kind": kind, **payload_view(payload)}] \
            if in_context else []
        # Linearizable commit: under the repo lock, re-read the branch tip.
        # If another writer moved it since we last synced, our new messages
        # are rebased onto the fresh tip instead of forking an orphaned
        # chain — every committed node stays reachable, always.
        with self.repo.locked():
            if self._branch is not None:
                tip = self.repo.get_branch(self._branch)
            else:
                raw = self.repo._read_text(self.repo.head_path).strip()
                tip = None if raw.startswith("ref:") else (raw or None)
            if tip != self._head:
                ours = self._messages[self._synced_len:]
                if tip is not None:
                    tip_node = self.repo.get_node(tip)
                    base = self.repo.get_context(tip_node.context_root)
                    base_root = tip_node.context_root
                    seq = tip_node.seq + 1
                else:
                    base, base_root = [], None
                    seq = 0
                self._messages = base + ours
                self._head = tip
                self._context_root = base_root
            elif seq is None:
                seq = self._seq
            if new_messages:
                self._messages.extend(new_messages)
                context_root = self.repo.put_context(self._context_root,
                                                     new_messages)
                self._context_root = context_root
            else:
                # Notes carry no context: reuse the current root verbatim.
                context_root = self._context_root
                if context_root is None:
                    context_root = self.repo.put_context(None, [])
                    self._context_root = context_root
            parents = [self._head] if self._head else []
            # Identity binding: a key may not sign under a second actor name.
            self.repo.check_identity_unlocked(
                self._signer.public_key.hex(), self.actor)
            node = create_node(
                parents=parents,
                actor=self.actor,
                kind=kind,
                payload=payload,
                context_root=context_root,
                seed=seed,
                seq=seq,
                signer=self._signer,
            )
            # The object write is atomic and idempotent; the lock above
            # serializes the ref update so parallel committers land in one
            # linear chain instead of orphaning each other.
            self.repo.put_node(node)
            if self._branch is not None:
                self.repo._set_branch_unlocked(self._branch, node.id,
                                               actor=self.actor)
            else:
                self.repo._write_text(self.repo.head_path, node.id + "\n")
        self._head = node.id
        self._synced_len = len(self._messages)
        self._seq = seq + 1
        return node

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------
    def system(self, text: str, **kw: Any) -> Node:
        return self._commit("system", {"text": text}, **kw)

    def user(self, text: str, **kw: Any) -> Node:
        return self._commit("user", {"text": text}, **kw)

    def inference(
        self,
        request: dict,
        response: dict,
        model: str,
        *,
        seed: int | None = None,
        usage: dict | None = None,
    ) -> Node:
        payload: dict[str, Any] = {
            "model": model,
            "request": request,
            "response": response,
        }
        if usage is not None:
            payload["usage"] = usage
        return self._commit("inference", payload, seed=seed)

    def tool_call(
        self, name: str, arguments: dict, *, call_id: str | None = None, **kw: Any
    ) -> Node:
        payload: dict[str, Any] = {"name": name, "arguments": arguments}
        if call_id is not None:
            payload["call_id"] = call_id
        return self._commit("tool_call", payload, **kw)

    def tool_result(
        self,
        name: str,
        result: Any,
        *,
        call_id: str | None = None,
        ok: bool = True,
        **kw: Any,
    ) -> Node:
        payload: dict[str, Any] = {"name": name, "result": result, "ok": ok}
        if call_id is not None:
            payload["call_id"] = call_id
        return self._commit("tool_result", payload, **kw)

    def observation(self, source: str, data: Any, **kw: Any) -> Node:
        return self._commit("observation", {"source": source, "data": data}, **kw)

    def note(self, text: str, **kw: Any) -> Node:
        return self._commit("note", {"text": text}, in_context=False, **kw)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------
    def branch(self, name: str) -> None:
        self.repo.create_branch(name, at=self._head)

    def checkout(self, ref: str, *, new_branch: str | None = None) -> str | None:
        oid = self.repo.checkout(ref, new_branch=new_branch)
        # Resolve the pin locally — never re-read the global HEAD file,
        # which a concurrent session may have moved under us.
        if new_branch is not None:
            self._branch = normalize_ref(new_branch)
        else:
            try:
                candidate: str | None = normalize_ref(ref)
            except RepoError:
                candidate = None
            if candidate is not None and candidate in self.repo.list_branches():
                self._branch = candidate
            else:
                self._branch = None  # detached
        self._sync_to_head()
        return oid

    def log(self, limit: int | None = None):
        return self.repo.walk(self._head, limit=limit)

    # ------------------------------------------------------------------
    # Unified client (record + execute)
    # ------------------------------------------------------------------
    def client(
        self,
        *,
        llm: Callable[..., Any] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
    ) -> "AgentClient":
        return AgentClient(self, llm=llm, tools=tools or {})


def payload_view(payload: dict) -> dict:
    """The context-visible projection of a commit payload."""
    view: dict[str, Any] = {}
    for key in ("text", "model", "request", "response", "name",
                "arguments", "result", "ok", "source", "data"):
        if key in payload:
            view[key] = payload[key]
    return view


class AgentClient:
    """Unified record-and-execute client.

    During normal runs this performs the real LLM / tool calls and records
    every step. :mod:`cairn.replay` provides a drop-in replacement that
    replays recorded outputs instead.
    """

    def __init__(
        self,
        session: Session,
        *,
        llm: Callable[..., Any] | None = None,
        tools: dict[str, Callable[..., Any]] | None = None,
    ):
        self.session = session
        self._llm = llm
        self._tools = tools or {}

    def system(self, text: str) -> Node:
        return self.session.system(text)

    def user(self, text: str) -> Node:
        return self.session.user(text)

    def note(self, text: str) -> Node:
        return self.session.note(text)

    def observation(self, source: str, data: Any) -> Node:
        return self.session.observation(source, data)

    def infer(
        self,
        request: dict,
        model: str = "unknown",
        *,
        seed: int | None = None,
        usage: dict | None = None,
    ) -> Any:
        if self._llm is None:
            raise RuntimeError("no LLM backend configured for recording client")
        response = self._llm(request, model=model, seed=seed)
        self.session.inference(request, response, model, seed=seed, usage=usage)
        return response

    def tool(self, name: str, arguments: dict) -> Any:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name!r}")
        call_id = secrets.token_hex(8)
        self.session.tool_call(name, arguments, call_id=call_id)
        try:
            result = self._tools[name](arguments)
        except Exception as exc:  # record the failure, then re-raise
            self.session.tool_result(name, {"error": str(exc)},
                                     call_id=call_id, ok=False)
            raise
        self.session.tool_result(name, result, call_id=call_id, ok=True)
        return result

    def stream(
        self,
        chunks,
        *,
        model: str,
        request: dict,
        seed: int | None = None,
        **kwargs: Any,
    ):
        """Wrap a sync chunk iterator; commits the assembled message at end.

        See :class:`cairn.aio.StreamRecorder`.
        """
        from .aio import StreamRecorder

        return StreamRecorder(self.session, chunks, model=model,
                              request=request, seed=seed, **kwargs)

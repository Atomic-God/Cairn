"""Content-addressed object store: the ``.cairn`` directory.

Layout::

    .cairn/
        config.json            # {"version": 1, "hash": "blake3", ...}
        objects/ab/cdef...     # zlib-compressed canonical JSON envelopes
        refs/heads/<branch>    # hex node id
        HEAD                   # "ref: refs/heads/main" or detached hex id

Objects are addressed by their BLAKE3 identity (node id / blob id), so
identical content is stored exactly once however many sessions reference it —
this is the zero-copy structural sharing at the heart of Cairn.
"""

from __future__ import annotations

import json
import os
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from . import codec, hash as hashmod
from .node import Node, compute_blob_id

CONFIG_VERSION = 1
MAIN_BRANCH = "main"

# Object envelope magic: the ASCII bytes "CAIR" (0x43 0x41 0x49 0x52),
# recorded as a JSON field so the envelope stays human-inspectable.
ENVELOPE_MAGIC = "CAIR"
ENVELOPE_VERSION = 1


class RepoError(Exception):
    pass


class BlobTooLargeError(RepoError):
    """A blob exceeded CAIRN_MAX_BLOB_BYTES."""


def _tmp_sibling(path: str) -> str:
    """A process-unique temporary sibling for atomic replacement writes."""
    return f"{path}.tmp.{os.getpid()}.{secrets.token_hex(4)}"


class RepoLock:
    """Cross-platform advisory exclusive lock for a repository.

    Uses ``flock(2)`` on POSIX and ``LockFile`` semantics via ``msvcrt`` on
    Windows, taken on ``.cairn/LOCK``. Guards every ref mutation (commit,
    checkout, branch create/delete) so parallel agents / processes cannot
    interleave and corrupt refs. Locks are *not* re-entrant: public methods
    take the lock and delegate to ``_*_unlocked`` internals.
    """

    def __init__(self, dotdir: str):
        self.path = os.path.join(dotdir, "LOCK")
        self._fd: int | None = None

    def __enter__(self) -> "RepoLock":
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            if os.name == "nt":
                import msvcrt

                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\x00")
                os.lseek(fd, 0, os.SEEK_SET)
                # Lock the first byte; blocks until acquired.
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_EX)
        except BaseException:
            os.close(fd)
            raise
        self._fd = fd
        return self

    def __exit__(self, *exc: object) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def normalize_ref(name: str) -> str:
    """Normalize a branch name to its POSIX internal form.

    Backslashes become forward slashes, redundant separators and ``.``
    segments are dropped, and ``..`` / empty names are rejected. The on-disk
    layout always uses the OS-native separator; only the *logical* name is
    normalized, so the same repository is addressable identically on
    POSIX and Windows.
    """
    if not isinstance(name, str):
        raise RepoError(f"invalid branch name: {name!r}")
    parts: list[str] = []
    for part in name.replace("\\", "/").split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            raise RepoError(f"invalid branch name: {name!r}")
        parts.append(part)
    normalized = "/".join(parts)
    _check_ref_name(normalized)
    return normalized


class Repo:
    def __init__(self, dotdir: str):
        self.dotdir = os.path.abspath(dotdir)
        self.objects = os.path.join(self.dotdir, "objects")
        self.refs_dir = os.path.join(self.dotdir, "refs", "heads")
        self.head_path = os.path.join(self.dotdir, "HEAD")
        self.config_path = os.path.join(self.dotdir, "config.json")
        self.logs_dir = os.path.join(self.dotdir, "logs", "refs", "heads")
        self.identities_path = os.path.join(self.dotdir, "identities.json")

    @contextmanager
    def locked(self) -> Iterator["Repo"]:
        """Hold the repository's advisory exclusive lock (see :class:`RepoLock`)."""
        with RepoLock(self.dotdir):
            yield self

    def clean_tmp(self, older_than_seconds: float = 3600) -> int:
        """Remove stale ``*.tmp.*`` files left by crashed writers.

        Only files older than ``older_than_seconds`` are removed, so a
        concurrently-running writer's in-flight temp file is never touched.
        Returns the number of files removed.
        """
        now = time.time()
        removed = 0
        for root, _, files in os.walk(self.dotdir):
            for f in files:
                if ".tmp." not in f:
                    continue
                path = os.path.join(root, f)
                try:
                    if now - os.path.getmtime(path) > older_than_seconds:
                        os.remove(path)
                        removed += 1
                except OSError:
                    pass
        return removed

    # ------------------------------------------------------------------
    # Construction / discovery
    # ------------------------------------------------------------------
    @classmethod
    def init(cls, path: str, *, exist_ok: bool = False) -> "Repo":
        dotdir = os.path.join(os.path.abspath(path), ".cairn")
        if os.path.exists(os.path.join(dotdir, "config.json")) and not exist_ok:
            raise RepoError(f"cairn repository already exists at {dotdir}")
        os.makedirs(os.path.join(dotdir, "objects"), exist_ok=True)
        os.makedirs(os.path.join(dotdir, "refs", "heads"), exist_ok=True)
        repo = cls(dotdir)
        config = {
            "version": CONFIG_VERSION,
            "hash": "blake3",
            "hash_backend": hashmod.backend(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "repo_id": hashmod.hexdigest(
                b"cairn-repo-v1\x00" + os.urandom(32)
            ),
        }
        repo._write_json(repo.config_path, config)
        # HEAD starts pointing at main, which is unborn until the first commit.
        repo._write_text(repo.head_path, f"ref: refs/heads/{MAIN_BRANCH}\n")
        return repo

    @classmethod
    def open(cls, path: str) -> "Repo":
        dotdir = os.path.join(os.path.abspath(path), ".cairn")
        if not os.path.exists(os.path.join(dotdir, "config.json")):
            raise RepoError(f"not a cairn repository: {dotdir} (run `cairn init`)")
        repo = cls(dotdir)
        repo.clean_tmp()
        return repo

    @classmethod
    def find(cls, start: str) -> "Repo":
        """Walk up from ``start`` looking for a ``.cairn`` directory."""
        cur = os.path.abspath(start)
        while True:
            dotdir = os.path.join(cur, ".cairn")
            if os.path.exists(os.path.join(dotdir, "config.json")):
                return cls(dotdir)
            parent = os.path.dirname(cur)
            if parent == cur:
                raise RepoError("not inside a cairn repository (no .cairn found)")
            cur = parent

    # ------------------------------------------------------------------
    # Objects
    # ------------------------------------------------------------------
    def _object_path(self, oid: str) -> str:
        return os.path.join(self.objects, oid[:2], oid[2:])

    def _write_object(self, oid: str, envelope: dict) -> None:
        path = self._object_path(oid)
        if os.path.exists(path):
            return  # content addressed: identical content already stored
        os.makedirs(os.path.dirname(path), exist_ok=True)
        framed = {"magic": ENVELOPE_MAGIC, "v": ENVELOPE_VERSION, **envelope}
        data = codec.compress(codec.canonical(framed))
        tmp = _tmp_sibling(path)
        try:
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, path)  # atomic: readers see old or new, never torn
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _read_object(self, oid: str) -> dict:
        path = self._object_path(oid)
        if not os.path.exists(path):
            raise RepoError(f"object not found: {oid}")
        with open(path, "rb") as f:
            data = codec.decompress(f.read())
        env = codec.canonical_loads(data)
        if env.get("magic") != ENVELOPE_MAGIC or env.get("v") != ENVELOPE_VERSION:
            raise RepoError(f"object {oid[:12]}…: bad magic/version (corrupt?)")
        return env

    def put_node(self, node: Node) -> str:
        node.verify_id()
        self._write_object(node.id, {"t": "node", "node": node.to_dict()})
        return node.id

    def get_node(self, oid: str) -> Node:
        env = self._read_object(self._resolve_prefix(oid))
        if env.get("t") != "node":
            raise RepoError(f"object {oid} is not a node")
        node = Node.from_dict(env["node"])  # verifies id against content
        if node.id != self._resolve_prefix(oid):
            raise RepoError(f"object address mismatch for {oid}")
        return node

    def put_blob(self, data: bytes) -> str:
        limit = int(os.environ.get("CAIRN_MAX_BLOB_BYTES",
                                   str(8 * 1024 * 1024)))
        if len(data) > limit:
            raise BlobTooLargeError(
                f"blob is {len(data)} bytes, over the {limit}-byte limit "
                f"(set CAIRN_MAX_BLOB_BYTES to raise it, or install the "
                f"native engine / `blake3` package so large payloads hash "
                f"fast enough to be practical)"
            )
        oid = compute_blob_id(data)
        self._write_object(oid, {"t": "blob", "data": codec.b64encode(data)})
        return oid

    def put_context(self, prev_root: str | None,
                    new_messages: list) -> str:
        """Store one context delta and return its root id.

        Each commit stores only the messages appended since the previous
        root, so commit cost is O(len(new)) — constant for a steady agent —
        instead of O(len(full history)).
        """
        from .node import encode_context_delta
        return self.put_blob(encode_context_delta(prev_root, new_messages))

    def get_context(self, root: str | None) -> list:
        """Reconstruct the full message list for a context root.

        Walks the delta chain newest-first. Blobs written by Cairn ≤ 0.1.x
        hold a bare message list (a full snapshot); those are treated as a
        chain of length one, so old repositories keep reading correctly.
        """
        if root is None:
            return []
        messages: list = []
        seen: set[str] = set()
        chunks: list[list] = []
        cur: str | None = self._resolve_prefix(root)
        while cur is not None:
            if cur in seen:
                raise RepoError(f"context chain cycle at {cur[:12]}…")
            seen.add(cur)
            obj = codec.canonical_loads(self.get_blob(cur))
            if isinstance(obj, dict) and "messages" in obj and "prev" in obj:
                chunk = obj["messages"]
                if not isinstance(chunk, list):
                    raise RepoError(f"bad context delta at {cur[:12]}…")
                chunks.append(chunk)
                nxt = obj["prev"]
                cur = self._resolve_prefix(nxt) if nxt else None
            elif isinstance(obj, list):
                chunks.append(obj)  # v0.1.x full snapshot
                cur = None
            else:
                raise RepoError(f"bad context object at {cur[:12]}…")
        for chunk in reversed(chunks):
            messages.extend(chunk)
        return messages

    def get_blob(self, oid: str) -> bytes:
        oid = self._resolve_prefix(oid)
        env = self._read_object(oid)
        if env.get("t") != "blob":
            raise RepoError(f"object {oid} is not a blob")
        data = codec.b64decode(env["data"])
        if compute_blob_id(data) != oid:
            raise RepoError(f"blob content mismatch for {oid}")
        return data

    def has(self, oid: str) -> bool:
        try:
            oid = self._resolve_prefix(oid)
        except RepoError:
            return False
        return os.path.exists(self._object_path(oid))

    def _resolve_prefix(self, prefix: str) -> str:
        """Resolve a full id or a unique hex prefix to a full id."""
        prefix = prefix.strip().lower()
        if len(prefix) == 64 and all(c in "0123456789abcdef" for c in prefix):
            return prefix
        if not prefix or not all(c in "0123456789abcdef" for c in prefix):
            raise RepoError(f"invalid object id: {prefix!r}")
        shard = os.path.join(self.objects, prefix[:2])
        matches = []
        if os.path.isdir(shard):
            for name in os.listdir(shard):
                if name.startswith(prefix[2:]):
                    matches.append(prefix[:2] + name)
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise RepoError(f"object not found: {prefix}")
        raise RepoError(f"ambiguous id prefix {prefix!r} ({len(matches)} matches)")

    def count_objects(self) -> int:
        n = 0
        for _, _, files in os.walk(self.objects):
            n += len([f for f in files if not f.endswith(".tmp")])
        return n

    # ------------------------------------------------------------------
    # Refs
    # ------------------------------------------------------------------
    def _write_text(self, path: str, text: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = _tmp_sibling(path)
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)  # atomic: readers see old or new, never torn
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _write_json(self, path: str, obj: dict) -> None:
        self._write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n")

    def _read_text(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    def current_branch(self) -> str | None:
        """Branch HEAD points at, or None if detached."""
        content = self._read_text(self.head_path).strip()
        if content.startswith("ref:"):
            ref = content.split(":", 1)[1].strip()
            if ref.startswith("refs/heads/"):
                return ref[len("refs/heads/"):]
            return ref
        return None

    def get_head(self) -> str | None:
        branch = self.current_branch()
        if branch is not None:
            return self.get_branch(branch)
        oid = self._read_text(self.head_path).strip()
        return oid or None

    def set_head(self, oid: str) -> None:
        """Move HEAD (following the current branch ref, if any)."""
        with self.locked():
            self._set_head_unlocked(oid)

    def _set_head_unlocked(self, oid: str) -> None:
        branch = self.current_branch()
        if branch is not None:
            self._set_branch_unlocked(branch, oid)
        else:
            self._write_text(self.head_path, oid + "\n")

    def _branch_path(self, name: str) -> str:
        return os.path.join(self.refs_dir, *normalize_ref(name).split("/"))

    def get_branch(self, name: str) -> str | None:
        path = self._branch_path(name)
        if not os.path.exists(path):
            return None
        return self._read_text(path).strip() or None

    def set_branch(self, name: str, oid: str) -> None:
        with self.locked():
            self._set_branch_unlocked(name, oid)

    def resolve_ref(self, ref: str | None) -> str | None:
        """Resolve a user-supplied ref to a node id.

        Accepts, in order: ``None`` (→ HEAD), a branch name, a full 64-hex
        node id, or a unique hex prefix. Returns ``None`` only when the
        repository has no commits yet (unborn HEAD). Raises
        :class:`RepoError` when the ref matches nothing.
        """
        if ref is None:
            return self.get_head()
        ref = ref.strip()
        # Branch name first: `cairn log main` should just work.
        try:
            branch = normalize_ref(ref)
        except RepoError:
            branch = None
        if branch is not None:
            oid = self.get_branch(branch)
            if oid is not None:
                return oid
        # Otherwise it must be a node id (full or unique prefix).
        return self._resolve_prefix(ref)

    def _set_branch_unlocked(self, name: str, oid: str,
                             *, actor: str | None = None) -> None:
        old = self.get_branch(name)
        self._write_text(self._branch_path(name), oid.strip() + "\n")
        self._append_reflog(name, old, oid.strip(), actor)

    def _append_reflog(self, name: str, old: str | None, new: str,
                       actor: str | None) -> None:
        """Append one line to the branch's reflog.

        The reflog is append-only: every ref movement — commits, checkouts,
        branch creation, and history rewrites — leaves a timestamped record,
        so a silent ``set_head`` rewind can always be detected afterwards.
        """
        path = os.path.join(self.logs_dir, *name.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        line = (f"{old or '0' * 64} {new} "
                f"{int(time.time())} {actor or '?'}\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_reflog(self, name: str) -> list[dict]:
        """Read a branch's reflog, oldest first. Empty list if none."""
        path = os.path.join(self.logs_dir, *normalize_ref(name).split("/"))
        entries: list[dict] = []
        if not os.path.isfile(path):
            return entries
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 3)
                if len(parts) != 4:
                    continue
                old, new, ts, actor = parts
                entries.append({"old": old, "new": new,
                                "time": int(ts), "actor": actor})
        return entries

    # ------------------------------------------------------------------
    # Identity registry: which signing key claimed which actor name first
    # ------------------------------------------------------------------
    def _read_identities_unlocked(self) -> dict:
        if not os.path.isfile(self.identities_path):
            return {"pubkeys": {}}
        try:
            with open(self.identities_path, encoding="utf-8") as f:
                reg = json.load(f)
            if isinstance(reg, dict) and isinstance(reg.get("pubkeys"), dict):
                return reg
        except (OSError, ValueError):
            pass
        return {"pubkeys": {}}

    def list_identities(self) -> dict:
        """pubkey hex -> {"actor": ..., "first_seen": ...} (read-only)."""
        with self.locked():
            return self._read_identities_unlocked()["pubkeys"]

    def check_identity_unlocked(self, pubkey_hex: str, actor: str) -> None:
        """Record which actor name a signing key commits under.

        Actor names are self-asserted labels — the Ed25519 key is the real
        identity. The registry records every (key, actor) pair ever seen so
        auditors can spot one key signing under several names (see
        :func:`cairn.verify.verify`, which surfaces these as notes).
        """
        reg = self._read_identities_unlocked()
        pubkeys = reg["pubkeys"]
        entry = pubkeys.get(pubkey_hex)
        if entry is None:
            pubkeys[pubkey_hex] = {"actor": actor, "actors_seen": [actor],
                                   "first_seen": int(time.time())}
        else:
            seen = entry.setdefault("actors_seen", [entry.get("actor", actor)])
            if actor not in seen:
                seen.append(actor)
            entry["actor"] = actor
        self._write_text(self.identities_path, json.dumps(reg, indent=2))

    def rebind_identity(self, pubkey_hex: str, actor: str) -> None:
        """Move a key's actor binding (explicit, auditable)."""
        with self.locked():
            reg = self._read_identities_unlocked()
            reg["pubkeys"][pubkey_hex] = {"actor": actor,
                                          "first_seen": int(time.time())}
            self._write_text(self.identities_path, json.dumps(reg, indent=2))

    # ------------------------------------------------------------------
    # Orphan detection
    # ------------------------------------------------------------------
    def iter_node_ids(self) -> Iterator[str]:
        """Yield the ids of every node object on disk (reachable or not)."""
        if not os.path.isdir(self.objects):
            return
        for shard in sorted(os.listdir(self.objects)):
            shard_path = os.path.join(self.objects, shard)
            if not os.path.isdir(shard_path):
                continue
            for name in sorted(os.listdir(shard_path)):
                oid = shard + name
                if len(oid) != 64:
                    continue
                try:
                    env = self._read_object(oid)
                except RepoError:
                    continue
                if env.get("t") == "node":
                    yield oid

    def find_orphans(self) -> list[str]:
        """Node ids on disk that no branch (or HEAD) can reach.

        Orphans are not corruption — they are usually the losers of a
        same-branch write race from before v0.2.0's linearizable commits —
        but they are invisible to log/verify/replay, so they are reported,
        not silently ignored.
        """
        reachable: set[str] = set()
        stack: list[str] = []
        for ref in self.list_branches():
            head = self.get_branch(ref)
            if head:
                stack.append(head)
        head = self.get_head()
        if head and head not in stack:
            stack.append(head)
        while stack:
            oid = stack.pop()
            if oid in reachable:
                continue
            try:
                node = self.get_node(oid)
            except RepoError:
                continue
            reachable.add(oid)
            stack.extend(node.parents)
        return sorted(oid for oid in self.iter_node_ids()
                      if oid not in reachable)

    def create_branch(self, name: str, at: str | None = None) -> None:
        name = normalize_ref(name)
        with self.locked():
            if self.get_branch(name) is not None:
                raise RepoError(f"branch {name!r} already exists")
            target = at or self.get_head()
            if target is None:
                raise RepoError("cannot create branch: no commits yet")
            self._set_branch_unlocked(name, target)

    def delete_branch(self, name: str) -> None:
        name = normalize_ref(name)
        with self.locked():
            if name == self.current_branch():
                raise RepoError(f"cannot delete the checked-out branch {name!r}")
            path = self._branch_path(name)
            if not os.path.isfile(path):
                raise RepoError(f"branch {name!r} does not exist")
            os.remove(path)
            # Prune newly-empty parent directories (best effort).
            parent = os.path.dirname(path)
            while parent != self.refs_dir and os.path.isdir(parent):
                try:
                    os.rmdir(parent)
                except OSError:
                    break
                parent = os.path.dirname(parent)

    def list_branches(self) -> list[str]:
        if not os.path.isdir(self.refs_dir):
            return []
        names = []
        for root, _, files in os.walk(self.refs_dir):
            for f in files:
                if ".tmp." in f:
                    continue
                full = os.path.join(root, f)
                rel = os.path.relpath(full, self.refs_dir)
                names.append(rel.replace(os.sep, "/"))
        return sorted(names)

    def checkout(self, ref: str, *, new_branch: str | None = None) -> str | None:
        """Point HEAD at ``ref`` (branch name or node id).

        Returns the new head node id, or ``None`` when checking out an
        unborn (commit-less) branch.
        """
        with self.locked():
            if new_branch:
                new_branch = normalize_ref(new_branch)
                oid = self._resolve_prefix(ref)
                if self.get_branch(new_branch) is not None:
                    raise RepoError(f"branch {new_branch!r} already exists")
                self._set_branch_unlocked(new_branch, oid)
                self._write_text(
                    self.head_path, f"ref: refs/heads/{new_branch}\n")
                return oid
            branches = self.list_branches()
            try:
                branch = normalize_ref(ref)
            except RepoError:
                branch = None
            if branch is not None and branch in branches:
                self._write_text(self.head_path, f"ref: refs/heads/{branch}\n")
                return self.get_branch(branch)
            oid = self._resolve_prefix(ref)  # detached
            self.get_node(oid)  # must exist and be a node
            self._write_text(self.head_path, oid + "\n")
            return oid

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------
    def walk(self, ref: str | None = None, limit: int | None = None) -> Iterator[Node]:
        """Yield nodes newest-first starting from ``ref`` (default HEAD).

        The order is a reverse topological sort: every node is yielded before
        its parents, which makes it suitable for ``cairn log`` even across
        merges and diamonds.
        """
        start = self.resolve_ref(ref)
        if start is None:
            return
        # 1. Collect the reachable subgraph.
        nodes: dict[str, Node] = {}
        stack = [start]
        while stack:
            oid = stack.pop()
            if oid in nodes:
                continue
            node = self.get_node(oid)
            nodes[oid] = node
            stack.extend(p for p in node.parents if p not in nodes)
        # 2. Post-order DFS (parents before children), then reverse.
        ordered: list[Node] = []
        visited: set[str] = set()
        work: list[tuple[str, bool]] = [(start, False)]
        while work:
            oid, expanded = work.pop()
            if oid in visited:
                continue
            if expanded:
                visited.add(oid)
                ordered.append(nodes[oid])
            else:
                work.append((oid, True))
                for p in nodes[oid].parents:
                    if p not in visited:
                        work.append((p, False))
        count = 0
        for node in reversed(ordered):
            yield node
            count += 1
            if limit is not None and count >= limit:
                return

    def reachable_ids(self, ref: str) -> set[str]:
        """All node ids reachable from ``ref`` (inclusive)."""
        return {n.id for n in self.walk(ref)}


def _check_ref_name(name: str) -> None:
    if not name or name.startswith(("/", ".")) or name.endswith("/"):
        raise RepoError(f"invalid branch name: {name!r}")
    for part in name.split("/"):
        if (not part or part in (".", "..") or part.endswith(".lock")
                or any(c in part for c in " ~^:?*[]\\")):
            raise RepoError(f"invalid branch name: {name!r}")

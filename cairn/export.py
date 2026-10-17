"""Portable bundles: export a trace (and its history) to a ``.car`` file.

A bundle is a gzipped tar archive containing every object reachable from the
exported ref, plus the branch heads. It is how traces move between machines,
get attached to bug reports, or get published for independent verification —
Cairn's answer to "fork it without re-running".
"""

from __future__ import annotations

import io
import json
import os
import tarfile
import tempfile

from . import codec
from .store import MAIN_BRANCH, Repo, RepoError

BUNDLE_VERSION = 1


def _reachable_objects(repo: Repo, ref: str) -> set[str]:
    oids: set[str] = set()
    for node in repo.walk(ref):
        oids.add(node.id)
        # The whole context delta chain, not just its head.
        cur: str | None = node.context_root
        seen: set[str] = set()
        while cur is not None and cur not in seen:
            seen.add(cur)
            oids.add(cur)
            try:
                obj = codec.canonical_loads(repo.get_blob(cur))
            except Exception:
                break
            if isinstance(obj, dict) and "prev" in obj:
                cur = obj["prev"]
            else:
                cur = None  # v0.1.x full snapshot: chain of one
    return oids


def export_bundle(repo: Repo, ref: str, out_path: str) -> dict:
    """Write a ``.car`` bundle of everything reachable from ``ref``.

    Returns a manifest dict ``{"ref": ..., "heads": {...}, "objects": n}``.
    """
    head = repo.resolve_ref(ref)
    if head is None:
        raise RepoError(f"cannot export {ref!r}: repository has no commits")
    repo.get_node(head)  # must be a node
    oids = _reachable_objects(repo, head)

    heads = {}
    for branch in repo.list_branches():
        target = repo.get_branch(branch)
        if target and target in oids or target == head:
            heads[branch] = target
    # Always include the exported ref's own resolution.
    if not heads:
        heads[MAIN_BRANCH] = head

    manifest = {
        "bundle_version": BUNDLE_VERSION,
        "exported_ref": head,
        "heads": heads,
        "objects": len(oids),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".car.tmp",
                                        dir=os.path.dirname(os.path.abspath(out_path)))
    try:
        with os.fdopen(tmp_fd, "wb") as f, \
                tarfile.open(fileobj=f, mode="w:gz") as tar:
            for oid in sorted(oids):
                src = repo._object_path(oid)
                if not os.path.exists(src):
                    raise RepoError(f"cannot export: missing object {oid[:12]}…")
                arcname = f"objects/{oid[:2]}/{oid[2:]}"
                tar.add(src, arcname=arcname)
            payload = json.dumps(manifest, indent=2, sort_keys=True).encode()
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return manifest


def import_bundle(dest_path: str, bundle_path: str, *, force: bool = False) -> Repo:
    """Import a ``.car`` bundle into the repository at ``dest_path``.

    Creates the repository if needed. Branch heads from the bundle are
    installed; with ``force=False`` an existing branch pointing at a
    different head raises :class:`RepoError` instead of being overwritten.
    """
    dest_path = os.path.abspath(dest_path)
    try:
        repo = Repo.open(dest_path)
    except RepoError:
        repo = Repo.init(dest_path)

    with tarfile.open(bundle_path, "r:gz") as tar:
        members = tar.getmembers()
        manifest_member = next(
            (m for m in members if m.name == "manifest.json"), None
        )
        if manifest_member is None:
            raise RepoError("not a cairn bundle: manifest.json missing")
        manifest = json.load(tar.extractfile(manifest_member))  # type: ignore[arg-type]

        for m in members:
            if not m.name.startswith("objects/"):
                continue
            oid = m.name[len("objects/"):].replace("/", "")
            dest = repo._object_path(oid)
            if os.path.exists(dest):
                continue
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            src = tar.extractfile(m)
            assert src is not None
            with open(dest, "wb") as f:
                f.write(src.read())

    for branch, head in manifest.get("heads", {}).items():
        existing = repo.get_branch(branch)
        if existing is not None and existing != head and not force:
            raise RepoError(
                f"branch {branch!r} already points at {existing[:12]}…; "
                "use force=True to overwrite"
            )
        repo.set_branch(branch, head)
    return repo


def read_manifest(bundle_path: str) -> dict:
    with tarfile.open(bundle_path, "r:gz") as tar:
        m = tar.getmember("manifest.json")
        f = tar.extractfile(m)
        assert f is not None
        return json.load(f)

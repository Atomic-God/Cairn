"""Cryptographic audit: verify the integrity of a Cairn repository.

:func:`verify` re-derives every node id from its preimage, checks every
Ed25519 signature, confirms every parent link resolves, and confirms every
context blob is present and matches its address. Any failure — a flipped bit
in the object store, a forged node, a missing parent — is reported with the
offending node id.

This is the primitive that makes Cairn traces admissible as audit trails:
verification is self-contained and needs no trusted third party.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .node import Node
from .signing import BadSignatureError
from .store import Repo


@dataclass
class VerifyReport:
    refs_checked: list[str]
    nodes_checked: int
    blobs_checked: int
    errors: list[str] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    identity_notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def verify(repo: Repo, refs: list[str] | None = None) -> VerifyReport:
    """Verify all nodes reachable from ``refs`` (default: every branch)."""
    if refs is None:
        refs = repo.list_branches()
    report = VerifyReport(refs_checked=list(refs), nodes_checked=0, blobs_checked=0)

    seen: set[str] = set()
    for ref in refs:
        try:
            head = repo.get_branch(ref)
        except Exception as exc:
            report.errors.append(f"ref {ref!r}: {exc}")
            continue
        if head is None:
            continue  # unborn branch: nothing to check
        try:
            start = repo._resolve_prefix(head)
        except Exception as exc:
            report.errors.append(f"ref {ref!r}: {exc}")
            continue
        stack = [start]
        while stack:
            oid = stack.pop()
            if oid in seen:
                continue
            seen.add(oid)
            try:
                node = repo.get_node(oid)
            except Exception as exc:
                report.errors.append(f"node {oid[:12]}…: cannot load: {exc}")
                continue
            report.nodes_checked += 1
            _check_node(repo, node, report)
            stack.extend(node.parents)
    # Orphaned nodes are not corruption, but they are invisible to every
    # other command — report them so no committed work goes unnoticed.
    try:
        report.orphans = repo.find_orphans()
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"orphan scan failed: {exc}")
    # Identity notes: one key signing under several actor names is worth an
    # auditor's attention (actor names are self-asserted; keys are not).
    try:
        for pubkey, entry in repo.list_identities().items():
            seen = entry.get("actors_seen", [entry.get("actor")])
            seen = [a for a in seen if a]
            if len(set(seen)) > 1:
                report.identity_notes.append(
                    f"key {pubkey[:16]}… signed as {', '.join(sorted(set(seen)))}"
                )
    except Exception as exc:  # noqa: BLE001
        report.errors.append(f"identity scan failed: {exc}")
    return report


def _check_node(repo: Repo, node: Node, report: VerifyReport) -> None:
    # 1. Content address: recompute the id from the preimage.
    try:
        node.verify_id()
    except ValueError as exc:
        report.errors.append(f"node {node.short()}: id mismatch ({exc})")
        return  # further checks are meaningless if the id is wrong
    # 2. Signature over (id, seq, wall_time).
    try:
        node.verify_signature()
    except BadSignatureError as exc:
        report.errors.append(f"node {node.short()}: bad signature ({exc})")
    except ValueError as exc:
        report.errors.append(f"node {node.short()}: bad signature ({exc})")
    # 3. Parents must exist as objects.
    for p in node.parents:
        if not repo.has(p):
            report.errors.append(
                f"node {node.short()}: missing parent {p[:12]}…"
            )
    # 4. Context blob must exist and match its address.
    try:
        blob = repo.get_blob(node.context_root)
        report.blobs_checked += 1
    except Exception as exc:
        report.errors.append(
            f"node {node.short()}: context blob missing/corrupt ({exc})"
        )

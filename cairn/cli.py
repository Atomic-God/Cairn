"""The ``cairn`` command-line interface.

All commands operate on the repository found by walking up from the current
directory (like git). Run ``cairn <command> --help`` for details.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from datetime import datetime, timezone

from . import export as exportmod
from . import keys, replay as replaymod
from .verify import verify as _verify_repo
from .store import MAIN_BRANCH, Repo, RepoError

VERSION = "0.1.0"


def _repo_here() -> Repo:
    try:
        return Repo.find(os.getcwd())
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        sys.exit(1)


def _short(oid: str | None) -> str:
    return oid[:12] if oid else "(unborn)"


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    try:
        repo = Repo.init(args.path or os.getcwd(), exist_ok=args.exist_ok)
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    # Ensure the user has an identity.
    _, pk = keys.load_or_create()
    print(f"Initialized empty cairn repository in {repo.dotdir}")
    print(f"Identity: {keys.key_fingerprint(pk)}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    repo = _repo_here()
    branch = repo.current_branch()
    head = repo.get_head()
    print(f"On branch {branch or '(detached HEAD)'}")
    print(f"Head: {_short(head)}")
    print(f"Objects: {repo.count_objects()}")
    branches = repo.list_branches()
    if branches:
        print("Branches: " + ", ".join(
            (b + "*" if b == branch else b) for b in branches
        ))
    return 0


def _fmt_node_oneline(node) -> str:
    return f"{node.short()} [{node.kind:>11}] seq={node.seq} {node.actor}"


def cmd_log(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        nodes = list(repo.walk(args.ref, limit=args.number))
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    if not nodes:
        print("(no commits yet)")
        return 0
    for node in nodes:
        if args.oneline:
            print(_fmt_node_oneline(node))
        else:
            print(f"node {node.id}")
            print(f"Author: {node.actor}  <key:{node.pubkey[:16]}…>")
            print(f"Date:   {node.wall_time}")
            print(f"Kind:   {node.kind}   Seq: {node.seq}   Seed: {node.seed}")
            if node.parents:
                print("Parents: " + " ".join(p[:12] for p in node.parents))
            print(f"Context: {node.context_root[:12]}…")
            summary = _payload_summary(node)
            if summary:
                print(f"        {summary}")
            print()
    return 0


def _payload_summary(node) -> str:
    p = node.payload if isinstance(node.payload, dict) else {}
    if node.kind in ("system", "user", "note"):
        t = str(p.get("text", ""))[:80].replace("\n", " ")
        return t
    if node.kind == "inference":
        return f"model={p.get('model')}"
    if node.kind == "tool_call":
        return f"{p.get('name')}({json.dumps(p.get('arguments', {}))[:60]})"
    if node.kind == "tool_result":
        return f"{p.get('name')} -> ok={p.get('ok')}"
    if node.kind == "observation":
        return f"source={p.get('source')}"
    return ""


def cmd_show(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        node = repo.get_node(repo.resolve_ref(args.ref))
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(node.to_dict(), indent=2, sort_keys=True))
    if args.context:
        try:
            messages = repo.get_context(node.context_root)
            print("--- context ---")
            print(json.dumps(messages, indent=2))
        except RepoError as exc:
            print(f"cairn: {exc}", file=sys.stderr)
            return 1
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        a = repo.get_node(repo.resolve_ref(args.a))
        b = repo.get_node(repo.resolve_ref(args.b))
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    changed = []
    for field in ("kind", "actor", "seed", "seq", "context_root"):
        va, vb = getattr(a, field), getattr(b, field)
        if va != vb:
            changed.append(f"{field}: {va} -> {vb}")
    pa = json.dumps(a.payload, sort_keys=True, indent=2).splitlines()
    pb = json.dumps(b.payload, sort_keys=True, indent=2).splitlines()
    import difflib
    payload_diff = list(difflib.unified_diff(
        pa, pb, fromfile=a.short(), tofile=b.short(), lineterm=""))
    print(f"diff {a.short()}..{b.short()}")
    for c in changed:
        print(f"  {c}")
    if payload_diff:
        print("  payload:")
        for line in payload_diff:
            print(f"  {line}")
    if not changed and not payload_diff:
        print("  (identical preimage fields)")
    return 0


def cmd_checkout(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        oid = repo.checkout(args.ref, new_branch=args.branch)
    except (RepoError, ValueError) as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    branch = repo.current_branch()
    if args.branch:
        print(f"Switched to a new branch '{args.branch}'")
    elif branch:
        print(f"Switched to branch '{branch}'")
    else:
        print(f"Detached HEAD at {_short(oid)}")
    return 0


def cmd_branch(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        if args.delete:
            repo.delete_branch(args.delete)
            print(f"Deleted branch '{args.delete}'")
        elif args.list or args.name is None:
            current = repo.current_branch()
            for b in repo.list_branches():
                head = repo.get_branch(b)
                marker = "*" if b == current else " "
                print(f"{marker} {b:20} {_short(head)}")
        else:
            at = args.at or repo.get_head()
            if at is None:
                print("cairn: no commits yet", file=sys.stderr)
                return 1
            repo.create_branch(args.name, at=at)
            print(f"Created branch '{args.name}' at {_short(at)}")
    except (RepoError, ValueError) as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    repo = _repo_here()
    report = _verify_repo(repo, refs=args.refs or None)
    print(f"Checked {report.nodes_checked} nodes, "
          f"{report.blobs_checked} context blobs "
          f"across refs: {', '.join(report.refs_checked) or '(none)'}")
    if report.orphans:
        print(f"Note: {len(report.orphans)} orphaned node(s) on disk, "
              f"reachable from no ref:")
        for oid in report.orphans[:10]:
            print(f"  - {oid[:16]}…")
        if len(report.orphans) > 10:
            print(f"  … and {len(report.orphans) - 10} more")
    if report.identity_notes:
        print("Identity notes (actor names are self-asserted; keys are not):")
        for n in report.identity_notes:
            print(f"  - {n}")
    if report.ok:
        print("OK: all node ids recomputed, all signatures valid, "
              "all parents and context blobs present.")
        return 0
    print("FAILED:")
    for e in report.errors:
        print(f"  - {e}")
    return 1


def _load_agent(spec: str):
    if ":" not in spec:
        raise ValueError("expected --agent as module:func")
    mod_name, func_name = spec.rsplit(":", 1)
    sys.path.insert(0, os.getcwd())
    mod = importlib.import_module(mod_name)
    fn = getattr(mod, func_name)
    if not callable(fn):
        raise ValueError(f"{spec} is not callable")
    return fn


def cmd_replay(args: argparse.Namespace) -> int:
    repo = _repo_here()
    if args.check:
        # Dry run: verify the trace replays deterministically in principle.
        try:
            trace = replaymod.linear_trace(repo, args.ref)
        except replaymod.ReplayError as exc:
            print(f"cairn: {exc}", file=sys.stderr)
            return 1
        # Dry run: check every node in the trace recomputes and verifies.
        ok = True
        for node in trace:
            try:
                node.verify_id()
                node.verify_signature()
            except Exception as exc:  # noqa: BLE001
                print(f"cairn: node {node.short()}: {exc}", file=sys.stderr)
                ok = False
        print(f"Trace of {len(trace)} nodes "
              f"({trace[0].short()}..{trace[-1].short()}): "
              f"{'replayable' if ok else 'CORRUPT'}")
        return 0 if ok else 1
    if not args.agent:
        print("cairn: need --agent module:func or --check", file=sys.stderr)
        return 2
    try:
        agent_fn = _load_agent(args.agent)
    except (ImportError, ValueError, AttributeError) as exc:
        print(f"cairn: cannot load agent: {exc}", file=sys.stderr)
        return 1
    try:
        report = replaymod.run(
            repo, args.ref, agent_fn,
            branch=args.branch, start=args.start,
        )
    except replaymod.ReplayError as exc:
        print(f"cairn: replay failed: {exc}", file=sys.stderr)
        return 1
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    print(f"Replayed {report.nodes_replayed} nodes onto branch '{report.branch}'")
    print(f"Node ids identical to recording: "
          f"{'YES — deterministic replay proven' if report.ids_match else 'no (forked trace)'}")
    for r in report.rebased_models:
        print(f"Rebased model: {r}")
    if report.result is not None:
        print(f"Agent returned: {report.result!r}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    repo = _repo_here()
    try:
        manifest = exportmod.export_bundle(repo, args.ref, args.output)
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {manifest['objects']} objects to {args.output}")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    try:
        repo = exportmod.import_bundle(args.dest or os.getcwd(),
                                       args.bundle, force=args.force)
    except RepoError as exc:
        print(f"cairn: {exc}", file=sys.stderr)
        return 1
    print(f"Imported into {repo.dotdir}")
    return 0


def cmd_key(args: argparse.Namespace) -> int:
    _, pk = keys.load_or_create()
    print(f"Fingerprint: {keys.key_fingerprint(pk)}")
    print(f"Public key:  {pk.hex()}")
    if args.armor:
        print(keys.export_public_key_armored())
    if args.rebind:
        from .store import Repo
        repo = _repo_here()
        repo.rebind_identity(pk.hex(), args.rebind)
        print(f"Rebound key {pk.hex()[:16]}… to actor {args.rebind!r}")
    elif not args.armor:
        from .store import Repo
        repo = _repo_here()
        entry = repo.list_identities().get(pk.hex())
        if entry:
            print(f"Bound actor: {entry['actor']!r}")
    return 0


def cmd_reflog(args: argparse.Namespace) -> int:
    from .store import Repo
    repo = _repo_here()
    entries = repo.read_reflog(args.branch)
    if not entries:
        print(f"(no reflog for branch {args.branch!r})")
        return 0
    for e in entries:
        ts = datetime.fromtimestamp(e["time"],
                                    tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        old = e["old"][:12] if e["old"] != "0" * 64 else "(create)"
        print(f"{old} -> {e['new'][:12]}  {ts}Z  {e['actor']}")
    return 0


def cmd_commit(args: argparse.Namespace) -> int:
    from .session import DEFAULT_ACTOR, Session

    repo = _repo_here()
    with Session(repo, actor=args.actor or DEFAULT_ACTOR) as s:
        if args.kind == "observation":
            node = s.observation(args.source, {"message": args.message})
        else:
            node = s.note(args.message)
    print(f"committed {node.short()} [{node.kind}]")
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cairn",
        description="Content-addressed state graph and deterministic replay "
                    "for AI agent executions.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("init", help="Create a new .cairn repository")
    sp.add_argument("path", nargs="?", default=None)
    sp.add_argument("--exist-ok", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("status", help="Show HEAD, branch, and object counts")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("log", help="Show the node history")
    sp.add_argument("ref", nargs="?", default=None)
    sp.add_argument("--oneline", action="store_true")
    sp.add_argument("-n", "--number", type=int, default=None)
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("show", help="Show a node in full")
    sp.add_argument("ref")
    sp.add_argument("--context", action="store_true",
                    help="Also print the context snapshot")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("diff", help="Diff two nodes' preimage fields")
    sp.add_argument("a")
    sp.add_argument("b")
    sp.set_defaults(func=cmd_diff)

    sp = sub.add_parser("checkout", help="Move HEAD to a branch or node")
    sp.add_argument("ref")
    sp.add_argument("-b", "--branch", default=None,
                    help="Create and switch to a new branch")
    sp.set_defaults(func=cmd_checkout)

    sp = sub.add_parser("branch", help="Create, delete, or list branches")
    sp.add_argument("name", nargs="?", default=None)
    sp.add_argument("at", nargs="?", default=None)
    sp.add_argument("-l", "--list", action="store_true")
    sp.add_argument("-d", "--delete", default=None)
    sp.set_defaults(func=cmd_branch)

    sp = sub.add_parser("verify", help="Cryptographically audit the repository")
    sp.add_argument("refs", nargs="*", default=None)
    sp.set_defaults(func=cmd_verify)

    sp = sub.add_parser("replay", help="Deterministically replay a trace")
    sp.add_argument("ref")
    sp.add_argument("--check", action="store_true",
                    help="Dry run: check the trace is replayable")
    sp.add_argument("--agent", default=None,
                    help="Agent entry point as module:func")
    sp.add_argument("--branch", default=None)
    sp.add_argument("--start", default=None,
                    help="Node id to resume after (skip trusted prefix)")
    sp.set_defaults(func=cmd_replay)

    sp = sub.add_parser("export", help="Export a trace to a .car bundle")
    sp.add_argument("ref")
    sp.add_argument("-o", "--output", required=True)
    sp.set_defaults(func=cmd_export)

    sp = sub.add_parser("import", help="Import a .car bundle")
    sp.add_argument("bundle")
    sp.add_argument("dest", nargs="?", default=None)
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("key", help="Show this machine's Cairn identity")
    sp.add_argument("--armor", action="store_true")
    sp.add_argument("--rebind", metavar="ACTOR",
                    help="Rebind this key to a different actor name")
    sp.set_defaults(func=cmd_key)
    sp = sub.add_parser("reflog", help="Show a branch's ref history")
    sp.add_argument("branch", nargs="?", default="main")
    sp.set_defaults(func=cmd_reflog)
    sp = sub.add_parser("commit", help="Record a note/observation node")
    sp.add_argument("-m", "--message", required=True,
                    help="Content of the note/observation")
    sp.add_argument("--kind", choices=["note", "observation"], default="note")
    sp.add_argument("--source", default="cli",
                    help="Observation source (with --kind observation)")
    sp.add_argument("--actor", default=None, help="Actor name for the node")
    sp.set_defaults(func=cmd_commit)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BrokenPipeError:
        return 0
    except KeyboardInterrupt:
        print("\ncairn: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

"""Cairn v0.1.0 adversarial stress harness.

Simulates thousands of agent sessions hitting one repository and reports
what breaks, what's slow, and what Cairn claims to handle but doesn't.

Usage:  .venv/bin/python stress/harness.py <scenario> [args]

Scenarios:
  swarm N P     - N processes x P commits each, separate branches (default 40x25)
  samebranch N  - N processes committing to the SAME branch (lock serialization)
  deep N        - one branch, N sequential commits; then time log/walk
  fanout N      - N branches from one node; time list/checkout
  bigpayload MB - commits carrying MB-sized context payloads
  tamper        - flip bytes in objects; check verify catches it
  adversarial   - weird inputs: unicode, huge seeds, empty, dup commits
"""

import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("CAIRN_KEY_FILE", os.path.join(tempfile.gettempdir(), "stress.key"))

from cairn import Session  # noqa: E402
from cairn import signing as signingmod  # noqa: E402

SK, _ = signingmod.generate_keypair()


def _worker_swarm(args):
    repo_path, worker_id, n_commits = args
    t0 = time.perf_counter()
    try:
        from cairn.store import Repo
        repo = Repo.find(repo_path)
        branch = f"w/{worker_id}"
        if branch not in repo.list_branches():
            repo.create_branch(branch)
        with Session.open(repo_path, actor=f"agent-{worker_id}",
                          branch=branch, secret_key=SK) as s:
            c = s.client()
            c.system(f"sys {worker_id}")
            for i in range(n_commits):
                c.note(f"worker {worker_id} step {i} " + "x" * 200)
        return (worker_id, True, time.perf_counter() - t0, None)
    except Exception as e:  # noqa: BLE001
        return (worker_id, False, time.perf_counter() - t0, f"{type(e).__name__}: {e}")


def scenario_swarm(nproc=40, per=25):
    d = tempfile.mkdtemp(prefix="cairn-swarm-")
    with Session.create(d, actor="init", secret_key=SK) as s:
        s.note("genesis")
    t0 = time.perf_counter()
    with mp.Pool(nproc) as pool:
        results = pool.map(_worker_swarm, [(d, i, per) for i in range(nproc)])
    dt = time.perf_counter() - t0
    ok = sum(1 for _, good, _, _ in results if good)
    fails = [(w, e) for w, good, _, e in results if not good]
    total_commits = nproc * (per + 1) + 1
    print(f"SWARM: {nproc} procs x {per} commits = {total_commits} nodes, "
          f"{ok}/{nproc} workers OK, wall {dt:.1f}s, "
          f"{total_commits/dt:.0f} commits/s")
    for w, e in fails[:5]:
        print(f"  FAIL worker {w}: {e}")
    # verify branch count
    from cairn.store import Repo
    repo = Repo.find(d)
    print(f"  branches: {len(repo.list_branches())}, "
          f"objects: {repo.count_objects()}")
    return d


def _worker_samebranch(args):
    repo_path, worker_id, n_commits = args
    try:
        with Session.open(repo_path, actor=f"agent-{worker_id}",
                          branch="shared", secret_key=SK) as s:
            for i in range(n_commits):
                s.note(f"shared {worker_id}/{i}")
        return (worker_id, True, None)
    except Exception as e:  # noqa: BLE001
        return (worker_id, False, f"{type(e).__name__}: {e}")


def scenario_samebranch(nproc=20, per=10):
    d = tempfile.mkdtemp(prefix="cairn-shared-")
    with Session.create(d, actor="init", secret_key=SK) as s:
        s.note("genesis")
    from cairn.store import Repo
    Repo.find(d).create_branch("shared")
    t0 = time.perf_counter()
    with mp.Pool(nproc) as pool:
        results = pool.map(_worker_samebranch, [(d, i, per) for i in range(nproc)])
    dt = time.perf_counter() - t0
    ok = sum(1 for _, good, _ in results if good)
    from cairn.store import Repo
    repo = Repo.find(d)
    chain = list(repo.walk(repo.get_head()))
    print(f"SAMEBRANCH: {nproc} procs x {per} on one branch: {ok}/{nproc} OK, "
          f"{dt:.1f}s, chain length {len(chain)} (expect {nproc*per+1})")
    for w, good, e in results:
        if not good:
            print(f"  FAIL worker {w}: {e}")


def scenario_deep(n=3000):
    d = tempfile.mkdtemp(prefix="cairn-deep-")
    with Session.create(d, actor="deep", secret_key=SK) as s:
        c = s.client()
        c.system("deep chain")
        t0 = time.perf_counter()
        for i in range(n):
            c.note(f"step {i}")
        dt = time.perf_counter() - t0
        head = s.head
    from cairn.store import Repo
    repo = Repo.find(d)
    t0 = time.perf_counter()
    chain = list(repo.walk(head))
    walk_dt = time.perf_counter() - t0
    print(f"DEEP: {n} commits in {dt:.1f}s ({n/dt:.0f}/s); "
          f"walk({len(chain)}) in {walk_dt:.2f}s")
    # log timing via CLI-ish path
    t0 = time.perf_counter()
    list(repo.walk(head))
    print(f"  second walk: {time.perf_counter()-t0:.2f}s")


def scenario_fanout(n=2000):
    d = tempfile.mkdtemp(prefix="cairn-fanout-")
    with Session.create(d, actor="fan", secret_key=SK) as s:
        s.note("genesis")
        base = s.head
        t0 = time.perf_counter()
        for i in range(n):
            s.checkout(base, new_branch=f"exp/{i:05d}")
            s.note(f"branch {i}")
        dt = time.perf_counter() - t0
    from cairn.store import Repo
    repo = Repo.find(d)
    t0 = time.perf_counter()
    branches = repo.list_branches()
    list_dt = time.perf_counter() - t0
    print(f"FANOUT: {n} branches created in {dt:.1f}s; "
          f"list_branches({len(branches)}) in {list_dt:.2f}s")


def scenario_bigpayload(mb=5, n=10):
    d = tempfile.mkdtemp(prefix="cairn-big-")
    blob = "Z" * (mb * 1024 * 1024)
    with Session.create(d, actor="big", secret_key=SK) as s:
        c = s.client()
        t0 = time.perf_counter()
        for i in range(n):
            c.note(blob[:100] + f" #{i}", )
            # big payload via observation data
            s.observation("sensor", {"chunk": blob, "i": i})
        dt = time.perf_counter() - t0
    import subprocess
    size = subprocess.run(["du", "-sb", d], capture_output=True, text=True).stdout
    print(f"BIGPAYLOAD: {n} x ~{mb}MB observations in {dt:.1f}s; repo size: {size.strip()}")


def scenario_tamper():
    d = tempfile.mkdtemp(prefix="cairn-tamper-")
    with Session.create(d, actor="t", secret_key=SK) as s:
        s.note("one")
        s.note("two")
        head = s.head
    from cairn.store import Repo
    from cairn import verify as verify_mod
    repo = Repo.find(d)
    # 1. flip a byte in a node object
    node = repo.get_node(head)
    import glob
    obj_path = os.path.join(d, ".cairn", "objects", head[:2], head[2:])
    with open(obj_path, "rb") as f:
        data = bytearray(f.read())
    data[len(data)//2] ^= 0xFF
    with open(obj_path, "wb") as f:
        f.write(bytes(data))
    try:
        repo.get_node(head)
        print("TAMPER-1: FAIL - corrupted object read without error!")
    except Exception as e:  # noqa: BLE001
        print(f"TAMPER-1: OK - corrupted object rejected ({type(e).__name__})")
    # 2. verify should catch it
    rep = verify_mod(repo)
    print(f"TAMPER-2: verify ok={rep.ok}")
    # 3. rewind a ref to an older node (equivocation by the same key)
    repo2_path = tempfile.mkdtemp(prefix="cairn-rewind-")
    with Session.create(repo2_path, actor="t", secret_key=SK) as s:
        s.note("a"); mid = s.head; s.note("b")
    r2 = Repo.find(repo2_path)
    r2.set_head(mid)  # history rewrite
    print(f"TAMPER-3: ref rewind allowed silently (head now {r2.get_head()[:8]}); "
          f"verify ok={verify_mod(r2).ok} <- no detection of history rewrite")


def scenario_adversarial():
    d = tempfile.mkdtemp(prefix="cairn-adv-")
    findings = []
    with Session.create(d, actor="adv", secret_key=SK) as s:
        c = s.client()
        # unicode / emoji
        c.note("héllo wörld 🎉\u0000null byte?")
        findings.append("unicode+null committed")
        # huge seed
        s.inference({"q": 1}, {"a": 1}, model="m", seed=2**64 - 1)
        findings.append("max uint64 seed ok")
        try:
            s.inference({"q": 1}, {"a": 1}, model="m", seed=2**64)
            findings.append("PROBLEM: seed=2**64 accepted")
        except ValueError:
            findings.append("seed=2**64 rejected")
        # empty payloads
        c.note("")
        findings.append("empty note ok")
        # deeply nested payload
        deep = {"a": 1}
        for _ in range(200):
            deep = {"nest": deep}
        try:
            c.note("x")
            s.observation("deep", deep)
            findings.append("200-deep nesting committed")
        except Exception as e:  # noqa: BLE001
            findings.append(f"deep nesting failed: {type(e).__name__}")
        # duplicate identical commit (idempotency)
        n1 = s.note("same")
        n2 = s.note("same")
        findings.append(f"dup commits: same id={n1.id == n2.id} "
                        f"(chain len grows: {n1.id != n2.id})")
        # float payload (canonical JSON floats banned?)
        try:
            s.observation("f", {"v": 0.1 + 0.2})
            findings.append("float payload accepted")
        except Exception as e:  # noqa: BLE001
            findings.append(f"float rejected: {type(e).__name__}")
    for f in findings:
        print(f"  ADV: {f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    args = sys.argv[2:]
    if which in ("swarm", "all"):
        scenario_swarm(int(args[0]) if args else 40, int(args[1]) if len(args) > 1 else 25)
    if which in ("samebranch", "all"):
        scenario_samebranch()
    if which in ("deep", "all"):
        scenario_deep(int(args[0]) if args else 3000)
    if which in ("fanout",):
        scenario_fanout(int(args[0]) if args else 2000)
    if which in ("bigpayload",):
        scenario_bigpayload()
    if which in ("tamper", "all"):
        scenario_tamper()
    if which in ("adversarial", "all"):
        scenario_adversarial()

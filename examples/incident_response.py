"""Flagship scenario: incident response with time-travel and safe remediation.

Run:  python examples/incident_response.py

No API keys, no network, no dependencies beyond Cairn itself.

The scenario, in seven steps:

  1. An autonomous SRE agent receives a high-latency alert.
  2. The agent searches the logs and observes database pool exhaustion.
  3. The agent takes a disastrous action: restart_database_cluster().
  4. Execution halts. A developer runs ``cairn checkout`` back to step 2.
  5. The developer changes the steering constraint: scale read replicas
     instead of restarting the cluster.
  6. The agent executes the alternative path, resolves the alert, logs success.
  7. ``cairn replay`` proves bit-identical offline verification in <2 seconds.

Every step is a signed, content-addressed node. The failed timeline is never
deleted — it stays on ``main`` as the audit trail of what went wrong.
"""

import json
import os
import sys
import tempfile
import time
# Runnable straight from a source checkout.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("CAIRN_KEY_FILE",
                       os.path.join(tempfile.gettempdir(), "cairn-incident.key"))

from cairn import Session, replay_run  # noqa: E402

SYS = ("You are an autonomous SRE agent. Diagnose production incidents and "
       "act with the least invasive effective remediation first.")
ALERT = {"service": "checkout-api", "p99_ms": 2400,
         "error_rate": 0.22, "region": "iad"}


# ------------------------------------------------------------------ mock world
class MockWorld:
    """Deterministic stand-ins for the LLM, the log store, and the infra."""

    def llm(self, request, model=None, seed=None):
        # The model always reaches the same diagnosis for this alert.
        return {"diagnosis": "primary db connection pool exhausted",
                "confidence": 0.94}

    def search_logs(self, args):
        return {"lines_scanned": 18412,
                "finding": "pool exhaustion: 500/500 connections held > 90s",
                "top_waiter": "checkout-api"}

    def restart_database_cluster(self, args):
        # THE DISASTER: restarting the cluster drops in-flight connections.
        return {"status": "disaster",
                "detail": "cluster restart dropped 412 in-flight connections",
                "p99_ms": 30000}

    def scale_read_replicas(self, args):
        return {"status": "ok", "p99_ms": 120,
                "detail": f"read replicas scaled to {args['replicas']}"}

    def tools(self):
        return {"search_logs": self.search_logs,
                "restart_database_cluster": self.restart_database_cluster,
                "scale_read_replicas": self.scale_read_replicas}


# ------------------------------------------------------------- agent policies
def investigate(client):
    """Steps 1-2: receive the alert, search logs, form a hypothesis."""
    client.system(SYS)
    client.observation("pagerduty", ALERT)
    logs = client.tool("search_logs",
                       {"service": "checkout-api", "window": "15m"})
    # NOTE: serialize deterministically (sort_keys). A bare str(dict) bakes
    # in insertion order, which does not survive the JSON round-trip through
    # the object store — the replay harness would (correctly) flag the
    # resulting string as a divergence.
    context = json.dumps({"alert": ALERT, "logs": logs}, sort_keys=True)
    plan = client.infer(
        {"messages": [{"role": "user", "content": context}]},
        model="sre-1", seed=7)
    return plan


def disastrous_path(client):
    """Step 3: follow the naive policy — restart the database cluster."""
    investigate(client)
    return client.tool("restart_database_cluster", {"cluster": "checkout-db"})


# The developer's steering constraint, changed in step 5.
STEERING = {"on_pool_exhaustion": "restart_database_cluster"}


def remediate_from_hypothesis(client):
    """Continue from the step-2 hypothesis: apply the steering constraint,
    act, and log the outcome."""
    action = STEERING["on_pool_exhaustion"]
    if action == "restart_database_cluster":
        result = client.tool("restart_database_cluster",
                             {"cluster": "checkout-db"})
    else:
        result = client.tool("scale_read_replicas",
                             {"cluster": "checkout-db", "replicas": 5})
    client.note("resolved: scaled read replicas instead of cluster restart; "
                "p99 back to 120ms")
    return result


def safe_path(client):
    """The full remediated timeline (used for replay verification)."""
    plan = investigate(client)
    assert plan["diagnosis"] == "primary db connection pool exhausted"
    return remediate_from_hypothesis(client)


# ------------------------------------------------------------------ scenario
def main(workdir=None):
    t0 = time.perf_counter()
    workdir = workdir or tempfile.mkdtemp(prefix="cairn-incident-")
    world = MockWorld()

    print("== step 1-3: the incident, and the disastrous action ==")
    with Session.create(workdir, actor="sre-agent") as s:
        client = s.client(llm=world.llm, tools=world.tools())
        result = disastrous_path(client)
        main_head = s.head
        repo = s.repo
    print(f"   restart result: {result['status']} — {result['detail']}")

    # Step 4: halt, and check out the hypothesis node (end of step 2).
    nodes = list(repo.walk(main_head))
    hypothesis = next(n for n in nodes if n.kind == "inference")
    print(f"== step 4: halt. checkout {hypothesis.short()} "
          f"(step 2 hypothesis), branch `safe-remediation` ==")

    # Step 5: the developer changes the steering constraint.
    STEERING["on_pool_exhaustion"] = "scale_read_replicas"
    print("== step 5: steering constraint updated: "
          "on_pool_exhaustion -> scale_read_replicas ==")

    print("== step 6: alternative path resolves the alert ==")
    with Session(repo, actor="sre-agent", branch="main") as s2:
        # Roll back to the step-2 hypothesis; the branch continues from the
        # restored context, so only the new remediation steps are recorded.
        s2.checkout(hypothesis.id, new_branch="safe-remediation")
        client2 = s2.client(llm=world.llm, tools=world.tools())
        result2 = remediate_from_hypothesis(client2)
        safe_head = s2.head
    print(f"   remediation result: {result2['status']} — p99 {result2['p99_ms']}ms")

    # Step 7: prove the safe timeline replays bit-identically, offline.
    print("== step 7: offline replay proves bit-identical verification ==")
    t_replay = time.perf_counter()
    report = replay_run(repo, safe_head, safe_path,
                        branch="replay/safe-remediation")
    replay_dt = time.perf_counter() - t_replay
    assert report.ids_match, "replay diverged!"
    assert replay_dt < 2.0, f"replay took {replay_dt:.2f}s, budget is 2s"
    print(f"   replayed {report.nodes_replayed} nodes in {replay_dt:.2f}s, "
          f"ids identical: {report.ids_match}")

    elapsed = time.perf_counter() - t0
    print(f"\ndone in {elapsed:.2f}s — all offline, zero API keys. "
          f"The disaster stays on `main` as the audit trail.")
    return {"elapsed": elapsed,
            "replay_seconds": replay_dt,
            "main_head": main_head,
            "safe_head": safe_head,
            "p99_ms": result2["p99_ms"]}


if __name__ == "__main__":
    summary = main()
    assert summary["p99_ms"] == 120
    assert summary["replay_seconds"] < 2.0

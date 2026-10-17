"""Zero-key "aha!" demo: an SRE agent diagnoses a mock outage, fails, time-travels.

Run:  python examples/mock_agent.py

No API keys, no network, no dependencies beyond Cairn itself. A 5-step agent
diagnosing a mock server outage:

  1. system prompt        — "You are an SRE…"
  2. observation          — PagerDuty alert: api-7 p99 latency 2.4s
  3. inference (sre-1)    — hypothesis: DB pool exhausted → restart service
  4. tool restart_service — FAILS: pool still exhausted after restart
  5. (rollback!) checkout step 3, branch `alt-rule`, apply the alternative
     rule (scale the pool instead) → p99 back to 180ms. Resolved.

The whole thing runs offline in well under 3 seconds.
"""

import os
import sys
import tempfile
import time

# Runnable straight from a source checkout.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.environ.setdefault("CAIRN_KEY_FILE",
                       os.path.join(tempfile.gettempdir(), "cairn-mock-demo.key"))

from cairn import Session, replay_run  # noqa: E402

SYS = "You are an SRE. Diagnose the outage, then act with the least invasive fix first."
ALERT = {"service": "api-7", "p99_ms": 2400, "error_rate": 0.18, "region": "iad"}


# ------------------------------------------------------------------ mock world
class MockWorld:
    """Deterministic stand-ins for the LLM and the infrastructure."""

    def __init__(self, rule="restart-first"):
        self.rule = rule

    def llm(self, request, model=None, seed=None):
        if self.rule == "restart-first":
            return {"hypothesis": "db connection pool exhausted",
                    "action": {"tool": "restart_service",
                               "args": {"service": "api-7"}}}
        return {"hypothesis": "db connection pool exhausted",
                "action": {"tool": "scale_pool",
                           "args": {"service": "api-7", "pool_size": 200}}}

    def restart_service(self, args):
        return {"status": "failed",
                "reason": "pool still exhausted after restart",
                "p99_ms": 2350}

    def scale_pool(self, args):
        return {"status": "ok", "p99_ms": 180,
                "detail": f"pool scaled to {args['pool_size']}"}

    def tools(self):
        return {"restart_service": self.restart_service,
                "scale_pool": self.scale_pool}


def diagnose(client):
    """The original agent: follows the model's suggested action."""
    client.system(SYS)
    client.observation("pagerduty", ALERT)
    plan = client.infer({"messages": [{"role": "user", "content": str(ALERT)}]},
                        model="sre-1", seed=4242)
    action = plan["action"]
    result = client.tool(action["tool"], action["args"])
    return plan, result


def remediate(client):
    """The alternative rule: reuse the diagnosis, scale instead of restart."""
    client.system(SYS)
    client.observation("pagerduty", ALERT)
    plan = client.infer({"messages": [{"role": "user", "content": str(ALERT)}]},
                        model="sre-1", seed=4242)
    assert plan["hypothesis"] == "db connection pool exhausted"
    result = client.tool("scale_pool", {"service": "api-7", "pool_size": 200})
    client.note("resolved via pool scaling, no restart needed")
    return result


def remediate_tail(client):
    """Continuation after rolling back to step 3 (context already restored)."""
    result = client.tool("scale_pool", {"service": "api-7", "pool_size": 200})
    client.note("resolved via pool scaling, no restart needed")
    return result


def main(workdir=None):
    t0 = time.perf_counter()
    workdir = workdir or tempfile.mkdtemp(prefix="cairn-mock-")
    world = MockWorld(rule="restart-first")

    print("== step 1-4: record the incident ==")
    with Session.create(workdir, actor="sre-agent") as s:
        client = s.client(llm=world.llm, tools=world.tools())
        plan, result = diagnose(client)
        head = s.head
        repo = s.repo
    print(f"   hypothesis: {plan['hypothesis']}")
    print(f"   action result: {result['status']} — {result['reason']}")

    nodes = list(repo.walk(head))
    failed = next(n for n in nodes if n.kind == "tool_result")
    assert failed.payload["result"]["status"] == "failed"
    step3 = next(n for n in nodes if n.kind == "inference")
    print(f"== rollback: checkout {step3.short()} (step 3), branch `alt-rule` ==")

    world2 = MockWorld(rule="scale-first")
    with Session(repo, actor="sre-agent", branch="main") as s2:
        # Roll back to the diagnosis (step 3); context is restored from the
        # node's blob, so the branch continues exactly where it left off.
        s2.checkout(step3.id, new_branch="alt-rule")
        client2 = s2.client(llm=world2.llm, tools=world2.tools())
        result2 = remediate_tail(client2)
        alt_head = s2.head
    print(f"   alternative result: {result2['status']} — p99 {result2['p99_ms']}ms")

    print("== prove the fix replays deterministically ==")
    report = replay_run(repo, alt_head, remediate, branch="replay/alt-rule")
    assert report.ids_match
    print(f"   replayed {report.nodes_replayed} nodes, ids identical: "
          f"{report.ids_match}")

    elapsed = time.perf_counter() - t0
    print(f"\ndone in {elapsed:.2f}s — all offline, zero API keys.")
    return {"elapsed": elapsed,
            "main_head": head,
            "alt_head": alt_head,
            "p99_ms": result2["p99_ms"]}


if __name__ == "__main__":
    summary = main()
    assert summary["p99_ms"] == 180

"""End-to-end Cairn demo: record, inspect, time-travel, and replay.

Run:  python examples/calculator_agent.py

Uses a temporary directory and fake backends (no network, no API keys),
so it works anywhere. It exercises the whole Phase-1 loop:

1. Record an agent trace with the unified client.
2. Inspect it with the Python API (like `cairn log`).
3. Time-travel: check out an earlier node and branch.
4. Deterministically replay the trace and prove id-equality.
5. Verify the cryptographic audit trail.
"""

import os
import sys
import tempfile

# Make the example runnable straight from a source checkout
# (unnecessary once `pip install cairn-core` has been run).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

os.environ.setdefault("CAIRN_KEY_FILE",
                       os.path.join(tempfile.gettempdir(), "cairn-demo.key"))

from cairn import Session, replay_run, verify  # noqa: E402


# ---------------------------------------------------------------- fake backends
def fake_llm(request, model=None, seed=None):
    """Pretend LLM: answers 'a+b?' style questions deterministically."""
    q = request["messages"][-1]["content"]
    return {"role": "assistant", "content": f"Let me compute {q}",
            "model": model, "seed": seed}


def fake_calc(arguments):
    return {"result": eval(arguments["expr"])}  # demo only; don't do this at home


# ------------------------------------------------------------- the agent logic
def agent(client):
    """Written ONCE against the client interface — works for record & replay."""
    client.system("You are a precise calculator. Use the calc tool.")
    client.user("What is 12*34?")
    reply = client.infer(
        {"messages": [{"role": "user", "content": "12*34?"}]},
        model="demo-llm-1", seed=20261003,
    )
    print("  model said:", reply["content"])
    out = client.tool("calc", {"expr": "12*34"})
    print("  tool said:", out)
    client.note("demo finished")
    return out["result"]


def main():
    tmp = tempfile.mkdtemp(prefix="cairn-demo-")
    print(f"repo: {tmp}")

    # 1. Record -----------------------------------------------------------
    print("\n[1] recording…")
    with Session.create(tmp, actor="demo-agent") as s:
        client = s.client(llm=fake_llm, tools={"calc": fake_calc})
        result = agent(client)
        head = s.head
        repo = s.repo
    print("  result:", result)

    # 2. Inspect -----------------------------------------------------------
    print("\n[2] log:")
    for n in repo.walk(head):
        print(f"  {n.short()} [{n.kind:>11}] seq={n.seq}")

    # 3. Time travel --------------------------------------------------------
    print("\n[3] time travel: back to the user message, new branch")
    nodes = list(repo.walk(head))
    user_node = next(n for n in nodes if n.kind == "user")
    with Session(repo, actor="demo-agent", branch="main") as s2:
        s2.checkout(user_node.id, new_branch="second-thought")
        s2.user("Actually, what is 13*37?", seed=7)
        print("  branched at", user_node.short(), "-> head", s2.head[:12])

    # 4. Replay --------------------------------------------------------------
    print("\n[4] deterministic replay (no network, no tokens)…")
    report = replay_run(repo, head, agent, branch="replay/demo")
    print(f"  replayed {report.nodes_replayed} nodes; "
          f"ids identical: {report.ids_match}")
    assert report.ids_match, "replay must be bit-identical"

    # 5. Audit ----------------------------------------------------------------
    print("\n[5] cryptographic audit…")
    r = verify(repo)
    print(f"  checked {r.nodes_checked} nodes: {'OK' if r.ok else 'FAILED'}")
    assert r.ok

    print("\ndone — every claim above is backed by content hashes.")


if __name__ == "__main__":
    main()

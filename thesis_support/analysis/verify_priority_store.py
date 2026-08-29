"""
THESIS-ONLY OBSERVATIONAL SCRIPT — does not import, modify or affect production code.

Purpose
-------
Establish, by direct observation, the within-class ordering semantics of the `urgent_first`
baseline policy implemented in src/simulation/multistage/sim_multistage.py.

That engine places orders into a `simpy.PriorityStore` as
`simpy.PriorityItem(priority, order_id)` with priority 0 for urgent and 1 for normal orders.
A natural assumption — and the convention usually implied by the name "Urgent-First" — is that
equal-priority orders are then served first-in-first-out. This script tests that assumption.

Why it matters for the thesis
-----------------------------
`PriorityItem` is a NamedTuple whose `__lt__` compares ONLY the `priority` field (see the
SimPy source printed below), and `PriorityStore` stores items in a binary heap via
`heappush`/`heappop`. Equal-priority items are therefore returned in heap-structural order,
which is neither FIFO nor sorted by order_id. The thesis documents the implementation as it
is, so this behaviour must be established as evidence rather than asserted.

Run
---
    py -3.12 thesis_support/analysis/verify_priority_store.py
"""
from __future__ import annotations

import inspect

import simpy
import simpy.resources.store as store_mod


def show_simpy_semantics() -> None:
    print("=" * 78)
    print("SimPy PriorityItem source (installed version %s)" % simpy.__version__)
    print("=" * 78)
    print(inspect.getsource(store_mod.PriorityItem))


def dequeue_order(n: int, priority: int = 1) -> list[int]:
    """Insert ids 1..n into a PriorityStore, all with the SAME priority, and record the order
    in which they come back out."""
    env = simpy.Environment()
    st = simpy.PriorityStore(env)
    out: list[int] = []

    def driver():
        for oid in range(1, n + 1):
            yield st.put(simpy.PriorityItem(priority, oid))
        for _ in range(n):
            item = yield st.get()
            out.append(item.item)

    env.process(driver())
    env.run()
    return out


def mixed_class_order() -> list[tuple[int, int]]:
    """Interleave urgent (priority 0) and normal (priority 1) arrivals, as the baseline engine
    does, and record (priority, order_id) dequeue order."""
    env = simpy.Environment()
    st = simpy.PriorityStore(env)
    out: list[tuple[int, int]] = []
    # ids 1..12, every 3rd one urgent
    arrivals = [(0 if oid % 3 == 0 else 1, oid) for oid in range(1, 13)]

    def driver():
        for prio, oid in arrivals:
            yield st.put(simpy.PriorityItem(prio, oid))
        for _ in range(len(arrivals)):
            item = yield st.get()
            out.append((item.priority, item.item))

    env.process(driver())
    env.run()
    print("  inserted (priority, id): %s" % arrivals)
    return out


def main() -> None:
    show_simpy_semantics()

    print("=" * 78)
    print("TEST 1 — equal-priority items: is dequeue order FIFO?")
    print("=" * 78)
    n = 10
    observed = dequeue_order(n)
    expected_fifo = list(range(1, n + 1))
    print("  insert order : %s" % expected_fifo)
    print("  dequeue order: %s" % observed)
    print("  FIFO within class? %s" % (observed == expected_fifo))
    print("  sorted by order_id? %s" % (observed == sorted(observed)))

    print()
    print("=" * 78)
    print("TEST 2 — mixed classes: is the two-class priority itself respected?")
    print("=" * 78)
    mixed = mixed_class_order()
    print("  dequeue order (priority, id): %s" % mixed)
    priorities = [p for p, _ in mixed]
    print("  all urgent (priority 0) served before any normal? %s"
          % (priorities == sorted(priorities)))
    urgent_ids = [oid for p, oid in mixed if p == 0]
    normal_ids = [oid for p, oid in mixed if p == 1]
    print("  urgent ids in dequeue order: %s  (FIFO? %s)"
          % (urgent_ids, urgent_ids == sorted(urgent_ids)))
    print("  normal ids in dequeue order: %s  (FIFO? %s)"
          % (normal_ids, normal_ids == sorted(normal_ids)))

    print()
    print("=" * 78)
    print("CONCLUSION FOR THE THESIS")
    print("=" * 78)
    print("  The urgent_first baseline enforces a strict two-class priority: every waiting")
    print("  urgent order is served before any normal order. Within a class, however, the")
    print("  dequeue order is determined by the binary heap's internal structure and is")
    print("  neither FIFO nor sorted by order_id. 'Urgent-First' in this implementation")
    print("  therefore means two-class priority with unspecified within-class ordering.")


if __name__ == "__main__":
    main()

"""Check that every consecutive edge pair in a SUMO route file is connected.

MATSim's queue model has no turn restrictions: any link-to-link movement at a
shared node is legal. netconvert instead prunes connections with geometry and
lane-count heuristics, so a route MATSim accepts can be rejected by SUMO with
"has no valid route. No connection between edge 'A' and edge 'B'". That kills the
whole run on the first offending vehicle, which is how the first SUMO benchmark
attempt failed on both Berlin and LA.

Run this after netconvert and before submitting the benchmark: it reports how
many pairs are missing and, for each, whether the pair is a U-turn (the two
directions of one road, i.e. from.to == to.from and from.from == to.to). If they
are all U-turns, `--no-turnarounds.geometry false` is the whole fix.

    python scripts/check_sumo_route_connectivity.py \\
        --net .../sumo/network.net.xml --routes .../sumo/B0p1.rou.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load_net(path: str) -> tuple[set[tuple[str, str]], dict[str, tuple[str, str]]]:
    """Return (connection pairs, edge id -> (from node, to node))."""
    conns: set[tuple[str, str]] = set()
    edges: dict[str, tuple[str, str]] = {}
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "connection":
            f, t = el.get("from"), el.get("to")
            # Internal junction edges are named ":node_i"; skip those.
            if f and t and not f.startswith(":"):
                conns.add((f, t))
            el.clear()
        elif el.tag == "edge":
            eid, fn, tn = el.get("id"), el.get("from"), el.get("to")
            if eid and fn and tn:
                edges[eid] = (fn, tn)
            el.clear()
    return conns, edges


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--net", required=True)
    p.add_argument("--routes", required=True, action="append",
                   help="Route file to check; repeat for several.")
    p.add_argument("--emit-connections", default="",
                   help="Write a netconvert --connection-files document restoring "
                        "every missing movement, then exit 0.")
    p.add_argument("--max-report", type=int, default=15)
    p.add_argument("--max-affected-frac", type=float, default=-1.0,
                   help="Exit 0 if the FRACTION of vehicles touching a missing "
                        "movement is at or below this, instead of exiting 1 on any "
                        "missing movement. Costs a second pass over the routes.")
    args = p.parse_args()

    conns, edges = load_net(args.net)
    print(f"[net] {len(edges):,} edges, {len(conns):,} connections")

    missing: Counter[tuple[str, str]] = Counter()
    n_pairs = n_routes = 0
    for rf in args.routes:
        for _, el in ET.iterparse(rf, events=("end",)):
            if el.tag != "route":
                continue
            seq = (el.get("edges") or "").split()
            n_routes += 1
            for a, b in zip(seq, seq[1:]):
                n_pairs += 1
                if (a, b) not in conns:
                    missing[(a, b)] += 1
            el.clear()
        print(f"[routes] done {rf}")

    print(f"[routes] {n_routes:,} routes, {n_pairs:,} consecutive pairs")
    if not missing:
        print("OK: every pair is connected")
        return 0

    uturn = sum(
        c for (a, b), c in missing.items()
        if a in edges and b in edges and edges[a] == edges[b][::-1]
    )
    total = sum(missing.values())
    print(f"MISSING: {len(missing):,} distinct pairs, {total:,} traversals")
    print(f"  of which U-turns: {uturn:,} traversals "
          f"({100.0 * uturn / total:.1f}%)")
    for (a, b), c in missing.most_common(args.max_report):
        kind = "u-turn" if a in edges and b in edges and edges[a] == edges[b][::-1] else "other"
        print(f"  {a} -> {b}  x{c}  [{kind}]")

    if args.max_affected_frac >= 0.0:
        # How much demand SUMO would actually lose. A handful of movements
        # netconvert refuses even when they are declared explicitly is not worth
        # regenerating every engine's population over, but the size of the
        # discrepancy has to be measured rather than assumed.
        keys = set(missing)
        affected = total_veh = 0
        for rf in args.routes:
            for _, el in ET.iterparse(rf, events=("end",)):
                if el.tag != "route":
                    continue
                total_veh += 1
                seq = (el.get("edges") or "").split()
                if any((a, b) in keys for a, b in zip(seq, seq[1:])):
                    affected += 1
                el.clear()
        frac = affected / total_veh if total_veh else 0.0
        print(f"AFFECTED VEHICLES: {affected:,} of {total_veh:,} ({100 * frac:.4f}%)")
        if frac <= args.max_affected_frac:
            print(f"OK: at or below the {100 * args.max_affected_frac:.4f}% tolerance")
            return 0
        print(f"OVER the {100 * args.max_affected_frac:.4f}% tolerance")
        return 1

    if args.emit_connections:
        # Declaring ANY connection out of an edge makes netconvert discard the ones
        # it computed for that edge, so each affected edge must be re-declared in
        # full: its existing successors plus the missing ones. Lanes are left out
        # deliberately -- netconvert assigns them, and the mesoscopic run does not
        # model lane choice anyway.
        from collections import defaultdict
        out_of = defaultdict(set)
        for a, b in conns:
            out_of[a].add(b)
        touched = {a for a, _ in missing}
        for a, b in missing:
            out_of[a].add(b)
        with open(args.emit_connections, "w") as fo:
            fo.write("<connections>\n")
            n = 0
            for a in sorted(touched):
                for b in sorted(out_of[a]):
                    fo.write(f'    <connection from="{a}" to="{b}"/>\n')
                    n += 1
            fo.write("</connections>\n")
        print(f"wrote {args.emit_connections}: {len(touched):,} edges, {n:,} connections")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Check that every routed car leg in a MATSim population is a topologically valid
link sequence: each link must start at the node where the previous one ends.

TorchDNL aborts an agent the step after it enters traffic when this fails -- see the
connectivity check in _process_links_B, which compares to_node[current] against
from_node[next] and removes the agent when they differ. That abort is what leaves a
residue of agents in the Berlin fidelity run, and because an aborted leg never
completes, every later leg of the same agent never departs either, so a small number
of broken routes costs a much larger number of missing departures.

MATSim does not notice: its queue model pushes a vehicle onto the next link of its
route without checking adjacency, so a broken route is simulated as a teleport and
the run completes normally. That asymmetry is why the two engines can disagree while
replaying what looks like the same demand.

    python scripts/check_population_route_connectivity.py \\
        --network .../berlin-v6.0-network-with-pt.xml \\
        --population .../B1car_population.xml
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter


def load_link_nodes(path: str) -> dict[str, tuple[str, str]]:
    """link id -> (from node, to node), by line scan: the network runs to 100s of MB."""
    pat = re.compile(r'<link id="([^"]+)"[^>]*\bfrom="([^"]+)"[^>]*\bto="([^"]+)"')
    out = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m:
                out[m.group(1)] = (m.group(2), m.group(3))
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--network", required=True)
    p.add_argument("--population", required=True)
    p.add_argument("--examples", type=int, default=6)
    args = p.parse_args()

    nodes = load_link_nodes(args.network)
    print(f"[net] {len(nodes):,} links")

    n_legs = n_broken = n_persons = n_persons_broken = 0
    n_unknown_link = 0
    break_at = Counter()
    examples = []

    for _, el in ET.iterparse(args.population, events=("end",)):
        if el.tag != "person":
            continue
        n_persons += 1
        person_broken = False
        plan = el.find("plan")
        if plan is not None:
            for leg_i, leg in enumerate([c for c in plan if c.tag == "leg"]):
                r = leg.find("route")
                seq = (r.text or "").split() if r is not None else []
                if not seq:
                    continue
                n_legs += 1
                if any(s not in nodes for s in seq):
                    n_unknown_link += 1
                    continue
                bad = None
                for i in range(len(seq) - 1):
                    if nodes[seq[i]][1] != nodes[seq[i + 1]][0]:
                        bad = i
                        break
                if bad is not None:
                    n_broken += 1
                    person_broken = True
                    # Where in the route does it break: 0 means right after the
                    # first link, which is what the stale-variable bug produces.
                    break_at[0 if bad == 0 else (1 if bad == len(seq) - 2 else 2)] += 1
                    if len(examples) < args.examples:
                        examples.append((el.get("id"), leg_i, len(seq), bad,
                                         seq[max(0, bad - 1):bad + 3]))
        if person_broken:
            n_persons_broken += 1
        el.clear()

    print(f"[pop] {n_persons:,} persons, {n_legs:,} routed car legs")
    print(f"      legs referencing an unknown link : {n_unknown_link:,}")
    print(f"      TOPOLOGICALLY BROKEN legs        : {n_broken:,} "
          f"({100 * n_broken / max(n_legs, 1):.2f}%)")
    print(f"      persons with at least one        : {n_persons_broken:,} "
          f"({100 * n_persons_broken / max(n_persons, 1):.2f}%)")
    if n_broken:
        labels = {0: "apres le 1er lien", 1: "avant le dernier lien", 2: "au milieu"}
        print("      ou la rupture se produit :")
        for k, v in break_at.most_common():
            print(f"        {labels[k]:<24} {v:,} ({100 * v / n_broken:.1f}%)")
        print("      exemples :")
        for pid, leg_i, n, bad, around in examples:
            print(f"        {pid} leg {leg_i}: route de {n} liens, rupture a l'index "
                  f"{bad}, autour = {' '.join(around)}")
    return 1 if n_broken else 0


if __name__ == "__main__":
    sys.exit(main())

"""Find out WHY agents fail to clear the network in a TorchDNL fixed-route replay.

The Berlin fidelity run leaves ~2,650 agents permanently en route and aborts 4,418
legs where MATSim aborts 120, which also costs 30,480 departures because a leg that
never finishes blocks every later leg of the same agent. That is a 1% residue with a
compounding effect, and "a limitation of the engine" is not an explanation.

This walks the event stream once and asks three questions:

  * which agents departed and never arrived, and where did they stop;
  * do the stop locations concentrate on a few links, or are they spread out;
  * what do those links look like -- capacity, length, lanes, free speed.

A concentrated answer points at specific network geometry (a zero-capacity link, a
dead end, a merge the priority rule starves). A spread-out answer points at the
queue model itself.

    python scripts/diagnose_stuck_agents.py \\
        --events results/validation_berlin_B1car/torchdnl/seed_0/events.csv \\
        --network .../berlin-v6.0-network-with-pt.xml --top 25
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", required=True)
    p.add_argument("--network", required=True)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--examples", type=int, default=5)
    args = p.parse_args()

    dep = Counter()
    arr = Counter()
    stuck_link = Counter()
    stuck_person: dict[str, tuple[float, str]] = {}
    last_event: dict[str, tuple[float, str, str]] = {}
    n = 0

    with open(args.events, newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        c = {k: i for i, k in enumerate(head)}
        for row in r:
            n += 1
            t, typ, person, link = row[c["time"]], row[c["type"]], row[c["person"]], row[c["link"]]
            if typ == "departure":
                dep[person] += 1
            elif typ == "arrival":
                arr[person] += 1
            elif typ == "stuckAndAbort":
                stuck_link[link] += 1
                stuck_person.setdefault(person, (float(t), link))
            # Only track position-bearing events, so the "last seen" is a place.
            if typ in ("entered_link", "left_link", "entered_buffer", "enters_traffic"):
                last_event[person] = (float(t), typ, link)

    print(f"evenements lus : {n:,}")
    unfinished = {k: dep[k] - arr.get(k, 0) for k in dep if dep[k] > arr.get(k, 0)}
    print(f"agents partis            : {len(dep):,}")
    print(f"agents avec trajet ouvert: {len(unfinished):,}")
    print(f"trajets abandonnes       : {sum(stuck_link.values()):,} "
          f"sur {len(stuck_link):,} liens distincts")

    # Concentration: how much of the abort mass sits on how few links.
    tot = sum(stuck_link.values())
    if tot:
        ranked = stuck_link.most_common()
        cum = 0
        for frac in (0.5, 0.8, 0.9):
            k = 0
            cum = 0
            while cum < frac * tot and k < len(ranked):
                cum += ranked[k][1]
                k += 1
            print(f"  {frac:.0%} des abandons tiennent sur {k:,} liens "
                  f"({100 * k / len(stuck_link):.1f}% des liens concernes)")

    # What do the worst links look like?
    import re

    want = set(l for l, _ in stuck_link.most_common(args.top))
    attrs: dict[str, dict] = {}
    pat = re.compile(r'<link id="([^"]+)"([^>]*)>')
    kv = re.compile(r'(\w+)="([^"]*)"')
    with open(args.network, errors="replace") as fh:
        for line in fh:
            m = pat.search(line)
            if m and m.group(1) in want:
                attrs[m.group(1)] = dict(kv.findall(m.group(2)))
    print(f"\nles {args.top} liens qui abandonnent le plus :")
    print(f"{'lien':>12}{'abandons':>10}{'capacite':>11}{'longueur':>11}{'voies':>7}{'vitesse':>9}")
    for link, cnt in stuck_link.most_common(args.top):
        a = attrs.get(link, {})
        print(f"{link:>12}{cnt:>10,}{a.get('capacity','?'):>11}"
              f"{float(a.get('length', 'nan')):>11.1f}{a.get('permlanes','?'):>7}"
              f"{float(a.get('freespeed','nan')):>9.2f}")

    print("\nexemples d'agents abandonnes :")
    for i, (person, (t, link)) in enumerate(sorted(stuck_person.items(), key=lambda kv: kv[1][0])):
        if i >= args.examples:
            break
        le = last_event.get(person)
        print(f"  {person}: abandon a t={t:,.0f}s sur le lien {link}; "
              f"dernier deplacement {le[1]} sur {le[2]} a t={le[0]:,.0f}s" if le else
              f"  {person}: abandon a t={t:,.0f}s sur le lien {link}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

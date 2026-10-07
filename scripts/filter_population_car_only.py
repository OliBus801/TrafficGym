"""Rewrite a MATSim population so that every kept plan is PURELY car: only car
legs, only the activities those legs actually connect.

Why this is necessary rather than cosmetic. Berlin's plans route a car trip as a
walk-access / car / walk-egress sandwich around two zero-duration placeholder
activities:

    activity home_86400   end_time="09:25:04"
    leg      walk
    activity "car interaction"  max_dur="00:00:00"
    leg      car   <route>...</route>
    activity "car interaction"  max_dur="00:00:00"
    leg      walk
    activity work_37800   end_time="16:47:10"
    ... and the same sandwich again for the trip home

That shape breaks the comparison at BOTH ends.

MATSim's side: it simulates the walk, pt, bike and ride legs as teleports. On
Berlin's 10% sample that is 874,828 non-car legs against 392,671 car ones -- MATSim
is charged for 2.2x more legs than TorchDNL ever sees, which violates the
benchmark's same-work rule and biases every speed-up in our favour.

TorchDNL's side, which is worse: both parsers collect one activity boundary per
activity but then keep only the FIRST (number of car legs - 1) of them
(scenario_loader.py and validation/matsim_io.py both do
`act_end_times[:num_boundaries]`). With the sandwich above, a two-car-leg agent
takes the zero-duration "car interaction" as the activity before its second leg,
so that leg departs the instant the first one arrives instead of at 16:47:10. The
afternoon trip is simulated in the morning. Nothing errors; the demand profile is
simply wrong, and multi-leg agents pile into the peak.

Stripping the file to car-only fixes both without touching the engine: the plan
becomes a clean alternating activity/leg chain, so the positional pairing the
parsers do is correct by construction, and MATSim is handed exactly the legs
TorchDNL replays.

Activities are dropped when they are pure routing placeholders -- by default any
whose type contains "interaction", which is MATSim's own convention for them. Real
activities are always kept, including when consecutive car legs leave two in a row.

    python scripts/filter_population_car_only.py \\
        --input  tamarl/data/scenarios/berlin/B1_population.xml \\
        --output tamarl/data/scenarios/berlin/B1car_population.xml
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET

HEADER_XMLDECL = '<?xml version="1.0" encoding="utf-8"?>\n'
FOOTER = "</population>\n"


def source_doctype(path: str, default_version: int = 5) -> str:
    """Echo the source file's DOCTYPE, so the output declares the same schema.

    Berlin is population_v6 while older scenarios are v5; hardcoding one made
    MATSim reject the other, and iterparse ignores the DTD so the mismatch only
    surfaced once MATSim read the file.
    """
    with open(path, errors="replace") as fh:
        for _ in range(10):
            line = fh.readline()
            if not line:
                break
            if "<!DOCTYPE" in line:
                return line if line.endswith("\n") else line + "\n"
    return ('<!DOCTYPE population SYSTEM '
            f'"http://www.matsim.org/files/dtd/population_v{default_version}.dtd">\n')


def selected_plan(person: ET.Element) -> ET.Element | None:
    first = None
    for child in person:
        if child.tag != "plan":
            continue
        if child.get("selected") == "yes":
            return child
        if first is None:
            first = child
    return first


def is_car_leg(el: ET.Element) -> bool:
    if el.get("mode") != "car":
        return False
    route = el.find("route")
    return route is not None and bool((route.text or "").strip())


def _collapse_activity_runs(items: list[ET.Element]) -> list[ET.Element]:
    """Force the plan into a strictly alternating activity / car-leg / activity chain.

    Removing the non-car legs can leave two real activities adjacent, because the
    trip between them was walked, cycled or taken by transit. On Berlin's 10%
    sample that happens 60,990 times. It matters because both parsers pair
    activities to legs POSITIONALLY and keep the first (legs - 1) boundaries, so a
    spare activity shifts every later boundary by one and reintroduces exactly the
    misalignment this filter exists to remove.

    Which activity of a run to keep follows from what the boundary is for:

      * a run that PRECEDES a car leg releases that leg, so the LAST activity of
        the run is the one whose end time matters;
      * the run after the final car leg is where the traveller arrives, so the
        FIRST of it is the real destination.

    The intermediate activities of a run are dropped outright. They were reached by
    a mode this comparison does not simulate, and keeping them would mean the file
    describes trips neither engine performs.
    """
    out: list[ET.Element] = []
    i, n = 0, len(items)
    seen_leg_after = any(x.tag == "leg" for x in items)
    while i < n:
        if items[i].tag == "leg":
            out.append(items[i])
            i += 1
            continue
        j = i
        while j < n and items[j].tag != "leg":
            j += 1
        run = items[i:j]
        trailing = j >= n  # no car leg after this run
        out.append(run[0] if trailing else run[-1])
        i = j
    del seen_leg_after
    return out


def rewrite_person(person: ET.Element, placeholder: str) -> bool:
    """Reduce the person to its car-only selected plan. False if nothing is left."""
    plan = selected_plan(person)
    if plan is None:
        return False

    kept: list[ET.Element] = []
    n_car = 0
    for el in list(plan):
        if el.tag == "leg":
            if is_car_leg(el):
                kept.append(el)
                n_car += 1
        elif el.tag in ("act", "activity"):
            if placeholder and placeholder in (el.get("type") or "").lower():
                continue
            kept.append(el)
        else:
            kept.append(el)

    if n_car == 0:
        return False

    # A plan must start and end on an activity. Leading or trailing legs can only
    # appear if the placeholder stripping removed a boundary activity that was not
    # flanked by a real one, which would leave a leg with nothing to depart from.
    while kept and kept[0].tag == "leg":
        kept.pop(0)
        n_car -= 1
    while kept and kept[-1].tag == "leg":
        kept.pop()
        n_car -= 1
    if n_car == 0:
        return False

    kept = _collapse_activity_runs(kept)

    for el in list(plan):
        plan.remove(el)
    for el in kept:
        plan.append(el)

    # Keep only the plan we rewrote; a second, unselected plan is dead weight in a
    # file that already runs to gigabytes, and neither engine looks at it.
    for child in list(person):
        if child.tag == "plan" and child is not plan:
            person.remove(child)
    plan.set("selected", "yes")
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--placeholder-marker", default="interaction",
                   help="Drop activities whose type contains this (case-insensitive). "
                        "MATSim names its routing placeholders '<mode> interaction'. "
                        "Pass an empty string to keep every activity.")
    args = p.parse_args()

    n_in = n_out = n_car_legs = n_acts = 0
    with open(args.output, "w") as out:
        out.write(HEADER_XMLDECL + source_doctype(args.input) + "<population>\n")
        for _, elem in ET.iterparse(args.input, events=("end",)):
            if elem.tag != "person":
                continue
            n_in += 1
            if rewrite_person(elem, args.placeholder_marker.lower()):
                n_out += 1
                plan = selected_plan(elem)
                n_car_legs += sum(1 for c in plan if c.tag == "leg")
                n_acts += sum(1 for c in plan if c.tag in ("act", "activity"))
                out.write(ET.tostring(elem, encoding="unicode"))
                out.write("\n")
            elem.clear()
            if n_in % 200000 == 0:
                print(f"  ... {n_in:,} read, {n_out:,} kept", flush=True)
        out.write(FOOTER)

    if n_out == 0:
        print("ABORT: no person survived the car-only filter", file=sys.stderr)
        return 1
    print(f"{n_in:,} persons in -> {n_out:,} kept ({100 * n_out / n_in:.1f}%)")
    print(f"  car legs   : {n_car_legs:,} ({n_car_legs / n_out:.2f} per person)")
    print(f"  activities : {n_acts:,} ({n_acts / n_out:.2f} per person)")
    if n_acts != n_car_legs + n_out:
        print(f"WARNING: expected activities = car legs + persons "
              f"({n_car_legs + n_out:,}); a plan is not a clean alternating chain",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

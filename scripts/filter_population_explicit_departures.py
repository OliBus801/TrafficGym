"""Keep only the agents whose demand BOTH engines interpret identically.

Why this exists
---------------
TorchDNL derives a leg's departure from an activity's `end_time`. Berlin v6.0
expresses most activity durations with `max_dur` instead: in the 10% sample only
25.2% of activities carry `end_time`, while 61.2% carry `max_dur` alone. Feeding
that population to both engines would not compare them on the same workload --
MATSim would honour `max_dur` and spread departures across the day, while
TorchDNL would see no departure time and dispatch those agents at t=0, producing
a midnight spike. The wall-clock numbers would differ because the demand differs,
which is precisely the confound the benchmark exists to avoid.

Materialising absolute end times from `max_dur` (walking each plan with a clock)
is the complete fix and a larger job. This script is the cheap, honest
alternative for a reconnaissance run: restrict the scenario to the agents whose
departure times are already explicit, and hand the SAME filtered file to both
engines. The result is a smaller but well-defined scenario -- reportable as
"Berlin 10%, agents with explicit departure times (N of M)" -- rather than a
larger ill-defined one.

Streams with iterparse and clears as it goes: the decompressed Berlin plans are
~3 GB, which a read-the-whole-file approach would turn into ~10 GB of RAM.

Usage:
    python scripts/filter_population_explicit_departures.py \\
        --input berlin/S0p1_population.xml --output berlin/S0p1exp_population.xml
"""

from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET

HEADER_XMLDECL = '<?xml version="1.0" encoding="utf-8"?>\n'
FOOTER = "</population>\n"

# MATSim resolves vehicleRefId against a vehicles container it builds from the
# scenario's own conventions. Berlin's routes name "<person>_car", which neither
# the default vehicle source nor modeVehicleTypesFromVehiclesData could resolve
# here, and cloning a person would in any case leave the copy pointing at the
# original's vehicle. TorchDNL never reads the attribute. Dropping it lets MATSim
# create one default vehicle per person and removes the whole class of problem;
# the simulated physics is unchanged because every vehicle uses the same type.
VEHICLE_REF_RE = re.compile(r'\s+vehicleRefId="[^"]*"')


def source_doctype(path, default_version=5):
    """Return the source file's DOCTYPE line, so the output declares the same schema.

    Hard-coding population_v5 silently produces files MATSim refuses: Berlin v6.0
    plans carry <attributes> blocks that only population_v6 defines. TorchDNL's
    iterparse ignores the DTD entirely, so the mismatch shows up only once the
    other engine reads the file.
    """
    with open(path, errors="replace") as f:
        for _ in range(5):
            line = f.readline()
            if "<!DOCTYPE" in line:
                return line if line.endswith("\n") else line + "\n"
    return (
        '<!DOCTYPE population SYSTEM '
        f'"http://www.matsim.org/files/dtd/population_v{default_version}.dtd">\n'
    )



def selected_plan(person: ET.Element) -> ET.Element | None:
    fallback = None
    for child in person:
        if child.tag == "plan":
            if child.get("selected") == "yes":
                return child
            if fallback is None:
                fallback = child
    return fallback


def keep(person: ET.Element) -> bool:
    """Keep a person iff both engines would agree on when and where they travel."""
    plan = selected_plan(person)
    if plan is None:
        return False
    first_act_has_end = False
    seen_act = False
    has_routed_car_leg = False
    for el in plan:
        if el.tag in ("act", "activity"):
            if not seen_act:
                seen_act = True
                # Either form of explicit timing is now understood by both
                # engines: an absolute end_time, or a relative max_dur that
                # TorchDNL turns into a wake-up via act_durations.
                first_act_has_end = (
                    el.get("end_time") is not None
                    or el.get("max_dur") is not None
                    or el.get("duration") is not None
                )
        elif el.tag == "leg" and el.get("mode") == "car":
            route = el.find("route")
            if route is not None and (route.text or "").strip():
                has_routed_car_leg = True
    return first_act_has_end and has_routed_car_leg


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--strip-vehicle-refs", action="store_true",
                   help="Remove vehicleRefId from routes (see VEHICLE_REF_RE)")
    args = p.parse_args()

    n_in = n_out = 0
    with open(args.output, "w", encoding="utf-8") as out:
        out.write(HEADER_XMLDECL + source_doctype(args.input) + "<population>\n")
        for _, elem in ET.iterparse(args.input, events=("end",)):
            if elem.tag != "person":
                continue
            n_in += 1
            if keep(elem):
                blob = ET.tostring(elem, encoding="unicode")
                if args.strip_vehicle_refs:
                    blob = VEHICLE_REF_RE.sub("", blob)
                out.write(blob)
                out.write("\n")
                n_out += 1
            elem.clear()
        out.write(FOOTER)

    pct = 100.0 * n_out / n_in if n_in else 0.0
    print(f"{n_in} persons in -> {n_out} kept ({pct:.1f}%) with an explicit first "
          f"departure and at least one routed car leg")
    print(f"wrote {args.output}")
    if n_out == 0:
        print("ABORT: nothing kept; the filter or the input format is wrong", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

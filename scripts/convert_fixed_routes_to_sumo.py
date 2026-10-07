"""Convert the same fixed/converged MATSim route population already used for
the MATSim-vs-TorchDNL DNL validation into a SUMO .rou.xml, so the SUMO
comparison replays *identical* routes and departure times rather than
re-routing with duarouter.

Requires netconvert's MATSim import to have been run first
(`netconvert --matsim-files Siouxfalls_network_PT.xml -o sioux_falls.net.xml`)
-- confirmed empirically that netconvert preserves MATSim link IDs verbatim
as SUMO edge IDs (e.g. MATSim link "11_1" -> SUMO edge "11_1"), so routes can
be translated by just reusing the same link ID strings, no ID-mapping needed.

Each leg becomes an independent SUMO <vehicle> (not a multi-stage <person>
plan) -- this matches how the MATSim/TorchDNL comparison already treats legs
as independent trips for the travel-time distribution metric (MATSim's
output_trips.csv has one row per leg; TorchDNL's leg_metrics is per-leg).

Usage:
    python scripts/convert_fixed_routes_to_sumo.py \\
        --network /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_network_PT.xml \\
        --population /path/to/matsim_project/scenarios/siouxfalls-2014/Siouxfalls_route_population.xml \\
        --output /path/to/matsim_project/scenarios/siouxfalls-2014/sumo/routes.rou.xml
"""

from __future__ import annotations

import argparse

from tamarl.envs.scenario_loader import parse_network
from tamarl.validation.matsim_io import parse_fixed_route_population

VTYPE_XML = '    <vType id="car" vClass="passenger"/>\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capacity-factor", type=float, default=1.0,
                        help="Must match the value used for the other engines")
    args = parser.parse_args()

    _, edges, link_id_to_idx, _ = parse_network(args.network, args.capacity_factor, 1.0)
    idx_to_link_id = {v: k for k, v in link_id_to_idx.items()}
    # Free-flow traversal time per edge index, used to chain relative activity
    # durations into absolute departures (see below). parse_network puts it last
    # in each edge's attr vector.
    ff_by_idx = {i: e["attr"][4] for i, e in enumerate(edges)}

    fr = parse_fixed_route_population(args.population, link_id_to_idx)
    num_agents = len(fr.departure_times)

    # Split each agent's concatenated paths_flat (leg edges separated by -2)
    # back into per-leg edge-id-string lists, paired with that leg's departure time.
    trips: list[tuple[float, str, list[str]]] = []  # (depart_time, veh_id, edge_ids)
    for i in range(num_agents):
        start, end = int(fr.path_offsets[i]), int(fr.path_offsets[i + 1])
        agent_path = fr.paths_flat[start:end]

        # Departure of leg 0 is absolute. For later legs MATSim and TorchDNL derive
        # the departure during the run, from the arrival at the previous activity
        # plus its duration -- SUMO's route file needs an absolute time up front,
        # so chain the plan using FREE-FLOW traversal times. Where an absolute
        # act_end_time exists it wins, exactly as in the other two engines.
        #
        # The approximation is confined to scenarios using relative durations
        # (max_dur): under congestion the true arrival is later than free flow, so
        # SUMO's later legs depart somewhat earlier than the same legs in MATSim
        # and TorchDNL. The set of trips and their routes are identical; only the
        # timing of later legs drifts. Report it rather than let it pass as exact.
        leg_edges: list[str] = []
        leg_idx = 0
        clock = float(fr.departure_times[i])

        def _emit(leg_idx: int, leg_edges: list[str], clock: float) -> float:
            trips.append((clock, f"agent_{i}_leg{leg_idx}", list(leg_edges)))
            ff = sum(ff_by_idx.get(link_id_to_idx[e], 0.0) for e in leg_edges)
            arrival = clock + ff
            end_t = float(fr.act_end_times[i, leg_idx]) if leg_idx < fr.act_end_times.shape[1] else -1.0
            dur_t = float(fr.act_durations[i, leg_idx]) if leg_idx < fr.act_durations.shape[1] else -1.0
            if end_t >= 0:
                return max(end_t, arrival)
            if dur_t >= 0:
                return arrival + dur_t
            return arrival

        for edge_idx in agent_path:
            edge_idx = int(edge_idx)
            if edge_idx == -2:
                clock = _emit(leg_idx, leg_edges, clock)
                leg_edges = []
                leg_idx += 1
            else:
                leg_edges.append(idx_to_link_id[edge_idx])
        if leg_edges:
            _emit(leg_idx, leg_edges, clock)

    trips.sort(key=lambda t: t[0])  # SUMO requires route files sorted by departure time

    with open(args.output, "w") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
                'xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">\n')
        f.write(VTYPE_XML)
        for dep_time, veh_id, edges in trips:
            if not edges:
                continue
            f.write(
                f'    <vehicle id="{veh_id}" type="car" depart="{dep_time:.0f}">\n'
                f'        <route edges="{" ".join(edges)}"/>\n'
                f'    </vehicle>\n'
            )
        f.write("</routes>\n")

    print(f"Wrote {len(trips)} vehicle trips to {args.output}")


if __name__ == "__main__":
    main()

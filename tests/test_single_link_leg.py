"""A leg whose route is a single link must complete, not be aborted as stuck.

Regression test for the defect found while validating TorchDNL against MATSim on
Berlin. _process_nodes_A classified a capacity-buffer agent as a mover whenever its
next_edge was anything but -1, but leg boundaries in paths_flat are marked -2. An
agent dispatched onto a one-link leg therefore had next_edge == -2, which
negative-indexed edge_endpoints, failed the downstream connectivity check and was
removed from the network one step after entering it.

It never showed up before because a one-link leg needs two activities on the SAME
link, which happens on a dense real network (1.9% of Berlin's car legs) and never
on Sioux Falls, Ingolstadt or grid_world. The symptom on Berlin was 4,605 aborted
legs out of 392,671 plus 30,480 departures that never happened, because an agent
whose leg is aborted never starts its later legs.
"""

import torch

from tamarl.core.torchdnl import TorchDNL

# Two links in series: edge 0 spans nodes (0, 1), edge 1 spans (1, 2).
EDGE_STATIC = torch.tensor(
    [[100, 10, 10, 3600, 1], [100, 10, 10, 3600, 1]], dtype=torch.float32
)
EDGE_ENDPOINTS = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)


def _run(paths_flat, path_offsets, num_legs, num_acts=1, steps=300):
    dnl = TorchDNL(
        EDGE_STATIC,
        None,
        device="cpu",
        stuck_threshold=10,
        edge_endpoints=EDGE_ENDPOINTS,
        paths_flat=torch.tensor(paths_flat, dtype=torch.long),
        path_offsets=torch.tensor(path_offsets, dtype=torch.long),
        num_legs=torch.tensor(num_legs, dtype=torch.long),
        act_durations=torch.full((len(num_legs), num_acts), 5, dtype=torch.long),
        act_end_times=torch.full((len(num_legs), num_acts), -1, dtype=torch.long),
        track_events=True,
    )
    for _ in range(steps):
        dnl.step()
    dnl.finalize_stuck_agents()
    return dnl


def test_single_link_leg_followed_by_another_leg_completes():
    # Leg 1 = [edge 0] alone, then leg 2 = [edge 0, edge 1].
    dnl = _run([0, -2, 0, 1, -1], [0, 5], [2])
    assert int(dnl.stuck_count) == 0, "a one-link leg was aborted as stuck"
    assert dnl.status.tolist() == [3], f"agent did not finish: status {dnl.status.tolist()}"
    tts = dnl.leg_metrics[0, :2, 1].tolist()
    assert all(t > 0 for t in tts), f"a leg recorded no travel time: {tts}"


def test_single_link_leg_as_the_only_leg_completes():
    # The -1 end-of-path sentinel was excluded from the mover mask but nothing
    # arrived the agent either, so it sat in the capacity buffer forever.
    dnl = _run([0, -1], [0, 2], [1])
    assert int(dnl.stuck_count) == 0
    assert dnl.status.tolist() == [3], f"agent did not finish: status {dnl.status.tolist()}"
    assert dnl.leg_metrics[0, 0, 1].item() > 0


def test_multi_link_legs_are_unchanged():
    """The fix must be inert for every leg longer than one link.

    This is what makes the correction safe to apply to a simulator whose published
    numbers were produced before it: a two-link leg still takes two steps.
    """
    one_leg = _run([0, 1, -1], [0, 3], [1])
    assert int(one_leg.stuck_count) == 0
    assert one_leg.leg_metrics[0, 0, 1].item() == 2.0

    two_legs = _run([0, 1, -2, 0, 1, -1], [0, 6], [2])
    assert int(two_legs.stuck_count) == 0
    assert two_legs.leg_metrics[0, :2, 1].tolist() == [2.0, 2.0]

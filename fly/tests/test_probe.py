import numpy as np
import pandas as pd

from fly.connectome import Connectome
from fly.probe import _adjacency, groups, hops_from


def _toy():
    """JO-A left (0, 1, 2) -> interneuron 3 -> left front-leg MN 4; KC 5 -> MBON 6 -> MN 4 with a weak edge."""
    neurons = pd.DataFrame({
        "bodyId": np.arange(10, 17),
        "type": ["JO-A1", "JO-A2", "JO-A1", "IN", "Ti flexor MN", "KCg", "MBON01"],
        "superclass": ["cb_sensory"] * 3 + ["vnc_intrinsic", "vnc_motor", "cb_intrinsic", "cb_intrinsic"],
        "somaSide": [None, None, None, "L", "L", "L", "L"],
        "rootSide": ["L", "L", "L", None, None, None, None],
        "exitNerve": [None, None, None, None, "ProLN", None, None],
        "entryNerve": [None] * 7,
        "subclass": [None] * 7,
    })
    return Connectome("toy", neurons, np.array([0, 1, 3, 3, 5, 6]), np.array([3, 3, 4, 5, 6, 4]),
                      np.array([10, 10, 10, 10, 10, 2], dtype=np.float32))


def test_groups_use_root_side_and_leg_nerves():
    cand, readouts, tracked = groups(_toy())
    assert cand["JO-A_L"].tolist() == [0, 1, 2]
    assert set(cand) == {"JO-A_L"}  # non-JO neurons never form a group
    assert readouts["front_left"].tolist() == [4] and len(readouts["front_right"]) == 0
    assert tracked["KC"].tolist() == [5] and tracked["MBON"].tolist() == [6]


def test_pick_cues_matches_groups_to_legs():
    from fly.drums import VOICES
    from fly.probe import LEGS, pick_cues

    def resp(**peaks):  # leg -> peak Hz; unlisted legs don't respond
        return {"groups": {leg: {"latency_ms": 20 if leg in peaks else None, "peak_hz": peaks.get(leg, 0.0)} for leg in LEGS}}

    sim = {f"g{i}": resp(front_left=10 + i, front_right=10 + i) for i in range(10)}
    sim["kicker"] = resp(hind_right=30)
    sim["pedal"] = resp(hind_left=30, front_right=30)
    cand = {name: np.array([i]) for i, name in enumerate(sim)}
    conn = Connectome("t", pd.DataFrame({"bodyId": np.arange(len(sim))}), np.array([], int), np.array([], int), np.array([], np.float32))
    picks = pick_cues({"sim": sim}, cand, conn)
    assert picks["kick"]["group"] == "kicker"
    hat = [picks["hat_open"], picks["hat_pedal"]]  # only "pedal" reaches the hind left leg, so one hat drum goes without
    assert sorted(p["group"] if p else "none" for p in hat) == ["none", "pedal"]
    assert len({p["group"] for p in picks.values() if p}) == sum(p is not None for p in picks.values())  # no reuse
    assert set(picks) == {v.name for v in VOICES}


def test_hops_all_and_strong():
    conn = _toy()
    d_all = hops_from(_adjacency(conn, 1), np.array([0, 1, 2]))
    assert d_all.tolist() == [0, 0, 0, 1, 2, 2, 3]
    d_strong = hops_from(_adjacency(conn, 5), np.array([6]))
    assert d_strong[4] == -1  # MBON -> MN has only 2 synapses

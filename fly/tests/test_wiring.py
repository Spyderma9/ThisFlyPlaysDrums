import json

import numpy as np
import pandas as pd

from fly.connectome import Connectome
from fly.drums import LEGS, VOICES
from fly.wiring import wire

# rows 0-11 cue (one per voice), 12-13 descending, 14-15 Kenyon cells, 16-19 leg MNs (one per leg), 20 other
N = 21


def _conn(tmp_path):
    types = [f"JO-{i}" for i in range(12)] + ["DNx", "DNy", "KCab", "KCg"] + ["Ti flexor MN"] * 4 + ["LN"]
    sup = ["sensory"] * 12 + ["descending_neuron"] * 2 + ["cb_intrinsic"] * 2 + ["vnc_motor"] * 4 + ["cb_intrinsic"]
    nerve = [None] * 16 + ["ProLN", "ProLN", "MetaLN", "MetaLN"] + [None]
    side = [None] * 16 + ["L", "R", "L", "R"] + [None]
    neurons = pd.DataFrame({"bodyId": np.arange(N) * 10 + 5, "type": types, "superclass": sup, "exitNerve": nerve,
                            "somaSide": side, "rootSide": [None] * N})
    rng = np.random.default_rng(0)
    pre, post = rng.integers(0, N, 600), rng.integers(0, N, 600)
    conn = Connectome("toy", neurons, pre, post, np.ones(600, dtype=np.float32))
    cues = tmp_path / "cues.json"
    cues.write_text(json.dumps({"drums": {v.name: {"bodyIds": [i * 10 + 5]} for i, v in enumerate(VOICES)}}))
    return conn, cues


def test_kc_cut_masks_and_shuffle(tmp_path):
    conn, cues = _conn(tmp_path)
    w = wire(conn, cues_path=cues)
    assert not np.isin(w.conn.post, [14, 15]).any() and w.kc_edges_cut == np.isin(conn.post, [14, 15]).sum()
    assert [int(w.cue_groups[v.name][0]) for v in VOICES] == list(range(12))
    assert w.plastic_mask.any() and (w.conn.pre[w.plastic_mask] < 12).all()
    assert np.isin(w.conn.post[w.plastic_mask], [12, 13]).all()
    assert (w.plastic_mask == ((w.conn.pre < 12) & np.isin(w.conn.post, [12, 13]))).all()
    assert [int(w.leg_mns[leg][0]) for leg in LEGS] == [16, 17, 18, 19]

    s = wire(conn, shuffle_seed=3, cues_path=cues)  # the shuffle keeps the cut and its exact degrees
    assert not np.isin(s.conn.post, [14, 15]).any()
    assert (np.bincount(s.conn.post, minlength=N) == np.bincount(w.conn.post, minlength=N)).all()
    assert (s.plastic_mask == ((s.conn.pre < 12) & np.isin(s.conn.post, [12, 13]))).all()

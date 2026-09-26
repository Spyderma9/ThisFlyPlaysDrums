import numpy as np
import pandas as pd

from fly.connectome import ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS, NEURON_COLUMNS, load_malecns


def _write_malecns(root):
    """Four neurons (given out of bodyId order) plus one glia fragment, 200, with no superclass."""
    ann = pd.DataFrame({c: pd.Series([None] * 5, dtype=object) for c in NEURON_COLUMNS})
    ann["bodyId"] = np.array([30, 10, 200, 20, 40], dtype=np.int64)
    ann["superclass"] = ["vnc_motor", "cb_sensory", None, "cb_intrinsic", "cb_intrinsic"]
    ann["exitNerve"] = ["ProLN", None, None, None, None]
    ann["somaLocation"] = [[3, 3, 3], None, [9, 9, 9], [2, 2, 2], None]
    ann["tosomaLocation"] = [None, None, None, None, [4, 4, 4]]
    ann.to_feather(root / ANNOTATIONS)

    pd.DataFrame({
        "body": np.array([10, 20, 30, 200], dtype=np.int64),  # 40 has no row at all
        "consensus_nt": ["acetylcholine", "GABA", "unclear", "glutamate"],
        "predicted_nt": ["acetylcholine", "gaba", "glutamate", "glutamate"],
    }).to_feather(root / NEUROTRANSMITTERS)

    pd.DataFrame({
        "body_pre": np.array([10, 20, 30, 40, 200, 10], dtype=np.int64),
        "body_post": np.array([20, 30, 10, 10, 10, 200], dtype=np.int64),
        "weight": np.array([5, 7, 2, 4, 9, 3], dtype=np.int64),
    }).to_feather(root / WEIGHTS)


def test_load_malecns_filters_indexes_and_signs(tmp_path):
    _write_malecns(tmp_path)
    conn = load_malecns(tmp_path)

    # Fragment 200 is gone and rows follow bodyId, so index i is the i-th smallest neuron id.
    assert conn.neurons["bodyId"].tolist() == [10, 20, 30, 40]
    assert conn.neurons.loc[2, "exitNerve"] == "ProLN"

    # Both edges touching 200 are dropped; the rest map to row indices.
    edges = sorted(zip(conn.pre.tolist(), conn.post.tolist(), conn.weight.tolist()))
    # 20 is GABA (any case) -> inhibitory; unclear (30) and missing (40) default to excitatory.
    assert edges == [(0, 1, 5.0), (1, 2, -7.0), (2, 0, 2.0), (3, 0, 4.0)]
    assert conn.weight.dtype == np.float32 and conn.pre.dtype == np.int64

    # Soma falls back to tosomaLocation, else NaN.
    xyz = conn.neurons[["x", "y", "z"]].to_numpy()
    np.testing.assert_array_equal(xyz[[0, 2, 3]], [[np.nan] * 3, [3, 3, 3], [4, 4, 4]])
    assert xyz[1].tolist() == [2, 2, 2]

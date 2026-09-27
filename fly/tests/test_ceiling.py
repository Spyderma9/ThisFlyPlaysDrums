import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("scipy")

from fly.brain import FLYBRAIN_CODE, Brain  # noqa: E402
from fly.ceiling import fit_dn_mn, record, ridge, smooth  # noqa: E402
from fly.connectome import Connectome  # noqa: E402


def test_fit_dn_mn_recovers_signed_weights_and_keeps_signs():
    rng = np.random.default_rng(0)
    dn = rng.gamma(2.0, 5.0, size=(4000, 3))  # DN rates, Hz
    # MN 0 <- DN 0 (excitatory, 2.0) and DN 1 (inhibitory, -0.5); MN 1 <- DN 2 (excitatory), but the target falls
    # when DN 2 rises, which an excitatory synapse can't do: its weight must stay at 0 (sign kept)
    y = np.stack([10 + 2.0 * dn[:, 0] - 0.5 * dn[:, 1], 30 - 1.0 * dn[:, 2]], axis=1)
    edges = [(0, 0, 5.0), (1, 0, -3.0), (2, 1, 4.0)]
    pred, w = fit_dn_mn(dn[:3000], y[:3000], dn[3000:], edges, 2)
    assert w[0] == pytest.approx(2.0, rel=1e-6) and w[1] == pytest.approx(-0.5, rel=1e-6) and w[2] == 0.0
    assert np.allclose(pred[:, 0], y[3000:, 0], atol=1e-6)
    assert np.allclose(pred[:, 1], y[:3000, 1].mean())  # no usable input: the constant (tone) only


def test_ridge_fits_a_linear_map_and_smooth_is_a_unit_gain_filter():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(500, 4))
    y = x @ np.array([[1.0], [-2.0], [0.0], [0.5]]) + 3.0
    assert np.allclose(ridge(x[:400], y[:400], x[400:], lams=(1e-8,))[1e-8], y[400:], atol=1e-4)
    step = np.ones((1, 400, 1))
    assert smooth(step, 20.0, 5)[0, -1, 0] == pytest.approx(1.0, abs=1e-6)


@pytest.mark.skipif(not FLYBRAIN_CODE.exists(), reason="fly-brain not cloned into fly/vendor/")
def test_record_filters_spikes_into_hz_every_stride():
    neurons = pd.DataFrame({"bodyId": [10, 11]})
    conn = Connectome("pair", neurons, pre=np.array([0]), post=np.array([1]), weight=np.array([300.0], dtype=np.float32))
    brain = Brain(conn, {"kick": np.array([0])}, batch=2, device="cpu")
    rates = torch.full((2, 300, 1), 200.0)
    rec = record(brain, rates, {"cue": np.array([0]), "post": np.array([1])}, tau_ms=20.0, stride=5, seed=0)
    assert rec["cue"].shape == (2, 60, 1) and rec["post"].shape == (2, 60, 1)
    assert 100 < rec["cue"][:, 20:].mean() < 400  # ~200 Hz Poisson drive reads as ~200 Hz
    assert rec["post"][:, 20:].mean() > 0

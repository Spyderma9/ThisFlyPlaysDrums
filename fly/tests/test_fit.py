import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("torch")

from fly.fit import fit_mn  # noqa: E402


def _inputs():
    return np.random.default_rng(0).gamma(2.0, 1e-3, size=(5000, 3)), np.array([40.0, -20.0, 10.0])


def test_fit_mn_recovers_weights_and_tone_that_keep_their_signs():
    u, w0 = _inputs()
    w, tone = fit_mn(u, u @ np.array([60.0, -40.0, 5.0]) + 0.3, w0)
    assert np.allclose(w, [60.0, -40.0, 5.0], rtol=1e-6) and tone == pytest.approx(0.3, abs=1e-6)


def test_fit_mn_never_flips_a_synapse():
    u, w0 = _inputs()
    w, _ = fit_mn(u, u @ np.array([60.0, -40.0, -10.0]) + 0.3, w0)  # synapse 2 would have to turn inhibitory
    assert w[2] == 0.0 and w[0] == pytest.approx(60.0, rel=2e-2) and w[1] == pytest.approx(-40.0, rel=2e-2)

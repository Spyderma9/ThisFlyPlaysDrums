import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("torch")

from fly.fit import fit_mn, fit_mn_gram, weights_at  # noqa: E402


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


def test_gram_solver_matches_the_direct_one_and_handles_silent_inputs():
    rng = np.random.default_rng(3)
    u = rng.gamma(2.0, 1e-3, size=(4000, 40))
    u[:, 7] = 0.0  # an input that never fires
    w0 = rng.choice([-1.0, 1.0], 40) * rng.uniform(1, 20, 40)
    target = u @ (w0 * rng.uniform(0, 3, 40)) + rng.normal(0, 1e-3, 4000) + 0.2
    target -= 5.0 * u[:, 3] * np.sign(w0[3])  # pushes synapse 3 towards the wrong sign
    w_direct, tone_direct = fit_mn(u, target, w0)
    w_gram, tone_gram = fit_mn_gram(u, target, w0)
    assert np.allclose(w_gram, w_direct, atol=1e-3 * np.abs(w_direct).max()) and tone_gram == pytest.approx(tone_direct, abs=1e-6)
    assert (np.sign(w_gram) * np.sign(w0) >= 0).all() and w_gram[7] == 0.0
    w_none, tone_none = fit_mn_gram(np.zeros((100, 3)), np.full(100, 0.4), np.ones(3))
    assert (w_none == 0).all() and tone_none == pytest.approx(0.4)


def test_gains_never_flip_a_synapse_and_zero_is_the_untrained_fly():
    w0 = np.array([4.0, -6.0, 2.0])
    fitted = np.array([1.0, -1.0, 0.0])  # all three weakened, one to 0
    assert np.array_equal(weights_at(0, w0, fitted, "all"), w0) and np.array_equal(weights_at(0, w0, fitted, "dn"), w0)
    assert np.allclose(weights_at(2, w0, fitted, "all"), [2.0, -2.0, 0.0])  # the fitted input, doubled
    assert np.allclose(weights_at(2, w0, fitted, "dn"), [0.0, 0.0, 0.0])  # 2 W' - W would flip all three: held at 0
    assert np.allclose(weights_at(0.5, w0, fitted, "dn"), [2.5, -3.5, 1.0])

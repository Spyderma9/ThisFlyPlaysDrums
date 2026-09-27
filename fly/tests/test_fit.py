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


def test_per_synapse_gains_scale_each_synapse_on_its_own():
    w0 = np.array([4.0, -6.0, 2.0])
    fitted = np.array([1.0, -1.0, 3.0])
    assert np.allclose(weights_at(np.array([2.0, 0.0, 1.0]), w0, fitted, "all"), [2.0, -6.0, 3.0])  # 0: untrained


def test_scale_cues_is_one_rule_for_every_drum_and_caps_at_one_spike_per_ms():
    from fly.encoder import MAX_RATE_HZ, scale_cues

    rates = np.array([[0.0, 50.0, 200.0]], dtype=np.float32)
    assert scale_cues(rates, 1.0) is rates
    assert np.allclose(scale_cues(rates, 3.0), [[0.0, 150.0, 600.0]])
    assert scale_cues(rates, 10.0).max() == MAX_RATE_HZ and scale_cues(rates, 10.0).dtype == np.float32


def test_validate_legs_matches_train_validate_and_splits_it_by_leg():
    import pandas as pd

    from fly.brain import FLYBRAIN_CODE, Brain
    from fly.connectome import Connectome
    from fly.decoder import Decoder
    from fly.drums import LEGS
    from fly.fit import validate_legs
    from fly.train import Loss, validate

    if not FLYBRAIN_CODE.exists():
        pytest.skip("fly-brain not cloned into fly/vendor/")
    import json
    import torch

    from fly.decoder import KIT
    types = ["Ti flexor MN", "Ti flexor MN", "Ti extensor MN", "Fe reductor MN", "Tr flexor MN", "Tr extensor MN"]
    leg_mns = {leg: np.arange(6) + 10 + 6 * k for k, leg in enumerate(LEGS)}
    dec = Decoder(leg_mns, {leg: types for leg in LEGS}, n_neurons=40, kit=json.loads(KIT.read_text()))
    rng = np.random.default_rng(0)
    pre, post = rng.integers(0, 10, 60), rng.integers(10, 34, 60)
    conn = Connectome("toy", pd.DataFrame({"bodyId": np.arange(40)}), pre=pre, post=post,
                      weight=rng.uniform(200, 400, 60).astype(np.float32))
    brain = Brain(conn, {"kick": np.arange(10)}, batch=2, device="cpu")
    loss_fn = Loss(dec)
    rates = torch.full((2, 120, 1), 200.0)
    qstar = dec.rest.expand(2, 120, 32).clone() + 0.05 * torch.as_tensor(rng.normal(size=(2, 120, 32)), dtype=torch.float32)
    total, legs = validate_legs(brain, dec, loss_fn, rates, qstar, 7)
    assert total == pytest.approx(validate(brain, dec, loss_fn, rates, qstar, 50, 7), rel=1e-5)
    assert legs.shape == (len(LEGS),) and legs.sum() == pytest.approx(total, rel=1e-9) and (legs > 0).all()

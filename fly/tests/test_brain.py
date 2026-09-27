import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fly.brain import FLYBRAIN_CODE, Brain  # noqa: E402
from fly.connectome import Connectome  # noqa: E402

pytestmark = pytest.mark.skipif(not FLYBRAIN_CODE.exists(), reason="fly-brain not cloned into fly/vendor/")


def _chain():
    """cue 0 -> relay 1 -> relay 2 -> motor 3, with 2 -> 3 as the one trainable edge."""
    neurons = pd.DataFrame({"bodyId": [10, 11, 12, 13]})
    conn = Connectome(
        "chain",
        neurons,
        pre=np.array([0, 1, 2]),
        post=np.array([1, 2, 3]),
        weight=np.array([300.0, 300.0, 300.0], dtype=np.float32),
    )
    return conn, np.array([False, False, True])


def _run(brain, steps, rate_hz=200.0):
    gen = torch.Generator().manual_seed(0)
    state = brain.init_state()
    rates = torch.tensor([[rate_hz]])
    out = []
    for _ in range(steps):
        state = brain.step(state, rates, generator=gen)
        out.append(state[2])  # spikes
    return torch.stack(out)  # [T, B, N]


def test_cue_reaches_motor_neuron_at_1ms():
    conn, mask = _chain()
    brain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu")
    with torch.no_grad():
        spikes = _run(brain, 100)
    first = [int(torch.nonzero(spikes[:, 0, i])[0]) for i in range(4)]
    assert first == sorted(first) and first[3] > first[0]


def test_gradient_reaches_trainable_edge_through_frozen_wiring():
    conn, _ = _chain()
    # Put the trainable edge upstream (1 -> 2) so the gradient has to cross the frozen 2 -> 3 edge.
    mask = np.array([False, True, False])
    brain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu")
    spikes = _run(brain, 60)
    spikes[:, 0, 3].sum().backward()
    assert brain.plastic.weight.grad is not None and brain.plastic.weight.grad.abs().sum() > 0


def test_wide_surrogate_changes_only_the_gradient():
    conn, _ = _chain()
    mask = np.array([False, True, False])
    grads, spikes = {}, {}
    for width in (None, 5.0):
        brain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu", surrogate_mv=width)
        out = _run(brain, 60)
        spikes[width] = out.detach()
        out[:, 0, 3].sum().backward()
        grads[width] = brain.plastic.weight.grad.abs().sum()
    assert torch.equal(spikes[None], spikes[5.0])  # the simulated fly is identical
    assert grads[5.0] > grads[None] > 0  # the gradient reaching the trainable edge is larger


def test_ff_credit_backprops_only_along_the_allowed_hop():
    """cue 0 -> 1 -> (plastic) 2 -> motor 3: with credit on the hop 2 -> 3 the plastic edge 1 -> 2 still learns;
    blocking that hop (credit only on 1 -> 3, which doesn't exist) leaves it no gradient. Forward is unchanged."""
    conn, _ = _chain()
    mask = np.array([False, True, False])
    out = {}
    for name, ff in (("full", None), ("hop", (np.array([2]), np.array([3]))), ("blocked", (np.array([1]), np.array([3])))):
        brain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu", ff_credit=ff)
        spikes = _run(brain, 60)
        spikes[:, 0, 3].sum().backward()
        out[name] = (spikes.detach(), float(brain.plastic.weight.grad.abs().sum()))
    assert torch.equal(out["full"][0], out["hop"][0]) and torch.equal(out["full"][0], out["blocked"][0])
    assert out["hop"][1] > 0 and out["blocked"][1] == 0


def test_zero_tone_changes_nothing_and_is_trainable():
    conn, mask = _chain()
    plain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu")
    toned = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu", tone_idx=np.array([3]))
    assert torch.equal(_run(plain, 60).detach(), _run(toned, 60).detach())
    with torch.no_grad():
        toned.tone.fill_(8.0)  # above the 7 mV gap: the motor neuron fires on its own
    assert _run(toned, 60)[:, 0, 3].sum() > _run(plain, 60)[:, 0, 3].sum()


def test_shuffle_keeps_degrees_and_signs():
    rng = np.random.default_rng(1)
    pre, post = rng.integers(0, 50, 400), rng.integers(0, 50, 400)
    weight = np.where(pre % 3 == 0, -1.0, 1.0).astype(np.float32) * rng.integers(1, 9, 400)
    conn = Connectome("r", pd.DataFrame({"bodyId": np.arange(50)}), pre, post, weight)
    shuf = conn.shuffled(seed=2)
    assert (np.bincount(shuf.pre, minlength=50) == np.bincount(pre, minlength=50)).all()
    assert (np.bincount(shuf.post, minlength=50) == np.bincount(post, minlength=50)).all()
    assert (np.sign(shuf.weight) == np.where(shuf.pre % 3 == 0, -1, 1)).all()
    assert not (shuf.post == post).all()

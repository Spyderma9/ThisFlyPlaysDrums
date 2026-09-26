import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fly.decoder import JOINTS, KIT, Decoder, lif_current  # noqa: E402
from fly.drums import LEGS  # noqa: E402

KIT_DATA = json.loads(KIT.read_text())
TYPES = ["Ti flexor MN", "Ti flexor MN", "Ti extensor MN", "Fe reductor MN", "MNhl59", "?"]


def _decoder():
    """Six motor neurons per leg (rows 10 + 6*leg ...), same types on every leg."""
    leg_mns = {leg: np.arange(6) + 10 + 6 * k for k, leg in enumerate(LEGS)}
    return Decoder(leg_mns, {leg: TYPES for leg in LEGS}, n_neurons=40, kit=KIT_DATA), leg_mns


def _servo(leg, joint):
    return LEGS.index(leg) * len(JOINTS) + JOINTS.index(joint)


def _drive(dec, spikes_per_step, steps=300):
    r = dec.init_state(1)
    for _ in range(steps):
        r, q = dec(r, spikes_per_step)
    return r, q


def test_silence_holds_rest_and_shapes():
    dec, _ = _decoder()
    r, q = dec(dec.init_state(2), torch.zeros(2, 40))
    assert r.shape == (2, 24) and q.shape == (2, 32)
    assert torch.allclose(q, dec.rest.expand(2, -1))


def test_flexors_and_extensors_move_the_joint_opposite_ways():
    dec, mns = _decoder()
    t = _servo("front_left", "tibia")
    sign = KIT_DATA["actuators"][t]["sign"]
    for col, direction in ((0, 1), (2, -1), (4, 0), (5, 0)):  # Ti flexor, Ti extensor, two unmapped types
        spikes = torch.zeros(1, 40)
        spikes[0, mns["front_left"][col]] = 1.0  # one spike every ms: ~1000 Hz after the filter settles
        _, q = _drive(dec, spikes)
        delta = float(q[0, t] - dec.rest[t])
        assert np.sign(delta) == sign * direction
        others = [i for i in range(32) if i not in (t, _servo("front_left", "femur_twist"))]
        assert torch.allclose(q[0, others], dec.rest[others])


def test_rate_filter_reads_steady_rate():
    dec, mns = _decoder()
    spikes = torch.zeros(1, 40)
    spikes[0, mns["hind_right"][0]] = 1.0
    r, _ = _drive(dec, spikes, steps=400)
    assert abs(float(r[0, 18]) - 1000.0) < 1.0


def test_pseudo_inverse_round_trip():
    dec, _ = _decoder()
    q = dec.rest.clone()
    q[_servo("front_right", "tibia")] += 0.2
    q[_servo("hind_left", "femur_twist")] += 0.1 * KIT_DATA["actuators"][_servo("hind_left", "femur_twist")]["sign"]
    back = dec.targets(dec.rates_for(q[None]))[0]
    assert torch.allclose(back, q, atol=1e-5)
    cur = dec.current_for(q[None])
    assert cur.shape == (1, 40) and cur[0, :10].abs().sum() == 0 and cur.abs().sum() > 0


def test_lif_current_gives_the_asked_rate():
    """A fly-brain LIF (dt 1 ms, tau 20 ms, gap 7 mV) driven by lif_current(f) fires at about f Hz."""
    for hz in (20.0, 60.0):
        cur, v, n = float(lif_current(torch.tensor(hz))), 0.0, 0
        for _ in range(2000):
            v += cur
            v -= v / 20.0
            if v > 7.0:
                n, v = n + 1, 0.0
        assert abs(n / 2 - hz) < 0.1 * hz
    assert float(lif_current(torch.tensor(-30.0))) == -float(lif_current(torch.tensor(30.0)))
    assert float(lif_current(torch.tensor(0.0))) == 0.0


def test_soft_limit_stays_in_range_and_keeps_gradient():
    dec, mns = _decoder()
    r = torch.zeros(1, 24, requires_grad=True)
    big = r + torch.nn.functional.one_hot(torch.tensor([2]), 24) * 100.0  # a Ti extensor rate twice the p99 of an untrained run
    q = dec.targets(big)
    t = _servo("front_left", "tibia")
    lo, hi = KIT_DATA["actuators"][t]["range"]
    qt = float(q[0, t].detach())
    assert lo <= qt <= hi and abs(qt - (lo if KIT_DATA["actuators"][t]["sign"] > 0 else hi)) < 0.05
    q[0, t].backward()
    assert r.grad[0, 2] != 0

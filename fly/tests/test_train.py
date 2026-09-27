import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fly.brain import FLYBRAIN_CODE, Brain  # noqa: E402
from fly.connectome import Connectome  # noqa: E402
from fly.decoder import JOINTS, KIT, Decoder  # noqa: E402
from fly.drums import LEGS  # noqa: E402
from fly.train import GAP_MS, Loss, Take, alpha_at, detach, keep_signs, load_weights, pack, run_window, save  # noqa: E402

KIT_DATA = json.loads(KIT.read_text())
TYPES = ["Ti flexor MN", "Ti flexor MN", "Ti extensor MN", "Fe reductor MN", "MNhl59", "?"]
TIBIA = LEGS.index("front_left") * len(JOINTS) + JOINTS.index("tibia")


def _take(n, value):
    return Take(f"t{n}", np.full((n, 2), value, np.float32), np.full((n, 32), value, np.float32), 1, 0)


def test_pack_keeps_every_take_once_and_balances_streams():
    takes = [_take(900, 1.0), _take(700, 2.0), _take(400, 3.0), _take(300, 4.0)]
    rates, q = pack(takes, 2, np.zeros(32), np.random.default_rng(0))
    assert rates.shape[0] == 2 and q.shape[:2] == rates.shape[:2] and q.shape[2] == 32
    for tk in takes:  # each take's steps appear exactly once, in one piece
        assert (rates[..., 0] == tk.rates[0, 0]).sum() == len(tk.rates)
    lengths = [(rates[b, :, 0] > 0).sum() for b in range(2)]
    assert abs(lengths[0] - lengths[1]) <= 300
    assert rates.shape[1] >= max(lengths) + GAP_MS


def test_alpha_anneals_then_stays_zero():
    assert alpha_at(0, 100, 0.5) == 1.0
    assert alpha_at(25, 100, 0.5) == pytest.approx(0.5)
    assert alpha_at(50, 100, 0.5) == 0.0 and alpha_at(99, 100, 0.5) == 0.0


def test_keep_signs_and_detach():
    w = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.0]))
    with torch.no_grad():
        w.copy_(torch.tensor([-1.0, 4.0, 0.5]))
    keep_signs(w, torch.tensor([1.0, -1.0, 1.0]))
    assert w.tolist() == [0.0, 0.0, 0.5]
    x = torch.ones(2, requires_grad=True) * 2
    out = detach((x, [x, None], 3))
    assert not out[0].requires_grad and not out[1][0].requires_grad and out[1][1] is None and out[2] == 3


def _decoder(n):
    leg_mns = {leg: np.arange(6) + 10 + 6 * k for k, leg in enumerate(LEGS)}
    return Decoder(leg_mns, {leg: TYPES for leg in LEGS}, n_neurons=n, kit=KIT_DATA)


def test_loss_ignores_undrivable_servos_and_weights_strokes():
    dec = _decoder(40)
    loss = Loss(dec)
    rest = dec.rest[None]
    assert float(loss(rest, rest)) == 0.0
    off = rest.clone()
    undriven = int(torch.nonzero(loss.driven == 0)[0])
    off[0, undriven] += 0.3
    assert float(loss(off, rest)) == 0.0
    miss = torch.zeros_like(rest)
    miss[0, TIBIA] = 0.05
    stroke = rest.clone()
    stroke[0, TIBIA] += 0.3 * KIT_DATA["actuators"][TIBIA]["sign"]  # q* away from rest: a stroke
    miss_at_rest = float(loss(rest + miss, rest))
    miss_in_stroke = float(loss(stroke + miss, stroke))
    assert miss_at_rest > 0 and miss_in_stroke == pytest.approx(5 * miss_at_rest)  # weight 1 + ACTIVE_W


def test_rest_on_pedal_holds_the_hat_closed_at_rest_and_keeps_ranges():
    from fly.strokes import STROKES
    from fly.train import rest_on_pedal

    dec = _decoder(40)
    strokes = json.loads(STROKES.read_text())
    rest0, lo0, hi0 = dec.rest.clone(), (dec.rest - dec.down).clone(), (dec.rest + dec.up).clone()
    rest_on_pedal(dec, strokes)
    k, n = LEGS.index("hind_left") * len(JOINTS), len(JOINTS)
    press = torch.tensor(strokes["pads"]["hat_pedal"]["legs"]["hind_left"]["soft"])
    q = dec.targets(dec.init_state(1))[0]  # no motor activity
    assert torch.allclose(q[k:k + n], torch.minimum(torch.maximum(press, lo0[k:k + n]), hi0[k:k + n]))
    other = torch.ones(32, dtype=torch.bool)
    other[k:k + n] = False
    assert torch.allclose(q[other], rest0[other])  # the other legs keep their rest
    assert torch.allclose(dec.rest - dec.down, lo0, atol=1e-3) and torch.allclose(dec.rest + dec.up, hi0, atol=1e-3)


@pytest.mark.skipif(not FLYBRAIN_CODE.exists(), reason="fly-brain not cloned into fly/vendor/")
def test_one_update_trains_only_the_plastic_edge_and_keeps_its_sign(tmp_path):
    """cue 0 -> (plastic) relay 1 -> the front-left Ti flexor motor neurons 10, 11."""
    neurons = pd.DataFrame({"bodyId": np.arange(40)})
    conn = Connectome("toy", neurons, pre=np.array([0, 1, 1]), post=np.array([1, 10, 11]),
                      weight=np.array([300.0, 300.0, 300.0], dtype=np.float32))
    mask = np.array([True, False, False])
    brain = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, batch=2, device="cpu")
    dec = _decoder(40)
    loss_fn = Loss(dec)
    rates = torch.full((2, 60, 1), 200.0)
    qstar = dec.rest.expand(2, 60, 32).clone()
    qstar[:, :, TIBIA] = dec.rest[TIBIA] + 0.3 * KIT_DATA["actuators"][TIBIA]["sign"]  # flex: ask the relay to fire
    w = brain.plastic.weight
    w0, sign0 = w.detach().clone(), torch.sign(w.detach().clone())
    frozen0 = brain.w.to_dense().clone()
    opt = torch.optim.Adam([w], lr=5.0)
    gen = torch.Generator().manual_seed(0)
    state, r, loss = run_window(brain, dec, loss_fn, brain.init_state(), dec.init_state(2), rates, qstar, 0.5, gen)
    opt.zero_grad()
    loss.backward()
    assert w.grad is not None and w.grad.abs().sum() > 0
    opt.step()
    keep_signs(w, sign0)
    assert not torch.equal(w.detach(), w0) and (torch.sign(w.detach()) * sign0 >= 0).all()
    assert torch.equal(brain.w.to_dense(), frozen0)
    save(tmp_path / "w.pt", brain, {"shuffle_seed": None})
    trained = w.detach().clone()
    with torch.no_grad():
        w.copy_(w0)
    load_weights(brain, tmp_path / "w.pt", None)
    assert torch.equal(w.detach(), trained)
    with pytest.raises(SystemExit):
        load_weights(brain, tmp_path / "w.pt", 1)
    with pytest.raises(SystemExit):  # trained on another plastic set
        load_weights(brain, tmp_path / "w.pt", None, "cue_dn+dn_mn")


@pytest.mark.skipif(not FLYBRAIN_CODE.exists(), reason="fly-brain not cloned into fly/vendor/")
def test_tone_is_saved_loaded_and_checked(tmp_path):
    from fly.train import checkpoint_settings

    neurons = pd.DataFrame({"bodyId": np.arange(20)})
    conn = Connectome("toy", neurons, pre=np.array([0, 1]), post=np.array([1, 10]), weight=np.array([300.0, 300.0], dtype=np.float32))
    mask = np.array([True, False])
    toned = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu", tone_idx=np.array([10, 11]))
    with torch.no_grad():
        toned.tone.copy_(torch.tensor([1.5, -0.5]))
    save(tmp_path / "t.pt", toned, {"shuffle_seed": None, "plastic": "cue_dn"})
    assert checkpoint_settings(tmp_path / "t.pt")["tone"] is True
    fresh = Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu", tone_idx=np.array([10, 11]))
    load_weights(fresh, tmp_path / "t.pt", None)
    assert torch.equal(fresh.tone.detach(), torch.tensor([1.5, -0.5]))
    with pytest.raises(SystemExit):  # a brain without tone can't take a toned checkpoint
        load_weights(Brain(conn, {"kick": np.array([0])}, plastic_mask=mask, device="cpu"), tmp_path / "t.pt", None)

import numpy as np

from fly.decoder import JOINTS
from fly.drums import LEGS
from fly.strokes import T_HOLD_MS, plan, stroke_poses, teacher

L, R = "front_left", "front_right"


def _entry(k):
    """Fake poses: every joint of soft/hard/raise at k, k + 0.2, k + 1."""
    return {"soft": [k] * 8, "hard": [k + 0.2] * 8, "raise": [k + 1.0] * 8,
            "timing": {"1": {"latency_ms": 10.0}, "127": {"latency_ms": 6.0}}}


PADS = {
    "snare": {"either": False, "legs": {L: _entry(1)}}, "xstick": {"either": False, "legs": {L: _entry(1)}},
    "hat": {"either": False, "legs": {R: _entry(2)}}, "ride": {"either": False, "legs": {R: _entry(2)}},
    "ride_bell": {"either": False, "legs": {R: _entry(2)}},
    "crash": {"either": False, "legs": {R: _entry(3), L: _entry(3)}},
    "tom1": {"either": True, "legs": {R: _entry(4), L: _entry(4)}},
    "kick": {"either": False, "legs": {"hind_right": _entry(5)}},
    "hat_pedal": {"either": False, "legs": {"hind_left": {**_entry(6), "pedal": {"latency_ms": 8.0, "release_ms": 2.0}}}},
}


def _voices(notes):
    return [(n.voice, n.leg, n.dropped) for n in sorted(notes, key=lambda n: (n.t_ms, n.voice))]


def test_sticking_resolves_clashes():
    notes = plan({"hat_closed": [(1.0, 90)], "crash": [(1.0, 100)]}, 0, PADS)
    assert _voices(notes) == [("crash", L, None), ("hat_closed", R, None)]  # the crash moves to the free left stick
    notes = plan({"hat_closed": [(1.0, 90)], "ride": [(1.005, 90)]}, 0, PADS)
    assert _voices(notes) == [("hat_closed", None, "stick busy"), ("ride", R, None)]  # one stick, the ride wins
    notes = plan({"snare": [(1.0, 90), (1.016, 60)]}, 0, PADS)
    assert [n.dropped for n in notes] == [None, "stick busy"]  # a flam on one stick keeps the first stroke


def test_middle_tom_goes_to_the_more_rested_stick():
    notes = plan({"hat_closed": [(1.0, 90)], "tom1": [(1.05, 90), (1.1, 90)]}, 0, PADS)
    assert [n.leg for n in notes if n.voice == "tom1"] == [L, R]


def test_teacher_strikes_on_the_note_and_holds_the_hat_closed():
    notes = plan({"snare": [(0.3, 127)], "hat_open": [(0.5, 90)], "kick": [(0.3, 64)]}, 200, PADS)
    q = teacher(notes, 1200, {"pads": PADS}, np.zeros(32))
    assert q.shape == (1200, 32)
    nj = len(JOINTS)
    left, hind_left = LEGS.index(L) * nj, LEGS.index("hind_left") * nj
    up, strike = stroke_poses(PADS["snare"]["legs"][L], 127)
    assert np.allclose(q[500 - 3, left:left + nj], strike) and np.allclose(q[500 - 6 - 10, left:left + nj], up)
    assert np.allclose(q[500 + T_HOLD_MS + 200, left:left + nj], 0)  # back at rest
    assert np.allclose(q[300, hind_left:hind_left + nj], 6)  # pedal held down (closed) after the pre-roll
    assert np.allclose(q[700 - 1, hind_left:hind_left + nj], 7)  # lifted for the open hat at 700 ms

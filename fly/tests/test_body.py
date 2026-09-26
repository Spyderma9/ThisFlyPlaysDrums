import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("flybody")

from fly.body import Body, pad_voice, strike, zone_at  # noqa: E402
from fly.decoder import JOINTS, KIT  # noqa: E402
from fly.drums import LEGS  # noqa: E402
from fly.strokes import STROKES  # noqa: E402

KIT_DATA = json.loads(KIT.read_text())
PADS = {p["name"]: p for p in KIT_DATA["pads"]}


@pytest.fixture(scope="module")
def body():
    return Body(KIT_DATA)


def _with(targets, leg, q):
    k = LEGS.index(leg)
    targets = targets.copy()
    targets[k * len(JOINTS):(k + 1) * len(JOINTS)] = q
    return targets


def test_rest_makes_no_sound(body):
    body.reset()
    assert sum((body.step(body.rest) for _ in range(150)), []) == []


def test_scripted_stick_into_pad_gives_one_hit(body):
    body.reset()
    hits = strike(body, "front_right", PADS["ride"]["reach"]["front_right"])
    assert [(h.pad, h.note, h.limb) for h in hits] == [("ride", 51, "front_right")]
    assert 1 <= hits[0].velocity <= 127 and hits[0].contact_speed > 0
    body.reset()
    hits = strike(body, "hind_right", PADS["kick"]["reach"]["hind_right"])
    assert [(h.note, h.voice) for h in hits] == [(36, "kick")]


def test_kit_is_the_td07_layout():
    sticks = [p["name"] for p in KIT_DATA["pads"] if p["kind"] == "stick"]
    assert sorted(sticks) == ["crash", "hat", "ride", "snare", "tom1", "tom2", "tom3"]
    assert {v: p["name"] for p in KIT_DATA["pads"] for v in p["targets"]}["xstick"] == "snare"
    assert {v: p["name"] for p in KIT_DATA["pads"] for v in p["targets"]}["ride_bell"] == "ride"


def test_a_contact_inside_a_zone_sounds_the_zone():
    ride = PADS["ride"]
    bell, bow = (np.array(ride["targets"][v][:2]) for v in ("ride_bell", "ride"))
    assert zone_at(ride["zones"], bell) == "ride_bell" and zone_at(ride["zones"], bow) is None
    assert pad_voice("ride", False, zone_at(ride["zones"], bell)) == "ride_bell"
    assert pad_voice("ride", False, zone_at(ride["zones"], bow)) == "ride"
    snare = PADS["snare"]
    assert zone_at(snare["zones"], np.array(snare["targets"]["xstick"][:2])) == "xstick"
    assert zone_at(snare["zones"], np.array(snare["targets"]["snare"][:2])) is None
    assert pad_voice("hat", True) == "hat_closed" and pad_voice("hat", False) == "hat_open"


@pytest.mark.skipif(not STROKES.exists(), reason="run python -m fly.strokes --calibrate")
@pytest.mark.parametrize("target", ["snare", "xstick", "ride", "ride_bell"])
def test_calibrated_strokes_sound_their_zone(body, target):
    from fly.strokes import _strike_timing, stroke_poses

    entry = json.loads(STROKES.read_text())["pads"][target]
    leg, poses = next(iter(entry["legs"].items()))
    t = _strike_timing(body, target, leg, *stroke_poses(poses, 96))
    assert t["latency_ms"] is not None and t["stray"] == []


@pytest.mark.skipif(not STROKES.exists(), reason="run python -m fly.strokes --calibrate")
def test_hat_note_follows_the_pedal(body):
    pedal = json.loads(STROKES.read_text())["pads"]["hat_pedal"]["legs"]["hind_left"]
    lift, hold, chick = (np.array(pedal[k]) for k in ("raise", "soft", "hard"))
    hat = PADS["hat"]["reach"]["front_right"]
    body.reset()
    assert [h.note for h in strike(body, "front_right", hat)] == [46]  # pedal up: open

    body.reset()
    hits = sum((body.step(_with(body.rest, "hind_left", lift)) for _ in range(60)), [])
    for k in range(120):  # a slow close is silent
        a = min(1.0, k / 100)
        hits += body.step(_with(body.rest, "hind_left", (1 - a) * lift + a * hold))
    assert hits == []
    both = _with(_with(body.rest, "hind_left", hold), "front_right", hat)
    hits = sum((body.step(both) for _ in range(60)), [])
    assert [(h.note, h.voice) for h in hits] == [(42, "hat_closed")]

    body.reset()
    sum((body.step(_with(body.rest, "hind_left", lift)) for _ in range(60)), [])
    hits = sum((body.step(_with(body.rest, "hind_left", chick)) for _ in range(40)), [])
    assert [(h.note, h.voice) for h in hits] == [(44, "hat_pedal")]  # a hard press is the pedal note

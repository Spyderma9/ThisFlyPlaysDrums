import json

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("flybody")

from fly.body import Body, strike  # noqa: E402
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
    hits = strike(body, "front_left", PADS["snare"]["reach"]["front_left"])
    assert [(h.pad, h.note, h.voice, h.limb) for h in hits] == [("snare", 38, "snare", "front_left")]
    assert 1 <= hits[0].velocity <= 127 and hits[0].contact_speed > 0
    body.reset()
    hits = strike(body, "hind_right", PADS["kick"]["reach"]["hind_right"])
    assert [(h.note, h.voice) for h in hits] == [(36, "kick")]


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

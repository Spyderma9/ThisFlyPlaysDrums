import json

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("flybody")

from fly.body import Body, strike  # noqa: E402
from fly.decoder import JOINTS, KIT  # noqa: E402
from fly.drums import LEGS  # noqa: E402

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


def test_hat_note_follows_the_pedal(body):
    hat = PADS["hat"]["reach"]["front_right"]
    body.reset()
    open_hits = strike(body, "front_right", hat)
    assert [h.note for h in open_hits] == [46]

    body.reset()
    pressed = _with(body.rest, "hind_left", PADS["hat_pedal"]["reach"]["hind_left"])
    hits = sum((body.step(pressed) for _ in range(80)), [])
    assert [h.note for h in hits] == [44]  # the pedal press itself
    both = _with(pressed, "front_right", hat)
    hits = sum((body.step(both) for _ in range(60)), [])
    assert [(h.note, h.voice) for h in hits] == [(42, "hat_closed")]

import json

import numpy as np
import pytest

from fly.decoder import KIT
from fly.scene import SHAPES, frame_state, kit_scene

KIT_DATA = json.loads(KIT.read_text())
PADS = {p["name"]: p for p in KIT_DATA["pads"]}


@pytest.fixture(scope="module")
def scene():
    return kit_scene(KIT_DATA)


def _prims(scene, pad=None, part=None, anim=None):
    return [(i, p) for i, p in enumerate(scene["prims"]) if (pad is None or p["pad"] == pad)
            and (part is None or p["part"] == part) and (anim is None or p["anim"] == anim)]


def _top(p):
    """z of an upright primitive's top (cylinder: half-height in size[1]; box, ellipsoid: z extent in size[2])."""
    return p["pos"][2] + p["size"][1 if p["type"] in ("cylinder", "capsule") else 2]


def test_every_pad_gets_a_struck_surface_on_its_physics_top_face(scene):
    for name, pad in PADS.items():
        (_, s), = _prims(scene, name, "strike")
        top = pad["pos"][2] + pad["half"][2]
        assert _top(s) == pytest.approx(top, abs=1e-9), name
        assert np.allclose(s["pos"][:2], pad["pos"][:2]), name
        if s["type"] == "cylinder":  # drum heads and cymbals: round, inside the square physics pad
            assert s["size"][0] <= pad["half"][0] + 1e-12, name


def test_primitives_are_well_formed(scene):
    for p in scene["prims"]:
        assert p["type"] in SHAPES
        assert len(p["pos"]) == 3 and len(p["size"]) == 3 and len(p["rgba"]) == 4
        assert np.linalg.norm(p["quat"]) == pytest.approx(1.0)
        assert all(s >= 0 for s in p["size"])
    assert {p["part"] for p in scene["prims"]} >= {"strike", "shell", "stand", "floor", "bass_drum", "beater"}


def test_nothing_covers_a_struck_surface(scene):
    """MuJoCo cylinders are solid: a rim reaching above the head hides the head (and its flash). Over a pad's footprint
    only cymbal bells, small and centred, rise above the struck face, by at most 0.0015 cm. (Beside it, e.g. the kick
    pedal's beater post, things may stand taller.)"""
    for name, pad in PADS.items():
        top = pad["pos"][2] + pad["half"][2]
        for _, p in _prims(scene, name):
            over = np.all(np.abs(np.subtract(p["pos"][:2], pad["pos"][:2])) < pad["half"][:2])
            if p["part"] == "bell":
                assert _top(p) - top < 0.0015, name
            elif over and p["part"] != "strike" and p["anim"] != "beater":  # the beater rises over the pedal's heel
                assert _top(p) <= top + 1e-9, (name, p["part"])


def _state(scene, t_ms, hits=(), touching=None):
    T = int(t_ms) + 1
    touching = np.zeros((T, len(scene["pads"])), dtype=bool) if touching is None else touching
    return frame_state(scene, t_ms, list(hits), touching)


def test_a_hit_flashes_its_drum_then_fades(scene):
    (i, _), = _prims(scene, "snare", "strike")
    (j, _), = _prims(scene, "tom1", "strike")
    base = _state(scene, 500)["rgba"]
    hit = [(100.0, "snare", 127)]
    assert not np.allclose(_state(scene, 110, hit)["rgba"][i], base[i])
    assert np.allclose(_state(scene, 110, hit)["rgba"][j], base[j])  # other drums untouched
    assert np.allclose(_state(scene, 90, hit)["rgba"][i], base[i])  # not before the hit
    assert np.allclose(_state(scene, 400, hit)["rgba"][i], base[i])  # faded


def test_a_hit_glows_so_bronze_on_yellow_still_shows(scene):
    (i, _), = _prims(scene, "hat", "strike")
    hit = [(100.0, "hat", 100)]
    assert _state(scene, 90, hit)["glow"][i] == 0
    assert _state(scene, 110, hit)["glow"][i] > 0.3
    assert _state(scene, 400, hit)["glow"][i] == 0


def test_louder_hits_flash_brighter(scene):
    (i, _), = _prims(scene, "snare", "strike")
    base = _state(scene, 500)["rgba"][i]
    soft = _state(scene, 110, [(100.0, "snare", 20)])["rgba"][i]
    loud = _state(scene, 110, [(100.0, "snare", 127)])["rgba"][i]
    assert np.abs(loud - base).sum() > np.abs(soft - base).sum() > 0


def test_a_struck_cymbal_wobbles_about_its_centre_and_settles(scene):
    (i, s), = _prims(scene, "crash", "strike")
    hit = [(100.0, "crash", 127)]
    st = _state(scene, 130, hit)
    assert not np.allclose(st["quat"][i], s["quat"])
    assert np.allclose(st["pos"][i], s["pos"], atol=1e-3)  # it tilts, it doesn't fly off
    assert np.allclose(_state(scene, 2500, hit)["quat"][i], s["quat"], atol=1e-3)


def test_the_hihat_opens_when_the_foot_is_off_its_pedal(scene):
    (i, _), = _prims(scene, "hat", "hat_bottom")
    (k, top), = _prims(scene, "hat", "strike")
    pedal = scene["pads"].index("hat_pedal")
    closed = np.zeros((200, len(scene["pads"])), dtype=bool)
    closed[:, pedal] = True
    z_closed = _state(scene, 199, touching=closed)["pos"][i][2]
    z_open = _state(scene, 199)["pos"][i][2]
    assert z_open < z_closed < top["pos"][2]
    assert _state(scene, 199)["pos"][k][2] == top["pos"][2]  # the top cymbal (the struck face) never moves


def test_the_kick_beater_swings_while_the_pedal_is_down(scene):
    (i, b), = _prims(scene, "kick", "beater")
    kick = scene["pads"].index("kick")
    down = np.zeros((200, len(scene["pads"])), dtype=bool)
    down[150:, kick] = True
    assert np.allclose(_state(scene, 100, touching=down)["quat"][i], b["quat"])
    assert not np.allclose(_state(scene, 199, touching=down)["quat"][i], b["quat"])


def _volume_points(p):
    """A 3x3x3 grid through the primitive's volume, in world coordinates."""
    from fly.scene import rotate

    g = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)], dtype=float)
    s = np.asarray(p["size"], dtype=float)
    if p["type"] in ("cylinder", "capsule"):
        local = g * np.array([s[0] / np.sqrt(2), s[0] / np.sqrt(2), s[1]])
    else:
        local = g * s
    return np.asarray(p["pos"]) + np.array([rotate(p["quat"], v) for v in local])


def test_the_kit_stays_clear_of_the_fly_at_rest(scene):
    pytest.importorskip("mujoco")
    pytest.importorskip("flybody")
    import mujoco

    from fly.body import Body, _clearance

    body = Body(KIT_DATA)
    body.m.geom_contype[body.pad_of >= 0] = 0  # measure against the fly only, not the physics pads
    mujoco.mj_forward(body.m, body.d)
    for p in scene["prims"]:
        if p["part"] in ("strike", "rim", "bell", "floor"):  # the struck faces sit on the physics pads, placed clear
            continue
        for x in _volume_points(p):
            assert _clearance(body.m, body.d, x) > 0, (p["pad"], p["part"], x)

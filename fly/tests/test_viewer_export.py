import json

import numpy as np
import pytest

from fly.viewer_export import touching_intervals


def test_touching_becomes_half_open_intervals_per_pad():
    t = np.zeros((8, 2), dtype=bool)
    t[2:4, 0] = True
    t[5:, 0] = True
    t[:, 1] = True
    assert touching_intervals(t, ["snare", "kick"]) == {"snare": [[2, 4], [5, 8]], "kick": [[0, 8]]}
    assert touching_intervals(np.zeros((3, 1), dtype=bool), ["hat"]) == {"hat": []}


@pytest.fixture(scope="module")
def body():
    pytest.importorskip("mujoco")
    pytest.importorskip("flybody")
    from fly.body import Body

    return Body()


def test_model_export_holds_every_visible_geom(tmp_path, body):
    from fly.viewer_export import export_model

    export_model(body, tmp_path)
    info = json.loads((tmp_path / "fly.json").read_text())
    blob = (tmp_path / "fly.bin").read_bytes()
    assert len(info["geoms"]) == 87  # flybody's 85 meshes + 2 sticks; invisible (alpha 0) and physics pads left out
    assert {g["type"] for g in info["geoms"]} == {"mesh", "capsule"}
    for mesh in info["meshes"]:
        n = mesh["count"]
        assert mesh["normal_offset"] + 3 * n <= len(blob) and mesh["position_offset"] + 12 * n <= len(blob)
        assert mesh["position_offset"] % 4 == 0
    assert info["bytes"] == len(blob)
    assert all(0 <= g["body"] < len(info["bodies"]) for g in info["geoms"])
    sticks = [g for g in info["geoms"] if g["type"] == "capsule"]
    assert sorted(g["name"] for g in sticks) == ["stick_front_left", "stick_front_right"]


def test_run_export_samples_the_recorded_poses(tmp_path, body):
    import mujoco

    from fly.loop import Recorder, play
    from fly.viewer_export import FRAME_T0_MS, export_model, export_run

    run = tmp_path / "run"
    run.mkdir()
    rec = Recorder(body, 25)
    body.reset()
    play(np.tile(body.rest, (25, 1)), body, on_step=rec)
    rec.save(run / "poses.npz", offset_ms=5.0)
    (run / "meta.json").write_text(json.dumps({"groove": "x.mid", "driver": "teacher", "hits": 0}))
    (run / "hits.json").write_text("[]")
    model = tmp_path / "model"
    export_model(body, model)
    export_run(run, model, frame_ms=2)

    v = json.loads((run / "viewer.json").read_text())
    bodies = json.loads((model / "fly.json").read_text())["bodies"]
    assert v["frame_ms"] == 2 and v["n_frames"] == 13 and v["n_bodies"] == len(bodies) and v["offset_ms"] == 5.0
    poses = np.fromfile(run / "poses.bin", dtype="<f4").reshape(v["n_frames"], len(bodies), 7)
    body.d.qpos[:] = rec.qpos[2 * 3]  # frame 3 = the pose recorded at step 6
    mujoco.mj_kinematics(body.m, body.d)
    k = body.m.body(bodies[5]).id
    assert np.allclose(poses[3, 5, :3], body.d.xpos[k], atol=1e-6)
    assert np.allclose(np.abs(poses[3, 5, 3:] @ body.d.xquat[k]), 1.0, atol=1e-5)  # same rotation (w x y z)
    assert v["t0_ms"] == FRAME_T0_MS

    scene = json.loads((run / "scene.json").read_text())
    assert len(scene["prims"]) > 50 and set(scene["touching"]) == set(scene["pads"])

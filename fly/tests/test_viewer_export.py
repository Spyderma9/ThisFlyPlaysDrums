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


def test_every_neuron_gets_one_group_and_highlights_win_over_regions():
    from fly.viewer_export import GROUPS, neuron_groups

    superclass = np.array(["ol_intrinsic", "cb_intrinsic", "vnc_intrinsic", "descending_neuron", "descending_neuron",
                           "vnc_motor", "cb_sensory", "ENS"], dtype=object)
    codes = neuron_groups(superclass, cue_groups={"snare": np.array([6])}, dn_plastic=np.array([4]),
                          leg_mns={"front_left": np.array([5])})
    name = {c: g["name"] for c, g in GROUPS.items()}
    assert [name[c] for c in codes] == ["optic lobe", "central brain", "nerve cord", "descending",
                                        "descending, learning", "leg motor: front left", "hearing: snare", "other"]


def test_spikes_export_round_trips_and_bins_count_every_spike(tmp_path):
    from fly.viewer_export import GROUPS, export_spikes

    ids = np.array([0, 2, 1, 2, 2], dtype=np.uint32)  # step 0: 0, 2; step 1: -; step 2: 1, 2; ... step 6: 2
    offsets = np.array([0, 2, 2, 4, 4, 4, 4, 5], dtype=np.uint32)
    np.savez_compressed(tmp_path / "spikes.npz", ids=ids, offsets=offsets, n_neurons=np.int64(3))
    codes = np.array([1, 2, 30], dtype=np.uint8)  # optic lobe, central brain, descending
    info = export_spikes(tmp_path, codes, bin_ms=5)
    raw = np.fromfile(tmp_path / "spikes.bin", dtype="<u4")
    assert info == {"steps": 7, "total": 5, "offsets": 8, "ids": 5}
    assert raw[:8].tolist() == offsets.tolist() and raw[8:].tolist() == ids.tolist()
    brain = json.loads((tmp_path / "brain.json").read_text())
    assert brain["bin_ms"] == 5
    by = {GROUPS[int(c)]["name"]: v for c, v in brain["bins"].items()}
    assert by["optic lobe"] == [1, 0] and by["central brain"] == [1, 0] and by["descending"] == [2, 1]


def test_a_neuron_without_a_soma_is_drawn_at_the_weighted_centre_of_its_targets():
    from fly.viewer_export import place_by_targets

    nan = np.nan
    xyz = np.array([[nan, nan, nan], [0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [nan, nan, nan], [nan, nan, nan]])
    pre = np.array([0, 0, 3])
    post = np.array([1, 2, 4])  # neuron 3's only target has no position either
    weight = np.array([1.0, -3.0, 5.0])  # inhibitory synapses still count, by size
    out = place_by_targets(xyz, pre, post, weight, np.array([0, 3]))
    assert np.allclose(out[0], [7.5, 0.0, 0.0])
    assert np.isnan(out[3]).all()  # nothing to place it by: it stays hidden
    assert np.allclose(out[1:3], xyz[1:3])  # neurons with a soma don't move

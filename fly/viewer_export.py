"""Export recorded runs for the browser viewer (viewer/): the fly's meshes once, then each run's poses and kit.

    python -m fly.viewer_export runs/viz/teacher_SD90 [runs/<id> ...] [--frame-ms 2]

Writes (all little-endian; units cm, z up, quaternions w x y z, as in MuJoCo):
  runs/model/fly.json + fly.bin   the fly's visible geoms (from fly.body.Body, so the sticks too)
      fly.json: bodies [names]; meshes [{position_offset, normal_offset, count}]; geoms [{name, body, type
                (mesh|capsule|...), mesh, size, pos, quat (in the body frame), rgba, specular, shininess}]; bytes
      fly.bin:  per mesh, unindexed triangles: float32 positions [count, 3], then after all positions, int8 normals
                [count, 3] (x127)
  runs/<id>/poses.bin    float32 [n_frames, n_bodies, 7]: each fly.json body's xpos xyz and xquat wxyz, frame k is the
                         recorded pose at sim time t0_ms + k * frame_ms
  runs/<id>/viewer.json  {frame_ms, t0_ms, n_frames, n_bodies, offset_ms (sim ms of score time 0), steps, groove,
                         driver, hits, model: "model/fly.json"}
  runs/<id>/scene.json   fly.scene.kit_scene() + touching: {pad: [[start, end), ...] sim ms}
  runs/viewer_index.json every exported run, for the viewer's run picker
hits.json and meta.json stay as fly.loop wrote them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FRAME_T0_MS = 1  # poses.npz row k is the pose after sim step k, i.e. at sim time k + 1 ms
DEFAULT_RGBA = (0.5, 0.5, 0.5, 1.0)  # MuJoCo's geom default: a geom left at it shows its material's colour


def touching_intervals(touching: np.ndarray, pads: list[str]) -> dict[str, list[list[int]]]:
    """[T, pads] bool -> {pad: [[start, end), ...]} in steps."""
    out = {}
    for k, pad in enumerate(pads):
        col = np.concatenate([[False], touching[:, k], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(col))
        out[pad] = [[int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]
    return out


def _visible(body) -> list[int]:
    m = body.m
    return [g for g in range(m.ngeom) if m.geom_group[g] <= 2 and body.pad_of[g] < 0 and _rgba(m, g)[3] > 0]


def _rgba(m, g) -> list[float]:
    mat = m.geom_matid[g]
    if mat >= 0 and np.allclose(m.geom_rgba[g], DEFAULT_RGBA):
        return [float(x) for x in m.mat_rgba[mat]]
    return [float(x) for x in m.geom_rgba[g]]


def export_model(body, out_dir: Path) -> dict:
    """The fly's visible geoms and meshes -> out_dir/fly.json + fly.bin. Returns fly.json's content."""
    import mujoco

    m = body.m
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    geoms = _visible(body)
    body_ids = sorted({int(m.geom_bodyid[g]) for g in geoms})
    body_index = {b: i for i, b in enumerate(body_ids)}
    mesh_ids = sorted({int(m.geom_dataid[g]) for g in geoms if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH})
    positions, normals, meshes, count0 = [], [], [], 0
    for i in mesh_ids:
        va, fa, na, nf = m.mesh_vertadr[i], m.mesh_faceadr[i], m.mesh_normaladr[i], m.mesh_facenum[i]
        faces = m.mesh_face[fa:fa + nf]
        fnorm = m.mesh_facenormal[fa:fa + nf]
        positions.append(m.mesh_vert[va + faces].reshape(-1, 3).astype("<f4"))
        n = m.mesh_normal[na + fnorm].reshape(-1, 3)
        n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        normals.append(np.round(n * 127).astype(np.int8))
        meshes.append({"name": m.mesh(i).name, "count": int(3 * nf), "first": count0})
        count0 += int(3 * nf)
    pos_bytes = 12 * count0
    for mesh in meshes:
        mesh["position_offset"] = 12 * mesh["first"]
        mesh["normal_offset"] = pos_bytes + 3 * mesh.pop("first")
    blob = b"".join(p.tobytes() for p in positions) + b"".join(n.tobytes() for n in normals)
    (out_dir / "fly.bin").write_bytes(blob)
    mesh_index = {i: k for k, i in enumerate(mesh_ids)}
    types = {int(v): k.replace("mjGEOM_", "").lower() for k, v in mujoco.mjtGeom.__members__.items()}
    info = {
        "units": "cm", "up": "z", "quat": "wxyz", "bytes": len(blob),
        "bodies": [m.body(b).name for b in body_ids],
        "meshes": meshes,
        "geoms": [{
            "name": m.geom(g).name, "body": body_index[int(m.geom_bodyid[g])], "type": types[int(m.geom_type[g])],
            "mesh": mesh_index.get(int(m.geom_dataid[g])) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH else None,
            "size": [float(x) for x in m.geom_size[g]], "pos": [float(x) for x in m.geom_pos[g]],
            "quat": [float(x) for x in m.geom_quat[g]], "rgba": _rgba(m, g),
            "specular": float(m.mat_specular[m.geom_matid[g]]) if m.geom_matid[g] >= 0 else 0.5,
            "shininess": float(m.mat_shininess[m.geom_matid[g]]) if m.geom_matid[g] >= 0 else 0.5,
        } for g in geoms],
    }
    (out_dir / "fly.json").write_text(json.dumps(info))
    return info


def export_run(run: Path, model_dir: Path, frame_ms: int = 2, body=None) -> dict:
    """poses.npz -> poses.bin, viewer.json, scene.json in the run dir. Returns viewer.json's content."""
    import mujoco

    from fly.body import Body
    from fly.loop import load_poses
    from fly.scene import kit_scene

    run = Path(run)
    body = body or Body()
    info = json.loads((Path(model_dir) / "fly.json").read_text())
    ids = [body.m.body(name).id for name in info["bodies"]]
    poses = load_poses(run)
    if poses["qpos"].shape[1] != body.m.nq or poses["pads"] != body.pads:
        raise SystemExit(f"{run} was recorded with a different kit or body than fly/kit.json")
    rows = np.arange(0, len(poses["qpos"]), frame_ms)
    out = np.zeros((len(rows), len(ids), 7), dtype="<f4")
    m, d = body.m, body.d
    for k, row in enumerate(rows):
        d.qpos[:] = poses["qpos"][row]
        mujoco.mj_kinematics(m, d)
        out[k, :, :3] = d.xpos[ids]
        out[k, :, 3:] = d.xquat[ids]
    out.tofile(run / "poses.bin")
    scene = kit_scene(body.kit)
    scene["touching"] = touching_intervals(poses["touching"], poses["pads"])
    (run / "scene.json").write_text(json.dumps(scene))
    meta = json.loads((run / "meta.json").read_text())
    viewer = {"frame_ms": frame_ms, "t0_ms": FRAME_T0_MS, "n_frames": len(rows), "n_bodies": len(ids),
              "offset_ms": poses["offset_ms"], "steps": len(poses["qpos"]), "groove": meta.get("groove"),
              "driver": meta.get("driver", "fly"), "weights": meta.get("weights"), "hits": meta.get("hits"),
              "model": "model/fly.json"}  # relative to runs/; the viewer loads runs/model/ (--model-dir for tests)
    (run / "viewer.json").write_text(json.dumps(viewer, indent=1))
    return viewer


def update_index(runs_root: Path) -> list[dict]:
    """runs/viewer_index.json: every run dir under runs_root that has a viewer.json, newest first."""
    runs_root = Path(runs_root)
    entries = []
    for v in runs_root.rglob("viewer.json"):
        info = json.loads(v.read_text())
        entries.append({"id": str(v.parent.relative_to(runs_root)), "groove": Path(info.get("groove") or "").name,
                        "driver": info.get("driver"), "hits": info.get("hits"), "mtime": v.stat().st_mtime})
    entries.sort(key=lambda e: -e["mtime"])
    (runs_root / "viewer_index.json").write_text(json.dumps(entries, indent=1))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run dirs with poses.npz (fly.loop --poses)")
    ap.add_argument("--frame-ms", type=int, default=2)
    ap.add_argument("--model-dir", type=Path, default=None, help="default: runs/model")
    args = ap.parse_args()

    from fly.body import Body
    from fly.connectome import REPO

    root = REPO / "runs"
    model_dir = args.model_dir or root / "model"
    body = Body()
    info = export_model(body, model_dir)
    print(f"model: {len(info['geoms'])} geoms, {len(info['bodies'])} bodies, {info['bytes'] / 1e6:.1f} MB -> {model_dir}")
    for run in args.runs:
        v = export_run(run, model_dir, args.frame_ms, body)
        size = (run / "poses.bin").stat().st_size / 1e6
        print(f"{run}: {v['n_frames']} frames every {args.frame_ms} ms ({size:.1f} MB)")
    print(f"index: {len(update_index(root))} runs in {root / 'viewer_index.json'}")


if __name__ == "__main__":
    main()

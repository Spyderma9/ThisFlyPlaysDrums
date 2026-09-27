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

The brain (runs recorded with --spikes):
  runs/model/neurons.bin + neurons.json   once: float32 [N, 3] soma xyz in um (NaN where MaleCNS has no soma
      position; row i is neuron i of the simulation), then uint8 [N] group codes (GROUPS, listed in neurons.json)
  runs/<id>/spikes.bin   uint32 offsets [steps + 1], then uint32 neuron ids: sim step t fired ids[offsets[t]:offsets[t+1]]
  runs/<id>/brain.json   {bin_ms, bins: {group code: spikes per bin}} for the panel's meters and sparklines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

FRAME_T0_MS = 1
VOXEL_UM = 0.008  # MaleCNS voxels are 8 nm


def _groups() -> dict[int, dict]:
    from fly.drums import LEGS, VOICES

    g = {0: {"name": "other", "kind": "region"}, 1: {"name": "optic lobe", "kind": "region"},
         2: {"name": "central brain", "kind": "region"}, 3: {"name": "nerve cord", "kind": "region"},
         30: {"name": "descending", "kind": "descending"},
         31: {"name": "descending, learning", "kind": "descending"}}  # receives trainable cue -> DN synapses
    for k, v in enumerate(VOICES):
        g[10 + k] = {"name": f"hearing: {v.name}", "kind": "hearing", "voice": v.name}
    for k, leg in enumerate(LEGS):
        g[40 + k] = {"name": f"leg motor: {leg.replace('_', ' ')}", "kind": "motor", "leg": leg}
    return g


GROUPS = _groups()
_CODE = {g.get("voice") or g.get("leg") or g["name"]: c for c, g in GROUPS.items()}


def neuron_groups(superclass: np.ndarray, cue_groups: dict, dn_plastic: np.ndarray, leg_mns: dict) -> np.ndarray:
    """uint8 GROUPS code per neuron: its region, unless it is a hearing (cue), descending or leg motor neuron."""
    sc = np.asarray(superclass, dtype=object).astype(str)
    codes = np.zeros(len(sc), dtype=np.uint8)
    codes[np.char.startswith(sc, "ol_") | np.char.startswith(sc, "visual_")] = _CODE["optic lobe"]
    codes[np.char.startswith(sc, "cb_")] = _CODE["central brain"]
    codes[np.char.startswith(sc, "vnc_") | np.char.startswith(sc, "ascending") | np.char.startswith(sc, "sensory_asc")] \
        = _CODE["nerve cord"]
    codes[np.char.startswith(sc, "descending")] = _CODE["descending"]
    codes[np.asarray(dn_plastic, dtype=np.int64)] = _CODE["descending, learning"]
    for leg, idx in leg_mns.items():
        codes[np.asarray(idx, dtype=np.int64)] = _CODE[leg]
    for voice, idx in cue_groups.items():
        codes[np.asarray(idx, dtype=np.int64)] = _CODE[voice]
    return codes


def place_by_targets(xyz: np.ndarray, pre: np.ndarray, post: np.ndarray, weight: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """A copy of xyz where each neuron in idx without a soma position sits at the |weight|-weighted centre of the
    positioned neurons it synapses onto. Johnston's organ (hearing) neurons need this: their cell bodies are in the
    antennae, outside the scanned CNS. Neurons with no positioned target stay NaN (not drawn)."""
    out = xyz.copy()
    todo = np.asarray(idx)[np.isnan(xyz[idx]).any(axis=1)]
    if not len(todo):
        return out
    has_pos = ~np.isnan(xyz).any(axis=1)
    keep = np.isin(pre, todo) & has_pos[post]
    p, q, w = pre[keep], post[keep], np.abs(weight[keep]).astype(np.float64)
    total = np.bincount(p, weights=w, minlength=len(xyz))
    for k in range(3):
        s = np.bincount(p, weights=w * xyz[q, k], minlength=len(xyz))
        placed = todo[total[todo] > 0]
        out[placed, k] = s[placed] / total[placed]
    return out


def export_neurons(out_dir: Path, wiring=None) -> dict:
    """runs/model/neurons.bin + neurons.json from the wiring every run uses (fly.wiring.wire)."""
    from fly.wiring import wire

    w = wiring or wire()
    n = w.conn.neurons
    xyz = n[["x", "y", "z"]].to_numpy(dtype=np.float64) * VOXEL_UM
    hearing = np.concatenate(list(w.cue_groups.values()))
    xyz = place_by_targets(xyz, w.conn.pre, w.conn.post, w.conn.weight, hearing)
    dn_plastic = np.unique(w.conn.post[w.plastic_mask])
    codes = neuron_groups(n["superclass"].to_numpy(), w.cue_groups, dn_plastic, w.leg_mns)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "neurons.bin").write_bytes(xyz.astype("<f4").tobytes() + codes.tobytes())
    ok = ~np.isnan(xyz).any(axis=1)
    info = {
        "count": int(len(n)), "positioned": int(ok.sum()), "units": "um", "axes": "MaleCNS voxel axes x y z, times 0.008",
        "min": np.nanmin(xyz, axis=0).round(2).tolist(), "max": np.nanmax(xyz, axis=0).round(2).tolist(),
        "groups": {str(c): {**g, "count": int((codes == c).sum())} for c, g in GROUPS.items()},
        "connectome": w.conn.name,
        "placed_by_targets": "hearing groups: Johnston's organ cell bodies are in the antennae, outside the scanned "
                             "CNS, so each is drawn at the synapse-weighted centre of the neurons it connects to",
    }
    (out_dir / "neurons.json").write_text(json.dumps(info, indent=1))
    return info


def export_spikes(run: Path, codes: np.ndarray, bin_ms: int = 5) -> dict:
    """spikes.npz -> spikes.bin (offsets then ids, uint32) and brain.json (spikes per group per bin_ms)."""
    from fly.loop import load_spikes

    run = Path(run)
    s = load_spikes(run)
    offsets, ids = s["offsets"].astype("<u4"), s["ids"].astype("<u4")
    (run / "spikes.bin").write_bytes(offsets.tobytes() + ids.tobytes())
    steps = len(offsets) - 1
    step_of = np.repeat(np.arange(steps), np.diff(offsets.astype(np.int64)))
    n_bins = -(-steps // bin_ms)
    group = codes[ids.astype(np.int64)]
    bins = {}
    for c in np.unique(group):
        bins[str(int(c))] = np.bincount(step_of[group == c] // bin_ms, minlength=n_bins).astype(int).tolist()
    (run / "brain.json").write_text(json.dumps({"bin_ms": bin_ms, "steps": steps, "bins": bins}))
    return {"steps": steps, "total": int(len(ids)), "offsets": int(len(offsets)), "ids": int(len(ids))}  # poses.npz row k is the pose after sim step k, i.e. at sim time k + 1 ms
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
    spikes = None
    if (run / "spikes.npz").exists():
        nb = Path(model_dir) / "neurons.bin"
        if not nb.exists():
            raise SystemExit(f"{run} has a brain recording: export the neurons first (--brain)")
        n = json.loads((Path(model_dir) / "neurons.json").read_text())["count"]
        codes = np.frombuffer(nb.read_bytes()[12 * n:], dtype=np.uint8)
        spikes = export_spikes(run, codes)
    viewer = {"frame_ms": frame_ms, "t0_ms": FRAME_T0_MS, "n_frames": len(rows), "n_bodies": len(ids),
              "offset_ms": poses["offset_ms"], "steps": len(poses["qpos"]), "groove": meta.get("groove"),
              "driver": meta.get("driver", "fly"), "weights": meta.get("weights"), "hits": meta.get("hits"),
              "alpha": meta.get("alpha"), "spikes": spikes, "brain_mode": meta.get("brain"),  # driving | listening
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
                        "driver": info.get("driver"), "alpha": info.get("alpha"), "brain": bool(info.get("spikes")),
                        "hits": info.get("hits"), "mtime": v.stat().st_mtime})
    entries.sort(key=lambda e: -e["mtime"])
    (runs_root / "viewer_index.json").write_text(json.dumps(entries, indent=1))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+", help="run dirs with poses.npz (fly.loop --poses)")
    ap.add_argument("--frame-ms", type=int, default=2)
    ap.add_argument("--model-dir", type=Path, default=None, help="default: runs/model")
    ap.add_argument("--brain", action="store_true", help="(re)build runs/model/neurons.* (needed once for brain runs)")
    args = ap.parse_args()

    from fly.body import Body
    from fly.connectome import REPO

    root = REPO / "runs"
    model_dir = args.model_dir or root / "model"
    body = Body()
    info = export_model(body, model_dir)
    print(f"model: {len(info['geoms'])} geoms, {len(info['bodies'])} bodies, {info['bytes'] / 1e6:.1f} MB -> {model_dir}")
    if args.brain or (any((r / "spikes.npz").exists() for r in args.runs) and not (model_dir / "neurons.bin").exists()):
        n = export_neurons(model_dir)
        print(f"neurons: {n['positioned']} of {n['count']} with a position -> {model_dir / 'neurons.bin'}")
    for run in args.runs:
        v = export_run(run, model_dir, args.frame_ms, body)
        size = (run / "poses.bin").stat().st_size / 1e6
        brain = f", {v['spikes']['total']} spikes ({(run / 'spikes.bin').stat().st_size / 1e6:.1f} MB)" if v["spikes"] else ""
        print(f"{run}: {v['n_frames']} frames every {args.frame_ms} ms ({size:.1f} MB){brain}")
    print(f"index: {len(update_index(root))} runs in {root / 'viewer_index.json'}")


if __name__ == "__main__":
    main()

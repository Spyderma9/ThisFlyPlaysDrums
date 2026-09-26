"""Render a recorded run (fly.loop --poses) as a video: the fly with its sticks at a real-looking drum kit (fly.scene).

    python -m fly.clip runs/<id> --out runs/clips/<id>.mp4 [--start 1 --seconds 4] [--speed 0.25] [--view front]
    python -m fly.clip runs/<fly id> --compare runs/<teacher id> --out cmp.mp4   # two runs side by side
    python -m fly.clip runs/<id> --still 2.0 --out still.png                     # one frame at score time 2.0 s

It renders the recording and doesn't re-simulate: each frame puts the body in its recorded pose (interpolated between
sim ms) and draws fly.scene's kit over the physics model, whose flat pads are hidden. Hits flash their drum, cymbals
rock, the hi-hat opens with the pedal and the kick beater swings. --speed 0.25 is 4x slow motion.
Video: H.264 MP4 through ffmpeg ($FFMPEG, else on PATH), or a GIF when the output ends in .gif or there's no ffmpeg.
Needs offscreen OpenGL (MUJOCO_GL=egl on the server).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

VIEWS = {  # MuJoCo free camera: lookat (cm), distance (cm), azimuth and elevation (deg). The fly faces +x.
    "front": ((0.0, 0.0, -0.17), 0.62, 200.0, -22.0),
    "three_quarter": ((0.0, 0.0, -0.16), 0.72, 235.0, -18.0),
    "side": ((-0.02, 0.0, -0.17), 0.75, 270.0, -12.0),
}
TYPES = {"cylinder": "mjGEOM_CYLINDER", "ellipsoid": "mjGEOM_ELLIPSOID", "box": "mjGEOM_BOX", "capsule": "mjGEOM_CAPSULE"}
LABEL_MS = 300  # how long a hit stays in a panel's list (sim ms)


class Panel:
    """One recorded run: its body in the recorded poses, the kit, hit flashes and a list of recent hits."""

    def __init__(self, run: Path, size: tuple[int, int], view: str, title: str | None = None):
        import mujoco

        from fly.body import Body
        from fly.loop import load_poses
        from fly.scene import kit_scene

        self.run, (self.w, self.h) = Path(run), size
        self.meta = json.loads((self.run / "meta.json").read_text())
        self.poses = load_poses(self.run)
        self.offset = self.poses["offset_ms"]
        self.body = Body()
        if self.poses["pads"] != self.body.pads or self.poses["qpos"].shape[1] != self.body.m.nq:
            raise SystemExit(f"{run} was recorded with a different kit or body than fly/kit.json")
        m = self.body.m
        m.geom_rgba[self.body.pad_of >= 0, 3] = 0  # the flat physics pads; fly.scene draws the drums in their place
        m.vis.headlight.ambient[:] = 0.32
        m.vis.headlight.diffuse[:] = 0.62
        m.vis.headlight.specular[:] = 0.25
        self.scene = kit_scene(self.body.kit)
        self.types = [int(getattr(mujoco.mjtGeom, TYPES[p["type"]])) for p in self.scene["prims"]]
        hits = json.loads((self.run / "hits.json").read_text())
        self.hits = [(h["t_s"] * 1000 + self.offset, h["pad"], h["velocity"]) for h in hits]
        self.labels = [(h["t_s"] * 1000 + self.offset, f"{h['voice']} ({_side(h['limb'])}) v{h['velocity']}") for h in hits]
        driver = self.meta.get("driver", "fly")
        self.title = title or (f"teacher's strokes" if driver == "teacher" else
                               f"the fly on its own{' (trained)' if self.meta.get('weights') else ' (untrained)'}")
        self.r = mujoco.Renderer(m, self.h, self.w, max_geom=m.ngeom + len(self.scene["prims"]) + 64)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.look(view)

    def look(self, view: str) -> None:
        lookat, dist, az, el = VIEWS[view]
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = lookat, dist, az, el

    @property
    def last_ms(self) -> float:
        return len(self.poses["qpos"]) - self.offset  # score time (ms) of the last recorded step

    def pose(self, t_ms: float) -> np.ndarray:
        """qpos at sim time t_ms: row k is the pose after step k, i.e. at k + 1 ms. Linear between rows (hinges only)."""
        q = self.poses["qpos"]
        x = float(np.clip(t_ms - 1, 0, len(q) - 1))
        i = int(x)
        j = min(i + 1, len(q) - 1)
        return q[i] + (x - i) * (q[j] - q[i])

    def frame(self, score_ms: float):
        import mujoco
        from PIL import Image, ImageDraw

        from fly.scene import frame_state

        t = score_ms + self.offset
        m, d = self.body.m, self.body.d
        d.qpos[:] = self.pose(t)
        mujoco.mj_kinematics(m, d)
        self.r.update_scene(d, camera=self.cam)
        scn = self.r.scene
        st = frame_state(self.scene, t, self.hits, self.poses["touching"])
        mat = np.zeros(9)
        for k, p in enumerate(self.scene["prims"]):
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mju_quat2Mat(mat, st["quat"][k])
            mujoco.mjv_initGeom(g, self.types[k], np.asarray(p["size"], dtype=float), st["pos"][k], mat,
                                st["rgba"][k].astype(np.float32))
            g.specular, g.shininess, g.reflectance = 0.6 * p["shine"], p["shine"], 0.0
            scn.ngeom += 1
        img = Image.fromarray(self.r.render())
        draw = ImageDraw.Draw(img)
        draw.text((10, 8), self.title, fill="white")
        recent = [label for at, label in self.labels if 0 <= t - at < LABEL_MS][-6:]
        for k, label in enumerate(recent):
            draw.text((10, 26 + 14 * k), label, fill=(255, 214, 90))
        return img


def _side(limb: str) -> str:
    return limb.replace("front_", "").replace("hind_", "foot ")


def ffmpeg_path() -> str | None:
    return os.environ.get("FFMPEG") or shutil.which("ffmpeg")


class Writer:
    """RGB frames -> H.264 MP4 (ffmpeg) or GIF (PIL)."""

    def __init__(self, out: Path, size: tuple[int, int], fps: int):
        self.out, self.fps, self.frames = out, fps, []
        exe = ffmpeg_path()
        self.proc = None
        if out.suffix.lower() != ".gif" and exe:
            self.proc = subprocess.Popen(
                [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
                 "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                 "-preset", "medium", "-movflags", "+faststart", str(out)], stdin=subprocess.PIPE)
        elif out.suffix.lower() != ".gif":
            self.out = out.with_suffix(".gif")
            print(f"no ffmpeg ($FFMPEG or PATH): writing {self.out} instead", flush=True)

    def add(self, img) -> None:
        if self.proc is not None:
            self.proc.stdin.write(np.asarray(img.convert("RGB")).tobytes())
        else:
            from PIL import Image

            self.frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=192))

    def close(self) -> Path:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise SystemExit("ffmpeg failed")
        elif self.frames:
            self.frames[0].save(self.out, save_all=True, append_images=self.frames[1:], duration=round(1000 / self.fps),
                                loop=0, optimize=True)
        return self.out


def compose(panels: list[Panel], score_ms: float, caption: str):
    from PIL import Image, ImageDraw

    imgs = [p.frame(score_ms) for p in panels]
    out = Image.new("RGB", (sum(i.width for i in imgs), imgs[0].height))
    x = 0
    for img in imgs:
        out.paste(img, (x, 0))
        x += img.width
    ImageDraw.Draw(out).text((10, out.height - 20), f"{score_ms / 1000:6.2f} s   {caption}", fill="white")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run dir with poses.npz (fly.loop --poses)")
    ap.add_argument("--out", type=Path, required=True, help=".mp4, .gif, or .png with --still")
    ap.add_argument("--compare", type=Path, default=None, help="a second run, shown on the right")
    ap.add_argument("--start", type=float, default=0.0, help="score time (s) the clip starts at")
    ap.add_argument("--seconds", type=float, default=None, help="score seconds to show (default: to the end)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed; 0.25 = 4x slow motion")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--view", choices=sorted(VIEWS), default="front")
    ap.add_argument("--size", default=None, help="WxH of each panel (default 1280x720, or 960x720 with --compare)")
    ap.add_argument("--still", type=float, nargs="+", default=None, metavar="S",
                    help="write one PNG per score time instead of a video (out: name.png -> name_<S>.png)")
    args = ap.parse_args()

    w, h = (int(v) for v in (args.size or ("960x720" if args.compare else "1280x720")).split("x"))
    w, h = w - w % 2, h - h % 2  # H.264 wants even sizes
    panels = [Panel(args.run, (w, h), args.view)]
    if args.compare is not None:
        panels.append(Panel(args.compare, (w, h), args.view))
    groove = Path(panels[0].meta.get("groove", "")).name
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.still:
        for s in args.still:
            out = args.out if len(args.still) == 1 else args.out.with_name(f"{args.out.stem}_{s:g}{args.out.suffix}")
            compose(panels, s * 1000, groove).save(out)
            print(f"wrote {out}")
        return

    end = min(p.last_ms for p in panels) / 1000
    seconds = end - args.start if args.seconds is None else min(args.seconds, end - args.start)
    n = int(seconds * args.fps / args.speed)
    caption = f"{groove}   " + ("real time" if args.speed == 1 else f"{1 / args.speed:g}x slow motion")
    writer = Writer(args.out, (w * len(panels), h), args.fps)
    print(f"rendering {n} frames ({seconds:.2f} s of score at {args.speed:g}x) -> {args.out}", flush=True)
    for k in range(n):
        writer.add(compose(panels, args.start * 1000 + k * args.speed * 1000 / args.fps, caption))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{n}", flush=True)
    print(f"wrote {writer.close()}")


if __name__ == "__main__":
    main()

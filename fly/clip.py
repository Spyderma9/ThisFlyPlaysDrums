"""A slow-motion clip of the fly playing a groove next to the teacher's strokes, both in the physics.

Left: the fly on its own (alpha = 0): encoder -> brain (optionally trained weights) -> decoder -> body, as fly.loop runs it.
Right: the teacher's q*(t) played straight into a second body (what training asks the fly for). Pads flash on a hit,
and each side lists its recent hits. For watching training, not for scoring (use fly.loop + human/score.py).

    python -m fly.clip --groove grooves/train/X.mid --weights runs/train/t3/last.pt --out runs/clips/t3.gif
    python -m fly.clip --groove X.mid --out untrained.gif --start 1 --seconds 2      # no --weights: untrained

Needs offscreen OpenGL (MUJOCO_GL=egl on the server).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLORS = {"snare": "#e63946", "hat": "#f4c430", "crash": "#f77f00", "tom1": "#4361ee", "tom2": "#7b2cbf",
          "tom3": "#2a9d8f", "ride": "#52b788", "kick": "#555555", "hat_pedal": "#999999"}
FLASH_MS = 40
LOOKAT, DISTANCE, AZIMUTH, ELEVATION = (0.03, 0.0, -0.15), 0.42, 200.0, -30.0


def _rgba(hex_color):
    return [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]


class Panel:
    """One body with its own renderer, pad colours, hit flashes and hit list."""

    def __init__(self, body, title: str, size=(480, 360)):
        import mujoco

        self.body, self.title, self.w, self.h = body, title, *size
        self.r = mujoco.Renderer(body.m, self.h, self.w)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:], self.cam.distance, self.cam.azimuth, self.cam.elevation = LOOKAT, DISTANCE, AZIMUTH, ELEVATION
        self.geom = {p: body.m.geom(f"pad_{p}").id for p in body.pads}
        for p, g in self.geom.items():
            body.m.geom_rgba[g] = _rgba(COLORS.get(p, "#cccccc"))
        self.flash, self.recent = {}, []

    def hits(self, t_ms: int, hits: list) -> None:
        for h in hits:
            self.flash[h.pad] = FLASH_MS
            side = h.limb.replace("front_", "").replace("hind_", "foot ")
            self.recent.append((t_ms, f"{h.voice} ({side}) v{h.velocity}"))
        for p in list(self.flash):
            self.flash[p] -= 1
            lit = self.flash[p] > 0
            self.body.m.geom_rgba[self.geom[p]] = [1, 1, 1, 1] if lit else _rgba(COLORS.get(p, "#cccccc"))
            if not lit:
                del self.flash[p]

    def frame(self, t_ms: int):
        from PIL import Image, ImageDraw

        self.r.update_scene(self.body.d, camera=self.cam)
        img = Image.fromarray(self.r.render())
        d = ImageDraw.Draw(img)
        d.text((8, 6), self.title, fill="white")
        for k, (_, label) in enumerate([x for x in self.recent if t_ms - x[0] < 300][-6:]):
            d.text((8, 24 + 13 * k), label, fill="yellow")
        return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--groove", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help=".gif")
    ap.add_argument("--weights", type=Path, default=None, help="trained plastic weights (fly.train); default untrained")
    ap.add_argument("--shuffled", type=int, default=None, metavar="SEED")
    ap.add_argument("--start", type=float, default=1.0, help="score time (s) the clip starts at")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--frame-ms", type=int, default=4, help="sim ms per frame; played at 25 fps (4 ms = 10x slow motion)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch

    from fly.body import Body
    from fly.decoder import Decoder
    from fly.encoder import encode, read_onsets
    from fly.loop import BURST_MS, default_lookahead_ms, simulate
    from fly.strokes import PREROLL_MS, STROKES, plan, teacher
    from fly.train import rest_on_pedal, wire_brain

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    strokes = json.loads(STROKES.read_text())
    enc = encode(args.groove, lookahead_ms=default_lookahead_ms(), burst_ms=BURST_MS, preroll_ms=PREROLL_MS)
    t0 = int(args.start * 1000 + enc.offset_ms)
    t1 = min(len(enc.rates), int((args.start + args.seconds) * 1000 + enc.offset_ms))

    wiring, brain, ck = wire_brain(args.weights, args.shuffled, device)  # the trained plastic set and tone, if any
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, wiring.conn.size).to(device)
    if ck is None or ck.get("rest_on_pedal", False):  # the posture fly.loop and fly.train use
        rest_on_pedal(decoder, strokes)
    label = f"fly on its own ({args.weights.parent.name}/{args.weights.name}, epoch {ck.get('epoch', '?')})" \
        if ck is not None else "fly on its own (untrained)"

    fly, tutor = Body(), Body()
    q_teacher = teacher(plan(read_onsets(args.groove), enc.offset_ms, strokes["pads"]), len(enc.rates), strokes,
                        decoder.rest.cpu().numpy())
    panels = (Panel(fly, label), Panel(tutor, "teacher's strokes (what training asks for)"))
    frames = []

    def on_step(t, state, q, hits):
        panels[0].hits(t, hits)
        panels[1].hits(t, tutor.step(q_teacher[t]))
        if t >= t0 and (t - t0) % args.frame_ms == 0:
            from PIL import Image, ImageDraw

            left, right = (p.frame(t) for p in panels)
            img = Image.new("RGB", (left.width + right.width, left.height))
            img.paste(left, (0, 0))
            img.paste(right, (left.width, 0))
            ImageDraw.Draw(img).text((8, img.height - 16), f"{(t - enc.offset_ms) / 1000:5.2f} s   "
                                     f"{args.groove.name}   {1000 / args.frame_ms / 25:.0f}x slow motion", fill="white")
            frames.append(img.convert("P", palette=Image.Palette.ADAPTIVE, colors=128))

    print(f"simulating {t1} ms of {args.groove.name} ({'trained' if ck else 'untrained'}) on {device}...", flush=True)
    simulate(enc.rates[:t1], brain, decoder, fly, seed=args.seed, device=device, on_step=on_step, log_every=1000)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=40, loop=0, optimize=True)
    fly_hits = sum(1 for _ in [x for x in panels[0].recent if t0 <= x[0] < t1])
    tutor_hits = sum(1 for _ in [x for x in panels[1].recent if t0 <= x[0] < t1])
    print(f"wrote {args.out}: {len(frames)} frames; hits in the clip: fly {fly_hits}, teacher {tutor_hits}")


if __name__ == "__main__":
    main()

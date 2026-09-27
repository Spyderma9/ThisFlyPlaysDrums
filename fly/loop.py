"""Co-simulation, one 1 ms step at a time: encoder rates -> brain -> decoder -> body -> contacts.

    python -m fly.loop --groove grooves/train/X.mid --alpha 0 --out runs/<id> [--seconds 5] [--shuffled SEED]
                       [--weights runs/train/<id>/best.pt]   # a trained fly (fly.train)
                       [--poses]                             # also record the body for fly.clip / the viewer
    python -m fly.loop --groove X.mid --teacher --out runs/<id> --poses   # q* straight into the body; CPU, no brain

Writes, all in score time (sim time - the encoder's offset; earlier hits are dropped):
  hits.mid   channel 10, the voice's out_note, velocity from contact speed, note_off 50 ms later
  hits.csv   t_ms,note,velocity (human/score.py reads it)
  hits.json  [{t_s, note, voice, velocity, contact_speed, pad, limb}]
  meta.json  groove, driver (fly or teacher), alpha, seed, offsets, wiring and kit summary, timings
  poses.npz  (--poses) qpos [T, nq] float32 after every sim ms, touching [T, pads] bool, pads, offset_ms (sim time)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import mido
import numpy as np
import torch

from fly.connectome import REPO
from fly.drums import DRUM_CHANNEL

BURST_MS = 50.0  # the probe's cue length


def default_lookahead_ms() -> float:
    """Cue -> motor latency + the teacher's stroke lead (fly/strokes.json), so a cue arrives in time to start a stroke."""
    from fly.strokes import CUE_LATENCY_MS, STROKES

    return json.loads(STROKES.read_text())["lookahead_ms"] if STROKES.exists() else CUE_LATENCY_MS
NOTE_MS = 50.0
TICKS_PER_BEAT, TEMPO = 1000, 500_000  # 120 bpm: one tick = 0.5 ms


def simulate(rates: np.ndarray, brain, decoder, body, seed: int = 0, device: str = "cpu",
             current=None, on_step=None, log_every: int = 0) -> list:
    """Run len(rates) steps. current(t, q) -> [1, N] mV or None (the teacher's alpha * I*, Phase 4);
    on_step(t, state, q, hits) sees every step (viewer recording)."""
    gen = torch.Generator(device=device).manual_seed(seed)
    state, r = brain.init_state(), decoder.init_state()
    drive = torch.as_tensor(rates, device=device)
    q = decoder.targets(r)
    hits, t0 = [], perf_counter()
    with torch.no_grad():
        for t in range(len(rates)):
            state = brain.step(state, drive[t:t + 1], None if current is None else current(t, q), generator=gen)
            r, q = decoder(r, state[2])
            new = body.step(q[0].cpu().numpy())
            hits += new
            if on_step is not None:
                on_step(t, state, q, new)
            if log_every and (t + 1) % log_every == 0:
                print(f"  {t + 1:6d} ms  {perf_counter() - t0:6.0f} s  {len(hits)} hits", flush=True)
    return hits


def play(q: np.ndarray, body, on_step=None) -> list:
    """Hold the body's servos at q[t] (e.g. the teacher's q*) for each 1 ms step. on_step(t, None, q[t], hits)."""
    hits = []
    for t in range(len(q)):
        new = body.step(q[t])
        hits += new
        if on_step is not None:
            on_step(t, None, q[t], new)
    return hits


class Recorder:
    """An on_step hook that keeps the body's pose and pad contacts after every sim ms (fly.clip, fly.viewer_export)."""

    def __init__(self, body, n_steps: int):
        self.body = body
        self.qpos = np.zeros((n_steps, body.m.nq), dtype=np.float32)
        self.touching = np.zeros((n_steps, len(body.pads)), dtype=bool)

    def __call__(self, t, *_):
        self.qpos[t] = self.body.d.qpos
        self.touching[t] = self.body.touching

    def save(self, path: Path, offset_ms: float) -> None:
        np.savez_compressed(path, qpos=self.qpos, touching=self.touching, pads=np.array(self.body.pads),
                            offset_ms=np.float64(offset_ms))


def load_poses(run: Path) -> dict:
    with np.load(Path(run) / "poses.npz") as z:
        return {"qpos": z["qpos"], "touching": z["touching"], "pads": [str(p) for p in z["pads"]],
                "offset_ms": float(z["offset_ms"])}


def score_time(hits: list, offset_ms: float) -> list[dict]:
    out = []
    for h in hits:
        t = h.t_ms - offset_ms
        if t >= 0:
            out.append({"t_s": round(t / 1000, 5), **{k: v for k, v in asdict(h).items() if k != "t_ms"}})
    return out


def write_midi(path: Path, hits: list[dict]) -> None:
    events = []  # (ms, order, message): note_off before note_on at the same time
    for i, h in enumerate(hits):
        on = h["t_s"] * 1000
        nxt = next((g["t_s"] * 1000 for g in hits[i + 1:] if g["note"] == h["note"]), np.inf)
        events.append((on, 1, mido.Message("note_on", channel=DRUM_CHANNEL, note=h["note"], velocity=h["velocity"])))
        events.append((min(on + NOTE_MS, nxt), 0, mido.Message("note_off", channel=DRUM_CHANNEL, note=h["note"], velocity=0)))
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    now = 0
    for ms, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        tick = int(round(mido.second2tick(ms / 1000, TICKS_PER_BEAT, TEMPO)))
        track.append(msg.copy(time=tick - now))
        now = tick
    mid.save(path)


def write_run(out: Path, hits: list[dict], meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "hits.json").write_text(json.dumps(hits, indent=1))
    with open(out / "hits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "note", "velocity"])
        for h in hits:
            w.writerow([round(h["t_s"] * 1000, 1), h["note"], h["velocity"]])
    write_midi(out / "hits.mid", hits)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def _groove_name(groove: Path) -> str:
    return str(groove.resolve().relative_to(REPO)) if groove.resolve().is_relative_to(REPO) else str(groove)


def _hit_summary(hits: list[dict]) -> dict:
    return {"hits": len(hits), "hits_by_voice": {v: sum(h["voice"] == v for h in hits) for v in sorted({h["voice"] for h in hits})}}


def teacher_run(groove: Path, out: Path, seconds: float | None = None, poses: bool = False,
                lookahead_ms: float | None = None, preroll_ms: float | None = None) -> tuple[list[dict], dict]:
    """The teacher's q* played straight into the body, no brain (fly.strokes' ceiling check, as a run dir).
    Returns (hits in score time, meta)."""
    from fly.body import Body
    from fly.encoder import encode, read_onsets
    from fly.strokes import PREROLL_MS, STROKES, plan, teacher

    strokes = json.loads(STROKES.read_text())
    lookahead = strokes["lookahead_ms"] if lookahead_ms is None else lookahead_ms
    preroll = PREROLL_MS if preroll_ms is None else preroll_ms
    enc = encode(groove, lookahead_ms=lookahead, preroll_ms=preroll)
    n_steps = len(enc.rates) if seconds is None else int((seconds * 1000 + enc.offset_ms) / enc.dt_ms)
    body = Body()
    notes = plan(read_onsets(groove), enc.offset_ms, strokes["pads"])
    q = teacher(notes, n_steps, strokes, body.rest)
    rec = Recorder(body, n_steps) if poses else None
    t0 = perf_counter()
    hits = score_time(play(q, body, on_step=rec), enc.offset_ms)
    sim_s = perf_counter() - t0
    meta = {
        "groove": _groove_name(Path(groove)), "driver": "teacher",
        "offset_ms": enc.offset_ms, "lookahead_ms": lookahead, "preroll_ms": preroll, "steps": n_steps, "seconds": seconds,
        "notes": len(notes), "dropped": sum(bool(n.dropped) for n in notes),
        "body": {"timestep": body.m.opt.timestep, "substeps": body.substeps},
        **_hit_summary(hits),
        "timings": {"sim_s": round(sim_s, 2), "wall_s_per_sim_s": round(sim_s / (n_steps / 1000), 2)},
    }
    write_run(Path(out), hits, meta)
    if rec is not None:
        rec.save(Path(out) / "poses.npz", enc.offset_ms)
    return hits, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groove", type=Path, required=True)
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seconds", type=float, default=None, help="stop after this much score time")
    ap.add_argument("--shuffled", type=int, default=None, metavar="SEED", help="degree-preserving shuffled control")
    ap.add_argument("--seed", type=int, default=0, help="Poisson input seed")
    ap.add_argument("--lookahead-ms", type=float, default=None, help="default: fly/strokes.json's lookahead")
    ap.add_argument("--preroll-ms", type=float, default=None, help="default: strokes.PREROLL_MS (the hat closes)")
    ap.add_argument("--burst-ms", type=float, default=BURST_MS)
    ap.add_argument("--weights", type=Path, default=None, help="trained plastic weights from fly.train (best.pt)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--poses", action="store_true", help="also write poses.npz (fly.clip, fly.viewer_export)")
    ap.add_argument("--teacher", action="store_true", help="play the teacher's q* into the body instead of the fly")
    args = ap.parse_args()
    if args.teacher:
        _, meta = teacher_run(args.groove, args.out, args.seconds, args.poses, args.lookahead_ms, args.preroll_ms)
        print(json.dumps({k: meta[k] for k in ("hits", "hits_by_voice", "dropped", "timings")}, indent=1))
        print(f"wrote {args.out}")
        return
    if args.alpha != 0:
        raise SystemExit("alpha > 0 needs the teacher strokes (Phase 3) and I* (Phase 4); only --alpha 0 runs now")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.body import Body
    from fly.decoder import Decoder
    from fly.encoder import encode
    from fly.strokes import PREROLL_MS, STROKES
    from fly.train import rest_on_pedal, wire_brain

    lookahead = args.lookahead_ms if args.lookahead_ms is not None else default_lookahead_ms()
    preroll = args.preroll_ms if args.preroll_ms is not None else PREROLL_MS

    timings, t0 = {}, perf_counter()
    enc = encode(args.groove, lookahead_ms=lookahead, burst_ms=args.burst_ms, preroll_ms=preroll)
    rates = enc.rates
    if args.seconds is not None:
        rates = rates[: int((args.seconds * 1000 + enc.offset_ms) / enc.dt_ms)]
    wiring, brain, ck = wire_brain(args.weights, args.shuffled, device)  # the trained plastic set and tone, if any
    timings["wiring_s"] = perf_counter() - t0
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, wiring.conn.size).to(device)
    on_pedal = ck is None or ck.get("rest_on_pedal", False)  # same posture as training; untrained runs match new ones
    if on_pedal:
        rest_on_pedal(decoder, json.loads(STROKES.read_text()))
    body = Body()
    timings["build_s"] = perf_counter() - t0 - timings["wiring_s"]
    print(f"{wiring.conn.name}: {len(rates)} steps on {device}, body dt {body.m.opt.timestep:g} s", flush=True)

    t1 = perf_counter()
    rec = Recorder(body, len(rates)) if args.poses else None
    raw = simulate(rates, brain, decoder, body, seed=args.seed, device=device, on_step=rec, log_every=1000)
    timings["sim_s"] = perf_counter() - t1
    timings["wall_s_per_sim_s"] = timings["sim_s"] / (len(rates) / 1000)
    hits = score_time(raw, enc.offset_ms)

    meta = {
        "groove": _groove_name(args.groove), "driver": "fly",
        "alpha": args.alpha, "seed": args.seed, "shuffled": args.shuffled, "device": device,
        "weights": None if args.weights is None else str(args.weights), "rest_on_pedal": on_pedal,
        "offset_ms": enc.offset_ms, "lookahead_ms": lookahead, "preroll_ms": preroll, "burst_ms": args.burst_ms,
        "steps": len(rates), "seconds": args.seconds,
        "wiring": wiring.summary(), "body": {"timestep": body.m.opt.timestep, "substeps": body.substeps},
        "hits": len(hits), "dropped_before_score_0": len(raw) - len(hits),
        "hits_by_voice": {v: sum(h["voice"] == v for h in hits) for v in sorted({h["voice"] for h in hits})},
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    write_run(args.out, hits, meta)
    if rec is not None:
        rec.save(args.out / "poses.npz", enc.offset_ms)
    print(json.dumps({k: meta[k] for k in ("hits", "hits_by_voice", "timings")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

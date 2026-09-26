"""Co-simulation, one 1 ms step at a time: encoder rates -> brain -> decoder -> body -> contacts.

    python -m fly.loop --groove grooves/train/X.mid --alpha 0 --out runs/<id> [--seconds 5] [--shuffled SEED]
                       [--weights runs/train/<id>/best.pt]   # a trained fly (fly.train)

Writes, all in score time (sim time - the encoder's offset; earlier hits are dropped):
  hits.mid   channel 10, the voice's out_note, velocity from contact speed, note_off 50 ms later
  hits.csv   t_ms,note,velocity (human/score.py reads it)
  hits.json  [{t_s, note, voice, velocity, contact_speed, pad, limb}]
  meta.json  groove, alpha, seed, offsets, wiring and kit summary, timings
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
    args = ap.parse_args()
    if args.alpha != 0:
        raise SystemExit("alpha > 0 needs the teacher strokes (Phase 3) and I* (Phase 4); only --alpha 0 runs now")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.body import Body
    from fly.decoder import Decoder
    from fly.encoder import encode
    from fly.strokes import PREROLL_MS
    from fly.wiring import wire

    lookahead = args.lookahead_ms if args.lookahead_ms is not None else default_lookahead_ms()
    preroll = args.preroll_ms if args.preroll_ms is not None else PREROLL_MS

    timings, t0 = {}, perf_counter()
    enc = encode(args.groove, lookahead_ms=lookahead, burst_ms=args.burst_ms, preroll_ms=preroll)
    rates = enc.rates
    if args.seconds is not None:
        rates = rates[: int((args.seconds * 1000 + enc.offset_ms) / enc.dt_ms)]
    wiring = wire(shuffle_seed=args.shuffled)
    timings["wiring_s"] = perf_counter() - t0
    brain = wiring.brain(device=device)
    if args.weights is not None:
        from fly.train import load_weights

        load_weights(brain, args.weights, args.shuffled)
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, wiring.conn.size).to(device)
    body = Body()
    timings["build_s"] = perf_counter() - t0 - timings["wiring_s"]
    print(f"{wiring.conn.name}: {len(rates)} steps on {device}, body dt {body.m.opt.timestep:g} s", flush=True)

    t1 = perf_counter()
    raw = simulate(rates, brain, decoder, body, seed=args.seed, device=device, log_every=1000)
    timings["sim_s"] = perf_counter() - t1
    timings["wall_s_per_sim_s"] = timings["sim_s"] / (len(rates) / 1000)
    hits = score_time(raw, enc.offset_ms)

    meta = {
        "groove": str(args.groove.resolve().relative_to(REPO)) if args.groove.resolve().is_relative_to(REPO) else str(args.groove),
        "alpha": args.alpha, "seed": args.seed, "shuffled": args.shuffled, "device": device,
        "weights": None if args.weights is None else str(args.weights),
        "offset_ms": enc.offset_ms, "lookahead_ms": lookahead, "preroll_ms": preroll, "burst_ms": args.burst_ms,
        "steps": len(rates), "seconds": args.seconds,
        "wiring": wiring.summary(), "body": {"timestep": body.m.opt.timestep, "substeps": body.substeps},
        "hits": len(hits), "dropped_before_score_0": len(raw) - len(hits),
        "hits_by_voice": {v: sum(h["voice"] == v for h in hits) for v in sorted({h["voice"] for h in hits})},
        "timings": {k: round(v, 2) for k, v in timings.items()},
    }
    write_run(args.out, hits, meta)
    print(json.dumps({k: meta[k] for k in ("hits", "hits_by_voice", "timings")}, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

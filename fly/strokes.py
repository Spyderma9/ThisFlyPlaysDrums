"""Teacher strokes (Phase 3): the joint targets q*(t) a drummer's teacher asks for, from any training .mid.

Never used on held-out grooves. The fly only ever sees q* through the loss and alpha * I* (Phase 4).

1. Poses (--calibrate -> fly/strokes.json): per strike target (kit pad["targets"]: pad centres, the ride's bow, and the
   cross-stick and bell zones) and leg, damped-least-squares IK on the tip site (stick tip, or
   claw for pedals) over the joints the decoder can drive (body.joint_bounds), so every pose is one the fly could
   produce. soft strike = tip DEPTH_CM into the pad; raise = tip RAISE_CM above it (toward the stick's rest tip if
   straight up is out of reach); hard strike = the swing continued FOLLOW_THROUGH past the soft target. The servos lag
   ~10 ms, so contact speed, and so loudness, follows how far past the surface the target sits. Timing is
   measured on the body: for velocities 1 to 127 (CAL_VELOCITIES), how long after the swing command the contact lands, how fast.
2. Sticking (plan): no drum belongs to one hand. Each target goes to the stick on its side (the right for the crash;
   the more rested stick for one in the middle), or to the other stick when that one is busy and the pad is within
   both sticks' reach, but never a second stick whose calibrated stroke clips other pads. Notes within MIN_GAP_MS on one stick are resolved most
   constrained first, then by PRIORITY; what can't be played is dropped and reported.
3. Stroke (teacher): raise (height and strike depth grow with velocity) for T_RAISE_MS -> swing to the strike pose, timed so contact
   lands on the note -> hold T_HOLD_MS -> the next stroke's raise, or (if it is more than REST_GAP_MS away) back up
   through this stroke's raise to rest.
   Kick: the same on its pedal. Hi-hat pedal: held down (closed, the soft depth); lifted for open hats; closed again
   slowly (silent, body.PEDAL_CHICK_CM_S); a 44 note is a lift then a press to the hard depth.

    python -m fly.strokes --calibrate
    python -m fly.strokes --ceiling [--takes 'grooves/train/*.mid'] [--jobs 16]   # q* straight into the body
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fly.decoder import JOINTS, KIT
from fly.drums import LEGS

STROKES = Path(__file__).with_name("strokes.json")
RAISE_CM = 0.035
DEPTH_CM = 0.004  # the soft (velocity 1) strike target sits this far past the pad surface
FOLLOW_THROUGH = 0.5  # velocity 127 keeps swinging this fraction of the raise->strike move past the soft target
HEIGHTS = (0.6, 1.0)  # raise height as a fraction of RAISE_CM, for velocity 1 .. 127
CAL_VELOCITIES = (1, 32, 64, 96, 127)
MAX_STRAYS = 1  # a second stick whose calibrated stroke clips other pads more than this is never used
MIDDLE_CM = 0.02  # a both-sticks pad within this of the midline goes to the more rested stick
T_RAISE_MS, T_HOLD_MS, REST_GAP_MS, MIN_GAP_MS = 35, 12, 120, 30
PEDAL_LIFT_CM, T_CLOSE_MS, T_OPEN_RING_MS = 0.02, 100, 30
PREROLL_MS = 200.0  # sim time before score time 0: legs settle and the hat closes
CUE_LATENCY_MS = 40.0  # cue -> motor-neuron latency measured in Phase 1
PRIORITY = ("crash", "ride", "ride_bell", "hat_open", "hat_closed", "tom1", "tom2", "tom3", "snare", "xstick")
VOICE_PAD = {"hat_closed": "hat", "hat_open": "hat", "hat_pedal": "hat_pedal"}
HAT_VOICES = ("hat_open", "hat_closed", "hat_pedal")


def pad_of(voice: str) -> str:
    return VOICE_PAD.get(voice, voice)


# ---------- IK and calibration ----------

def ik(m, leg: str, target: np.ndarray, q0: np.ndarray, bounds: np.ndarray, iters: int = 400, lam: float = 1e-5):
    """Leg servo targets [8] putting the leg's tip site at `target` (cm), within `bounds` [8, 2]. -> (q, error cm)."""
    import mujoco

    from fly.body import SUFFIX, _set, rest_pose, tip_site

    d = mujoco.MjData(m)
    rest_pose(m, d)
    site = m.site(tip_site(leg)).id
    dofs = [m.jnt_dofadr[m.joint(f"{j}_{SUFFIX[leg]}").id] for j in JOINTS[:-1]]
    free = (bounds[:-1, 1] > bounds[:-1, 0]).astype(float)
    q = np.clip(np.asarray(q0, dtype=float)[:-1], bounds[:-1, 0], bounds[:-1, 1])
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        for j, joint in enumerate(JOINTS[:-1]):
            _set(m, d, leg, joint, q[j])
        mujoco.mj_fwdPosition(m, d)
        err = target - d.site_xpos[site]
        if np.linalg.norm(err) < 1e-4:
            break
        mujoco.mj_jacSite(m, d, jacp, None, site)
        jac = jacp[:, dofs] * free
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(3), err)
        q = np.clip(q + np.clip(dq, -0.1, 0.1), bounds[:-1, 0], bounds[:-1, 1])
    return np.append(q, 0.0), float(np.linalg.norm(err))


def _leg_targets(body, leg: str, q: np.ndarray) -> np.ndarray:
    k = LEGS.index(leg)
    t = body.rest.copy()
    t[k * len(JOINTS):(k + 1) * len(JOINTS)] = q
    return t


def _run(body, targets, ms: int) -> list:
    return sum((body.step(targets) for _ in range(ms)), [])


def sounds(hit, target: str) -> bool:
    """Does this hit play `target` (a strokes.json pad key)? The hat target is either hat note; zones must match."""
    return hit.pad == "hat" if target == "hat" else hit.voice == target


def _strike_timing(body, target: str, leg: str, q_raise, q_strike, settle_ms: int = 60, max_ms: int = 40) -> dict:
    """The whole stroke from rest: raise, strike, back through the raise to rest. -> ms from the swing command until
    the target sounds, contact speed, and any other hit (another pad or zone, or this one again) as strays."""
    body.reset()
    stray = [h.voice for h in _run(body, _leg_targets(body, leg, q_raise), settle_ms)]
    t0 = body.t_ms
    hits = _run(body, _leg_targets(body, leg, q_strike), max_ms)
    hits += _run(body, _leg_targets(body, leg, q_raise), T_RAISE_MS) + _run(body, body.rest, 60)
    mine = [h for h in hits if sounds(h, target)]
    return {"latency_ms": round(mine[0].t_ms - t0, 2) if mine else None,
            "speed_cm_s": round(mine[0].contact_speed, 3) if mine else None,
            "stray": stray + [h.voice for h in hits if h is not (mine[0] if mine else None)]}


def _pedal_timing(body, q_lift, q_press, q_chick) -> dict:
    """Hi-hat pedal: a hard press from lift (a 44), a slow close to the hold depth (silent), and how long a lift takes
    to open the hat."""
    hp = body.pads.index("hat_pedal")
    fast = _strike_timing(body, "hat_pedal", "hind_left", q_lift, q_chick)
    body.reset()
    _run(body, _leg_targets(body, "hind_left", q_lift), 60)
    t0, closed_at, sounded = body.t_ms, None, []
    for k in range(T_CLOSE_MS + 60):
        a = min(1.0, k / T_CLOSE_MS)
        sounded += body.step(_leg_targets(body, "hind_left", (1 - a) * q_lift + a * q_press))
        if closed_at is None and body.touching[hp]:
            closed_at = body.t_ms - t0
    _run(body, _leg_targets(body, "hind_left", q_press), 40)
    t1, opened_at = body.t_ms, None
    for _ in range(80):
        body.step(_leg_targets(body, "hind_left", q_lift))
        if opened_at is None and not body.touching[hp]:
            opened_at = body.t_ms - t1
    return {**fast, "slow_close_ms": closed_at, "slow_close_sounded": [h.note for h in sounded], "release_ms": opened_at}


def calibrate() -> dict:
    from fly.body import Body, joint_bounds, tip_site

    kit = json.loads(KIT.read_text())
    body = Body(kit)
    signs = {a["name"]: a for a in kit["actuators"]}
    allowed = {tuple(k.split(".")): set(v) for k, v in kit["driven"].items()}
    out = {"about": "Generated by `python -m fly.strokes --calibrate` from fly/kit.json. Poses are 8 servo targets "
                    "(decoder.JOINTS order) for one leg: soft/hard strikes (velocity 1/127) and the full raise. "
                    "timing[v]: swing command -> contact from velocity v's raise. pads[p].legs is in preference order.",
           "raise_cm": RAISE_CM, "depth_cm": DEPTH_CM, "follow_through": FOLLOW_THROUGH, "heights": HEIGHTS, "pads": {}}
    body.reset()
    rest_tip = {leg: body.d.site_xpos[body.m.site(tip_site(leg)).id].copy() for leg in LEGS}
    targets = [(p, name, np.array(point)) for p in kit["pads"] for name, point in p["targets"].items()]
    for p, name, top in targets:  # one entry per strike point: pad centres, the ride's bow, and the zones
        n = np.array(p["normal"])
        y = float(top[1])
        legs = sorted(p["legs"], key=lambda leg: (leg != "front_right") if name == "crash" or y < 0 else leg != "front_left")
        pad = out["pads"][name] = {"either": name != "crash" and len(legs) > 1 and abs(y) < MIDDLE_CM, "legs": {}}
        for leg in legs:
            bounds = joint_bounds(signs, allowed, leg)
            soft, e1 = ik(body.m, leg, top - n * DEPTH_CM, p["reach"][leg], bounds)
            if p["kind"] == "pedal":
                ups = [n * (PEDAL_LIFT_CM if p["name"] == "hat_pedal" else RAISE_CM)]
            else:  # candidate raise directions: straight up, toward the rest tip, between; the cleanest stroke wins
                back = (rest_tip[leg] - top) / np.linalg.norm(rest_tip[leg] - top)
                ups = [RAISE_CM * v / np.linalg.norm(v) for v in (n, back, n + back, 2 * n + back)]
            best = None
            for up in ups:
                raise_, e3 = ik(body.m, leg, top + up, soft, bounds)
                if e3 > 0.3 * np.linalg.norm(up) and best is not None:
                    continue
                hard = np.clip(soft + FOLLOW_THROUGH * (soft - raise_), bounds[:, 0], bounds[:, 1])  # keep swinging past
                entry = {"soft": soft.round(4).tolist(), "hard": hard.round(4).tolist(), "raise": raise_.round(4).tolist(),
                         "ik_error_cm": [round(e1, 5), round(e3, 5)]}
                if name == "hat_pedal":
                    entry["pedal"] = _pedal_timing(body, raise_, soft, hard)
                    best = (0, entry)
                    break
                entry["timing"] = {str(v): _strike_timing(body, name, leg, *stroke_poses(entry, v)) for v in CAL_VELOCITIES}
                entry["strays"] = sum(len(t["stray"]) + 3 * (t["latency_ms"] is None) for t in entry["timing"].values())
                if best is None or entry["strays"] < best[0]:
                    best = (entry["strays"], entry)
                if entry["strays"] == 0:
                    break
            entry = best[1]
            pad["legs"][leg] = entry
            t = entry.get("timing") or {"44": entry["pedal"]}
            print(f"  {name:10s} {leg:11s} ik {entry['ik_error_cm']} cm  " + "  ".join(
                f"v{v}: {x['latency_ms']} ms {x['speed_cm_s']} cm/s{' stray ' + ','.join(x['stray']) if x['stray'] else ''}"
                for v, x in t.items()), flush=True)
        if len(pad["legs"]) > 1:  # a stick that clips other pads plays it only when the cleaner one is busy
            order = sorted(pad["legs"], key=lambda leg: pad["legs"][leg]["strays"])
            if pad["legs"][order[0]]["strays"] < pad["legs"][order[1]]["strays"]:
                pad["legs"] = {leg: pad["legs"][leg] for leg in order}
                pad["either"] = False
    lat = [t["latency_ms"] for pad in out["pads"].values() for e in pad["legs"].values()
           for t in e.get("timing", {}).values() if t["latency_ms"] is not None]
    out["stroke_lead_ms"] = round(T_RAISE_MS + float(np.max(lat)), 1)
    out["lookahead_ms"] = round(CUE_LATENCY_MS + out["stroke_lead_ms"], 1)
    return out


# ---------- sticking ----------

@dataclass
class Note:
    t_ms: float  # sim time the note is due
    voice: str
    velocity: int
    leg: str | None = None
    dropped: str | None = None


def plan(onsets: dict[str, list[tuple[float, int]]], offset_ms: float, pads: dict) -> list[Note]:
    """Assign each note a leg. pads: strokes.json "pads" ({pad: {"legs": {leg: ...} in preference order, "either"}})."""
    notes = [Note(t * 1000 + offset_ms, v, vel) for v, hits in onsets.items() for t, vel in hits]
    notes.sort(key=lambda n: n.t_ms)
    for n in notes:
        if n.voice == "kick":
            n.leg = "hind_right"
        elif n.voice == "hat_pedal":
            n.leg = "hind_left"
    busy: dict[str, list[float]] = {"front_left": [], "front_right": []}
    stick = [n for n in notes if n.leg is None]
    i = 0
    while i < len(stick):  # clusters of notes closer than MIN_GAP_MS
        j = i + 1
        while j < len(stick) and stick[j].t_ms - stick[j - 1].t_ms < MIN_GAP_MS:
            j += 1
        group = sorted(stick[i:j], key=lambda n: (len(pads[pad_of(n.voice)]["legs"]), PRIORITY.index(n.voice), n.t_ms))
        for n in group:
            def gap(leg):
                return min((abs(n.t_ms - t) for t in busy[leg]), default=np.inf)
            pad = pads[pad_of(n.voice)]
            # preference order; never a stroke that clips other pads, unless every stick's does: then the cleanest
            clean = [leg for leg, e in pad["legs"].items() if e.get("strays", 0) <= MAX_STRAYS]
            clean = clean or [min(pad["legs"], key=lambda leg: pad["legs"][leg].get("strays", 0))]
            free = [leg for leg in clean if gap(leg) >= MIN_GAP_MS]
            if not free:
                n.dropped = "stick busy"
                continue
            if pad["either"] and len(free) > 1:
                free.sort(key=lambda leg: (-min(gap(leg), 1e6), leg != "front_right"))  # the more rested; ties go right
            n.leg = free[0]
            busy[n.leg].append(n.t_ms)
        i = j
    return notes


# ---------- q*(t) ----------

def stroke_poses(entry: dict, velocity: int) -> tuple[np.ndarray, np.ndarray]:
    """(raise, strike) servo targets [8] for a note of this velocity."""
    a = (velocity - 1) / 126
    soft, hard, up = (np.array(entry[k]) for k in ("soft", "hard", "raise"))
    h = HEIGHTS[0] + (HEIGHTS[1] - HEIGHTS[0]) * a
    return soft + h * (up - soft), soft + a * (hard - soft)


def _latency(entry: dict, velocity: int) -> float:
    ok = [(float(v), t["latency_ms"]) for v, t in entry["timing"].items() if t["latency_ms"] is not None]
    return float(np.interp(velocity, [a for a, _ in ok], [b for _, b in ok])) if ok else 15.0


def teacher(notes: list[Note], n_steps: int, strokes: dict, rest: np.ndarray) -> np.ndarray:
    """q* [n_steps, 32] servo targets (drums.LEGS x decoder.JOINTS), sim ms steps."""
    nj = len(JOINTS)
    q = np.tile(rest.astype(np.float32), (n_steps, 1))
    pads = strokes["pads"]

    def paint(leg, a, b, pose, pose_b=None):
        a, b = max(0, int(round(a))), min(n_steps, int(round(b)))
        if b <= a:
            return
        k = LEGS.index(leg) * nj
        if pose_b is None:
            q[a:b, k:k + nj] = pose
        else:
            w = np.linspace(0, 1, b - a, endpoint=False)[:, None]
            q[a:b, k:k + nj] = (1 - w) * np.asarray(pose) + w * np.asarray(pose_b)

    for leg in ("front_left", "front_right", "hind_right"):
        mine = [n for n in notes if n.leg == leg and not n.dropped]
        prev_end, prev_up = -np.inf, None
        for n in mine:
            e = pads[pad_of(n.voice)]["legs"][leg]
            up, strike = stroke_poses(e, n.velocity)
            swing = n.t_ms - _latency(e, n.velocity)
            lift = swing - T_RAISE_MS
            start = prev_end if lift - prev_end <= REST_GAP_MS else lift
            if start == prev_end and prev_up is not None:  # rise over the last pad first, then travel at height
                mid = prev_end + min(T_RAISE_MS, (swing - prev_end) / 2)
                paint(leg, prev_end, mid, prev_up)
                start = mid
            paint(leg, max(start, prev_end), swing, up)
            prev_up = up
            paint(leg, max(swing, prev_end), n.t_ms + T_HOLD_MS, strike)
            prev_end = n.t_ms + T_HOLD_MS
            paint(leg, prev_end, prev_end + T_RAISE_MS, up)  # back out the way it came; a next stroke paints over it

    e = pads["hat_pedal"]["legs"]["hind_left"]
    press, chick, lift, timing = np.array(e["soft"]), np.array(e["hard"]), np.array(e["raise"]), e["pedal"]
    fast = timing["latency_ms"] or 15.0
    release = timing["release_ms"] or 15.0
    k = LEGS.index("hind_left") * nj
    q[:, k:k + nj] = press  # held closed by default
    paint("hind_left", 0, 40, lift)
    paint("hind_left", 40, 40 + T_CLOSE_MS, lift, press)  # the silent first close, in the pre-roll
    hat = [n for n in notes if n.voice in HAT_VOICES and not n.dropped]
    for i, n in enumerate(hat):
        if n.voice == "hat_open":
            nxt = next((m.t_ms for m in hat[i + 1:] if m.voice != "hat_open"), n_steps + T_CLOSE_MS)
            if i and hat[i - 1].voice == "hat_open":
                a = hat[i - 1].t_ms
            else:
                a = n.t_ms - release - 10
            close = max(n.t_ms + T_OPEN_RING_MS, nxt - T_CLOSE_MS - 20)
            ramp = max(15.0, min(T_CLOSE_MS, nxt - 10 - close))
            paint("hind_left", a, close, lift)
            paint("hind_left", close, close + ramp, lift, press)
        elif n.voice == "hat_pedal":
            down = n.t_ms - fast
            paint("hind_left", down - release - 15, down, lift)
            paint("hind_left", down, n.t_ms + T_HOLD_MS, chick)
    return q


# ---------- ceiling check ----------

def ceiling_run(take: str, out_root: str, seconds: float | None = None) -> dict:
    from fly.body import Body
    from fly.encoder import encode, read_onsets
    from fly.loop import score_time, write_run

    strokes = json.loads(STROKES.read_text())
    enc = encode(take, lookahead_ms=strokes["lookahead_ms"], preroll_ms=PREROLL_MS)
    onsets = read_onsets(take)
    n_steps = len(enc.rates) if seconds is None else int(seconds * 1000 + enc.offset_ms)
    body = Body()
    notes = plan(onsets, enc.offset_ms, strokes["pads"])
    q = teacher(notes, n_steps, strokes, body.rest)
    raw = []
    for t in range(n_steps):
        raw += body.step(q[t])
    hits = score_time(raw, enc.offset_ms)
    out = Path(out_root) / Path(take).stem
    dropped = [{"t_s": round((n.t_ms - enc.offset_ms) / 1000, 4), "voice": n.voice, "why": n.dropped} for n in notes if n.dropped]
    write_run(out, hits, {"take": take, "offset_ms": enc.offset_ms, "steps": n_steps, "notes": len(notes),
                          "dropped": dropped, "hits": len(hits)})
    return {"take": take, "out": str(out), "notes": len(notes), "dropped": len(dropped), "hits": len(hits)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--ceiling", action="store_true")
    ap.add_argument("--takes", default="grooves/train/*.mid")
    ap.add_argument("--out", default="runs/ceiling")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    if args.calibrate:
        res = calibrate()
        STROKES.write_text(json.dumps(res, indent=1))
        print(f"stroke lead {res['stroke_lead_ms']} ms -> lookahead {res['lookahead_ms']} ms; wrote {STROKES}")
    if args.ceiling:
        from glob import glob
        from multiprocessing import Pool

        takes = sorted(glob(args.takes))
        with Pool(args.jobs) as pool:
            rows = pool.starmap(ceiling_run, [(t, args.out, args.seconds) for t in takes])
        for r in rows:
            print(f"  {Path(r['take']).name:48s} notes {r['notes']:4d} dropped {r['dropped']:3d} hits {r['hits']:4d}")
        cmd = [sys.executable, "human/score.py"]
        for r in rows:  # one comparison per take (each has its own reference)
            subprocess.run(cmd + [r["take"], str(Path(r["out"]) / "hits.csv"), "--csv", str(Path(r["out"]) / "score.csv")],
                           check=False, stdout=subprocess.DEVNULL)
        print(f"wrote {args.out}/<take>/ (hits files, meta.json, score.csv)")


if __name__ == "__main__":
    main()

"""The drum kit's look: real drums drawn around fly.body's physics pads, for fly.clip (MuJoCo) and viewer/ (Three.js).

Visual only. The physics kit is flat boxes (fly/kit.json) and hits come only from contact with them. Each pad becomes a
real drum whose struck surface *is* the pad's top face (part "strike": same centre, same top, round and inside the
square), so a stick that looks like it hits the drum is touching the pad. Only cymbal bells rise above it (0.0012 cm).
Shells, stands, the bass drum and the floor hang below or behind, clear of the fly.

Primitives follow MuJoCo's conventions, so fly.clip hands them to mjv_initGeom as they are; viewer/ converts them:
  type  cylinder (size: radius, half-height), ellipsoid (radii), box (half-extents), capsule (radius, half-length);
        the local axis is z
  pos   centre, cm, world frame     quat  w x y z     rgba     shine  0 matte .. 1 polished metal
  anim  what frame_state changes: "cymbal" (tilts about `pivot` on a hit), "hat_bottom" (drops when the hi-hat opens),
        "beater" (swings about `pivot` while the kick pedal is down), "" (static). flash: lights up on the pad's hits.
frame_state(scene, t_ms, hits, touching) gives every primitive's pose and colour at a sim time. viewer/main.js
implements the same rules from the constants in scene["anim"].
"""

from __future__ import annotations

import numpy as np

SHAPES = ("cylinder", "ellipsoid", "box", "capsule")

# colours
HEAD = (0.93, 0.91, 0.84, 1.0)
SHELL = (0.52, 0.04, 0.08, 1.0)  # one kit, deep red lacquer
CHROME = (0.80, 0.81, 0.84, 1.0)
BRONZE = (0.80, 0.60, 0.24, 1.0)
STAND = (0.08, 0.08, 0.09, 1.0)
PEDAL = (0.30, 0.31, 0.33, 1.0)
FELT = (0.88, 0.87, 0.82, 1.0)
FLOOR = (0.11, 0.09, 0.10, 1.0)
VOICE_COLORS = {"snare": "#e63946", "hat": "#f4c430", "crash": "#f77f00", "tom1": "#4361ee", "tom2": "#7b2cbf",
                "tom3": "#2a9d8f", "ride": "#52b788", "kick": "#6ee7f2", "hat_pedal": "#f4c430"}

# sizes, cm (pads are 0.03-0.044 wide; the fly is ~0.25 long)
SKIN = 0.0004  # half-thickness of a drum head
RIM_R, RIM_HALF, RIM_RISE = 1.06, 0.0010, -0.0002  # rim radius / head radius, half-height, top vs the head's top
# (MuJoCo cylinders are solid, so a rim above the head would cover it: it sits just below, showing only its ring)
SHELL_DEPTH = {"snare": 0.7, "tom1": 1.0, "tom2": 1.15, "tom3": 1.3}  # x the head radius
LUGS, LUG = 6, (0.0012, 0.0012, 0.0030)
CYMBAL_HALF = 0.0005
BELL_R, BELL_RISE = 0.22, 0.0012  # bell radius / cymbal radius; how far it rises above the struck face
HAT_CLOSED, HAT_OPEN = 0.0003, 0.006  # gap between the hi-hat's cymbals, closed and fully open
STAND_R, LEG_SPREAD = 0.0012, 0.018
FLOOR_DROP = 0.09  # the floor sits this far below the lowest pad top
FOOT_BOARD = (1.0, 0.6, 0.0008)  # pedal footboard half-extents: x, y as fractions of the pedal pad's half, z in cm
BEATER_LEN, BEATER_R, BALL_R = 0.028, 0.0008, 0.0035
BEATER_REST_DEG, BEATER_HIT_DEG = -15.0, 40.0  # from vertical, + toward the bass drum (-x)
BASS_R, BASS_DEPTH = 0.042, 0.045

ANIM = {"flash_ms": 80.0, "wobble_deg": 6.0, "wobble_tau_ms": 300.0, "wobble_period_ms": 150.0,
        "hat_ease_ms": 20.0, "pedal_ease_ms": 12.0, "hat_open": HAT_OPEN,
        "beater_rest_deg": BEATER_REST_DEG, "beater_hit_deg": BEATER_HIT_DEG}


# ---------- small quaternion helpers (w, x, y, z) ----------

def quat_axis_angle(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle / 2), *(np.sin(angle / 2) * axis)])


def quat_mul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rotate(q, v) -> np.ndarray:
    """Rotate vector v by unit quaternion q."""
    w, u = q[0], np.asarray(q[1:], dtype=float)
    v = np.asarray(v, dtype=float)
    return v + 2 * np.cross(u, np.cross(u, v) + w * v)


def quat_z_to(d) -> np.ndarray:
    """The rotation taking +z to direction d."""
    d = np.asarray(d, dtype=float)
    d = d / np.linalg.norm(d)
    c = float(d[2])
    if c < -1 + 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    axis = np.cross([0.0, 0.0, 1.0], d)
    if np.linalg.norm(axis) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat_axis_angle(axis, float(np.arccos(np.clip(c, -1, 1))))


UP = np.array([1.0, 0.0, 0.0, 0.0])


# ---------- building the kit ----------

def _hex(c: str, alpha: float = 1.0) -> list[float]:
    return [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [alpha]


class _Kit:
    def __init__(self):
        self.prims: list[dict] = []

    def add(self, pad, part, type_, size, pos, quat=UP, rgba=STAND, shine=0.2, anim="", flash=False, **extra):
        self.prims.append({"pad": pad, "part": part, "type": type_, "size": [float(s) for s in size],
                           "pos": [float(x) for x in pos], "quat": [float(x) for x in quat],
                           "rgba": [float(x) for x in rgba], "shine": shine, "anim": anim, "flash": flash, **extra})

    def rod(self, pad, part, a, b, r=STAND_R, rgba=STAND, shine=0.3, **kw):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        self.add(pad, part, "capsule", (r, np.linalg.norm(b - a) / 2, 0), (a + b) / 2, quat_z_to(b - a), rgba, shine, **kw)

    def stand(self, pad, top, floor_z):
        """A vertical stand from `top` down to the floor, on a tripod."""
        top = np.asarray(top, dtype=float)
        foot = np.array([top[0], top[1], floor_z])
        self.rod(pad, "stand", top, foot)
        knee = foot + np.array([0, 0, min(0.3 * (top[2] - floor_z), 0.02)])
        for k in range(3):
            a = 2 * np.pi * k / 3 + np.pi / 6
            self.rod(pad, "stand", knee, foot + LEG_SPREAD * np.array([np.cos(a), np.sin(a), 0]), r=STAND_R * 0.8)


def _drum(kit: _Kit, name: str, xy, top: float, r: float, floor_z: float) -> None:
    depth = SHELL_DEPTH[name] * r
    kit.add(name, "strike", "cylinder", (r, SKIN, 0), (*xy, top - SKIN), rgba=HEAD, shine=0.1, flash=True)
    kit.add(name, "rim", "cylinder", (RIM_R * r, RIM_HALF, 0), (*xy, top + RIM_RISE - RIM_HALF), rgba=CHROME, shine=0.9)
    shell_mid = top - 2 * SKIN - depth / 2
    kit.add(name, "shell", "cylinder", (0.985 * r, depth / 2, 0), (*xy, shell_mid), rgba=SHELL, shine=0.6)
    kit.add(name, "bottom_rim", "cylinder", (RIM_R * r, RIM_HALF, 0), (*xy, top - 2 * SKIN - depth), rgba=CHROME, shine=0.9)
    for k in range(LUGS):
        a = 2 * np.pi * k / LUGS
        kit.add(name, "lug", "ellipsoid", LUG, (xy[0] + r * np.cos(a), xy[1] + r * np.sin(a), shell_mid),
                rgba=CHROME, shine=0.9)
    kit.stand(name, (*xy, top - 2 * SKIN - depth - RIM_HALF), floor_z)


def _cymbal(kit: _Kit, name: str, xy, top: float, r: float, anim: str, floor_z: float | None, tilt_axis) -> None:
    pivot = [float(xy[0]), float(xy[1]), float(top)]
    kw = {"anim": anim, "pivot": pivot, "axis": [float(a) for a in tilt_axis]} if anim else {}
    kit.add(name, "strike", "cylinder", (r, CYMBAL_HALF, 0), (*xy, top - CYMBAL_HALF), rgba=BRONZE, shine=0.8,
            flash=True, **kw)
    kit.add(name, "bell", "ellipsoid", (BELL_R * r, BELL_R * r, BELL_RISE), (*xy, top), rgba=BRONZE, shine=0.8, **kw)
    if floor_z is not None:
        kit.stand(name, (*xy, top - 2 * CYMBAL_HALF), floor_z)


def _tilt_axis(xy) -> np.ndarray:
    """Horizontal axis a struck cymbal rocks about: across the line from the fly (the origin) to the cymbal."""
    d = np.asarray(xy, dtype=float)
    d = d / max(float(np.linalg.norm(d)), 1e-9)
    return np.array([-d[1], d[0], 0.0])


def _pedal(kit: _Kit, name: str, pad: dict, floor_z: float) -> float:
    """Footboard (its top = the pedal pad's top) on a base plate, raised on a stand. Returns the footboard's top z."""
    (x, y, z), h = pad["pos"], pad["half"]
    top = z + h[2]
    fb = (FOOT_BOARD[0] * h[0], FOOT_BOARD[1] * h[1], FOOT_BOARD[2])
    kit.add(name, "strike", "box", fb, (x, y, top - fb[2]), rgba=PEDAL, shine=0.7, flash=True)
    base = top - 2 * fb[2] - 0.004
    kit.add(name, "pedal_base", "box", (1.1 * h[0], 0.8 * h[1], 0.0012), (x, y, base - 0.0012), rgba=PEDAL, shine=0.5)
    kit.rod(name, "pedal_frame", (x - h[0], y, base), (x - h[0], y, top), r=0.0010, rgba=PEDAL, shine=0.7)
    kit.stand(name, (x, y, base - 0.0024), floor_z)
    return top


def kit_scene(kit: dict) -> dict:
    """Every primitive of the kit's look, in world coordinates, plus the animation constants (see module doc)."""
    pads = {p["name"]: p for p in kit["pads"]}
    tops = {n: p["pos"][2] + p["half"][2] for n, p in pads.items()}
    out = _Kit()

    # kick: pedal at the pad, beater on an axle at its back edge, the bass drum behind (-x), sitting on the floor
    kick = pads["kick"]
    (kx, ky, _), kh = kick["pos"], kick["half"]
    pivot = np.array([kx - kh[0] - 0.003, ky, tops["kick"] + 0.010])
    hit = np.radians(BEATER_HIT_DEG)
    head_x = pivot[0] - BEATER_LEN * np.sin(hit) - BALL_R
    bass_z = pivot[2] + BEATER_LEN * np.cos(hit)
    floor_z = min(min(tops.values()) - FLOOR_DROP, bass_z - BASS_R - 0.004)

    out.add("", "floor", "box", (0.6, 0.6, 0.002), (0.0, 0.0, floor_z - 0.002), rgba=FLOOR, shine=0.0)
    for name in SHELL_DEPTH:
        p = pads[name]
        _drum(out, name, p["pos"][:2], tops[name], p["half"][0], floor_z)
    for name in ("crash", "ride"):
        p = pads[name]
        _cymbal(out, name, p["pos"][:2], tops[name], p["half"][0], "cymbal", floor_z, _tilt_axis(p["pos"][:2]))

    hat = pads["hat"]
    hxy, hr, htop = hat["pos"][:2], hat["half"][0], tops["hat"]
    _cymbal(out, "hat", hxy, htop, hr, "", None, (1, 0, 0))  # the top cymbal is the struck face; it doesn't move
    closed_z = htop - 3 * CYMBAL_HALF - HAT_CLOSED
    out.add("hat", "hat_bottom", "cylinder", (hr, CYMBAL_HALF, 0), (*hxy, closed_z), rgba=BRONZE, shine=0.8,
            anim="hat_bottom")
    rod_top = closed_z - CYMBAL_HALF - HAT_OPEN
    out.stand("hat", (*hxy, rod_top), floor_z)

    _pedal(out, "hat_pedal", pads["hat_pedal"], floor_z)
    # a remote hi-hat: the pedal is on the far side of the fly, so a cable runs along the floor to the stand
    hp = pads["hat_pedal"]["pos"]
    cable = [(hp[0] - pads["hat_pedal"]["half"][0], hp[1], floor_z + 0.0015), (hxy[0], hxy[1], floor_z + 0.0015)]
    out.rod("hat_pedal", "cable", *cable, r=0.0008)

    _pedal(out, "kick", kick, floor_z)
    out.rod("kick", "pedal_frame", (pivot[0], ky - 0.6 * kh[1], floor_z), pivot + [0, -0.6 * kh[1], 0], r=0.0010,
            rgba=PEDAL, shine=0.7)
    rest = np.radians(BEATER_REST_DEG)
    d = np.array([-np.sin(rest), 0.0, np.cos(rest)])
    kw = {"anim": "beater", "pivot": [float(v) for v in pivot], "axis": [0.0, -1.0, 0.0]}
    q = quat_axis_angle([0, -1, 0], rest)
    out.add("kick", "beater", "capsule", (BEATER_R, BEATER_LEN / 2, 0), pivot + d * BEATER_LEN / 2, q, CHROME, 0.9, **kw)
    out.add("kick", "beater_ball", "ellipsoid", (BALL_R, BALL_R, BALL_R), pivot + d * BEATER_LEN, q, FELT, 0.1, **kw)

    side = quat_z_to([1, 0, 0])  # a bass drum lies on its side, heads facing the pedal (+x) and back
    bass_c = np.array([head_x - BASS_DEPTH / 2, ky, bass_z])
    out.add("kick", "bass_drum", "cylinder", (BASS_R, BASS_DEPTH / 2, 0), bass_c, side, SHELL, 0.6)
    out.add("kick", "bass_head", "cylinder", (0.97 * BASS_R, SKIN, 0), (head_x + SKIN, ky, bass_z), side, HEAD, 0.1,
            flash=True)
    for x in (head_x, head_x - BASS_DEPTH):
        out.add("kick", "bass_hoop", "cylinder", (1.04 * BASS_R, 0.0018, 0), (x, ky, bass_z), side, CHROME, 0.9)
    for s in (-1, 1):  # spurs
        a = bass_c + np.array([0.3 * BASS_DEPTH, s * 0.8 * BASS_R, -0.3 * BASS_R])
        out.rod("kick", "spur", a, a + np.array([0.004, s * 0.012, floor_z - a[2]]), r=0.0010, rgba=CHROME, shine=0.9)

    return {"prims": out.prims, "pads": [p["name"] for p in kit["pads"]], "floor_z": float(floor_z),
            "voice_colors": VOICE_COLORS, "anim": dict(ANIM)}


# ---------- animation ----------

def _eased(touching: np.ndarray, t: int, col: int, ease_ms: float) -> float:
    """Fraction of the last `ease_ms` sim ms the pad was touched (0..1): a pedal moves, it doesn't teleport."""
    if len(touching) == 0:
        return 0.0
    t = int(np.clip(t, 0, len(touching) - 1))
    w = max(1, int(round(ease_ms)))
    return float(touching[max(0, t - w + 1):t + 1, col].mean())


def frame_state(scene: dict, t_ms: float, hits: list[tuple[float, str, int]], touching: np.ndarray) -> dict:
    """Pose, colour and glow (0..1, light it gives off; a bronze cymbal flashing yellow needs it to show) of every
    primitive at sim time t_ms. hits: (sim t_ms, pad, velocity), any order. touching: [T, pads] per sim ms."""
    a = scene["anim"]
    prims = scene["prims"]
    rgba = np.array([p["rgba"] for p in prims], dtype=float)
    pos = np.array([p["pos"] for p in prims], dtype=float)
    quat = np.array([p["quat"] for p in prims], dtype=float)
    glow = np.zeros(len(prims))
    last: dict[str, tuple[float, int]] = {}
    for t, pad, vel in hits:
        if t <= t_ms and (pad not in last or t >= last[pad][0]):
            last[pad] = (t, vel)
    col = {n: i for i, n in enumerate(scene["pads"])}
    t_row = int(t_ms)
    hat_open = 1.0 - _eased(touching, t_row, col["hat_pedal"], a["hat_ease_ms"])
    kick_down = _eased(touching, t_row, col["kick"], a["pedal_ease_ms"])
    for i, p in enumerate(prims):
        hit = last.get(p["pad"])
        dt = t_ms - hit[0] if hit else np.inf
        if p["flash"] and dt < a["flash_ms"]:
            amount = (hit[1] / 127) ** 0.7 * (1 - dt / a["flash_ms"])
            target = np.array(_hex(scene["voice_colors"][p["pad"]]))
            target[:3] = 0.65 * target[:3] + 0.35
            rgba[i] = rgba[i] + amount * (target - rgba[i])
            glow[i] = amount
        angle = 0.0
        if p["anim"] == "cymbal" and np.isfinite(dt):
            angle = np.radians(a["wobble_deg"]) * hit[1] / 127 * np.exp(-dt / a["wobble_tau_ms"]) \
                * np.sin(2 * np.pi * dt / a["wobble_period_ms"])
        elif p["anim"] == "beater":
            angle = np.radians(a["beater_hit_deg"] - a["beater_rest_deg"]) * kick_down
        elif p["anim"] == "hat_bottom":
            pos[i, 2] -= a["hat_open"] * hat_open
        if angle:
            r = quat_axis_angle(p["axis"], angle)
            pivot = np.asarray(p["pivot"])
            pos[i] = pivot + rotate(r, pos[i] - pivot)
            quat[i] = quat_mul(r, quat[i])
    return {"rgba": rgba, "pos": pos, "quat": quat, "glow": glow}

"""MuJoCo fly body (flybody) pinned at the thorax, holding two drumsticks, at a 12-drum kit.

A hit only counts on contact between a pad and the fly (a stick or a playing leg). Contact speed sets MIDI velocity.

- Pinned: flybody's free joint is deleted, so the thorax is welded to the world. Units are cm, gravity -981.
- Rest: the playing legs start in READY (a standing pose with the front femurs raised); everything else, wings
  and middle legs included, holds flybody's springref (its folded flight posture).
- Sticks: a capsule gripped at the end of each front leg's first tarsal segment, mounted so that in READY it points
  along STICK_DIR (forward, a little down and in). The tarsal segments past the grip lose their collisions.
- Kit (fly/kit.json, from --build-kit): 9 stick pads (the hat pad sounds 42 or 46) and 2 pedals
  (hind right = kick, hind left = hi-hat). Only playing-leg and stick geoms collide with the kit.
- Control: Body.step(targets[32]) sets the 32 leg servos (drums.LEGS x decoder.JOINTS) and runs 1 ms of physics.
  Every other actuator holds flybody's rest pose (joint springref).
- Hits: a pad's first contact after DEBOUNCE_MS untouched records a hit, then the pad is refractory for REFRACTORY_MS. The hat pad
  sounds 42 while the hind-left leg is on the hat pedal, else 46. Pedal presses send 36 (kick) and 44 (hat), but a
  hat-pedal press slower than PEDAL_CHICK_CM_S is silent: it only closes the hat.

    python -m fly.body --build-kit   # measure joint signs, place the kit, check reach; writes fly/kit.json
    python -m fly.body --bench       # scripted strokes at 1e-4 vs 2e-4 s: hits, drift and speed
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import flybody
import mujoco
import numpy as np

from fly.decoder import JOINTS, KIT
from fly.drums import LEGS, VOICES

FRUITFLY_XML = Path(flybody.__file__).parent / "fruitfly" / "assets" / "fruitfly.xml"
SUFFIX = {"front_left": "T1_left", "front_right": "T1_right", "hind_left": "T3_left", "hind_right": "T3_right"}
ACTUATORS = tuple(f"{j}_{SUFFIX[leg]}" for leg in LEGS for j in JOINTS)
SEGMENTS = ("coxa", "femur", "tibia", "tarsus", "tarsus2", "tarsus3", "tarsus4", "claw")
STICK_LEGS = ("front_left", "front_right")
STICK_LENGTH, STICK_RADIUS, STICK_MASS = 0.1, 0.003, 5e-7  # cm, cm, g (the fly is ~0.25 cm long)
STICK_DIR = {"front_left": (1.0, -0.3, -0.35), "front_right": (1.0, 0.3, -0.35)}  # world direction in READY
READY = {"front": {"femur": 0.5}, "hind": {}}  # playing-leg joint angles at rest (rad); unlisted joints are 0
PAD_HALF = (0.015, 0.015, 0.002)
PEDAL_HALF = (0.02, 0.02, 0.002)
PAD_GAP = 0.015  # cm between pad edges
DROP_MIN, DROP_MAX = 0.015, 0.045  # cm a pad sits below the rest tip of the stick (or foot) that plays it
CLEARANCE = 0.005  # cm between a pad and the fly at rest
KIT_BIT = 2  # collision bit shared by the kit and the playing legs
REFRACTORY_MS = 30.0
DEBOUNCE_MS = 10.0  # a pad must have been untouched this long for a new contact to count (no chatter re-triggers)
PEDAL_CHICK_CM_S = 0.8  # a hat-pedal press this fast or faster sends 44; a slower one just closes the hat
VEL_PER_CM_S = 45.0  # MIDI velocity per cm/s of normal contact speed: the teacher strokes span ~0.5-2.8 cm/s
# stick pads in placement order (most played first), with the legs that play each; then pedals
STICK_PADS = {"snare": ("front_left",), "hat": ("front_right",), "crash": STICK_LEGS, "tom1": STICK_LEGS,
              "tom2": STICK_LEGS, "tom3": STICK_LEGS, "ride": ("front_right",), "ride_bell": ("front_right",),
              "xstick": ("front_left",)}  # crash: either stick, so it can sound with the hat or ride
PEDALS = {"kick": "hind_right", "hat_pedal": "hind_left"}
VOICE = {v.name: v for v in VOICES}
# anatomical "+" of each joint (decoder.MN_JOINT's direction) as a geometric test on the leg tip
ANATOMY = {"coxa_abduct": "adduct", "coxa_twist": "forward", "coxa": "forward", "femur_twist": "backward",
           "femur": "up", "tibia": "flex", "tarsus": "flex", "tarsus2": "flex"}  # femur: Tr flexion levates the leg
FLEX_PARENT = {"femur": "coxa", "tibia": "femur", "tarsus": "tibia", "tarsus2": "tarsus"}


@dataclass
class Hit:
    t_ms: float  # sim time
    pad: str
    voice: str
    note: int
    velocity: int
    contact_speed: float  # cm/s, normal to the pad
    limb: str


def pad_voice(pad: str, hat_closed: bool) -> str:
    return ("hat_closed" if hat_closed else "hat_open") if pad == "hat" else pad


# ---------- model ----------

def build_spec(kit: dict | None = None) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(FRUITFLY_XML))
    spec.delete(spec.joint("free"))
    if kit is not None:
        spec.option.timestep = kit["timestep"]
    m0 = spec.compile()  # the tarsus orientation in READY sets each stick's mount
    d0 = mujoco.MjData(m0)
    rest_pose(m0, d0)
    # Welded to the world, the thorax loses MuJoCo's parent-child filter; restore it. The head/proboscis contacts
    # (present in flybody too) only cost solver time.
    for b in range(1, m0.nbody):
        if m0.body_parentid[b] == m0.body("thorax").id:
            spec.add_exclude(bodyname1="thorax", bodyname2=m0.body(b).name)
    for pair in (("head", "haustellum"), ("head", "labrum_left"), ("head", "labrum_right"), ("labrum_left", "labrum_right")):
        spec.add_exclude(bodyname1=pair[0], bodyname2=pair[1])
    # Edit existing geoms before adding any: spec.geom(name) can return the wrong element after an add.
    for leg in LEGS:
        for k, seg in enumerate(SEGMENTS):
            g = spec.geom(f"tarsal_claw_{SUFFIX[leg]}_collision" if seg == "claw" else f"{seg}_{SUFFIX[leg]}_collision")
            if leg in STICK_LEGS and k >= 4:  # the stick covers the distal tarsus
                g.contype = g.conaffinity = 0
            elif g.contype:
                g.contype |= KIT_BIT
                g.conaffinity |= KIT_BIT
    for leg in STICK_LEGS:
        s = SUFFIX[leg]
        tarsus = spec.body(f"tarsus_{s}")
        grip = np.array(spec.body(f"tarsus2_{s}").pos)
        world = np.array(STICK_DIR[leg]) / np.linalg.norm(STICK_DIR[leg])
        tip = grip + d0.xmat[m0.body(f"tarsus_{s}").id].reshape(3, 3).T @ world * STICK_LENGTH
        tarsus.add_geom(name=f"stick_{leg}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[STICK_RADIUS, 0, 0],
                        fromto=[*grip, *tip], mass=STICK_MASS, contype=1 | KIT_BIT, conaffinity=1 | KIT_BIT,
                        condim=1, group=1, rgba=[0.78, 0.6, 0.38, 1])
        tarsus.add_site(name=f"stick_tip_{leg}", pos=tip, size=[0.004, 0, 0])
        tarsus.add_site(name=f"stick_grip_{leg}", pos=grip, size=[0.002, 0, 0])
    for p in (kit or {}).get("pads", ()):
        quat = np.zeros(4)
        mujoco.mju_quatZ2Vec(quat, np.array(p["normal"], dtype=float))
        spec.worldbody.add_geom(name=f"pad_{p['name']}", type=mujoco.mjtGeom.mjGEOM_BOX, size=p["half"], pos=p["pos"],
                                quat=quat, contype=KIT_BIT, conaffinity=KIT_BIT, condim=1, group=1,
                                rgba=[0.2, 0.2, 0.22, 1] if p["kind"] == "pedal" else [0.85, 0.85, 0.8, 1])
    return spec


def rest_pose(m: mujoco.MjModel, d: mujoco.MjData) -> np.ndarray:
    """Playing legs in READY, every other joint at its springref. Returns the ctrl that holds it."""
    mujoco.mj_resetData(m, d)
    d.qpos[:] = m.qpos_spring
    for leg in LEGS:
        ready = READY[leg.split("_")[0]]
        for joint in JOINTS[:-1]:
            _set(m, d, leg, joint, ready.get(joint, 0.0))
        _set(m, d, leg, "tarsus2", 0.0)
    mujoco.mj_forward(m, d)
    ctrl = np.zeros(m.nu)
    for i in range(m.nu):
        if m.actuator_biastype[i] == mujoco.mjtBias.mjBIAS_AFFINE:  # position servo: ctrl is the target
            ctrl[i] = d.actuator_length[i]
    return np.clip(ctrl, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])


def _tarsal_chain(m: mujoco.MjModel, leg: str) -> list[tuple[int, float]]:
    t = m.tendon(f"tarsus2_{SUFFIX[leg]}").id
    adr, num = m.tendon_adr[t], m.tendon_num[t]
    return [(int(m.wrap_objid[k]), float(m.wrap_prm[k])) for k in range(adr, adr + num)]


def _set(m, d, leg: str, joint: str, value: float) -> None:
    """Set a joint (or the tarsus2 tendon, spread over its joints by coefficient) in d.qpos."""
    if joint == "tarsus2":
        chain = _tarsal_chain(m, leg)
        norm = sum(c * c for _, c in chain)
        for j, c in chain:
            d.qpos[m.jnt_qposadr[j]] = value * c / norm
    else:
        d.qpos[m.jnt_qposadr[m.joint(f"{joint}_{SUFFIX[leg]}").id]] = value


def _tip(m, d, leg: str) -> np.ndarray:
    return d.site_xpos[m.site(f"claw_{SUFFIX[leg]}").id].copy()


def joint_signs(m: mujoco.MjModel) -> dict[str, dict]:
    """Measure, per servo, whether +angle on flybody's axis is the anatomical "+" (ANATOMY). Kinematics only.

    flex: the flexed end of the range is the one that brings the leg tip closest to the parent segment's base.
    Around rest, +-0.2 rad: forward/backward = the tip's x (anterior) change, up = its z change,
    adduct = |y| shrinking (toward the midline)."""
    d = mujoco.MjData(m)
    out = {}
    for leg in LEGS:
        for joint in JOINTS:
            name = f"{joint}_{SUFFIX[leg]}"
            lo, hi = m.actuator_ctrlrange[m.actuator(name).id]
            if joint == "tarsus2":
                lo, hi = -0.3, 0.3
            rest_pose(m, d)
            rest = float(d.actuator_length[m.actuator(name).id])
            kind = ANATOMY[joint]

            def at(v):
                rest_pose(m, d)
                _set(m, d, leg, joint, v)
                mujoco.mj_kinematics(m, d)
                return _tip(m, d, leg), d.xpos[m.body(f"{FLEX_PARENT.get(joint, 'coxa')}_{SUFFIX[leg]}").id].copy()

            if kind == "flex":
                (t_lo, base_lo), (t_hi, base_hi) = at(lo), at(hi)
                d_lo, d_hi = np.linalg.norm(t_lo - base_lo), np.linalg.norm(t_hi - base_hi)
                sign, evidence = (1 if d_hi < d_lo else -1), {"tip_to_parent_at_lo": d_lo, "tip_to_parent_at_hi": d_hi}
            else:
                (t_minus, _), (t_plus, _) = at(max(lo, rest - 0.2)), at(min(hi, rest + 0.2))
                if kind == "adduct":
                    change = abs(t_minus[1]) - abs(t_plus[1])  # > 0: + moves toward the midline
                elif kind == "up":
                    change = t_plus[2] - t_minus[2]
                else:
                    change = (t_plus[0] - t_minus[0]) * (1 if kind == "forward" else -1)
                sign, evidence = (1 if change > 0 else -1), {"tip_change_cm": float(change)}
            out[name] = {"sign": sign, "rest": rest, "range": [float(lo), float(hi)] if joint != "tarsus2"
                         else [float(x) for x in m.actuator_ctrlrange[m.actuator(name).id]],
                         "test": kind, **{k: round(float(v), 5) for k, v in evidence.items()}}
    return out


# ---------- the body ----------

class Body:
    def __init__(self, kit: dict | None = None):
        self.kit = kit if kit is not None else json.loads(KIT.read_text())
        self.m = build_spec(self.kit).compile()
        self.d = mujoco.MjData(self.m)
        self.substeps = int(self.kit["substeps"])
        self.dt_ms = self.m.opt.timestep * 1000.0
        self.act = np.array([self.m.actuator(n).id for n in ACTUATORS])
        self.lo, self.hi = self.m.actuator_ctrlrange[self.act].T
        self.pads = [p["name"] for p in self.kit["pads"]]
        self.pad_of = np.full(self.m.ngeom, -1)
        for i, p in enumerate(self.pads):
            self.pad_of[self.m.geom(f"pad_{p}").id] = i
        self.limb_of = np.full(self.m.ngeom, -1)
        for g in range(self.m.ngeom):
            b = self.m.body(self.m.geom_bodyid[g]).name
            for k, leg in enumerate(LEGS):
                if b.endswith(SUFFIX[leg]) and self.m.geom_contype[g] & KIT_BIT:
                    self.limb_of[g] = k
        self.hat_pedal = self.pads.index("hat_pedal")
        self._jacp = np.zeros((3, self.m.nv))
        self.reset()

    def reset(self) -> None:
        self.rest_ctrl = rest_pose(self.m, self.d)
        self.d.ctrl[:] = self.rest_ctrl
        self.t_ms = 0.0
        self.touching = np.zeros(len(self.pads), dtype=bool)
        self.last_hit = np.full(len(self.pads), -np.inf)
        self.last_touch = np.full(len(self.pads), -np.inf)

    @property
    def rest(self) -> np.ndarray:
        return self.rest_ctrl[self.act]

    def step(self, targets: np.ndarray) -> list[Hit]:
        """Hold the 32 leg servos at `targets` (rad, clipped to range) for 1 ms. Returns the hits in that ms."""
        self.d.ctrl[self.act] = np.clip(targets, self.lo, self.hi)
        hits = []
        for k in range(self.substeps):
            qvel = self.d.qvel.copy()
            mujoco.mj_step(self.m, self.d)
            hits += self._contacts(qvel, self.t_ms + (k + 1) * self.dt_ms)
        self.t_ms += self.substeps * self.dt_ms
        return hits

    def _speed(self, i: int, geom: int, qvel: np.ndarray) -> float:
        """Normal speed of the fly geom at contact i, from the velocities just before this substep."""
        c = self.d.contact[i]
        mujoco.mj_jac(self.m, self.d, self._jacp, None, c.pos, self.m.geom_bodyid[geom])
        return float(abs(self._jacp @ qvel @ c.frame[:3]))

    def _contacts(self, qvel: np.ndarray, t_ms: float) -> list[Hit]:
        now = np.zeros(len(self.pads), dtype=bool)
        speed = np.zeros(len(self.pads))
        limb = np.full(len(self.pads), -1)
        n = self.d.ncon
        if n:
            geoms = self.d.contact.geom[:n]
            pad, player = self.pad_of[geoms], self.limb_of[geoms]
            for i in np.flatnonzero(((pad[:, 0] >= 0) & (player[:, 1] >= 0)) | ((pad[:, 1] >= 0) & (player[:, 0] >= 0))):
                side = 0 if pad[i, 0] >= 0 else 1
                p, g = pad[i, side], geoms[i, 1 - side]
                if not self.touching[p]:
                    s = self._speed(i, g, qvel)
                    if s >= speed[p]:
                        speed[p], limb[p] = s, player[i, 1 - side]
                now[p] = True
        hits = []
        for p in np.flatnonzero(now & ~self.touching):
            if p == self.hat_pedal and speed[p] < PEDAL_CHICK_CM_S:
                continue  # a gentle close is silent, like the TD-07's pedal
            if t_ms - self.last_hit[p] >= REFRACTORY_MS and t_ms - self.last_touch[p] >= DEBOUNCE_MS:
                self.last_hit[p] = t_ms
                voice = VOICE[pad_voice(self.pads[p], now[self.hat_pedal])]
                hits.append(Hit(t_ms, self.pads[p], voice.name, voice.out_note,
                                int(np.clip(round(VEL_PER_CM_S * speed[p]), 1, 127)), float(speed[p]), LEGS[limb[p]]))
        self.last_touch[self.touching] = t_ms - self.dt_ms  # the last substep each released pad was still touched
        self.touching = now
        return hits

    def pad_contacts(self) -> list[tuple[str, str]]:
        """(pad, other geom) for every contact touching the kit right now (for checks)."""
        out = []
        for i in range(self.d.ncon):
            for a, b in ((0, 1), (1, 0)):
                g = self.d.contact.geom[i]
                if self.pad_of[g[a]] >= 0:
                    out.append((self.pads[self.pad_of[g[a]]], self.m.geom(g[b]).name))
        return out


# ---------- kit placement ----------

def _driven(mn_types: dict[str, list[str]]) -> dict[tuple[str, str], set[int]]:
    from fly.decoder import directions

    return directions(mn_types)


def tip_site(leg: str) -> str:
    return f"stick_tip_{leg}" if leg in STICK_LEGS else f"claw_{SUFFIX[leg]}"


def joint_bounds(signs: dict, allowed: dict, leg: str) -> np.ndarray:
    """[8, 2] angle bounds the decoder can reach per joint of a leg: a joint moves only in the anatomical directions
    its motor neurons drive (one-sided joints stop at rest); undriven joints and tarsus2 stay at rest.
    signs: {servo name: {sign, range, rest}} (kit actuators); allowed: {(leg, joint): {+1, -1}}."""
    out = np.zeros((len(JOINTS), 2))
    for j, joint in enumerate(JOINTS):
        info = signs[f"{joint}_{SUFFIX[leg]}"]
        lo, hi, rest = *info["range"], info["rest"]
        dirs = allowed.get((leg, joint), set()) if joint != "tarsus2" else set()
        angle_dirs = {s * info["sign"] for s in dirs}
        out[j] = (lo if -1 in angle_dirs else rest), (hi if 1 in angle_dirs else rest)
    return out


def _workspace(m, leg: str, signs: dict, allowed: dict, n: int, rng):
    """Random decoder-reachable configurations of one leg -> (servo targets [n, 8], tip positions [n, 3],
    stick grip positions [n, 3]; the tip again for a leg without a stick)."""
    d = mujoco.MjData(m)
    site = m.site(tip_site(leg)).id
    grip = m.site(f"stick_grip_{leg}").id if leg in STICK_LEGS else site
    bounds = joint_bounds(signs, allowed, leg)
    qs, tips, grips = np.zeros((n, len(JOINTS))), np.zeros((n, 3)), np.zeros((n, 3))
    rest_pose(m, d)
    base = d.qpos.copy()
    for k in range(n):
        d.qpos[:] = base
        qs[k] = rng.uniform(bounds[:, 0], bounds[:, 1])
        for j, joint in enumerate(JOINTS):
            _set(m, d, leg, joint, qs[k, j])
        mujoco.mj_kinematics(m, d)
        tips[k], grips[k] = d.site_xpos[site], d.site_xpos[grip]
    return qs, tips, grips


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + t * ab)))


def _clearance(m, d, p: np.ndarray) -> float:
    """Distance from point p to the nearest colliding geom of the fly (capsules exact, others by bounding sphere)."""
    best = np.inf
    for g in np.flatnonzero(m.geom_contype != 0):
        c = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            t = np.clip(np.dot(p - c, half) / max(np.dot(half, half), 1e-12), -1, 1)
            best = min(best, np.linalg.norm(p - (c + t * half)) - m.geom_size[g, 0])
        else:
            best = min(best, np.linalg.norm(p - c) - m.geom_rbound[g])
    return best


def place_kit(m, signs: dict, allowed: dict, n: int = 6000, seed: int = 0) -> list[dict]:
    """Drums face up under the READY stick tips, between DROP_MIN and DROP_MAX below them, each at a point the
    decoder can reach, packed around the playing stick's rest tip (toms around both). Pedals go under the hind feet.
    A stick striking one pad must keep its shaft clear of every other pad."""
    from scipy.spatial import cKDTree

    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    rest_pose(m, d)
    ws = {leg: _workspace(m, leg, signs, allowed, n, rng) for leg in LEGS}
    rest_tip = {leg: d.site_xpos[m.site(f"stick_tip_{leg}" if leg in STICK_LEGS else f"claw_{SUFFIX[leg]}").id].copy()
                for leg in LEGS}
    pads: list[dict] = []
    shafts: list[tuple[np.ndarray, np.ndarray]] = []  # stick shafts (grip, tip) at every placed pad's strike
    up = [0.0, 0.0, 1.0]

    def radius(half):
        return float(np.linalg.norm(half[:2]))

    def free(p, half):
        r = radius(half)
        return (_clearance(m, d, p) >= r + CLEARANCE
                and all(np.linalg.norm(np.array(q["pos"]) - p) >= r + radius(q["half"]) + PAD_GAP for q in pads)
                and all(_seg_dist(p, a, b) >= r + STICK_RADIUS + CLEARANCE for a, b in shafts))

    def add(name, kind, legs, pts, half):
        home = np.mean([rest_tip[leg] for leg in legs], axis=0)
        for k in np.argsort(np.linalg.norm((pts - home)[:, :2], axis=1)):  # nearest to straight below home
            if not free(pts[k], half):
                continue
            reach, mine = {}, []
            for leg in legs:
                j = int(np.argmin(np.linalg.norm(ws[leg][1] - pts[k], axis=1)))
                reach[leg] = [round(float(x), 4) for x in ws[leg][0][j]]
                if leg in STICK_LEGS:
                    mine.append((ws[leg][2][j], ws[leg][1][j]))
            if any(_seg_dist(np.array(q["pos"]), a, b) < radius(q["half"]) + STICK_RADIUS + CLEARANCE
                   for q in pads for a, b in mine):
                continue
            shafts.extend(mine)
            pads.append({"name": name, "kind": kind, "legs": list(legs), "pos": [round(float(x), 5) for x in pts[k]],
                         "normal": up, "half": list(half), "reach": reach})
            return
        raise RuntimeError(f"no free reachable spot for {name}")

    for name, legs in STICK_PADS.items():
        pts = ws[legs[0]][1]
        z0 = min(rest_tip[leg][2] for leg in legs)
        keep = (pts[:, 2] < z0 - DROP_MIN) & (pts[:, 2] > z0 - DROP_MAX)
        if len(legs) > 1:  # both sticks must reach it
            keep &= cKDTree(ws[legs[1]][1]).query(pts)[0] < 0.004
        add(name, "stick", legs, pts[keep], PAD_HALF)
    for name, leg in PEDALS.items():
        pts = ws[leg][1]
        add(name, "pedal", (leg,), pts[pts[:, 2] < rest_tip[leg][2] - DROP_MIN], PEDAL_HALF)
    return pads


def strike(body: Body, leg: str, q: list[float], ms: int = 60, back_ms: int = 60) -> list[Hit]:
    """Scripted stroke: drive one leg's 8 servos to q (others at rest), hold, then return to rest."""
    k = LEGS.index(leg)
    target = body.rest.copy()
    target[k * len(JOINTS):(k + 1) * len(JOINTS)] = q
    hits = []
    for _ in range(ms):
        hits += body.step(target)
    for _ in range(back_ms):
        hits += body.step(body.rest)
    return hits


def build_kit(n: int = 6000, seed: int = 0, timestep: float = 1e-4) -> dict:
    import pandas as pd

    from fly.connectome import ANNOTATIONS, MALECNS_DIR
    from fly.decoder import unmapped
    from fly.wiring import leg_motor_neurons

    ann = pd.read_feather(MALECNS_DIR / ANNOTATIONS)
    ann = ann[ann["superclass"].notna()].sort_values("bodyId").reset_index(drop=True)
    _, mn_types = leg_motor_neurons(ann)
    allowed = _driven(mn_types)

    m = build_spec().compile()
    signs = joint_signs(m)
    kit = {
        "about": "Generated by `python -m fly.body --build-kit`. Units cm. Actuators in drums.LEGS x decoder.JOINTS "
                 "order; sign = +1 when +angle on flybody's axis is the anatomical + of decoder.MN_JOINT. rest = springref.",
        "timestep": timestep, "substeps": int(round(1e-3 / timestep)),
        "ready": READY,
        "stick": {"length": STICK_LENGTH, "radius": STICK_RADIUS, "mass": STICK_MASS, "dir_in_ready": STICK_DIR},
        "sign_left_right_mismatch": [f"{j}_{pair}" for j in JOINTS for pair in ("T1", "T3")
                                     if signs[f"{j}_{pair}_left"]["sign"] != signs[f"{j}_{pair}_right"]["sign"]],
        "actuators": [{"name": name, "leg": leg, "joint": joint, **signs[name]}
                      for leg in LEGS for joint in JOINTS for name in [f"{joint}_{SUFFIX[leg]}"]],
        "driven": {f"{leg}.{joint}": sorted(s) for (leg, joint), s in sorted(allowed.items())},
        "unmapped_mn_types": unmapped(mn_types),
        "pads": place_kit(m, signs, allowed, n, seed),
    }
    body = Body(kit)
    for _ in range(300):  # the fly at rest must not touch the kit
        body.step(body.rest)
    kit["rest_contacts"] = sorted(set(body.pad_contacts()))
    for p in kit["pads"]:
        p["reach_ok"] = {}
        for leg, q in p["reach"].items():
            body.reset()
            hits = strike(body, leg, q)
            p["reach_ok"][leg] = [h.pad for h in hits]
    return kit


def bench(kit: dict) -> dict:
    """Every pad's scripted strokes at 1e-4 s and 2e-4 s: hits, tip drift between the two, warnings, wall time."""
    out = {}
    tips = {}
    for ts in (1e-4, 2e-4):
        b = Body({**kit, "timestep": ts, "substeps": int(round(1e-3 / ts))})
        rows, track, t0 = [], [], perf_counter()
        for p in kit["pads"]:
            for leg, q in p["reach"].items():
                b.reset()
                hits = strike(b, leg, q)
                rows.append({"pad": p["name"], "leg": leg, "hits": [(h.pad, round(h.t_ms, 1), h.velocity) for h in hits]})
                track.append(b.d.site_xpos[b.m.site(f"stick_tip_{leg}" if leg in STICK_LEGS else f"claw_{SUFFIX[leg]}").id].copy())
        wall = perf_counter() - t0
        sim_s = sum(len(p["reach"]) for p in kit["pads"]) * 0.12
        out[str(ts)] = {"wall_s_per_sim_s": round(wall / sim_s, 2), "warnings": int(b.d.warning.number.sum()),
                        "nan": bool(np.isnan(b.d.qpos).any()), "strokes": rows}
        tips[ts] = np.array(track)
    out["final_tip_drift_cm"] = float(np.abs(tips[1e-4] - tips[2e-4]).max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-kit", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--timestep", type=float, default=1e-4, help="kit timestep when building")
    ap.add_argument("--samples", type=int, default=6000)
    args = ap.parse_args()
    if args.build_kit:
        kit = build_kit(args.samples, timestep=args.timestep)
        KIT.write_text(json.dumps(kit, indent=1))
        for a in kit["actuators"]:
            print(f"  {a['name']:22s} sign {a['sign']:+d}  rest {a['rest']:+.3f}  {a['test']}")
        for p in kit["pads"]:
            print(f"  {p['name']:10s} {p['kind']:6s} pos {p['pos']} reach {p['reach_ok']}")
        print("rest contacts:", kit["rest_contacts"])
        print(f"wrote {KIT}")
    if args.bench:
        res = bench(json.loads(KIT.read_text()))
        print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

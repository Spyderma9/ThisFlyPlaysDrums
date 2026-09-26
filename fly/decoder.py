"""Fixed decoder: motor-neuron spikes -> joint-angle targets for the body's 32 leg position servos (D5).

Rule-based, like muscles. It never sees the target groove.

1. Rates: an exponential filter (tau_ms) on each leg motor neuron's spikes, scaled so a steady f Hz train reads f.
2. Per leg and joint: drive = mean rate of the joint's "+" types - mean rate of its "-" types (Hz). "+" is flexion or
   the direction the muscle names (MN_JOINT); `sign` maps that anatomical direction onto flybody's joint axis and is
   measured by `python -m fly.body --build-kit`. Joints with no motor neurons hold rest.
3. Soft limit: x = sign * gain * drive, target = rest + side * tanh(x / side), where side is the room from rest to
   the range end x heads for. Smooth, so gradients never vanish at a limit; slope 1 at rest.
   gain: `hz_third_range` Hz of drive asks for a third of the joint's range (before the tanh).
4. The pseudo-inverse maps joint targets back to motor-neuron rates and then to currents I* (mV per step, the
   LIF's rate -> current inverse), which the teacher injects as alpha * I* (Phase 4).

Joint order, ranges, rest pose and signs come from fly/kit.json, so this module needs no MuJoCo.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

KIT = Path(__file__).with_name("kit.json")
JOINTS = ("coxa_abduct", "coxa_twist", "coxa", "femur_twist", "femur", "tibia", "tarsus", "tarsus2")

# Motor-neuron type (MaleCNS `type` minus " MN") -> ((joint, +1 = flexion/named direction, -1 = against), ...)
# Sternotrochanter: the sternotrochanter extensor (stte) depresses the trochanter, so it opposes Tr flexor.
MN_JOINT = {
    "Ti flexor": (("tibia", 1),), "Acc. ti flexor": (("tibia", 1),), "Ti extensor": (("tibia", -1),),
    "Tr flexor": (("femur", 1),), "Acc. tr flexor": (("femur", 1),),
    "Tr extensor": (("femur", -1),), "Sternotrochanter": (("femur", -1),),
    "Fe reductor": (("femur_twist", 1),),
    "Sternal anterior rotator": (("coxa_twist", 1),), "Sternal posterior rotator": (("coxa_twist", -1),),
    "Pleural promotor": (("coxa", 1),), "Pleural remotor": (("coxa", -1),),
    "Pleural remotor/abductor": (("coxa", -1), ("coxa_abduct", -1)),
    "Sternal adductor": (("coxa_abduct", 1),), "Sternal abductor": (("coxa_abduct", -1),),
    "Ta depressor": (("tarsus", 1),), "Ta levator": (("tarsus", -1),),
    "ltm": (("tarsus2", 1),), "ltm1-tibia": (("tarsus2", 1),), "ltm2-femur": (("tarsus2", 1),),
}

# fly-brain LIF at dt = 1 ms: v += I; v += (g - (v - rest)) / tauMem. From rest a constant I (mV/step) spikes
# after n steps where 19 I (1 - 0.95^n) = threshold gap, so rate 1000/n needs I = gap / (19 (1 - 0.95^n)).
_V_GAP, _TAU_MEM = 7.0, 20.0
HZ_THIRD_RANGE = 30.0  # ~p90-p95 of the nonzero drive in an untrained run (Phase 2); hand-set once


def mapping(mn_type: str):
    return MN_JOINT.get(mn_type.removesuffix(" MN"), ())


def directions(mn_types: dict[str, list[str]]) -> dict[tuple[str, str], set[int]]:
    """(leg, joint) -> anatomical directions the decoder can drive, from the motor-neuron types present."""
    out: dict[tuple[str, str], set[int]] = {}
    for leg, types in mn_types.items():
        for t in types:
            for joint, s in mapping(t):
                out.setdefault((leg, joint), set()).add(s)
    return out


def unmapped(mn_types: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    out = {}
    for leg, types in mn_types.items():
        miss = [t for t in types if not mapping(t)]
        out[leg] = {t: miss.count(t) for t in sorted(set(miss))}
    return out


def lif_current(rates: torch.Tensor) -> torch.Tensor:
    """Constant current (mV/step) that makes a resting fly-brain LIF fire at `rates` Hz; odd in the rate, so a
    negative rate becomes an inhibiting current. Below 1 Hz it ramps linearly to 0."""
    decay = 1 - 1 / _TAU_MEM
    mag = rates.abs().clamp(min=1.0)
    cur = _V_GAP / ((_TAU_MEM - 1) * (1 - decay ** (1000.0 / mag)))
    return torch.sign(rates) * cur * rates.abs().clamp(max=1.0)


class Decoder(nn.Module):
    def __init__(self, leg_mns: dict[str, np.ndarray], mn_types: dict[str, list[str]], n_neurons: int,
                 kit: dict | None = None, tau_ms: float = 20.0, hz_third_range: float = HZ_THIRD_RANGE,
                 dt_ms: float = 1.0):
        super().__init__()
        kit = kit if kit is not None else json.loads(KIT.read_text())
        acts = kit["actuators"]  # 32 leg servos, drums.LEGS x JOINTS order
        row = {(a["leg"], a["joint"]): i for i, a in enumerate(acts)}
        legs = list(leg_mns)
        idx = np.concatenate([leg_mns[leg] for leg in legs]).astype(np.int64)
        types = [t for leg in legs for t in mn_types[leg]]
        owner = [leg for leg in legs for _ in leg_mns[leg]]

        a = np.zeros((len(acts), len(idx)), dtype=np.float32)  # mean(+ types) - mean(- types), per servo
        for col, (leg, t) in enumerate(zip(owner, types)):
            for joint, s in mapping(t):
                a[row[leg, joint], col] = s
        for r in range(len(acts)):
            for s in (1, -1):
                sel = a[r] == s
                if sel.any():
                    a[r, sel] = s / sel.sum()
        lo, hi = (np.array([x["range"][k] for x in acts], dtype=np.float32) for k in (0, 1))
        rest = np.array([x["rest"] for x in acts], dtype=np.float32)
        gain = np.array([x["sign"] for x in acts], dtype=np.float32) * (hi - lo) / 3 / hz_third_range
        eff = gain[:, None] * a  # rates [M] -> pre-limit joint offsets x [32]

        self.names = [x["name"] for x in acts]
        self.n_neurons, self.decay, self.dt_ms = n_neurons, float(np.exp(-dt_ms / tau_ms)), dt_ms
        self.register_buffer("idx", torch.as_tensor(idx))
        self.register_buffer("drive_mat", torch.as_tensor(a))
        self.register_buffer("eff", torch.as_tensor(eff))
        self.register_buffer("pinv", torch.as_tensor(np.linalg.pinv(eff).astype(np.float32)))
        self.register_buffer("rest", torch.as_tensor(rest))
        self.register_buffer("up", torch.as_tensor(np.maximum(hi - rest, 1e-3)))  # room above and below rest
        self.register_buffer("down", torch.as_tensor(np.maximum(rest - lo, 1e-3)))

    def init_state(self, batch: int = 1) -> torch.Tensor:
        return self.idx.new_zeros(batch, len(self.idx), dtype=torch.float32)

    def forward(self, rates: torch.Tensor, spikes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """rates [B, M] filter state, spikes [B, N] -> (new rates, joint targets [B, 32])."""
        rates = rates * self.decay + spikes[:, self.idx] * ((1 - self.decay) * 1000.0 / self.dt_ms)
        return rates, self.targets(rates)

    def drive(self, rates: torch.Tensor) -> torch.Tensor:
        """Per-servo drive [B, 32] in Hz: mean rate of + types - mean rate of - types."""
        return rates @ self.drive_mat.t()

    def targets(self, rates: torch.Tensor) -> torch.Tensor:
        x = rates @ self.eff.t()
        side = torch.where(x >= 0, self.up, self.down)
        return self.rest + side * torch.tanh(x / side)

    def rates_for(self, q: torch.Tensor) -> torch.Tensor:
        """Least-norm motor-neuron rates [B, M] (Hz, signed) whose decoded targets are q [B, 32] (q is kept 1% inside
        the range, where the soft limit is still invertible)."""
        dq = q - self.rest
        side = torch.where(dq >= 0, self.up, self.down)
        x = side * torch.atanh((dq / side).clamp(-0.99, 0.99))
        return x @ self.pinv.t()

    def current_for(self, q: torch.Tensor) -> torch.Tensor:
        """I* [B, N] in mV per step: the current into each leg motor neuron that asks for rates_for(q)."""
        cur = q.new_zeros(q.shape[0], self.n_neurons)
        return cur.index_copy(1, self.idx, lif_current(self.rates_for(q)))

"""Everything the brain needs, built one way so loop, train and evaluate can't drift apart.

    load_malecns -> cut every edge onto a Kenyon cell (D8) -> optional degree-preserving shuffle
    -> cue groups from cues.json (D2) -> plastic mask: cue -> descending neuron (D3) -> leg motor neurons.

The KC cut comes before the shuffle, so the control has the same cut and exactly the same degrees as the
network it controls. Masks are always computed on the (possibly shuffled) edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fly.connectome import Connectome, load_malecns
from fly.drums import LEGS, VOICES
from fly.probe import LEGS as LEG_NERVES, _side

CUES = Path(__file__).with_name("cues.json")
DESCENDING = "descending_neuron"
PLASTIC = ("cue_dn", "cue_dn+dn_mn")  # trainable sets: D3's cue -> DN, optionally plus DN -> leg motor neurons


@dataclass
class Wiring:
    conn: Connectome
    cue_groups: dict[str, np.ndarray]  # voice -> neuron rows, drums.VOICES order
    plastic_mask: np.ndarray  # bool per edge: the trainable set (see PLASTIC)
    leg_mns: dict[str, np.ndarray]  # leg -> motor-neuron rows, drums.LEGS order
    mn_types: dict[str, list[str]]  # leg -> cell type per motor neuron, same order ("?" if unnamed)
    kc_edges_cut: int
    shuffle_seed: int | None
    plastic: str = "cue_dn"
    plastic_counts: dict = field(default_factory=dict)

    def brain(self, batch: int = 1, device: str = "cuda", surrogate_mv: float | None = None, tone: bool = False,
              ff_credit: bool = False):
        """ff_credit: backprop only one hop, leg motor neurons -> descending neurons (training; forward unchanged)."""
        from fly.brain import Brain

        mns = np.concatenate(list(self.leg_mns.values()))
        dns = np.flatnonzero((self.conn.neurons["superclass"] == DESCENDING).to_numpy())
        return Brain(self.conn, self.cue_groups, self.plastic_mask, batch=batch, device=device, surrogate_mv=surrogate_mv,
                     tone_idx=mns if tone else None, ff_credit=(dns, mns) if ff_credit else None)

    def summary(self) -> dict:
        return {
            "connectome": self.conn.name, "neurons": self.conn.size, "edges": len(self.conn.pre),
            "kc_edges_cut": self.kc_edges_cut, "shuffle_seed": self.shuffle_seed,
            "plastic": self.plastic, "plastic_edges": int(self.plastic_mask.sum()), "plastic_counts": self.plastic_counts,
            "cue_groups": {v: len(i) for v, i in self.cue_groups.items()},
            "leg_mns": {leg: len(i) for leg, i in self.leg_mns.items()},
        }


def cut_kc_inputs(conn: Connectome) -> tuple[Connectome, int]:
    kc = conn.neurons["type"].fillna("").astype(str).str.startswith("KC").to_numpy()
    keep = ~kc[conn.post]
    return Connectome(conn.name, conn.neurons, conn.pre[keep], conn.post[keep], conn.weight[keep]), int((~keep).sum())


def leg_motor_neurons(neurons: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, list[str]]]:
    side, sup = _side(neurons), neurons["superclass"].fillna("").astype(str)
    nerve, types = neurons["exitNerve"].fillna(""), neurons["type"].fillna("?").astype(str)
    idx = {leg: np.flatnonzero(((sup == "vnc_motor") & (nerve == LEG_NERVES[leg][0]) & (side == LEG_NERVES[leg][1])).to_numpy())
           for leg in LEGS}
    return idx, {leg: types.iloc[i].tolist() for leg, i in idx.items()}


def cue_groups(neurons: pd.DataFrame, cues_path: Path = CUES) -> dict[str, np.ndarray]:
    drums = json.loads(cues_path.read_text())["drums"]
    row = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"].to_numpy())
    missing = {v.name: [b for b in drums[v.name]["bodyIds"] if b not in row.index] for v in VOICES}
    if any(missing.values()):
        raise KeyError(f"cue bodyIds not in the connectome: {missing}")
    return {v.name: row.loc[drums[v.name]["bodyIds"]].to_numpy() for v in VOICES}


def wire(conn: Connectome | None = None, shuffle_seed: int | None = None, cues_path: Path = CUES,
         plastic: str = "cue_dn") -> Wiring:
    if plastic not in PLASTIC:
        raise ValueError(f"plastic must be one of {PLASTIC}, not {plastic!r}")
    conn, n_cut = cut_kc_inputs(conn if conn is not None else load_malecns())
    if shuffle_seed is not None:
        conn = conn.shuffled(shuffle_seed)
    cues = cue_groups(conn.neurons, cues_path)
    is_cue = np.zeros(conn.size, dtype=bool)
    is_cue[np.concatenate(list(cues.values()))] = True
    is_dn = (conn.neurons["superclass"] == DESCENDING).to_numpy()
    mns, types = leg_motor_neurons(conn.neurons)
    is_mn = np.zeros(conn.size, dtype=bool)
    is_mn[np.concatenate(list(mns.values()))] = True
    classes = {"cue_dn": is_cue[conn.pre] & is_dn[conn.post]}  # D3
    if plastic == "cue_dn+dn_mn":  # also descending -> playing-leg motor neurons: which legs and joints each DN drives
        classes["dn_mn"] = is_dn[conn.pre] & is_mn[conn.post]
    mask = np.logical_or.reduce(list(classes.values()))
    counts = {k: int(m.sum()) for k, m in classes.items()}
    return Wiring(conn, cues, mask, mns, types, n_cut, shuffle_seed, plastic, counts)

"""Everything the brain needs, built one way so loop, train and evaluate can't drift apart.

    load_malecns -> cut every edge onto a Kenyon cell (D8) -> optional degree-preserving shuffle
    -> cue groups from cues.json (D2) -> plastic mask: cue -> descending neuron (D3) -> leg motor neurons.

The KC cut comes before the shuffle, so the control has the same cut and exactly the same degrees as the
network it controls. Masks are always computed on the (possibly shuffled) edges.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fly.connectome import Connectome, load_malecns
from fly.drums import LEGS, VOICES
from fly.probe import LEGS as LEG_NERVES, _side

CUES = Path(__file__).with_name("cues.json")
DESCENDING = "descending_neuron"


@dataclass
class Wiring:
    conn: Connectome
    cue_groups: dict[str, np.ndarray]  # voice -> neuron rows, drums.VOICES order
    plastic_mask: np.ndarray  # bool per edge: pre is a cue neuron, post a descending neuron
    leg_mns: dict[str, np.ndarray]  # leg -> motor-neuron rows, drums.LEGS order
    mn_types: dict[str, list[str]]  # leg -> cell type per motor neuron, same order ("?" if unnamed)
    kc_edges_cut: int
    shuffle_seed: int | None

    def brain(self, batch: int = 1, device: str = "cuda"):
        from fly.brain import Brain

        return Brain(self.conn, self.cue_groups, self.plastic_mask, batch=batch, device=device)

    def summary(self) -> dict:
        return {
            "connectome": self.conn.name, "neurons": self.conn.size, "edges": len(self.conn.pre),
            "kc_edges_cut": self.kc_edges_cut, "shuffle_seed": self.shuffle_seed,
            "plastic_edges": int(self.plastic_mask.sum()),
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


def wire(conn: Connectome | None = None, shuffle_seed: int | None = None, cues_path: Path = CUES) -> Wiring:
    conn, n_cut = cut_kc_inputs(conn if conn is not None else load_malecns())
    if shuffle_seed is not None:
        conn = conn.shuffled(shuffle_seed)
    cues = cue_groups(conn.neurons, cues_path)
    is_cue = np.zeros(conn.size, dtype=bool)
    is_cue[np.concatenate(list(cues.values()))] = True
    is_dn = (conn.neurons["superclass"] == DESCENDING).to_numpy()
    plastic = is_cue[conn.pre] & is_dn[conn.post]
    mns, types = leg_motor_neurons(conn.neurons)
    return Wiring(conn, cues, plastic, mns, types, n_cut, shuffle_seed)

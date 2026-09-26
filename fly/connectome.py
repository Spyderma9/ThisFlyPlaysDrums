"""Connectome loading: MaleCNS v1.0 via neuPrint (main), FlyWire v783 from fly-brain's files (fallback).

Both give a Connectome: a neuron table whose row order is the matrix index, and signed synapse-count
edges (pre, post, weight). Sign follows Shiu et al.: GABA, glutamate and histamine inhibit, everything
else excites. fly-brain's wScale is applied later in brain.py, not here.

    python -m fly.connectome            # pull MaleCNS into data/malecns/ (resumes if interrupted)
    python -m fly.connectome --summary  # print neuron counts by superclass and the exit nerves seen
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
MALECNS_DIR = DATA / "malecns"
FLYBRAIN_DATA = REPO / "fly" / "vendor" / "fly-brain" / "data"

DATASET = "male-cns:v1.0"
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}
NEURON_COLUMNS = [
    "bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "somaNeuromere",
    "entryNerve", "exitNerve", "consensusNt", "predictedNt", "somaLocation", "status",
]


@dataclass
class Connectome:
    name: str
    neurons: pd.DataFrame  # row i is matrix index i
    pre: np.ndarray  # int64 matrix indices
    post: np.ndarray
    weight: np.ndarray  # float32 signed synapse counts

    @property
    def size(self) -> int:
        return len(self.neurons)

    def to_torch(self, device="cpu", transpose=False):
        """Sparse CSR matrix W[post, pre] (or W.T with transpose=True) on `device`."""
        import torch

        rows, cols = (self.pre, self.post) if transpose else (self.post, self.pre)
        coo = torch.sparse_coo_tensor(
            np.stack([rows, cols]), torch.from_numpy(self.weight), (self.size, self.size)
        ).coalesce()
        return coo.to_sparse_csr().to(device)

    def shuffled(self, seed: int = 0) -> Connectome:
        """Degree-preserving control: permute post endpoints across edges.

        Every neuron keeps its exact in- and out-degree, and each edge keeps its presynaptic sign
        (Dale's law). The same neuron IDs, so cue groups, motor neurons and the trainable set carry over.
        """
        rng = np.random.default_rng(seed)
        return Connectome(f"{self.name}-shuffled{seed}", self.neurons, self.pre, rng.permutation(self.post), self.weight)


def _client():
    from dotenv import load_dotenv
    from neuprint import Client

    load_dotenv(REPO / ".env")
    token = os.environ.get("NEUPRINT_TOKEN", "").strip()
    if not token:
        raise SystemExit("NEUPRINT_TOKEN is empty. Paste your neuprint.janelia.org token into .env.")
    return Client("neuprint.janelia.org", dataset=DATASET, token=token)


def _sign(nt: pd.Series) -> np.ndarray:
    return np.where(nt.fillna("").str.lower().isin(INHIBITORY_NT), -1.0, 1.0).astype(np.float32)


def fetch_malecns(batch: int = 2000) -> None:
    """Pull neurons and all neuron-to-neuron edges into data/malecns/. Edge batches resume if interrupted."""
    from neuprint import NeuronCriteria, fetch_adjacencies, fetch_neurons

    client = _client()
    MALECNS_DIR.mkdir(parents=True, exist_ok=True)
    neurons_path = MALECNS_DIR / "neurons.parquet"
    if not neurons_path.exists():
        print("Fetching neurons...")
        df, _ = fetch_neurons(NeuronCriteria(), omit_rois=True, client=client)
        missing = [c for c in NEURON_COLUMNS if c not in df.columns]
        if missing:
            print(f"  note: columns not in {DATASET}: {missing}")
        if "somaLocation" in df.columns:
            loc = df["somaLocation"].apply(lambda p: p if isinstance(p, (list, tuple)) and len(p) == 3 else [np.nan] * 3)
            df[["x", "y", "z"]] = pd.DataFrame(loc.tolist(), index=df.index)
        keep = [c for c in NEURON_COLUMNS if c in df.columns and c != "somaLocation"] + [c for c in "xyz" if c in df.columns]
        df[keep].sort_values("bodyId").reset_index(drop=True).to_parquet(neurons_path)
    neurons = pd.read_parquet(neurons_path)
    print(f"{len(neurons):,} neurons")

    edge_dir = MALECNS_DIR / "edges"
    edge_dir.mkdir(exist_ok=True)
    ids = neurons["bodyId"].to_numpy()
    n_batches = int(np.ceil(len(ids) / batch))
    for b in range(n_batches):
        out = edge_dir / f"{b:04d}.parquet"
        if out.exists():
            continue
        chunk = ids[b * batch : (b + 1) * batch].tolist()
        _, conn = fetch_adjacencies(
            NeuronCriteria(bodyId=chunk), None, omit_rois=True, properties=[], client=client
        )
        conn[["bodyId_pre", "bodyId_post", "weight"]].to_parquet(out)
        print(f"  edges batch {b + 1}/{n_batches}: {len(conn):,}")


def load_malecns() -> Connectome:
    neurons = pd.read_parquet(MALECNS_DIR / "neurons.parquet")
    edges = pd.concat(pd.read_parquet(p) for p in sorted((MALECNS_DIR / "edges").glob("*.parquet")))
    index = pd.Series(np.arange(len(neurons)), index=neurons["bodyId"])
    edges = edges[edges["bodyId_post"].isin(index.index)]  # drop edges onto fragments outside the table
    pre = index[edges["bodyId_pre"]].to_numpy()
    post = index[edges["bodyId_post"]].to_numpy()
    nt = neurons["consensusNt"] if "consensusNt" in neurons else neurons.get("predictedNt", pd.Series(index=neurons.index, dtype=str))
    weight = edges["weight"].to_numpy(np.float32) * _sign(nt)[pre]
    return Connectome("malecns-v1.0", neurons, pre, post, weight)


def load_flywire() -> Connectome:
    """FlyWire v783 as shipped with fly-brain (already signed). Brain only: no leg motor neurons."""
    names = pd.read_csv(FLYBRAIN_DATA / "2025_Completeness_783.csv")
    conn = pd.read_parquet(FLYBRAIN_DATA / "2025_Connectivity_783.parquet")
    neurons = pd.DataFrame({"bodyId": names.iloc[:, 0].to_numpy()})
    return Connectome(
        "flywire-v783",
        neurons,
        conn["Presynaptic_Index"].to_numpy(np.int64),
        conn["Postsynaptic_Index"].to_numpy(np.int64),
        conn["Excitatory x Connectivity"].to_numpy(np.float32),
    )


def summary(conn: Connectome) -> None:
    n = conn.neurons
    print(f"{conn.name}: {conn.size:,} neurons, {len(conn.pre):,} edges, "
          f"{(conn.weight < 0).mean():.1%} inhibitory")
    for col in ("superclass", "exitNerve"):
        if col in n:
            print(f"\n{col}:\n{n[col].value_counts(dropna=False).head(40).to_string()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true", help="print counts instead of fetching")
    ap.add_argument("--flywire", action="store_true", help="summarize the FlyWire fallback instead")
    args = ap.parse_args()
    if args.flywire:
        summary(load_flywire())
    elif args.summary:
        summary(load_malecns())
    else:
        fetch_malecns()

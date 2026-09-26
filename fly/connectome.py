"""Connectome loading: MaleCNS v1.0 bulk download (main), FlyWire v783 from fly-brain's files (fallback).

Both give a Connectome: a neuron table whose row order is the matrix index, and signed synapse-count
edges (pre, post, weight). Sign follows Shiu et al.: GABA, glutamate and histamine inhibit, everything
else excites (including an unclear or missing neurotransmitter). fly-brain's wScale is applied later in
brain.py, not here.

MaleCNS v1.0 (CC-BY) comes as flat feather files from Janelia's bucket. A body counts as a neuron when it
has a superclass; that drops glia, orphans and other fragments (166,700 of 211,577 bodies). Soma positions
are raw voxel coordinates (8 nm), from somaLocation or else tosomaLocation, NaN for neurons with neither.

    python -m fly.connectome            # download MaleCNS into data/malecns/ (resumes if interrupted)
    python -m fly.connectome --summary  # print neuron counts by superclass and the exit nerves seen
"""

from __future__ import annotations

import argparse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
MALECNS_DIR = DATA / "malecns"
FLYBRAIN_DATA = REPO / "fly" / "vendor" / "fly-brain" / "data"

MALECNS_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"  # 14.5 MB
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"  # 43 MB
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"  # 1.05 GB: body_pre, body_post, weight
INHIBITORY_NT = {"gaba", "glutamate", "histamine"}
NEURON_COLUMNS = [
    "bodyId", "type", "instance", "superclass", "class", "subclass", "somaSide", "somaNeuromere",
    "entryNerve", "exitNerve", "receptorType", "flywireType", "mancType", "status",
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


def _sign(nt: pd.Series) -> np.ndarray:
    return np.where(nt.fillna("").str.lower().isin(INHIBITORY_NT), -1.0, 1.0).astype(np.float32)


def _download(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    """Fetch url to dest via dest.part, resuming a partial .part with an HTTP Range request."""
    if dest.exists():
        return
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    with urllib.request.urlopen(req) as resp:
        if have and resp.status != 206:  # server ignored the range: start over
            have = 0
        total = have + int(resp.headers.get("Content-Length", 0))
        with open(part, "ab" if have else "wb") as f:
            done = have
            while block := resp.read(chunk):
                f.write(block)
                done += len(block)
                print(f"\r  {dest.name}: {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
    print()
    part.rename(dest)


def fetch_malecns(root: Path = MALECNS_DIR) -> None:
    """Download the three MaleCNS v1.0 files into root. Re-running resumes or skips finished files."""
    root.mkdir(parents=True, exist_ok=True)
    for name in (ANNOTATIONS, NEUROTRANSMITTERS, WEIGHTS):
        _download(MALECNS_URL + name, root / name)


def _xyz(loc) -> list:
    return list(loc) if loc is not None and len(loc) == 3 else [np.nan] * 3


def load_malecns(root: Path = MALECNS_DIR) -> Connectome:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.ipc  # noqa: F401  (makes pa.ipc available)

    ann = pd.read_feather(root / ANNOTATIONS)
    ann = ann[ann["superclass"].notna()].sort_values("bodyId").reset_index(drop=True)
    neurons = ann[NEURON_COLUMNS].copy()
    soma = ann["somaLocation"].where(ann["somaLocation"].notna(), ann["tosomaLocation"])
    neurons[["x", "y", "z"]] = np.array([_xyz(p) for p in soma], dtype=np.float64).reshape(-1, 3)

    nt = pd.read_feather(root / NEUROTRANSMITTERS, columns=["body", "consensus_nt", "predicted_nt"])
    neurons = neurons.merge(nt.rename(columns={"body": "bodyId"}), on="bodyId", how="left")

    ids = neurons["bodyId"].to_numpy()  # sorted, so searchsorted maps bodyId -> row index
    edges = pa.ipc.open_file(root / WEIGHTS).read_all()  # feather v2 is Arrow IPC: body_pre, body_post, weight
    known = pa.array(ids)  # drop edges touching fragments outside the neuron table
    edges = edges.filter(pc.and_(pc.is_in(edges["body_pre"], known), pc.is_in(edges["body_post"], known)))
    pre = np.searchsorted(ids, edges["body_pre"].to_numpy())
    post = np.searchsorted(ids, edges["body_post"].to_numpy())
    weight = edges["weight"].to_numpy().astype(np.float32) * _sign(neurons["consensus_nt"])[pre]
    return Connectome("malecns-v1.0", neurons, pre.astype(np.int64), post.astype(np.int64), weight)


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

"""Phase 1 probe: which input neurons reach the playing legs, and how fast.

Graph pass: fewest synapse hops from each candidate cue group to each readout group, all synapses and
strong ones only (>= STRONG synapses), and whether the Kenyon cell -> MBON step can sit on the way.
Simulation pass: drive one candidate at a time (--rate Hz for --stim-ms, --trials trials batched) and
time how each readout group responds, plus Kenyon cells and MBONs.

Candidates are Johnston's organ families (JO-A..F, per side) and, in MaleCNS, leg proprioceptors
(chordotonal organs, campaniform sensilla, hair plates) per leg pair and side.
Readouts: MaleCNS = motor neurons of the four playing legs (front L/R hold sticks, hind R kick,
hind L hi-hat). FlyWire has no leg motor neurons, so its readouts are descending neurons per side.

    python -m fly.probe                          # MaleCNS
    python -m fly.probe --connectome flywire     # fallback
    python -m fly.probe --graph-only             # no GPU

Writes runs/probe/<connectome>/report.json and PNG charts.
"""

from __future__ import annotations

import argparse
import json
from time import perf_counter

import numpy as np
import pandas as pd
import scipy.sparse as sp

from fly.connectome import REPO, Connectome, load_flywire, load_malecns

STRONG = 5  # synapses; the "strong path" graph keeps only edges at least this heavy
MAX_HOPS = 8
BIN_MS = 5
MIN_GROUP = 3
LEGS = {"front_left": ("ProLN", "L"), "front_right": ("ProLN", "R"), "hind_left": ("MetaLN", "L"), "hind_right": ("MetaLN", "R")}
LEG_ROLE = {"front_left": "left stick", "front_right": "right stick", "hind_left": "hi-hat pedal", "hind_right": "kick"}
PROPRIO = {"chordotonal organ": "chordo", "campaniform sensilla": "campani", "hair plate": "hairplate"}
LEG_PAIR = {"ProLN": "front", "MesoLN": "mid", "MetaLN": "hind"}


# ---------- groups ----------

def _side(n: pd.DataFrame) -> pd.Series:
    side = n["somaSide"] if "somaSide" in n else pd.Series(index=n.index, dtype=object)
    if "rootSide" in n:
        side = side.fillna(n["rootSide"])
    return side.where(side.isin(["L", "R"]))


def groups(conn: Connectome, fine_jo: bool = False) -> tuple[dict, dict, dict]:
    """-> candidates {name: idx}, readouts {name: idx}, tracked {"KC": idx, "MBON": idx}.

    fine_jo: one candidate per JO cell type and side (JO-EV1_L, ...) and no proprioceptors,
    instead of one per JO family (JO-E_L, ...) plus proprioceptors.
    """
    n = conn.neurons
    side = _side(n)
    types = n["type"].fillna("").astype(str)
    sup = n["superclass"].fillna("").astype(str)

    cand = {}
    if fine_jo:
        fam = types.where(types.str.match(r"^JO-[A-F]") & ~types.str.contains("unclear"))
    else:
        fam = "JO-" + types.str.extract(r"^JO-([A-F])", expand=False)
    for f in sorted(fam.dropna().unique()):
        for s in "LR":
            cand[f"{f}_{s}"] = np.flatnonzero(((fam == f) & (side == s)).to_numpy())
    if "entryNerve" in n and "subclass" in n and not fine_jo:
        sub, nerve = n["subclass"].fillna(""), n["entryNerve"].fillna("")
        for name, short in PROPRIO.items():
            for nv, pair in LEG_PAIR.items():
                for s in "LR":
                    idx = np.flatnonzero(((sub == name) & (nerve == nv) & (side == s)).to_numpy())
                    cand[f"{short}_{pair}_{s}"] = idx
    cand = {k: v for k, v in sorted(cand.items()) if len(v) >= MIN_GROUP}

    if "exitNerve" in n and (sup == "vnc_motor").any():
        nerve = n["exitNerve"].fillna("")
        readouts = {leg: np.flatnonzero(((sup == "vnc_motor") & (nerve == nv) & (side == s)).to_numpy()) for leg, (nv, s) in LEGS.items()}
    else:
        dn = sup.str.startswith("descending").to_numpy()
        readouts = {f"DN_{s}": np.flatnonzero(dn & (side == s).to_numpy()) for s in "LR"}
    tracked = {"KC": np.flatnonzero(types.str.startswith("KC").to_numpy()), "MBON": np.flatnonzero(types.str.startswith("MBON").to_numpy())}
    return cand, readouts, tracked


# ---------- graph pass ----------

def _adjacency(conn: Connectome, min_syn: int) -> sp.csr_matrix:
    keep = np.abs(conn.weight) >= min_syn
    return sp.csr_matrix((np.ones(keep.sum(), dtype=bool), (conn.pre[keep], conn.post[keep])), shape=(conn.size, conn.size))


def hops_from(adj: sp.csr_matrix, sources: np.ndarray) -> np.ndarray:
    """Fewest hops from any source to every neuron (sources = 0, unreached = -1), up to MAX_HOPS."""
    dist = np.full(adj.shape[0], -1, dtype=np.int16)
    dist[sources] = 0
    frontier = np.unique(sources)
    for h in range(1, MAX_HOPS + 1):
        nxt = np.unique(adj[frontier].indices)
        nxt = nxt[dist[nxt] < 0]
        if not len(nxt):
            break
        dist[nxt] = h
        frontier = nxt
    return dist


def _min_hops(dist: np.ndarray, idx: np.ndarray):
    d = dist[idx]
    d = d[d >= 0]
    return int(d.min()) if len(d) else None


def graph_pass(conn: Connectome, cand: dict, readouts: dict, tracked: dict) -> dict:
    out = {}
    for label, min_syn in (("all", 1), ("strong", STRONG)):
        adj = _adjacency(conn, min_syn)
        mbon_dist = hops_from(adj, tracked["MBON"]) if len(tracked["MBON"]) else None
        rows = {}
        for name, idx in cand.items():
            dist = hops_from(adj, idx)
            row = {}
            for r, ridx in readouts.items():
                reached = dist[ridx]
                row[r] = {"min_hops": _min_hops(dist, ridx), "frac_reached": round(float((reached >= 0).mean()), 3) if len(ridx) else 0.0}
                if mbon_dist is not None:
                    to_kc, from_mbon = _min_hops(dist, tracked["KC"]), _min_hops(mbon_dist, ridx)
                    row[r]["via_kc_mbon"] = None if to_kc is None or from_mbon is None else to_kc + 1 + from_mbon
            rows[name] = row
        out[label] = rows
    return out


# ---------- simulation pass ----------

def sim_pass(conn: Connectome, cand: dict, readouts: dict, tracked: dict, rate: float, stim_ms: int,
             pre_ms: int, post_ms: int, trials: int, device: str) -> dict:
    import torch

    from fly.brain import Brain

    brain = Brain(conn, cand, batch=trials, device=device)
    names = list(cand)
    groups_ = {**readouts, **tracked}
    gnames = list(groups_)
    mat = torch.zeros(conn.size, len(gnames), device=device)
    for j, g in enumerate(gnames):
        mat[torch.as_tensor(groups_[g], device=device), j] = 1.0 / max(len(groups_[g]), 1)
    ro_idx = torch.as_tensor(np.concatenate(list(readouts.values())), device=device)
    steps = pre_ms + stim_ms + post_ms

    results = {}
    for v, name in enumerate(names):
        state = brain.init_state()
        drive = torch.zeros(trials, len(names), device=device)
        gen = torch.Generator(device=device).manual_seed(v)
        trace = torch.zeros(steps, len(gnames), device=device)  # mean spikes per neuron per step, over trials
        per_neuron = torch.zeros(len(ro_idx), device=device)
        t0 = perf_counter()
        with torch.no_grad():
            for t in range(steps):
                drive[:, v] = rate if pre_ms <= t < pre_ms + stim_ms else 0.0
                state = brain.step(state, drive, generator=gen)
                spikes = state[2]
                trace[t] = (spikes @ mat).mean(0)
                if t >= pre_ms:
                    per_neuron += spikes[:, ro_idx].sum(0)
        trace = trace.cpu().numpy() * 1000.0  # Hz per neuron
        per_neuron = (per_neuron / (trials * (stim_ms + post_ms) / 1000.0)).cpu().numpy()
        nb = steps // BIN_MS
        binned = trace[: nb * BIN_MS].reshape(nb, BIN_MS, -1).mean(1)
        onset_bin = pre_ms // BIN_MS
        res = {"wall_s": round(perf_counter() - t0, 2), "groups": {}}
        for j, g in enumerate(gnames):
            base = binned[:onset_bin, j]
            thresh = base.mean() + 3 * base.std() + 0.5
            resp = binned[onset_bin:, j]
            above = np.flatnonzero(resp > thresh)
            res["groups"][g] = {
                "latency_ms": int(above[0] * BIN_MS) if len(above) else None,
                "peak_hz": round(float(resp.max()), 2),
                "t_peak_ms": int(resp.argmax() * BIN_MS),
                "baseline_hz": round(float(base.mean()), 2),
                "late_hz": round(float(binned[-100 // BIN_MS:, j].mean()), 2),  # last 100 ms: should fall back to baseline

                "trace_hz": [round(float(x), 2) for x in binned[:, j]],
            }
        off = 0
        for r, ridx in readouts.items():
            rates = per_neuron[off: off + len(ridx)]
            off += len(ridx)
            res["groups"][r]["n_active_5hz"] = int((rates >= 5).sum())
            top = np.argsort(-rates)[:8]
            res["groups"][r]["top_neurons"] = [
                {"bodyId": int(conn.neurons["bodyId"].iat[ridx[i]]), "type": str(conn.neurons["type"].iat[ridx[i]]), "hz": round(float(rates[i]), 1)}
                for i in top if rates[i] > 0
            ]
        results[name] = res
        print(f"  {name:22s} {res['wall_s']:5.1f}s  " + "  ".join(
            f"{r}:{res['groups'][r]['latency_ms']}ms/{res['groups'][r]['peak_hz']}Hz" for r in readouts)
            + "  " + "  ".join(f"{g} peak {res['groups'][g]['peak_hz']}/late {res['groups'][g]['late_hz']}Hz" for g in tracked))
    return results


# ---------- cue choice (D2) ----------

MAX_LATENCY_MS = 60


def pick_cues(report: dict, cand: dict, conn: Connectome) -> dict:
    """One candidate group per drum. A group's score for a drum is its weakest peak rate over the legs that
    play that drum, counting only legs it reaches within MAX_LATENCY_MS. Most constrained drums pick first."""
    from fly.drums import VOICES

    sim = report["sim"]

    def score(c, limbs):
        g = sim[c]["groups"]
        vals = [g[leg]["peak_hz"] if g[leg]["latency_ms"] is not None and g[leg]["latency_ms"] <= MAX_LATENCY_MS else 0.0 for leg in limbs]
        return min(vals)

    viable = {v.name: sorted((c for c in sim if score(c, v.limbs) > 0), key=lambda c: -score(c, v.limbs)) for v in VOICES}
    used, picks = set(), {}
    for v in sorted(VOICES, key=lambda v: len(viable[v.name])):
        options = [c for c in viable[v.name] if c not in used]
        if not options:
            picks[v.name] = None
            continue
        c = options[0]
        used.add(c)
        g = sim[c]["groups"]
        picks[v.name] = {
            "group": c, "limbs": list(v.limbs), "score_hz": score(c, v.limbs),
            "bodyIds": [int(b) for b in conn.neurons["bodyId"].iloc[cand[c]]],
            "legs": {leg: {"latency_ms": g[leg]["latency_ms"], "peak_hz": g[leg]["peak_hz"]} for leg in LEGS},
        }
    return {v.name: picks[v.name] for v in VOICES}


# ---------- charts ----------

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]  # validated categorical order


def _heatmap(ax, data, rows, cols, title, fmt, cmap, empty="—"):
    import numpy.ma as ma

    arr = np.array([[np.nan if x is None else x for x in r] for r in data], dtype=float)
    ax.imshow(ma.masked_invalid(arr), cmap=cmap, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = arr[i, j]
            ax.text(j, i, empty if np.isnan(v) else fmt(v), ha="center", va="center", fontsize=8,
                    color=INK if np.isnan(v) or v <= np.nanmax(arr) * 0.6 else "white")
    ax.set_xticks(range(len(cols)), cols, fontsize=8, color=INK)
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)


def charts(report: dict, out_dir) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])
    blues_r = blues.reversed()
    ro = list(report["readouts"])
    cols = [f"{r}\n({LEG_ROLE.get(r, r)})" for r in ro]
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})

    if "graph" in report:
        cand = list(report["graph"]["all"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(cand) + 1.8))
        for ax, label in zip(axes, ("all", "strong")):
            data = [[report["graph"][label][c][r]["min_hops"] for r in ro] for c in cand]
            _heatmap(ax, data, cand, cols, f"Fewest synapse hops, {label} synapses" + (f" (≥{STRONG})" if label == "strong" else ""), lambda v: f"{v:.0f}", blues_r)
        fig.tight_layout()
        fig.savefig(out_dir / "hops.png", dpi=150)
        plt.close(fig)

    if "sim" not in report:
        return
    sim = report["sim"]
    cand = list(sim)
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(cand) + 1.8))
    lat = [[sim[c]["groups"][r]["latency_ms"] for r in ro] for c in cand]
    peak = [[sim[c]["groups"][r]["peak_hz"] or None for r in ro] for c in cand]
    _heatmap(axes[0], lat, cand, cols, "Response latency after cue onset (ms)", lambda v: f"{v:.0f}", blues_r)
    _heatmap(axes[1], peak, cand, cols, "Peak rate, mean per motor neuron (Hz)", lambda v: f"{v:.1f}", blues)
    fig.tight_layout()
    fig.savefig(out_dir / "response.png", dpi=150)
    plt.close(fig)

    def score(c):  # most legs reached first, then fastest mean latency
        lats = [x for x in (sim[c]["groups"][r]["latency_ms"] for r in ro) if x is not None]
        return (len(lats), -(sum(lats) / len(lats)) if lats else 0.0)
    top = sorted(cand, key=score, reverse=True)[:6]
    meta = report["meta"]
    t = (np.arange(len(sim[top[0]]["groups"][ro[0]]["trace_hz"])) * BIN_MS) - meta["pre_ms"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for ax, c in zip(axes.flat, top):
        ax.axvspan(0, meta["stim_ms"], color=GRID, lw=0)
        for k, r in enumerate(ro):
            ax.plot(t, sim[c]["groups"][r]["trace_hz"], color=SERIES[k], lw=2, label=LEG_ROLE.get(r, r))
        ax.set_title(c, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors=MUTED, labelsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("ms from cue onset (grey = cue on)", fontsize=8, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("Hz per motor neuron", fontsize=8, color=MUTED)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "timecourse.png", dpi=150)
    plt.close(fig)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", choices=["malecns", "flywire"], default="malecns")
    ap.add_argument("--rate", type=float, default=200.0, help="cue firing rate (Hz)")
    ap.add_argument("--stim-ms", type=int, default=50)
    ap.add_argument("--pre-ms", type=int, default=50)
    ap.add_argument("--post-ms", type=int, default=250)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--graph-only", action="store_true")
    ap.add_argument("--no-graph", action="store_true", help="skip the graph pass (e.g. for gain tests)")
    ap.add_argument("--fine-jo", action="store_true", help="one candidate per JO cell type and side; also picks a cue group per drum")
    ap.add_argument("--only", nargs="+", help="simulate just these candidate groups")
    ap.add_argument("--weight-scale", type=float, default=1.0, help="multiply every synapse weight (gain test)")
    ap.add_argument("--drop-kc-kc", action="store_true", help="remove Kenyon cell -> Kenyon cell edges")
    ap.add_argument("--drop-kc-in", action="store_true", help="remove every edge onto Kenyon cells (silences the mushroom body)")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    conn = load_malecns() if args.connectome == "malecns" else load_flywire(annotate=True)
    if args.drop_kc_kc or args.drop_kc_in:
        kc = conn.neurons["type"].fillna("").astype(str).str.startswith("KC").to_numpy()
        keep = ~kc[conn.post] if args.drop_kc_in else ~(kc[conn.pre] & kc[conn.post])
        conn = Connectome(conn.name, conn.neurons, conn.pre[keep], conn.post[keep], conn.weight[keep])
    if args.weight_scale != 1.0:
        conn = Connectome(conn.name, conn.neurons, conn.pre, conn.post, conn.weight * np.float32(args.weight_scale))
    cand, readouts, tracked = groups(conn, fine_jo=args.fine_jo)
    if args.only:
        cand = {k: v for k, v in cand.items() if k in args.only}
    print(f"{conn.name}: {len(cand)} candidate groups, readouts " + ", ".join(f"{k} {len(v)}" for k, v in readouts.items())
          + f", KC {len(tracked['KC'])}, MBON {len(tracked['MBON'])}")
    report = {
        "meta": {"connectome": conn.name, "rate_hz": args.rate, "stim_ms": args.stim_ms, "pre_ms": args.pre_ms,
                 "post_ms": args.post_ms, "trials": args.trials, "bin_ms": BIN_MS, "strong_synapses": STRONG,
                 "weight_scale": args.weight_scale, "drop_kc_kc": args.drop_kc_kc, "drop_kc_in": args.drop_kc_in,
                 "note": "all candidate neurons get fly-brain's no-refractory treatment for stimulated cells"},
        "candidates": {k: {"n": len(v), "types": sorted(set(conn.neurons["type"].iloc[v].dropna().astype(str)))[:12]} for k, v in cand.items()},
        "readouts": {k: {"n": len(v), "role": LEG_ROLE.get(k, k),
                         "types": conn.neurons["type"].iloc[v].fillna("?").value_counts().to_dict()} for k, v in readouts.items()},
    }
    if not args.no_graph:
        t0 = perf_counter()
        report["graph"] = graph_pass(conn, cand, readouts, tracked)
        print(f"graph pass {perf_counter() - t0:.0f}s")

    if not args.graph_only:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        report["sim"] = sim_pass(conn, cand, readouts, tracked, args.rate, args.stim_ms, args.pre_ms, args.post_ms, args.trials, device)

    out_dir = REPO / "runs" / "probe" / (args.connectome + (f"_{args.tag}" if args.tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    if args.fine_jo and "sim" in report:
        cues = {"meta": report["meta"], "max_latency_ms": MAX_LATENCY_MS, "drums": pick_cues(report, cand, conn)}
        (out_dir / "cues.json").write_text(json.dumps(cues, indent=1))
        for drum, p in cues["drums"].items():
            print(f"  {drum:11s} " + (f"{p['group']:14s} n={len(p['bodyIds']):3d} score {p['score_hz']:.1f} Hz" if p else "NO VIABLE GROUP"))
    charts(report, out_dir)
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()

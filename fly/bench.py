"""Phase 0 checks on the server: our 1 ms wrapper against fly-brain's own 0.1 ms model, and speed/VRAM.

    python -m fly.bench sugar                      # FlyWire sugar GRNs at 200 Hz, fly-brain (0.1 ms) vs Brain (1 ms)
    python -m fly.bench speed                      # MaleCNS: VRAM and wall-clock per simulated second
    python -m fly.bench speed --min-synapses 3     # same, dropping edges with fewer than 3 synapses

Results are printed and written to runs/bench/<mode>.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fly.brain import Brain, _flybrain
from fly.connectome import REPO, Connectome, load_flywire, load_malecns

OUT = REPO / "runs" / "bench"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _vram_gb() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"vram_used_gb": round((total - free) / 1024**3, 2), "torch_peak_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2)}


def _flybrain_rates(idx: np.ndarray, rate_hz: float, t_s: float, trials: int, device: str, seed: int) -> tuple[np.ndarray, float]:
    """Per-neuron firing rates (Hz) from fly-brain's own TorchModel at its native dt."""
    fb = _flybrain()
    weights = fb.get_weights(str(fb.path_con), str(fb.path_comp), str(fb.path_wt), csr=True).to(device)
    model = fb.TorchModel(trials, weights.shape[0], fb.DT, fb.MODEL_PARAMS, weights, exc_indices=list(idx), device=device)
    state = model.state_init()
    rates = torch.zeros(trials, weights.shape[0], device=device)
    rates[:, idx] = rate_hz
    gen = torch.Generator(device=device).manual_seed(seed)
    counts = torch.zeros(weights.shape[0], device=device)
    steps = int(round(t_s * 1000 / fb.DT))
    _sync()
    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            state = model(rates, *state, generator=gen)
            counts += state[2].sum(0)
    _sync()
    return (counts / (trials * t_s)).cpu().numpy(), perf_counter() - t0


def _brain_rates(conn: Connectome, idx: np.ndarray, rate_hz: float, t_s: float, trials: int, device: str, seed: int) -> tuple[np.ndarray, float]:
    """Per-neuron firing rates (Hz) through our Brain wrapper at dt = 1 ms."""
    brain = Brain(conn, {"stim": idx}, batch=trials, device=device)
    state = brain.init_state()
    voice = torch.full((trials, 1), rate_hz, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)
    counts = torch.zeros(conn.size, device=device)
    steps = int(round(t_s * 1000))
    _sync()
    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            state = brain.step(state, voice, generator=gen)
            counts += state[2].sum(0)
    _sync()
    return (counts / (trials * t_s)).cpu().numpy(), perf_counter() - t0


def sugar(t_s: float, trials: int, device: str) -> dict:
    fb = _flybrain()
    exp = fb.get_experiment("sugar")
    conn = load_flywire()
    index = {fid: i for i, fid in enumerate(conn.neurons["bodyId"])}
    idx = np.array([index[n] for n in exp["neu_exc"]])
    ref, ref_wall = _flybrain_rates(idx, exp["stim_rate"], t_s, trials, device, seed=0)
    ours, our_wall = _brain_rates(conn, idx, exp["stim_rate"], t_s, trials, device, seed=1)

    others = np.setdiff1d(np.arange(conn.size), idx)  # the stimulated GRNs match trivially
    active = others[(ref[others] > 0) | (ours[others] > 0)]
    top = active[np.argsort(-ref[active])][:20]
    strong = others[ref[others] >= 5]  # rates reliable enough to compare over 1 s
    result = {
        "t_s": t_s, "trials": trials, "stim_hz": exp["stim_rate"], "n_stimulated": len(idx),
        "active_flybrain": int((ref[others] > 0).sum()), "active_ours": int((ours[others] > 0).sum()),
        "pearson_active": float(np.corrcoef(ref[active], ours[active])[0, 1]) if len(active) > 1 else None,
        "n_strong": len(strong),
        "median_ratio_strong": float(np.median(ours[strong] / ref[strong])) if len(strong) else None,
        "wall_s_per_sim_s_flybrain_0.1ms": ref_wall / t_s, "wall_s_per_sim_s_ours_1ms": our_wall / t_s,
        "top20": [{"bodyId": int(conn.neurons["bodyId"].iat[i]), "flybrain_hz": round(float(ref[i]), 1), "ours_hz": round(float(ours[i]), 1)} for i in top],
        **_vram_gb(),
    }
    return result


def speed(connectome: str, t_s: float, batch: int, grad_steps: int, min_synapses: int, device: str) -> dict:
    t0 = perf_counter()
    conn = load_malecns() if connectome == "malecns" else load_flywire()
    load_s = perf_counter() - t0
    n_edges_all = len(conn.pre)
    if min_synapses > 1:
        keep = np.abs(conn.weight) >= min_synapses
        conn = Connectome(conn.name, conn.neurons, conn.pre[keep], conn.post[keep], conn.weight[keep])

    types = conn.neurons["type"].fillna("") if "type" in conn.neurons else None
    if types is not None and types.str.startswith("KC").any():
        kc = types.str.startswith("KC").to_numpy()
        mbon = types.str.startswith("MBON").to_numpy()
        plastic = kc[conn.pre] & mbon[conn.post]  # the D3 candidate, for a realistic gradient check
    else:
        plastic = None
    cue = np.arange(20)  # placeholder cue group; D2 picks the real one after the probe

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = perf_counter()
    brain = Brain(conn, {"cue": cue}, plastic_mask=plastic, batch=batch, device=device)
    build_s = perf_counter() - t0
    after_build = _vram_gb()

    state = brain.init_state()
    voice = torch.full((batch, 1), 200.0, device=device)
    steps = int(round(t_s * 1000))
    _sync()
    t0 = perf_counter()
    with torch.no_grad():
        for _ in range(steps):
            state = brain.step(state, voice)
    _sync()
    sim_s = perf_counter() - t0
    after_sim = _vram_gb()

    grad = {}
    if plastic is not None and plastic.any() and grad_steps:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        state = brain.init_state()
        _sync()
        t0 = perf_counter()
        total = 0.0
        for _ in range(grad_steps):
            state = brain.step(state, voice)
            total = total + state[3].mean()  # voltage: stand-in for the decoded-angle loss
        total.backward()
        _sync()
        grad = {"grad_steps": grad_steps, "grad_wall_s": round(perf_counter() - t0, 2),
                "grad_norm": float(brain.plastic.weight.grad.norm()), **{f"grad_{k}": v for k, v in _vram_gb().items()}}

    return {
        "connectome": conn.name, "neurons": conn.size, "edges_all": n_edges_all, "edges_used": len(conn.pre),
        "min_synapses": min_synapses, "plastic_edges": int(plastic.sum()) if plastic is not None else 0,
        "load_s": round(load_s, 1), "build_s": round(build_s, 1), "batch": batch, "t_s": t_s,
        "wall_s_per_sim_s": round(sim_s / t_s, 2), "after_build": after_build, "after_sim": after_sim, **grad,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sugar", "speed"])
    ap.add_argument("--t", type=float, default=1.0, help="simulated seconds")
    ap.add_argument("--trials", type=int, default=8, help="sugar: trials batched together")
    ap.add_argument("--batch", type=int, default=1, help="speed: batch size")
    ap.add_argument("--grad-steps", type=int, default=150, help="speed: truncated-backprop window to test (ms)")
    ap.add_argument("--min-synapses", type=int, default=1, help="speed: drop edges weaker than this")
    ap.add_argument("--connectome", choices=["malecns", "flywire"], default="malecns")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.mode == "sugar":
        res = sugar(args.t, args.trials, args.device)
    else:
        res = speed(args.connectome, args.t, args.batch, args.grad_steps, args.min_synapses, args.device)
    OUT.mkdir(parents=True, exist_ok=True)
    name = args.mode if args.mode == "sugar" else f"speed_{args.connectome}_min{args.min_synapses}_b{args.batch}"
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

"""Phase 4b: learn the DN -> leg-MN synapses and motor-neuron tone in one least-squares step (no gradients).

Why: gradient training (fly.train with ff credit) removes the untrained fly's twitching but settles just above
hold-still (runs/train/s1). Descent through spikes favours going quiet over a stroke that might land off time.

The rule. The untrained fly plays the training takes at alpha = 0. For every descending neuron d that synapses onto a
leg motor neuron, u_d(t) is the voltage input it delivers per unit synapse weight: fly-brain's synapse (5 ms decay)
times wScale / tauMem, i.e. 6.875e-5 mV per step per (weight x Hz). Each leg MN m gets new DN weights W' (every
synapse keeps its sign, Dale's law) and a constant tone b_m that solve

    min  sum_t ( sum_d W'_dm u_d(t) + b_m  -  [ sum_d W_dm u_d(t) + I*_m(t) ] )^2

so the synapses take over the teacher's current I*(q*), what fly.train injects at alpha = 1 (training loss ~0.001
there). Non-negative least squares per MN on sign-flipped inputs; the tone is the fit's constant. The rest of the
connectome, the encoder and the decoder are unchanged, and the target groove never reaches the fly.

The LIF isn't linear, so the change (W' - W, b) is scaled by each --gains value and validated at alpha = 0 on the
slice fly.train validates on; the best is saved as best.pt, loadable by fly.loop like any fly.train checkpoint.

    python -m fly.fit --out runs/train/f1                     # ~12 min on the 3060 Ti
    python -m fly.fit --out runs/train/f1_shuf --shuffled 1   # the control, fit the same way
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fly.ceiling import record
from fly.train import VAL_SEED, Loss, hold_still, load_take, pack, rest_on_pedal, save, take_paths, validate

SYN_DECAY = 0.8  # fly-brain: conductance *= 1 - dt / tauSyn (5 ms) each step


def fit_mn(u: np.ndarray, target: np.ndarray, w0: np.ndarray):
    """u [T, k] per-unit-weight inputs of one MN's k DN synapses, target [T] its current, w0 [k] signed weights.
    -> (new signed weights [k], tone): NNLS on sign-flipped inputs after centring."""
    from scipy.optimize import nnls

    signs = np.sign(w0)
    a = u * signs
    ma, mt = a.mean(0), target.mean()
    mag, _ = nnls(a - ma, target - mt)
    return mag * signs, float(mt - ma @ mag)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--takes", default="grooves/train/*.mid", help="glob, or several separated by commas")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=40.0, help="fit data per stream")
    ap.add_argument("--stride", type=int, default=2, help="keep every stride-th ms for the fit")
    ap.add_argument("--gains", default="0.5,1,2", help="scales of the fitted change to validate")
    ap.add_argument("--val-ms", type=int, default=10_000)
    ap.add_argument("--window-ms", type=int, default=300, help="validation window (state carries; only speed)")
    ap.add_argument("--shuffled", type=int, default=None, metavar="SEED")
    ap.add_argument("--seed", type=int, default=0, help="packing and Poisson seed of the fit data")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.decoder import Decoder, lif_current
    from fly.loop import BURST_MS, default_lookahead_ms
    from fly.strokes import PREROLL_MS, STROKES
    from fly.wiring import wire

    paths = take_paths(args.takes)
    held_out = [p for p in paths if "heldout" in Path(p).parts]
    if held_out or not paths:
        raise SystemExit(f"never fit on held-out grooves: {held_out}" if held_out else f"no takes match {args.takes}")
    t0 = perf_counter()
    lookahead = default_lookahead_ms()
    wiring = wire(shuffle_seed=args.shuffled, plastic="cue_dn+dn_mn")
    conn = wiring.conn
    brain = wiring.brain(batch=args.batch, device=device, tone=True)
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, conn.size).to(device)
    strokes = json.loads(STROKES.read_text())
    rest_on_pedal(decoder, strokes)
    loss_fn = Loss(decoder)
    rest = decoder.rest.cpu().numpy()
    takes = [load_take(p, strokes, rest, lookahead, BURST_MS, PREROLL_MS) for p in paths]

    # the DN -> leg-MN synapses, in PlasticEdges order
    mn_idx = decoder.idx.cpu().numpy()
    col_of_mn = {int(m): k for k, m in enumerate(mn_idx)}
    p_pre, p_post = conn.pre[wiring.plastic_mask], conn.post[wiring.plastic_mask]
    is_mn = np.isin(p_post, mn_idx)
    dn_mn = np.flatnonzero(is_mn)  # positions in brain.plastic.weight
    dns = np.unique(p_pre[dn_mn])
    col_of_dn = {int(d): k for k, d in enumerate(dns)}
    w0_all = brain.plastic.weight.detach().cpu().numpy().copy()
    print(f"{conn.name}{'' if args.shuffled is None else f' shuffled {args.shuffled}'}: {len(dn_mn)} DN->MN synapses "
          f"from {len(dns)} DNs onto {len(set(p_post[dn_mn].tolist()))} of {len(mn_idx)} leg MNs; {len(takes)} takes; "
          f"built in {perf_counter() - t0:.0f} s", flush=True)

    # fit data: a packing of the training takes other than the validation one
    rates_np, q_np = pack(takes, args.batch, rest, np.random.default_rng(args.seed))
    n = min(rates_np.shape[1], int(args.seconds * 1000))
    rates = torch.as_tensor(rates_np[:, :n], device=device)
    t1 = perf_counter()
    tau = -1.0 / math.log(SYN_DECAY)  # record() filters with exp(-1 / tau) and reads in Hz
    rec = record(brain, rates, {"dn": dns}, tau, args.stride, args.seed)["dn"]
    per_unit = brain.scale / brain.params["tauMem"] / ((1 - SYN_DECAY) * 1000.0)  # Hz-filtered -> mV/step per weight
    u = (rec * per_unit).reshape(-1, len(dns))  # [B * T', n_dn]
    with torch.no_grad():
        q = torch.as_tensor(q_np[:, :n:args.stride][:, :rec.shape[1]], device=device).reshape(-1, q_np.shape[-1])
        i_star = torch.cat([lif_current(decoder.rates_for(c)) for c in q.split(8192)]).cpu().numpy()  # I* at the MNs
    print(f"recorded {n / 1000:.0f} s x {args.batch} streams in {perf_counter() - t1:.0f} s", flush=True)

    # per MN: new weights and tone
    new_w = w0_all.copy()
    tone = np.zeros(len(mn_idx), dtype=np.float32)
    by_mn: dict[int, list[int]] = {}
    for pos in dn_mn:
        by_mn.setdefault(col_of_mn[int(p_post[pos])], []).append(int(pos))
    for m, positions in by_mn.items():
        cols = [col_of_dn[int(p_pre[p])] for p in positions]
        w0 = w0_all[positions]
        target = u[:, cols] @ w0 + i_star[:, m]
        new_w[positions], tone[m] = fit_mn(u[:, cols], target, w0)
    no_dn = [m for m in range(len(mn_idx)) if m not in by_mn]
    tone[no_dn] = i_star[:, no_dn].mean(0)  # MNs no DN reaches: tone only
    change = np.abs(new_w[dn_mn] - w0_all[dn_mn])
    print(f"fit: median |W' - W| {np.median(change):.2f}, max {change.max():.1f} (original median "
          f"{np.median(np.abs(w0_all[dn_mn])):.1f}); {int((new_w[dn_mn] == 0).sum())} synapses at 0; tone "
          f"{tone.min():+.3f} .. {tone.max():+.3f} mV/step", flush=True)

    # validate each gain at alpha 0 on fly.train's validation slice
    val_rates, val_q = pack(takes, args.batch, rest, np.random.default_rng(VAL_SEED))
    n_val = min(args.val_ms, val_rates.shape[1])
    val_rates = torch.as_tensor(val_rates[:, :n_val], device=device)
    val_q = torch.as_tensor(val_q[:, :n_val], device=device)
    hold = hold_still(decoder, loss_fn, val_q)
    settings = {"shuffle_seed": args.shuffled, "lookahead_ms": lookahead, "burst_ms": BURST_MS, "preroll_ms": PREROLL_MS,
                "rest_on_pedal": True, "plastic": "cue_dn+dn_mn", "surrogate_mv": None, "ff_credit": False,
                "method": "lstsq DN->MN + tone (fly.fit)"}
    args.out.mkdir(parents=True, exist_ok=True)
    results = {"hold_still_val": hold}
    best = (math.inf, None)
    for gain in [0.0] + [float(g) for g in args.gains.split(",")]:
        with torch.no_grad():
            brain.plastic.weight.copy_(torch.as_tensor(w0_all + gain * (new_w - w0_all), device=device))
            brain.tone.copy_(torch.as_tensor(gain * tone, device=device))
        t = perf_counter()
        val = validate(brain, decoder, loss_fn, val_rates, val_q, args.window_ms, VAL_SEED)
        results[f"gain {gain:g}"] = val
        print(f"  gain {gain:<4g} validation (alpha 0): {val:.5f}  {'BELOW' if val < hold else 'above'} hold-still "
              f"{hold:.5f}  [{perf_counter() - t:.0f} s]", flush=True)
        if gain > 0 and val < best[0]:
            best = (val, gain)
            save(args.out / "best.pt", brain, settings, gain=gain, val_loss=val)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, **settings, "results": results, "best_gain": best[1],
            "dn_mn_synapses": len(dn_mn), "dns": len(dns), "takes": paths}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    np.savez(args.out / "fit.npz", positions=dn_mn, w0=w0_all[dn_mn], w=new_w[dn_mn], tone=tone)
    print(f"best: gain {best[1]:g}, validation {best[0]:.5f} (hold-still {hold:.5f}); wrote {args.out} "
          f"({(perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

"""Phase 4b: learn the synapses onto the leg motor neurons, and their tone, in one least-squares step (no gradients).

Why: the leg MNs are driven mostly by nerve-cord premotor interneurons. The direct DN -> MN synapses give them only
~15% of the drive they need to fire, so training cue -> DN and DN -> MN (fly.train s1, `--inputs dn` here: f1) can
only quiet the fly. Setting half the DN -> MN synapses to 0 barely changed it. `--inputs all` (default) refits every
synapse onto the 203 playing-leg MNs (plastic set "mn_in"): existing connections only, each keeping its sign
(Dale's law), plus a constant tone per MN. The learning then controls the MNs' whole synaptic input.

The rule. The untrained fly plays the training takes at alpha = 0. For every neuron n that synapses onto a leg MN,
u_n(t) is the voltage input it delivers per unit synapse weight: fly-brain's synapse (5 ms decay) times
wScale / tauMem, i.e. 6.875e-5 mV per step per (weight x Hz). Each leg MN m gets new input weights W' and a tone b_m:

    all:  min  sum_t ( sum_n W'_nm u_n(t) + b_m  -  I*_m(t) )^2                      the whole input becomes I*
    dn:   min  sum_t ( sum_d W'_dm u_d(t) + b_m  -  [ sum_d W_dm u_d(t) + I*_m(t) ] )^2   DN input takes over I*

I*(q*) is the teacher's current, what fly.train injects at alpha = 1 (training loss ~0.001 there); it is 0 at rest,
so an MN whose whole input is I* stays silent between strokes. Non-negative least squares per MN on sign-flipped,
centred inputs (via the k x k normal equations); the constant is the tone. The body gives the brain no feedback and
the MNs barely feed back into it, so re-weighting the MNs' inputs leaves the recorded activity as it was: the fit is
on the distribution the fly will play from. MN -> MN synapses are the exception (their input is what changes), so
they are set to 0 rather than fit. The rest of the connectome, the encoder and the decoder are unchanged,
and the target groove never reaches the fly.

The fit uses the first --fit-frac of the recorded time; R^2 on the rest shows whether it generalises. The LIF isn't
linear, so the fit is scaled by each --gains value (all: W' and b; dn: the change W' - W and b; see weights_at) and
validated at alpha = 0 on the slice fly.train validates on; the best is saved as best.pt, loadable by fly.loop like
any fly.train checkpoint.

    python -m fly.fit --out runs/train/f2                     # ~15 min on the 3060 Ti
    python -m fly.fit --out runs/train/f2_shuf --shuffled 1   # the control, fit the same way
    python -m fly.fit --out runs/train/f1 --inputs dn         # DN -> MN synapses only (f1: no effect)
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
PLASTIC_FOR = {"all": "mn_in", "dn": "cue_dn+dn_mn"}


def fit_mn(u: np.ndarray, target: np.ndarray, w0: np.ndarray):
    """u [T, k] per-unit-weight inputs of one MN's k synapses, target [T] its current, w0 [k] signed weights.
    -> (new signed weights [k], tone): NNLS on sign-flipped inputs after centring."""
    from scipy.optimize import nnls

    signs = np.sign(w0)
    a = u * signs
    ma, mt = a.mean(0), target.mean()
    mag, _ = nnls(a - ma, target - mt)
    return mag * signs, float(mt - ma @ mag)


def fit_mn_gram(u: np.ndarray, target: np.ndarray, w0: np.ndarray, ridge: float = 1e-8):
    """fit_mn's problem solved on the k x k normal equations: fast when T >> k (hundreds of inputs, 10^4+ samples).
    ||A w - y||^2 = ||L^T w - L^-1 c||^2 + const for A^T A = L L^T, c = A^T y; `ridge` (relative to the mean
    diagonal) keeps L defined when an input never fires."""
    from scipy.linalg import solve_triangular
    from scipy.optimize import nnls

    signs = np.sign(w0)
    a = u.astype(np.float64) * signs
    ma, mt = a.mean(0), float(target.mean())
    a -= ma
    g = a.T @ a
    scale = np.trace(g) / len(g)
    if scale <= 0:  # no input ever fired: the tone alone
        return np.zeros_like(w0, dtype=np.float64), mt
    g[np.diag_indices_from(g)] += ridge * scale
    low = np.linalg.cholesky(g)
    mag, _ = nnls(low.T, solve_triangular(low, a.T @ (target - mt), lower=True))
    return mag * signs, float(mt - ma @ mag)


def weights_at(gain: float, w0: np.ndarray, fitted: np.ndarray, inputs: str) -> np.ndarray:
    """The plastic weights at one gain; gain 0 is the untrained fly. all: the fitted input scaled (it is the MN's whole
    input); dn: the change from the original weights scaled. Either way no synapse changes sign (Dale's law)."""
    w = gain * fitted if inputs == "all" else w0 + gain * (fitted - w0)
    if gain == 0:
        w = w0.copy()
    return np.where(np.sign(w0) >= 0, np.maximum(w, 0), np.minimum(w, 0)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", default="all", choices=tuple(PLASTIC_FOR),
                    help="refit every synapse onto the leg MNs (all) or only the DN -> MN ones (dn)")
    ap.add_argument("--takes", default="grooves/train/*.mid", help="glob, or several separated by commas")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=40.0, help="fit data per stream")
    ap.add_argument("--stride", type=int, default=4, help="keep every stride-th ms for the fit")
    ap.add_argument("--fit-frac", type=float, default=0.7, help="fit on this share of the time, R^2 on the rest")
    ap.add_argument("--gains", default="1,1.5,2,3,5",
                    help="scales of the fit to validate (above 1 lifts stroke currents the fit shrank below threshold)")
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
    from fly.wiring import DESCENDING, wire

    paths = take_paths(args.takes)
    held_out = [p for p in paths if "heldout" in Path(p).parts]
    if held_out or not paths:
        raise SystemExit(f"never fit on held-out grooves: {held_out}" if held_out else f"no takes match {args.takes}")
    t0 = perf_counter()
    lookahead = default_lookahead_ms()
    plastic = PLASTIC_FOR[args.inputs]
    wiring = wire(shuffle_seed=args.shuffled, plastic=plastic)
    conn = wiring.conn
    brain = wiring.brain(batch=args.batch, device=device, tone=True)
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, conn.size).to(device)
    strokes = json.loads(STROKES.read_text())
    rest_on_pedal(decoder, strokes)
    loss_fn = Loss(decoder)
    rest = decoder.rest.cpu().numpy()
    takes = [load_take(p, strokes, rest, lookahead, BURST_MS, PREROLL_MS) for p in paths]

    # the synapses onto leg MNs that get refit, in PlasticEdges order
    mn_idx = decoder.idx.cpu().numpy()
    col_of_mn = {int(m): k for k, m in enumerate(mn_idx)}
    p_pre, p_post = conn.pre[wiring.plastic_mask], conn.post[wiring.plastic_mask]
    onto_mn = np.flatnonzero(np.isin(p_post, mn_idx))  # positions in brain.plastic.weight
    pres = np.unique(p_pre[onto_mn])
    col_of_pre = {int(n): k for k, n in enumerate(pres)}
    n_dn = int((conn.neurons["superclass"].to_numpy()[pres] == DESCENDING).sum())
    w0_all = brain.plastic.weight.detach().cpu().numpy().copy()
    print(f"{conn.name}{'' if args.shuffled is None else f' shuffled {args.shuffled}'} ({args.inputs}): "
          f"{len(onto_mn)} synapses onto {len(set(p_post[onto_mn].tolist()))} of {len(mn_idx)} leg MNs from "
          f"{len(pres)} neurons ({n_dn} DNs); {len(takes)} takes; built in {perf_counter() - t0:.0f} s", flush=True)

    # fit data: a packing of the training takes other than the validation one
    rates_np, q_np = pack(takes, args.batch, rest, np.random.default_rng(args.seed))
    n = min(rates_np.shape[1], int(args.seconds * 1000))
    rates = torch.as_tensor(rates_np[:, :n], device=device)
    t1 = perf_counter()
    tau = -1.0 / math.log(SYN_DECAY)  # record() filters with exp(-1 / tau) and reads in Hz
    u = record(brain, rates, {"pre": pres}, tau, args.stride, args.seed)["pre"]  # [B, T', n_pre]
    u *= brain.scale / brain.params["tauMem"] / ((1 - SYN_DECAY) * 1000.0)  # -> mV/step per unit weight
    with torch.no_grad():
        q = torch.as_tensor(q_np[:, :n:args.stride][:, :u.shape[1]], device=device)
        i_star = torch.cat([lif_current(decoder.rates_for(c)) for c in q.reshape(-1, q.shape[-1]).split(8192)])
        i_star = i_star.reshape(*q.shape[:2], -1).cpu().numpy()  # [B, T', M] the teacher's current at each MN
    split = int(args.fit_frac * u.shape[1])
    print(f"recorded {n / 1000:.0f} s x {args.batch} streams ({u.nbytes / 2**30:.1f} GB) in {perf_counter() - t1:.0f} s",
          flush=True)

    # per MN: new input weights and tone, fit on the first part of the time, scored on the rest
    t2 = perf_counter()
    new_w = w0_all.copy()
    tone = np.zeros(len(mn_idx), dtype=np.float32)
    by_mn: dict[int, list[int]] = {}
    for pos in onto_mn:
        by_mn.setdefault(col_of_mn[int(p_post[pos])], []).append(int(pos))
    ss_res = ss_tot = 0.0
    is_leg_mn = np.isin(p_pre, mn_idx)  # MN -> MN synapses: their input changes once the fit is in, so they go to 0
    new_w[onto_mn[is_leg_mn[onto_mn]]] = 0.0
    for m in range(len(mn_idx)):
        positions = [p for p in by_mn.get(m, []) if not is_leg_mn[p]]
        cols = [col_of_pre[int(p_pre[p])] for p in positions]
        w0 = w0_all[positions]
        x = u[:, :, cols]  # [B, T', k]
        target = i_star[:, :, m] + (x @ w0 if args.inputs == "dn" else 0.0)
        n_fit, n_test = x.shape[0] * split, x.shape[0] * (x.shape[1] - split)  # explicit: k may be 0
        x_fit, y_fit = x[:, :split].reshape(n_fit, len(cols)), target[:, :split].reshape(n_fit)
        x_test, y_test = x[:, split:].reshape(n_test, len(cols)), target[:, split:].reshape(n_test)
        if positions:
            w, tone[m] = fit_mn_gram(x_fit, y_fit, w0)
            new_w[positions] = w
        else:  # nothing synapses onto this MN: tone only
            w, tone[m] = np.zeros(0), float(y_fit.mean())
        pred = x_test @ w + tone[m]
        ss_res += float(((y_test - pred) ** 2).sum())
        ss_tot += float(((y_test - y_fit.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    change = np.abs(new_w[onto_mn] - w0_all[onto_mn])
    print(f"fit in {perf_counter() - t2:.0f} s: R^2 of the MN currents on held-out time {r2:+.3f}; median |W' - W| "
          f"{np.median(change):.2f}, max {change.max():.1f} (original median {np.median(np.abs(w0_all[onto_mn])):.1f}); "
          f"{int((new_w[onto_mn] == 0).sum())} of {len(onto_mn)} synapses at 0 ({int(is_leg_mn[onto_mn].sum())} "
          f"MN -> MN); tone {tone.min():+.3f} .. "
          f"{tone.max():+.3f} mV/step", flush=True)
    del u

    # validate each gain at alpha 0 on fly.train's validation slice
    val_rates, val_q = pack(takes, args.batch, rest, np.random.default_rng(VAL_SEED))
    n_val = min(args.val_ms, val_rates.shape[1])
    val_rates = torch.as_tensor(val_rates[:, :n_val], device=device)
    val_q = torch.as_tensor(val_q[:, :n_val], device=device)
    hold = hold_still(decoder, loss_fn, val_q)
    settings = {"shuffle_seed": args.shuffled, "lookahead_ms": lookahead, "burst_ms": BURST_MS, "preroll_ms": PREROLL_MS,
                "rest_on_pedal": True, "plastic": plastic, "surrogate_mv": None, "ff_credit": False,
                "method": f"lstsq {args.inputs} inputs onto leg MNs + tone (fly.fit)"}
    args.out.mkdir(parents=True, exist_ok=True)
    results = {"hold_still_val": hold, "r2_heldout_time": r2}
    best = (math.inf, None)
    for gain in [0.0] + [float(g) for g in args.gains.split(",")]:
        w = weights_at(gain, w0_all, new_w, args.inputs)
        with torch.no_grad():
            brain.plastic.weight.copy_(torch.as_tensor(w, device=device))
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
            "synapses": len(onto_mn), "presynaptic_neurons": len(pres), "presynaptic_dns": n_dn, "takes": paths}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    np.savez(args.out / "fit.npz", positions=onto_mn, w0=w0_all[onto_mn], w=new_w[onto_mn], tone=tone)
    print(f"best: gain {best[1]:g}, validation {best[0]:.5f} (hold-still {hold:.5f}); wrote {args.out} "
          f"({(perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

"""Could learning the DN -> leg-MN synapses make the fly drum? Linear ceilings on the untrained fly, no training.

The untrained fly (alpha = 0) plays the training takes. Every ms the descending neurons' (DNs) and leg motor
neurons' (MNs) spikes are filtered (exponential, `tau` ms, in Hz) and kept every `stride` ms. The target is the
teacher's MN rates, decoder.rates_for(q*). Each readout is fit on the first 70% of the time and scored on the last
30% with the training loss itself (decoded targets vs q*; predicted rates clipped at 0), next to hold-still and the
untrained fly:

  cues       ridge from the cue rates (the score, shifted early), at several filter lengths: a perfect pathway
  all DNs    ridge from every DN: is the groove in the DN population at all?
  DN->MN     each MN from its own presynaptic DNs only, signs kept, plus a constant (tone): the most that
             re-weighting the 3,006 DN -> leg-MN synapses could do, if MN rate followed its input linearly
  untrained  the fly's own MN rates

A readout "below hold-still" means that information reaches that layer in time; "DN->MN" below hold-still means
the synapses fly.train changes are enough in principle. Nothing here is the fly playing: it's a bound.

    python -m fly.ceiling                              # ~3 min on the 3060 Ti
    python -m fly.ceiling --takes 'grooves/train/SD_*.mid' --seconds 10
Writes runs/ceiling/<tag>.json and the DN->MN fit (edge rows, weights) to runs/ceiling/<tag>_dn_mn.npz.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fly.train import VAL_SEED, Loss, hold_still, load_take, pack, rest_on_pedal, take_paths


def record(brain, rates: torch.Tensor, idx: dict, tau_ms: float, stride: int, seed: int) -> dict:
    """Forward at alpha 0; -> {group: [B, T // stride, n] filtered rates in Hz (float32, CPU)}."""
    decay = float(np.exp(-1.0 / tau_ms))
    gen = torch.Generator(device=rates.device).manual_seed(seed)
    state = brain.init_state()
    idx = {g: torch.as_tensor(i, dtype=torch.long, device=rates.device) for g, i in idx.items()}
    filt = {g: torch.zeros(rates.shape[0], len(i), device=rates.device) for g, i in idx.items()}
    out = {g: [] for g in idx}
    with torch.no_grad():
        for t in range(rates.shape[1]):
            state = brain.step(state, rates[:, t], None, generator=gen)
            for g, i in idx.items():
                filt[g] = filt[g] * decay + state[2][:, i] * ((1 - decay) * 1000.0)
                if t % stride == 0:
                    out[g].append(filt[g].cpu())
    return {g: torch.stack(v, 1).numpy() for g, v in out.items()}


def smooth(x: np.ndarray, tau_ms: float, stride: int) -> np.ndarray:
    """Exponential filter along time of x [B, T, n] sampled every `stride` ms."""
    a = float(np.exp(-stride / tau_ms))
    y, acc = np.empty_like(x), np.zeros_like(x[:, 0])
    for t in range(x.shape[1]):
        acc = a * acc + (1 - a) * x[:, t]
        y[:, t] = acc
    return y


def ridge(x_fit, y_fit, x_test, lams=(1e-4, 1e-2, 1)) -> dict:
    """-> {lam: prediction on x_test}; lam relative to the mean feature variance."""
    mx, my = x_fit.mean(0), y_fit.mean(0)
    xf, yf = x_fit - mx, y_fit - my
    gram = xf.T @ xf
    scale = np.trace(gram) / gram.shape[0]
    rhs = xf.T @ yf
    return {lam: (x_test - mx) @ np.linalg.solve(gram + lam * scale * np.eye(len(gram)), rhs) + my for lam in lams}


def fit_dn_mn(dn_fit, y_fit, dn_test, edges: list[tuple[int, int, float]], n_mn: int):
    """Per MN, non-negative least squares on its presynaptic DNs' rates times each synapse's sign, after centring
    (the constant is the MN's tone). -> (prediction [T, n_mn], weights per edge in edges order)."""
    from scipy.optimize import nnls

    pred = np.tile(y_fit.mean(0), (len(dn_test), 1))
    weights = np.zeros(len(edges))
    by_mn: dict[int, list[int]] = {}
    for k, (_, m, _) in enumerate(edges):
        by_mn.setdefault(m, []).append(k)
    for m, ks in by_mn.items():
        cols = [edges[k][0] for k in ks]
        signs = np.array([np.sign(edges[k][2]) for k in ks])
        a_fit, a_test = dn_fit[:, cols] * signs, dn_test[:, cols] * signs
        ma, my = a_fit.mean(0), y_fit[:, m].mean()
        w, _ = nnls(a_fit - ma, y_fit[:, m] - my)
        pred[:, m] = (a_test - ma) @ w + my
        weights[ks] = w * signs
    return pred, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", default="grooves/train/*.mid", help="glob, or several separated by commas")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=20.0, help="per stream")
    ap.add_argument("--tau-ms", type=float, default=20.0, help="spike filter (the decoder's)")
    ap.add_argument("--stride", type=int, default=5, help="keep every stride-th ms")
    ap.add_argument("--fit-frac", type=float, default=0.7)
    ap.add_argument("--tag", default="ceiling")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.decoder import Decoder
    from fly.loop import BURST_MS, default_lookahead_ms
    from fly.strokes import PREROLL_MS, STROKES
    from fly.wiring import DESCENDING, wire

    t0 = perf_counter()
    wiring = wire(plastic="cue_dn+dn_mn")
    conn = wiring.conn
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, conn.size).to(device)
    strokes = json.loads(STROKES.read_text())
    rest_on_pedal(decoder, strokes)
    loss_fn = Loss(decoder)
    rest = decoder.rest.cpu().numpy()
    takes = [load_take(p, strokes, rest, default_lookahead_ms(), BURST_MS, PREROLL_MS) for p in take_paths(args.takes)]
    rates_np, q_np = pack(takes, args.batch, rest, np.random.default_rng(VAL_SEED))
    n = min(rates_np.shape[1], int(args.seconds * 1000))
    rates = torch.as_tensor(rates_np[:, :n], device=device)
    qstar = torch.as_tensor(q_np[:, :n], device=device)

    mn_idx = decoder.idx.cpu().numpy()
    dn_idx = np.flatnonzero((conn.neurons["superclass"] == DESCENDING).to_numpy())
    brain = wiring.brain(batch=args.batch, device=device)
    print(f"{len(takes)} takes, {n / 1000:.0f} s x {args.batch} streams, {len(dn_idx)} DNs, {len(mn_idx)} leg MNs; "
          f"built in {perf_counter() - t0:.0f} s", flush=True)

    t1 = perf_counter()
    rec = record(brain, rates, {"dn": dn_idx, "mn": mn_idx}, args.tau_ms, args.stride, VAL_SEED)
    print(f"recorded in {perf_counter() - t1:.0f} s", flush=True)

    s = args.stride
    q_s = qstar[:, ::s][:, :rec["dn"].shape[1]]
    target = decoder.rates_for(q_s.reshape(-1, q_s.shape[-1])).reshape(*q_s.shape[:2], -1).cpu().numpy()
    cues = rates_np[:, :n:s][:, :rec["dn"].shape[1]].astype(np.float32)
    cue_feats = np.concatenate([smooth(cues, tau, s) for tau in (20, 60, 150)], axis=-1)
    split = int(args.fit_frac * target.shape[1])

    def flat(x, part):  # [B, T, n] -> [B * T_part, n]
        x = x[:, :split] if part == "fit" else x[:, split:]
        return x.reshape(-1, x.shape[-1])

    q_test = q_s[:, split:].reshape(-1, q_s.shape[-1])

    def solo(pred_rates: np.ndarray) -> float:  # a firing rate can't be negative
        with torch.no_grad():
            q = decoder.targets(torch.as_tensor(np.maximum(pred_rates, 0), dtype=torch.float32, device=device))
            return float(loss_fn(q, q_test))

    y_fit, y_test = flat(target, "fit"), flat(target, "test")
    report = {"args": vars(args), "takes": len(takes), "dns": len(dn_idx), "leg_mns": len(mn_idx),
              "hold_still": hold_still(decoder, loss_fn, q_s[:, split:]), "untrained": solo(flat(rec["mn"], "test"))}
    for name, x in (("cues", cue_feats), ("all DNs", rec["dn"])):
        preds = ridge(flat(x, "fit"), y_fit, flat(x, "test"))
        losses = {lam: solo(p) for lam, p in preds.items()}
        report[name] = min(losses.values())
        report[f"{name} by lambda"] = losses

    is_dn = np.zeros(conn.size, dtype=bool)
    is_dn[dn_idx] = True
    col_of_dn = {int(d): k for k, d in enumerate(dn_idx)}
    col_of_mn = {int(m): k for k, m in enumerate(mn_idx)}
    rows = np.flatnonzero(wiring.plastic_mask & is_dn[conn.pre] & np.isin(conn.post, mn_idx))
    edges = [(col_of_dn[int(conn.pre[r])], col_of_mn[int(conn.post[r])], float(conn.weight[r])) for r in rows]
    pred, weights = fit_dn_mn(flat(rec["dn"], "fit"), y_fit, flat(rec["dn"], "test"), edges, len(mn_idx))
    report["DN->MN"] = solo(pred)
    report["dn_mn_edges"] = len(edges)
    report["mns_with_dn_input"] = len({m for _, m, _ in edges})

    hold = report["hold_still"]
    print(f"\nsolo loss on the last {1 - args.fit_frac:.0%} of the time (training loss, alpha 0):")
    for name in ("hold_still", "untrained", "cues", "all DNs", "DN->MN"):
        v = report[name]
        tag = "" if name == "hold_still" else f"  {'BELOW' if v < hold else 'above'} hold-still ({v / hold - 1:+.1%})"
        print(f"  {name:<12}{v:.5f}{tag}")
    print(f"  ({report['dn_mn_edges']} DN->MN edges onto {report['mns_with_dn_input']} of {len(mn_idx)} leg MNs)")

    out = Path("runs/ceiling")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.tag}.json").write_text(json.dumps(report, indent=1, default=str))
    np.savez(out / f"{args.tag}_dn_mn.npz", edge_rows=rows, fit_weights=weights, original=conn.weight[rows])
    print(f"wrote {out / args.tag}.json ({(perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

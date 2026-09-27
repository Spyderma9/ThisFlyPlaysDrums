"""Why doesn't training move the fly's solo score? Gradient health checks on the real wiring, no training.

Tests the candidate explanations for flat training (t3-t5), each with its own measurement:
  C1 surrogate too narrow   membrane-voltage distance to threshold in DNs and leg MNs; gradient stats per surrogate width
  C2 reachability           solo loss with every cue -> DN weight zeroed (can the trainable set even reach hold-still?)
  C3 truncation             the same gradient stats with 300 ms windows
  C4 optimizer noise        agreement of gradients across Poisson seeds (cosine, signal fraction)
  C5 validation noise       untrained validation under three Poisson seeds
  C6 broken gradient path   fraction of trainable edges with a non-zero gradient
and a line search per config: does a step along the averaged gradient lower the solo loss at all?

The forward simulation doesn't depend on the surrogate or on which edges are marked trainable (at the untrained
weights), so warm-up states are computed once per (seed, window) and reused by every config.

    python -m fly.diagnose                          # ~10 min on the 3060 Ti
    python -m fly.diagnose --takes 'grooves/train/SD_*.mid' --seeds 2 --no-val     # quick
Writes runs/diagnose/<tag>.json and prints a table.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from fly.train import VAL_SEED, Loss, detach, load_take, pack, rest_on_pedal, run_window, take_paths, validate


def _warm(brain, decoder, rates, starts, seeds, device):
    """Forward (no grad) from 0 to each window start. -> {(seed, start): (state, decoder rates, generator state)}."""
    out = {}
    for seed in seeds:
        gen = torch.Generator(device=device).manual_seed(seed)
        state, r = brain.init_state(), decoder.init_state(rates.shape[0])
        t = 0
        with torch.no_grad():
            for start in sorted(starts):
                for k in range(t, start):
                    state = brain.step(state, rates[:, k], None, generator=gen)
                    r, _ = decoder(r, state[2])
                t = start
                out[seed, start] = (detach(tuple(x.clone() if torch.is_tensor(x) else x for x in state)), r.clone(),
                                    gen.get_state())
    return out


def _voltage_report(brain, decoder, rates, warm, starts, window, idx: dict, device, threshold_gap=7.0):
    """Share of (neuron, ms) samples within 1 / 3 / 7 mV of threshold, per group, over the windows (forward only)."""
    v_thr = brain.params["vThreshold"]
    counts = {g: np.zeros(4) for g in idx}
    spikes = {g: 0.0 for g in idx}
    with torch.no_grad():
        for (seed, start), (state, r, gstate) in warm.items():
            gen = torch.Generator(device=device)
            gen.set_state(gstate)
            s = state
            for k in range(start, start + window):
                s = brain.step(s, rates[:, k], None, generator=gen)
                gap = (v_thr - s[3]).clamp(min=0)
                for g, i in idx.items():
                    gi = gap[:, i]
                    counts[g] += np.array([gi.numel(), (gi < 1).sum().item(), (gi < 3).sum().item(), (gi < threshold_gap).sum().item()])
                    spikes[g] += s[2][:, i].sum().item()
    n_ms = len(warm) * window * rates.shape[0]
    return {g: {"within_1mV": c[1] / c[0], "within_3mV": c[2] / c[0], "within_7mV": c[3] / c[0],
                "rate_hz": spikes[g] / (len(idx[g]) * n_ms) * 1000} for g, c in counts.items()}


def _grads(brain, decoder, loss_fn, rates, qstar, warm, window, device):
    """Gradient of the solo (alpha = 0) loss w.r.t. the plastic weights, per (seed, window)."""
    w = brain.plastic.weight
    out = {}
    for (seed, start), (state, r, gstate) in warm.items():
        gen = torch.Generator(device=device)
        gen.set_state(gstate)
        w.grad = None
        _, _, loss = run_window(brain, decoder, loss_fn, state, r, rates[:, start:start + window],
                                qstar[:, start:start + window], 0.0, gen)
        loss.backward()
        out[seed, start] = w.grad.detach().clone()
    return out


def _solo_loss(brain, decoder, loss_fn, rates, qstar, warm, window, device) -> float:
    total = 0.0
    with torch.no_grad():
        for (seed, start), (state, r, gstate) in warm.items():
            gen = torch.Generator(device=device)
            gen.set_state(gstate)
            total += float(run_window(brain, decoder, loss_fn, state, r, rates[:, start:start + window],
                                      qstar[:, start:start + window], 0.0, gen)[2])
    return total / len(warm)


def _grad_stats(g: dict, classes: dict) -> dict:
    stack = torch.stack(list(g.values()))  # [S*K, E]
    seeds = sorted({s for s, _ in g})
    starts = sorted({t for _, t in g})
    cos = []
    for t in starts:  # agreement between Poisson seeds on the same window
        vs = [g[s, t] for s in seeds]
        for a, b in itertools.combinations(vs, 2):
            if a.norm() > 0 and b.norm() > 0:
                cos.append(float(torch.dot(a, b) / (a.norm() * b.norm())))
    mean = stack.mean(0)
    signal = float(mean.pow(2).sum() / stack.pow(2).sum(1).mean()) if stack.abs().sum() > 0 else 0.0
    out = {"nonzero_frac": float((stack != 0).float().mean()), "grad_norm": float(stack.norm(dim=1).mean()),
           "seed_cosine": float(np.mean(cos)) if cos else 0.0, "signal_frac": signal, "per_class": {}}
    for name, m in classes.items():
        sub = stack[:, m]
        out["per_class"][name] = {"edges": int(m.sum()), "nonzero_frac": float((sub != 0).float().mean()),
                                  "grad_norm": float(sub.norm(dim=1).mean())}
    return out


def _line_search(brain, decoder, loss_fn, rates, qstar, warm, window, direction, device, steps=(0.5, 1, 2, 4)) -> dict:
    """Step every weight by up to `step` synapse units along -direction (sign-locked), measure the solo loss."""
    w = brain.plastic.weight
    w0, sign0 = w.detach().clone(), torch.sign(w.detach())
    base = _solo_loss(brain, decoder, loss_fn, rates, qstar, warm, window, device)
    scale = direction.abs().max()
    res = {"base": base}
    for step in steps:
        with torch.no_grad():
            new = w0 - step * direction / scale if scale > 0 else w0
            w.copy_(torch.where(sign0 >= 0, new.clamp(min=0), new.clamp(max=0)))
        res[str(step)] = _solo_loss(brain, decoder, loss_fn, rates, qstar, warm, window, device)
    with torch.no_grad():
        w.copy_(w0)
    res["best_rel_change"] = min(res[str(s)] for s in steps) / base - 1 if base > 0 else 0.0
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--takes", default="grooves/train/*.mid")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--windows", type=int, default=4, help="windows per seed, spread over the first seconds")
    ap.add_argument("--surrogates", default="1,3,7", help="surrogate widths in mV (1 = fly-brain's)")
    ap.add_argument("--plastic", default="cue_dn,cue_dn+dn_mn")
    ap.add_argument("--credit", default="ff", help="'full' (BPTT through every loop), 'ff' (one hop MN -> DN), or both")
    ap.add_argument("--tones", default="-3,-2,-1,-0.5,0.5", help="uniform leg-MN tone values (mV/step) for the tone scan")
    ap.add_argument("--long-window-ms", type=int, default=300, help="also test this window (C3) at the middle surrogate")
    ap.add_argument("--no-val", action="store_true", help="skip the full validations (C2 zeroed weights, C5 seed noise)")
    ap.add_argument("--val-ms", type=int, default=10_000)
    ap.add_argument("--tag", default="diagnose")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.connectome import load_malecns
    from fly.decoder import Decoder
    from fly.loop import BURST_MS, default_lookahead_ms
    from fly.strokes import PREROLL_MS, STROKES
    from fly.wiring import wire

    t0 = perf_counter()
    conn = load_malecns()
    strokes = json.loads(STROKES.read_text())
    base_wiring = wire(conn, plastic="cue_dn")
    decoder = Decoder(base_wiring.leg_mns, base_wiring.mn_types, base_wiring.conn.size).to(device)
    rest_on_pedal(decoder, strokes)
    loss_fn = Loss(decoder)
    rest = decoder.rest.cpu().numpy()
    takes = [load_take(p, strokes, rest, default_lookahead_ms(), BURST_MS, PREROLL_MS) for p in take_paths(args.takes)]
    rates_np, q_np = pack(takes, args.batch, rest, np.random.default_rng(VAL_SEED))
    n = min(rates_np.shape[1], max(args.val_ms, 4000))
    rates = torch.as_tensor(rates_np[:, :n], device=device)
    qstar = torch.as_tensor(q_np[:, :n], device=device)
    widths = [float(x) for x in args.surrogates.split(",")]
    plastics = args.plastic.split(",")
    window = 150
    starts = [400 + i * 700 for i in range(args.windows)]  # after the pre-roll, spread over ~3 s
    if starts[-1] + args.long_window_ms > n:
        raise SystemExit(f"groove slice too short ({n} ms) for {args.windows} windows")
    seeds = list(range(1, args.seeds + 1))
    report = {"args": vars(args), "device": device, "starts_ms": starts, "seeds": seeds, "configs": {}}
    print(f"loaded in {perf_counter() - t0:.0f} s; {len(takes)} takes, windows at {starts} ms x seeds {seeds}", flush=True)

    # hold-still: the decoder at rest (no motor activity) against the teacher, over the same windows
    with torch.no_grad():
        hold = np.mean([float(loss_fn(decoder.rest.expand(args.batch, -1), qstar[:, k]))
                        for s in starts for k in range(s, s + window)])
    report["hold_still_windows"] = hold

    brain0 = base_wiring.brain(batch=args.batch, device=device)
    warm = _warm(brain0, decoder, rates, starts, seeds, device)
    mn_idx = torch.as_tensor(np.concatenate(list(base_wiring.leg_mns.values())), device=device)
    dn_idx = torch.as_tensor(np.flatnonzero((conn.neurons["superclass"] == "descending_neuron").to_numpy()), device=device)
    report["voltage"] = _voltage_report(brain0, decoder, rates, warm, starts, window, {"DN": dn_idx, "leg_MN": mn_idx}, device)
    print(f"voltage (share of neuron-ms within 1/3/7 mV of threshold; rate): "
          + "  ".join(f"{g}: {v['within_1mV']:.4f}/{v['within_3mV']:.4f}/{v['within_7mV']:.3f}, {v['rate_hz']:.2f} Hz"
                      for g, v in report["voltage"].items()), flush=True)
    report["untrained_solo_windows"] = _solo_loss(brain0, decoder, loss_fn, rates, qstar, warm, window, device)
    print(f"solo loss over the windows: untrained {report['untrained_solo_windows']:.5f}, hold-still {hold:.5f}", flush=True)

    # C2: zero every cue -> DN weight
    with torch.no_grad():
        w0 = brain0.plastic.weight.detach().clone()
        brain0.plastic.weight.zero_()
    report["zeroed_cue_dn_windows"] = _solo_loss(brain0, decoder, loss_fn, rates, qstar, warm, window, device)
    with torch.no_grad():
        brain0.plastic.weight.copy_(w0)
    print(f"C2: cue->DN weights all zero: {report['zeroed_cue_dn_windows']:.5f}", flush=True)
    del brain0

    # tone scan: can quieting (or exciting) every leg motor neuron move the solo loss toward hold-still?
    brain_t = base_wiring.brain(batch=args.batch, device=device, tone=True)
    report["tone_scan"] = {}
    for tone in [float(x) for x in args.tones.split(",")]:
        with torch.no_grad():
            brain_t.tone.fill_(tone)
        report["tone_scan"][tone] = _solo_loss(brain_t, decoder, loss_fn, rates, qstar, warm, window, device)
    print("tone scan (uniform leg-MN tone -> solo loss): "
          + "  ".join(f"{k:+g}: {v:.5f}" for k, v in report["tone_scan"].items()), flush=True)
    del brain_t

    # C1 / C3 / C4 / C6: gradient stats and a line search per config
    credits = args.credit.split(",")
    configs = [(p, wdt, window, c) for c in credits for p in plastics for wdt in widths]
    configs.append((plastics[-1], widths[len(widths) // 2], args.long_window_ms, credits[-1]))
    for plastic, width, win, credit in configs:
        t = perf_counter()
        wiring = base_wiring if plastic == "cue_dn" else wire(conn, plastic=plastic)
        brain = wiring.brain(batch=args.batch, device=device, surrogate_mv=None if width == 1 else width,
                             ff_credit=credit == "ff")
        classes = {}
        if plastic != "cue_dn":  # which plastic edges belong to which class, in PlasticEdges order
            pre, post = wiring.conn.pre[wiring.plastic_mask], wiring.conn.post[wiring.plastic_mask]
            is_mn = np.zeros(wiring.conn.size, dtype=bool)
            is_mn[np.concatenate(list(wiring.leg_mns.values()))] = True
            classes = {"dn_mn": torch.as_tensor(is_mn[post], device=device),
                       "cue_dn": torch.as_tensor(~is_mn[post], device=device)}
        g = _grads(brain, decoder, loss_fn, rates, qstar, warm, win, device)
        stats = _grad_stats(g, classes)
        direction = torch.stack(list(g.values())).mean(0)
        stats["line_search"] = _line_search(brain, decoder, loss_fn, rates, qstar, warm, win, direction, device)
        key = f"{plastic} | {credit} | surrogate {width:g} mV | window {win} ms"
        report["configs"][key] = stats
        ls = stats["line_search"]
        print(f"{key:<52} nonzero {stats['nonzero_frac']:.3f}  |g| {stats['grad_norm']:.2e}  "
              f"seed-cos {stats['seed_cosine']:+.2f}  signal {stats['signal_frac']:.2f}  "
              f"line search {ls['base']:.5f} -> best {min(ls[str(s)] for s in (0.5, 1, 2, 4)):.5f} "
              f"({ls['best_rel_change']:+.1%})  [{perf_counter() - t:.0f} s]", flush=True)
        for name, c in stats["per_class"].items():
            print(f"    {name}: {c['edges']} edges, nonzero {c['nonzero_frac']:.3f}, |g| {c['grad_norm']:.2e}")
        del brain

    if not args.no_val:  # C5 seed noise and C2 on the full validation slice (the numbers training reports)
        brain = base_wiring.brain(batch=args.batch, device=device)
        vr, vq = rates[:, :args.val_ms], qstar[:, :args.val_ms]
        report["val_untrained_by_seed"] = {s: validate(brain, decoder, loss_fn, vr, vq, 150, s) for s in (VAL_SEED, 1, 2)}
        with torch.no_grad():
            brain.plastic.weight.zero_()
        report["val_zeroed_cue_dn"] = validate(brain, decoder, loss_fn, vr, vq, 150, VAL_SEED)
        print(f"C5 untrained validation by seed: {report['val_untrained_by_seed']}  |  "
              f"C2 zeroed cue->DN: {report['val_zeroed_cue_dn']:.5f}", flush=True)

    out = Path("runs/diagnose") / f"{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, default=str))
    print(f"wrote {out} ({(perf_counter() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()

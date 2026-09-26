"""Phase 4: teacher forcing. Only the cue -> descending-neuron synapses learn (D3).

Each training take (grooves/train/*.mid, never grooves/heldout/) becomes
  - cue rates [T, 12], encoded exactly as fly.loop does (lookahead, burst, pre-roll), and
  - the teacher's servo targets q*(t) [T, 32] from fly.strokes (plan + teacher).
The brain runs with alpha * I*(q*) injected into the leg motor neurons. The loss is the gap between the fly's own
decoded joint targets and q*, over the servos the decoder can drive, weighted up where q* is away from rest (the
strokes). MuJoCo isn't differentiable, so the body isn't in the loop. Plastic weights keep their sign (Dale's law).

Batching: takes are shuffled and packed into `batch` parallel streams with GAP_MS of silence between them, so the
state carries across windows with no per-element resets. Truncated backprop: one update per `window_ms` steps, then
the state is detached. alpha falls linearly from 1 to 0 over the first `anneal` fraction of all updates, then stays 0.
Before training and after every epoch, a validation pass runs at alpha = 0 with no gradients (the fly on its own, on
the first `val_ms` of a fixed packing): that loss is the number the D7 gate looks at.

    python -m fly.train --out runs/train/smoke --max-windows 20        # a few minutes: does the loss move?
    python -m fly.train --out runs/train/t1 --epochs 6
    python -m fly.train --out runs/train/t1_shuf --epochs 6 --shuffled 1   # the control, trained the same way

Writes <out>/log.csv (every update), val.csv, meta.json, last.pt and best.pt (plastic weights + settings).
A trained fly plays with: python -m fly.loop --groove X.mid --out runs/<id> --weights <out>/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

GAP_MS = 500  # silence between packed takes
ACTIVE_FRAC, ACTIVE_W = 0.01, 4.0  # q* this far from rest (fraction of the range) counts as a stroke: weight 1 + 4
VAL_SEED = 12345


@dataclass
class Take:
    path: str
    rates: np.ndarray  # [T, 12] Hz
    q: np.ndarray  # [T, 32] teacher servo targets
    notes: int
    dropped: int


def rest_on_pedal(decoder, strokes: dict) -> None:
    """The hind-left foot rests on the hi-hat pedal, held closed, like a drummer's: that becomes the leg's rest pose,
    so motor activity only has to lift it (open hats, chicks). The fly's cues are bursts, so it can't hold a pose for
    minutes; without this the teacher's held pedal was 82% of the loss. Fixed posture, the same for every groove.
    Training and fly.loop both apply it, so the fly is tested in the posture it learned in."""
    from fly.decoder import JOINTS
    from fly.drums import LEGS

    k = LEGS.index("hind_left") * len(JOINTS)
    lo, hi = decoder.rest - decoder.down, decoder.rest + decoder.up
    rest = decoder.rest.clone()
    rest[k:k + len(JOINTS)] = torch.as_tensor(strokes["pads"]["hat_pedal"]["legs"]["hind_left"]["soft"], dtype=rest.dtype)
    rest = torch.minimum(torch.maximum(rest, lo), hi)
    with torch.no_grad():
        decoder.rest.copy_(rest)
        decoder.up.copy_((hi - rest).clamp(min=1e-3))
        decoder.down.copy_((rest - lo).clamp(min=1e-3))


def load_take(path: str, strokes: dict, rest: np.ndarray, lookahead_ms: float, burst_ms: float, preroll_ms: float) -> Take:
    from fly.encoder import encode, read_onsets
    from fly.strokes import plan, teacher

    enc = encode(Path(path), lookahead_ms=lookahead_ms, burst_ms=burst_ms, preroll_ms=preroll_ms)
    notes = plan(read_onsets(Path(path)), enc.offset_ms, strokes["pads"])
    q = teacher(notes, len(enc.rates), strokes, rest)
    return Take(str(path), enc.rates.astype(np.float32), q.astype(np.float32), len(notes), sum(bool(n.dropped) for n in notes))


def pack(takes: list[Take], batch: int, rest: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Takes -> `batch` streams of similar length (longest first into the shortest stream; each stream's order then
    shuffled). Returns rates [B, T, V] and q* [B, T, 32]; gaps and padding are silent and hold the last pose."""
    streams, lengths = [[] for _ in range(batch)], [0] * batch
    for i in sorted(rng.permutation(len(takes)), key=lambda i: -len(takes[i].rates)):
        b = int(np.argmin(lengths))
        streams[b].append(i)
        lengths[b] += len(takes[i].rates) + GAP_MS
    n_steps, n_voices = max(lengths), takes[0].rates.shape[1]
    rates = np.zeros((batch, n_steps, n_voices), dtype=np.float32)
    q = np.tile(rest.astype(np.float32), (batch, n_steps, 1))
    for b, order in enumerate(streams):
        rng.shuffle(order)
        t = 0
        for i in order:
            tk = takes[i]
            n = len(tk.rates)
            rates[b, t:t + n], q[b, t:t + n] = tk.rates, tk.q
            q[b, t + n:] = tk.q[-1]  # hold the take's last pose through the gap (and any padding)
            t += n + GAP_MS
    return rates, q


def alpha_at(update: int, total: int, anneal: float) -> float:
    return max(0.0, 1.0 - update / max(1.0, anneal * total))


def detach(x):
    if torch.is_tensor(x):
        return x.detach()
    if isinstance(x, (list, tuple)):
        return type(x)(detach(v) for v in x)
    return x


class Loss:
    """Weighted squared error between decoded targets and q*, in units of each servo's range, drivable servos only."""

    def __init__(self, decoder):
        self.rest, self.span = decoder.rest, decoder.up + decoder.down
        self.driven = (decoder.eff.abs().sum(1) > 0).float()

    def __call__(self, q: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err = ((q - target) / self.span) ** 2
        w = 1.0 + ACTIVE_W * (((target - self.rest).abs() / self.span) > ACTIVE_FRAC).float()
        return (err * w * self.driven).sum() / (self.driven.sum() * q.shape[0])


def run_window(brain, decoder, loss_fn, state, r, rates: torch.Tensor, qstar: torch.Tensor, alpha: float, gen):
    """Steps through rates [B, W, V] with alpha * I*(q*) injected; -> (state, decoder rates, mean loss)."""
    total = 0.0
    for t in range(rates.shape[1]):
        cur = alpha * decoder.current_for(qstar[:, t]) if alpha > 0 else None
        state = brain.step(state, rates[:, t], cur, generator=gen)
        r, q = decoder(r, state[2])
        total = total + loss_fn(q, qstar[:, t])
    return state, r, total / rates.shape[1]


def keep_signs(weight: torch.nn.Parameter, sign0: torch.Tensor) -> None:
    with torch.no_grad():
        weight.copy_(torch.where(sign0 >= 0, weight.clamp(min=0), weight.clamp(max=0)))


def validate(brain, decoder, loss_fn, rates: torch.Tensor, qstar: torch.Tensor, window: int, seed: int) -> float:
    gen = torch.Generator(device=rates.device).manual_seed(seed)
    state, r, losses = brain.init_state(), decoder.init_state(rates.shape[0]), []
    with torch.no_grad():
        for s in range(0, rates.shape[1], window):
            state, r, loss = run_window(brain, decoder, loss_fn, state, r, rates[:, s:s + window], qstar[:, s:s + window], 0.0, gen)
            losses.append(float(loss) * min(window, rates.shape[1] - s))
    return sum(losses) / rates.shape[1]


def load_weights(brain, path: Path, shuffle_seed: int | None) -> dict:
    ck = torch.load(path, map_location=brain.plastic.weight.device)
    if ck["shuffle_seed"] != shuffle_seed:
        raise SystemExit(f"{path} was trained with shuffle seed {ck['shuffle_seed']}, not {shuffle_seed}")
    if ck["plastic_weight"].shape != brain.plastic.weight.shape:
        raise SystemExit(f"{path}: {ck['plastic_weight'].shape[0]} plastic edges, this wiring has {brain.plastic.weight.shape[0]}")
    with torch.no_grad():
        brain.plastic.weight.copy_(ck["plastic_weight"])
    return ck


def save(path: Path, brain, settings: dict, **extra) -> None:
    torch.save({"plastic_weight": brain.plastic.weight.detach().cpu(), **settings, **extra}, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--takes", default="grooves/train/*.mid")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--window-ms", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.05, help="Adam step, in synapse-weight units")
    ap.add_argument("--anneal", type=float, default=0.7, help="fraction of all updates over which alpha goes 1 -> 0")
    ap.add_argument("--val-ms", type=int, default=10_000, help="validation length per stream (alpha = 0)")
    ap.add_argument("--max-windows", type=int, default=None, help="stop after this many updates (smoke test)")
    ap.add_argument("--shuffled", type=int, default=None, metavar="SEED", help="train the shuffled-connectome control")
    ap.add_argument("--init", type=Path, default=None, help="start from these weights (resume)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    from fly.decoder import Decoder
    from fly.loop import BURST_MS, default_lookahead_ms
    from fly.strokes import PREROLL_MS, STROKES
    from fly.wiring import wire

    takes_paths = sorted(glob(args.takes))
    held_out = [p for p in takes_paths if "heldout" in Path(p).parts]
    if held_out or not takes_paths:
        raise SystemExit(f"never train on held-out grooves: {held_out}" if held_out else f"no takes match {args.takes}")
    lookahead = default_lookahead_ms()
    settings = {"shuffle_seed": args.shuffled, "lookahead_ms": lookahead, "burst_ms": BURST_MS, "preroll_ms": PREROLL_MS,
                "rest_on_pedal": True}

    t0 = perf_counter()
    wiring = wire(shuffle_seed=args.shuffled)
    brain = wiring.brain(batch=args.batch, device=device)
    decoder = Decoder(wiring.leg_mns, wiring.mn_types, wiring.conn.size).to(device)
    if args.init is not None:
        load_weights(brain, args.init, args.shuffled)
    strokes = json.loads(STROKES.read_text())
    rest_on_pedal(decoder, strokes)
    loss_fn = Loss(decoder)
    rest = decoder.rest.cpu().numpy()
    takes = [load_take(p, strokes, rest, lookahead, BURST_MS, PREROLL_MS) for p in takes_paths]
    print(f"{wiring.conn.name}{'' if args.shuffled is None else f' shuffled {args.shuffled}'}: "
          f"{int(wiring.plastic_mask.sum())} plastic edges, {len(takes)} takes "
          f"({sum(len(t.rates) for t in takes) / 1000:.0f} s, {sum(t.dropped for t in takes)} notes the teacher can't play), "
          f"built in {perf_counter() - t0:.0f} s", flush=True)

    val_rates, val_q = pack(takes, args.batch, rest, np.random.default_rng(VAL_SEED))
    n_val = min(args.val_ms, val_rates.shape[1])
    val_rates = torch.as_tensor(val_rates[:, :n_val], device=device)
    val_q = torch.as_tensor(val_q[:, :n_val], device=device)

    w = brain.plastic.weight
    sign0 = torch.sign(w.detach().clone())
    opt = torch.optim.Adam([w], lr=args.lr)
    steps_per_epoch = pack(takes, args.batch, rest, np.random.default_rng(args.seed))[0].shape[1]
    per_epoch = math.ceil(steps_per_epoch / args.window_ms)
    total = per_epoch * args.epochs if args.max_windows is None else min(args.max_windows, per_epoch * args.epochs)

    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"args": {k: str(v) for k, v in vars(args).items()}, "device": device, **settings,
            "wiring": wiring.summary(), "updates": total, "updates_per_epoch": per_epoch,
            "steps_per_epoch": steps_per_epoch, "takes": [{"path": t.path, "steps": len(t.rates), "notes": t.notes,
                                                           "dropped": t.dropped} for t in takes]}
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    log = open(args.out / "log.csv", "w", newline="")
    log_w = csv.writer(log)
    log_w.writerow(["update", "epoch", "alpha", "loss", "seconds"])
    val_f = open(args.out / "val.csv", "w", newline="")
    val_w = csv.writer(val_f)
    val_w.writerow(["epoch", "update", "val_loss_alpha0", "seconds"])

    def check(epoch: int, update: int, best: float) -> float:
        t = perf_counter()
        val = validate(brain, decoder, loss_fn, val_rates, val_q, args.window_ms, VAL_SEED)
        val_w.writerow([epoch, update, f"{val:.6f}", round(perf_counter() - t0, 1)])
        val_f.flush()
        print(f"  validation (alpha 0, {n_val / 1000:.0f} s x {args.batch}): {val:.5f}"
              f"{'  (best)' if val < best else ''}  [{perf_counter() - t:.0f} s]", flush=True)
        save(args.out / "last.pt", brain, settings, epoch=epoch, update=update, val_loss=val)
        if val < best:
            save(args.out / "best.pt", brain, settings, epoch=epoch, update=update, val_loss=val)
        return min(val, best)

    print(f"{total} updates ({per_epoch} per epoch of {steps_per_epoch / 1000:.0f} s x {args.batch} streams), "
          f"{args.window_ms} ms windows, lr {args.lr}, alpha 1 -> 0 over {args.anneal:.0%}", flush=True)
    best = check(0, 0, math.inf)
    update, t_train = 0, perf_counter()
    for epoch in range(1, args.epochs + 1):
        rates, qstar = pack(takes, args.batch, rest, np.random.default_rng(args.seed + epoch))
        rates, qstar = torch.as_tensor(rates, device=device), torch.as_tensor(qstar, device=device)
        gen = torch.Generator(device=device).manual_seed(args.seed + epoch)
        state, r = brain.init_state(), decoder.init_state(args.batch)
        for s in range(0, rates.shape[1], args.window_ms):
            if update >= total:
                break
            alpha = alpha_at(update, total, args.anneal)
            state, r, loss = run_window(brain, decoder, loss_fn, state, r, rates[:, s:s + args.window_ms],
                                        qstar[:, s:s + args.window_ms], alpha, gen)
            if not torch.isfinite(loss):
                save(args.out / "last.pt", brain, settings, epoch=epoch, update=update)
                raise SystemExit(f"loss is {float(loss)} at update {update}; stopped (weights saved to last.pt)")
            opt.zero_grad()
            loss.backward()
            opt.step()
            keep_signs(w, sign0)
            state, r = detach(state), r.detach()
            update += 1
            elapsed = perf_counter() - t_train
            log_w.writerow([update, epoch, f"{alpha:.4f}", f"{float(loss):.6f}", round(elapsed, 1)])
            if update % args.log_every == 0 or update == 1:
                log.flush()
                mem = f", {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device == "cuda" else ""
                eta = elapsed / update * (total - update)
                print(f"  update {update:5d}/{total}  epoch {epoch}  alpha {alpha:.3f}  loss {float(loss):.5f}  "
                      f"{elapsed / update:.1f} s/update{mem}  ETA {eta / 3600:.1f} h", flush=True)
        best = check(epoch, update, best)
        if update >= total:
            break
    log.close()
    val_f.close()
    print(f"done: {update} updates in {(perf_counter() - t_train) / 3600:.2f} h, best validation {best:.5f}; wrote {args.out}")


if __name__ == "__main__":
    main()

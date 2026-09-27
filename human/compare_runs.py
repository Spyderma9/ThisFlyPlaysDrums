"""Phase 5: compare fly runs (trained, shuffled control, untrained, teacher) on the held-out grooves.

Whole-groove F1 gives one number per groove, too few to test anything, so each groove is cut into segments of
--segment-s seconds (of the score) and every run is scored per segment. Conditions are compared on the same
segments (paired): mean F1 difference, a 95% bootstrap interval, d_z, and a sign-flip permutation p-value,
Holm-adjusted over the comparisons against the baseline.

Planned before any held-out run (confirmatory): trained vs shuffled, per-segment F1, tolerance 60 ms, 10 s
segments, two-sided. Everything else (untrained, teacher, per-drum numbers) is descriptive or exploratory.

Layout: one fly.loop run dir per condition and groove, runs/eval/<condition>/<groove stem>[_s<seed>]/ (each has
meta.json, which names its groove, and hits.csv). Several seeds of one condition are averaged per segment.

    python human/compare_runs.py runs/eval                                  # every condition vs the first baseline found
    python human/compare_runs.py runs/eval --baseline shuffled --csv runs/eval/segments.csv

Matching is score.py's (same drum, within --tolerance ms, edges fold onto their drum), with no --align: when the
fly plays is part of what's scored. Caveat printed with the table: segments of one run share its state, and one
Poisson seed per condition doesn't measure seed-to-seed spread.

Chance: a fly that hits a lot matches some notes by luck. Each run's own hits are slid by a random offset (at least
1 s, wrapping around the end) and re-scored --shifts times: same hits, same rhythm, no alignment with the score.
F1 above that is F1 from playing *this* groove; p is the share of shifts that scored at least as well.
"""
import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from drum_map import DRUMS
from score import TOLERANCE_MS, load_hits, match, score, stats

REPO = Path(__file__).resolve().parents[1]
BASELINES = ("shuffled", "untrained")  # the default comparison, first one present
MIN_SHIFT_MS = 1000.0


def chance(ref, played, tol, n_shifts, rng):
    """Scores of the run's own hits slid by random offsets (wrapping): -> ([overall F1], {note: [F1]}, [hits])."""
    if not played:
        return [0.0] * n_shifts, {}, [0] * n_shifts
    period = max(max(t for t, _, _ in ref), max(t for t, _, _ in played)) + tol
    f1s, hits, per_drum = [], [], defaultdict(list)
    for _ in range(n_shifts):
        off = rng.uniform(MIN_SHIFT_MS, period - MIN_SHIFT_MS)
        overall, drums = score(ref, sorted(((t + off) % period, n, v) for t, n, v in played), tol)
        f1s.append(overall["f1"])
        hits.append(overall["hit"])
        for note, s in drums.items():
            per_drum[note].append(s["f1"])
    return f1s, per_drum, hits


def segment_scores(ref, played, tol, segment_ms):
    """-> [stats per segment]. Matched pairs count in their reference note's segment (so a hit a few ms over the
    boundary still counts), unmatched played hits in their own; hits after the score's end count in its last."""
    last = int(max(t for t, _, _ in ref) // segment_ms)

    def seg(t):
        return min(max(int(t // segment_ms), 0), last)

    seg_ref, seg_played, seg_pairs = defaultdict(list), defaultdict(list), defaultdict(list)
    matched = set()
    for r, p in match(ref, played, tol):
        seg_pairs[seg(r[0])].append((r, p))
        seg_played[seg(r[0])].append(p)
        matched.add(p)
    for h in ref:
        seg_ref[seg(h[0])].append(h)
    for h in played:
        if h not in matched:
            seg_played[seg(h[0])].append(h)
    return [stats(seg_ref[k], seg_played[k], seg_pairs[k]) for k in range(last + 1)]


def load_runs(root: Path):
    """-> {condition: {groove: [run dirs]}} from root/<condition>/<run>/meta.json."""
    runs = defaultdict(lambda: defaultdict(list))
    for meta_path in sorted(root.glob("*/*/meta.json")):
        meta = json.loads(meta_path.read_text())
        runs[meta_path.parent.parent.name][meta["groove"]].append(meta_path.parent)
    return runs


def paired(diffs, n_boot=10_000, n_perm=20_000, seed=0):
    """Mean of paired differences, 95% bootstrap interval, two-sided sign-flip permutation p."""
    rng = random.Random(seed)
    mean = statistics.fmean(diffs)
    boots = sorted(statistics.fmean(rng.choices(diffs, k=len(diffs))) for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    extreme = sum(abs(statistics.fmean(d if rng.random() < 0.5 else -d for d in diffs)) >= abs(mean) - 1e-12
                  for _ in range(n_perm))
    return mean, lo, hi, (extreme + 1) / (n_perm + 1)


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="runs/eval: one dir per condition, one run dir per groove inside")
    ap.add_argument("--baseline", default=None, help=f"condition to compare against (default: first of {BASELINES})")
    ap.add_argument("--segment-s", type=float, default=10.0)
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_MS)
    ap.add_argument("--csv", type=Path, help="also write one row per condition, groove and segment")
    ap.add_argument("--shifts", type=int, default=200, help="random time shifts per run for the chance line (0: skip)")
    args = ap.parse_args()
    rng = random.Random(0)

    runs = load_runs(args.root)
    if not runs:
        sys.exit(f"no run dirs with meta.json under {args.root}/<condition>/")
    grooves = sorted({g for by_groove in runs.values() for g in by_groove})
    refs = {g: load_hits(REPO / g) for g in grooves}

    # per condition: whole-groove totals and per-segment F1 (averaged over seeds)
    seg_f1, totals, rows, luck = {}, {}, [], {}
    for cond, by_groove in sorted(runs.items()):
        seg_f1[cond], counts = {}, defaultdict(int)
        obs, null, drum_obs, drum_null, drum_played = [], [], defaultdict(list), defaultdict(list), defaultdict(int)
        for g in grooves:
            for run in by_groove.get(g, []):
                played = load_hits(run / "hits.csv")
                overall = stats(refs[g], played, match(refs[g], played, args.tolerance))
                for k in ("ref", "played", "hit"):
                    counts[k] += overall[k]
                if args.shifts:
                    f1s, null_drums, null_hits = chance(refs[g], played, args.tolerance, args.shifts, rng)
                    obs.append((overall["f1"], overall["hit"]))
                    null.append((f1s, null_hits))
                    for note, s in score(refs[g], played, args.tolerance)[1].items():
                        drum_obs[note].append(s["f1"])
                        drum_null[note].append(null_drums.get(note, [0.0] * args.shifts))
                        drum_played[note] += s["played"]
                for i, s in enumerate(segment_scores(refs[g], played, args.tolerance, args.segment_s * 1000)):
                    seg_f1[cond].setdefault((g, i), []).append(s["f1"])
                    rows.append([cond, g, run.name, i, s["ref"], s["played"], s["hit"], f"{s['f1']:.4f}"])
        seg_f1[cond] = {k: statistics.fmean(v) for k, v in seg_f1[cond].items()}
        p = counts["hit"] / counts["played"] if counts["played"] else 0.0
        r = counts["hit"] / counts["ref"] if counts["ref"] else 0.0
        totals[cond] = {"runs": sum(len(v) for v in by_groove.values()), "grooves": len(by_groove), **counts,
                        "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if counts["hit"] else 0.0}
        if obs:  # pooled over runs: the i-th shift of every run together is one draw of a luck-only fly
            obs_f1 = statistics.fmean(f for f, _ in obs)
            null_f1 = [statistics.fmean(f1s[i] for f1s, _ in null) for i in range(args.shifts)]
            per_drum = {}
            for note, xs in drum_obs.items():
                if drum_played[note] >= 5:
                    per_drum[note] = (statistics.fmean(xs), statistics.fmean(statistics.fmean(n) for n in drum_null[note]))
            luck[cond] = {"f1": obs_f1, "chance_f1": statistics.fmean(null_f1),
                          "p": (sum(x >= obs_f1 - 1e-12 for x in null_f1) + 1) / (args.shifts + 1),
                          "hits": sum(h for _, h in obs),
                          "chance_hits": statistics.fmean(sum(hs[i] for _, hs in null) for i in range(args.shifts)),
                          "per_drum": per_drum}

    print(f"held-out: {', '.join(grooves)}; tolerance {args.tolerance:g} ms; {args.segment_s:g} s segments\n")
    print(f"{'condition':<14}{'runs':>5}{'ref':>7}{'played':>8}{'hit':>6}{'prec':>7}{'recall':>8}{'F1':>7}"
          f"   segment F1: n, mean, SD, median")
    for cond, t in totals.items():
        f1s = list(seg_f1[cond].values())
        sd = statistics.stdev(f1s) if len(f1s) > 1 else float("nan")
        print(f"{cond:<14}{t['runs']:>5}{t['ref']:>7}{t['played']:>8}{t['hit']:>6}{t['precision']:>7.3f}"
              f"{t['recall']:>8.3f}{t['f1']:>7.3f}   {len(f1s)}, {statistics.fmean(f1s):.3f}, {sd:.3f}, "
              f"{statistics.median(f1s):.3f}")

    if luck:
        print(f"\nchance: each run's own hits slid to random times ({args.shifts} shifts of at least "
              f"{MIN_SHIFT_MS / 1000:g} s; F1 is the mean over runs, p the share of shifts scoring as well):")
        print(f"{'condition':<14}{'F1':>7}{'chance':>8}{'p':>9}{'hits':>7}{'chance':>8}   per drum played 5+ times: F1 / chance")
        for cond, c in luck.items():
            drums = "  ".join(f"{DRUMS.get(n, n)} {o:.2f}/{z:.2f}" for n, (o, z) in sorted(c["per_drum"].items()))
            print(f"{cond:<14}{c['f1']:>7.3f}{c['chance_f1']:>8.3f}{fmt_p(c['p']):>9}{c['hits']:>7}"
                  f"{c['chance_hits']:>8.1f}   {drums}")

    base = args.baseline or next((b for b in BASELINES if b in runs), None)
    if base is None or base not in runs:
        print(f"\nno baseline condition ({args.baseline or ' or '.join(BASELINES)}) under {args.root}; no paired tests")
    else:
        tests = []
        for cond in sorted(runs):
            keys = sorted(seg_f1[cond].keys() & seg_f1[base].keys())
            if cond != base and len(keys) >= 2:
                diffs = [seg_f1[cond][k] - seg_f1[base][k] for k in keys]
                sd = statistics.stdev(diffs)
                tests.append((cond, diffs, *paired(diffs), statistics.fmean(diffs) / sd if sd > 0 else float("nan")))
        print(f"\npaired against {base}, per segment (F1 difference; + means better than {base}), "
              f"Holm-adjusted over {len(tests)}:")
        for (cond, diffs, mean, lo, hi, p, dz), p_adj in zip(tests, holm([t[5] for t in tests])):
            tag = "  <- planned test" if (cond, base) == ("trained", "shuffled") else ""
            print(f"  {cond:<14}{mean:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  d_z {dz:+.2f}  p {fmt_p(p)}  "
                  f"Holm p {fmt_p(p_adj)}  better on {sum(d > 0 for d in diffs)}/{len(diffs)} segments{tag}")
        print("  (sign-flip permutation, 20,000 draws; bootstrap 10,000; seed 0. Segments of one run share its state, "
              "and one\n   Poisson seed per condition doesn't measure seed spread: a second seed per condition tightens "
              "this.)")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["condition", "groove", "run", "segment", "ref", "played", "hit", "f1"])
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()

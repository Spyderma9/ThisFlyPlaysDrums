"""Score played drum hits (the fly's, or a human take) against a reference score.

Each played hit is paired with the nearest reference note on the same drum within --tolerance ms.
Per drum it reports hits, misses, extra hits, F1, timing (played minus reference: + is late) and
loudness (mean velocity error and correlation). Edges and rims count as their drum.

Reference: sheet music (.musicxml / .mxl / .txt), .mid, or a hits .csv.
Played:    a hits .csv (t_ms, note, velocity), a take .csv, or a .mid. Several files -> a comparison table.

Usage:
    python human/score.py grooves/heldout/groove1.mid runs/<id>/hits.csv
    python human/score.py grooves/heldout/groove1.mid runs/fly/hits.csv runs/shuffled/hits.csv
    python human/score.py takes/take_X.musicxml takes/take_X.csv --align     # human baseline
"""
import argparse
import bisect
import csv
import statistics
import sys
from pathlib import Path

from drum_map import DRUMS, SAME_DRUM, HitFilter
from sheet_to_midi import SHEET_EXTS, convert, read_midi

TOLERANCE_MS = 60  # a played hit further than this from every reference note on its drum is an extra hit


# ---------- loading ----------

def load_hits(path):
    """-> sorted [(t_ms, note, velocity)] with edges and rims folded onto their drum."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        rows = list(csv.DictReader(open(path, newline="")))
        if rows and "type" in rows[0]:  # a raw take from midi_capture.py: drop fake hits
            hit_filter = HitFilter()
            raw = [(float(r["t_ms"]), int(r["note"]), int(r["velocity"])) for r in rows
                   if r["type"] == "note_on" and int(r["velocity"]) > 0]
            raw = [h for h in raw if not hit_filter.check(*h)]
        else:
            raw = [(float(r["t_ms"]), int(r["note"]), int(r["velocity"])) for r in rows]
    elif ext in (".mid", ".midi"):
        raw, _ = read_midi(path)
    elif ext in SHEET_EXTS:
        events, _ = convert(path)
        raw = [(t, m.note, m.velocity) for t, m in events if m.type == "note_on"]
    else:
        sys.exit(f"{path}: unsupported file type {ext!r}")
    return sorted((t, SAME_DRUM.get(n, n), v) for t, n, v in raw)


# ---------- matching ----------

def match(ref, played, tol, offset=0.0):
    """Pair played hits with reference notes on the same drum, closest pairs first.

    Returns [(ref_hit, played_hit)] with played times shifted by -offset.
    """
    pairs = []
    for note in {n for _, n, _ in ref} | {n for _, n, _ in played}:
        r = [h for h in ref if h[1] == note]
        p = [(t - offset, n, v) for t, n, v in played if n == note]
        p_times = [t for t, _, _ in p]
        candidates = []
        for i, (rt, _, _) in enumerate(r):
            lo, hi = bisect.bisect_left(p_times, rt - tol), bisect.bisect_right(p_times, rt + tol)
            candidates += [(abs(p_times[j] - rt), i, j) for j in range(lo, hi)]
        used_r, used_p = set(), set()
        for _, i, j in sorted(candidates):
            if i not in used_r and j not in used_p:
                used_r.add(i)
                used_p.add(j)
                pairs.append((r[i], p[j]))
    return pairs


def best_offset(ref, played, tol):
    """The constant shift of the played hits that lines up the most notes (for takes that start late)."""
    candidates = {0.0}
    for rt, rn, _ in ref[:20]:
        candidates |= {pt - rt for pt, pn, _ in played[:20] if pn == rn}

    def quality(off):
        pairs = match(ref, played, tol, off)
        return len(pairs), -sum(abs(p[0] - r[0]) for r, p in pairs)

    off = max(candidates, key=quality)
    pairs = match(ref, played, tol, off)
    if pairs:  # centre the shift on the median error so timing stats aren't skewed by the anchor note
        off += statistics.median(p[0] - r[0] for r, p in pairs)
    return off


# ---------- stats ----------

def stats(ref, played, pairs):
    n_ref, n_played, n_hit = len(ref), len(played), len(pairs)
    precision = n_hit / n_played if n_played else 0.0
    recall = n_hit / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if n_hit else 0.0
    dts = [p[0] - r[0] for r, p in pairs]
    rv, pv = [r[2] for r, _ in pairs], [p[2] for _, p in pairs]
    corr = None
    if len(pairs) > 2 and len(set(rv)) > 1 and len(set(pv)) > 1:
        corr = statistics.correlation(rv, pv)
    return {
        "ref": n_ref, "played": n_played, "hit": n_hit, "miss": n_ref - n_hit, "extra": n_played - n_hit,
        "precision": precision, "recall": recall, "f1": f1,
        "timing_mean": statistics.fmean(dts) if dts else None,
        "timing_sd": statistics.pstdev(dts) if len(dts) > 1 else None,
        "timing_abs": statistics.fmean(abs(d) for d in dts) if dts else None,
        "vel_err": statistics.fmean(abs(a - b) for a, b in zip(rv, pv)) if pairs else None,
        "vel_corr": corr,
    }


def score(ref, played, tol=TOLERANCE_MS, offset=0.0):
    """-> (overall stats, {note: stats}). Reusable from other scripts."""
    pairs = match(ref, played, tol, offset)
    shifted = [(t - offset, n, v) for t, n, v in played]
    per_drum = {}
    for note in sorted({n for _, n, _ in ref} | {n for _, n, _ in played}):
        per_drum[note] = stats([h for h in ref if h[1] == note], [h for h in shifted if h[1] == note],
                               [pr for pr in pairs if pr[0][1] == note])
    return stats(ref, shifted, pairs), per_drum


# ---------- output ----------

def fmt_timing(s):
    if s["timing_mean"] is None:
        return "-"
    sd = f" sd {s['timing_sd']:.0f}" if s["timing_sd"] is not None else ""
    return f"{round(s['timing_mean']):+d}{sd}"


def fmt_vel(s):
    if s["vel_err"] is None:
        return "-"
    corr = f", corr {s['vel_corr']:.2f}" if s["vel_corr"] is not None else ""
    return f"err {s['vel_err']:.0f}{corr}"


def print_detail(overall, per_drum):
    print(f"{'drum':<14}{'ref':>5}{'played':>8}{'hit':>6}{'miss':>6}{'extra':>7}{'F1':>7}   {'timing ms':<12}velocity")
    for note, s in per_drum.items():
        name = DRUMS.get(note, str(note))
        print(f"{name:<14}{s['ref']:>5}{s['played']:>8}{s['hit']:>6}{s['miss']:>6}{s['extra']:>7}{s['f1']:>7.2f}"
              f"   {fmt_timing(s):<12}{fmt_vel(s)}")
    s = overall
    print(f"{'all':<14}{s['ref']:>5}{s['played']:>8}{s['hit']:>6}{s['miss']:>6}{s['extra']:>7}{s['f1']:>7.2f}"
          f"   {fmt_timing(s):<12}{fmt_vel(s)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reference", help="the score: .musicxml / .mxl / .txt / .mid / hits .csv")
    ap.add_argument("played", nargs="+", help="hits .csv, take .csv or .mid; several for a comparison table")
    ap.add_argument("--tolerance", type=float, default=TOLERANCE_MS,
                    help=f"max ms between a played hit and its note (default {TOLERANCE_MS})")
    ap.add_argument("--offset", type=float, default=0.0, help="subtract this many ms from played times")
    ap.add_argument("--align", action="store_true",
                    help="find the constant shift that lines the played hits up best (for takes that start late)")
    ap.add_argument("--csv", metavar="FILE", help="also write one summary row per played file to FILE")
    args = ap.parse_args()

    ref = load_hits(args.reference)
    if not ref:
        sys.exit(f"{args.reference}: no notes in the reference")
    print(f"Reference: {args.reference} ({len(ref)} notes, {(ref[-1][0] - ref[0][0]) / 1000:.1f} s)")
    print(f"Tolerance: {args.tolerance:g} ms either side. Timing is played minus reference (+ means late), "
          f"sd is its spread.\n")

    rows = []
    for path in args.played:
        played = load_hits(path)
        offset = best_offset(ref, played, args.tolerance) if args.align and played else args.offset
        overall, per_drum = score(ref, played, args.tolerance, offset)
        rows.append((path, offset, overall))
        if len(args.played) == 1:
            shift = f", shifted {offset:+.0f} ms" if offset else ""
            print(f"Played: {path} ({len(played)} hits{shift})\n")
            print_detail(overall, per_drum)
            if not args.align and played and overall["f1"] < 0.5:
                aligned = score(ref, played, args.tolerance, best_offset(ref, played, args.tolerance))[0]
                if aligned["f1"] > overall["f1"] + 0.2:
                    print(f"\nnote: with --align the F1 would be {aligned['f1']:.2f}; the files may start at different times")

    if len(args.played) > 1:
        labels = [f"{Path(p).parent.name}/{Path(p).name}" for p, _, _ in rows]  # e.g. <run id>/hits.csv
        width = max(len(label) for label in labels) + 2
        print(f"{'played':<{width}}{'hit':>6}{'miss':>6}{'extra':>7}{'F1':>7}   {'timing ms':<12}velocity")
        for label, (path, offset, s) in zip(labels, rows):
            print(f"{label:<{width}}{s['hit']:>6}{s['miss']:>6}{s['extra']:>7}{s['f1']:>7.2f}   {fmt_timing(s):<12}{fmt_vel(s)}")

    if args.csv:
        fields = ["played", "offset_ms", "ref", "played_hits", "hit", "miss", "extra", "precision", "recall", "f1",
                  "timing_mean_ms", "timing_sd_ms", "timing_abs_ms", "vel_err", "vel_corr"]
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for path, offset, s in rows:
                w.writerow([path, f"{offset:.1f}", s["ref"], s["played"], s["hit"], s["miss"], s["extra"]]
                           + [None if s[k] is None else f"{s[k]:.3f}" for k in
                              ("precision", "recall", "f1", "timing_mean", "timing_sd", "timing_abs", "vel_err", "vel_corr")])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()

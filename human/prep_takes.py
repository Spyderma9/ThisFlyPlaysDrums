"""Package recorded takes as clean material for the fly: training takes, or held-out test grooves.

Each take is cleaned (crosstalk, kick bounces and stray touches dropped; edges and rims folded onto
their drum), trimmed to start LEAD_MS before the first hit, and saved to grooves/train/ (or --dir) as a
channel-10 .mid with TD-07 note numbers. <dir>/index.csv lists every file with its tempo, length and drums.
A take already used in another groove folder is refused, so held-out grooves can't leak into training.

Usage:
    python human/prep_takes.py takes/take_20260926_031545.csv --name feet_kick_pedal
    python human/prep_takes.py takes/*.csv                    # names made from the drums and tempo
    python human/prep_takes.py takes/take_20260926_031545.csv --note "feet only, no click"
    python human/prep_takes.py takes/take_20260926_115406.csv --dir grooves/heldout --name heldout_1
"""
import argparse
import csv
import glob
import sys
from collections import Counter
from pathlib import Path

import mido

from drum_map import DRUM_CHANNEL, DRUMS, GRID_NAMES, SAME_DRUM, HitFilter, write_midi
from sheet_to_midi import NOTE_MS, read_midi

GROOVES = Path("grooves")
OUT_DIR = GROOVES / "train"
INDEX_FIELDS = ["file", "source", "bpm", "seconds", "hits", "drums", "note"]
LEAD_MS = 1000  # silence before the first hit, so the encoder has room to cue it early

SHORT = {}
for _name, _note in GRID_NAMES.items():
    SHORT.setdefault(_note, _name)  # first grid name for each note: BD, SD, HH, ...


def load_hits(path):
    """-> [(t_ms, note, velocity)] with fake hits dropped and edges/rims folded, plus dropped counts."""
    if path.suffix.lower() == ".csv":
        raw = [(float(r["t_ms"]), int(r["note"]), int(r["velocity"])) for r in csv.DictReader(open(path))
               if r["type"] == "note_on" and int(r["velocity"]) > 0]
    else:
        raw, _ = read_midi(path)
    hit_filter, hits, dropped = HitFilter(), [], Counter()
    for t, note, vel in raw:
        reason = hit_filter.check(t, note, vel)
        note = SAME_DRUM.get(note, note)
        if reason or note not in DRUMS:
            dropped[reason or f"unknown note {note}"] += 1
            continue
        hits.append((t, note, vel))
    return hits, dropped


def estimate_bpm(times):
    """Tempo whose 16th grid fits the hits best, relative to the grid size (80-180 BPM). Approximate."""
    def misfit(bpm):
        step = 60000 / bpm / 4
        return min(sum(abs((t - off) / step - round((t - off) / step)) for t in times)
                   for off in (step * i / 20 for i in range(20)))

    return min((b / 2 for b in range(160, 361)), key=misfit)


def prep(path, name, note, out_dir):
    hits, dropped = load_hits(path)
    if not hits:
        print(f"{path}: no drum hits, skipped")
        return None
    shift = hits[0][0] - LEAD_MS
    events = []
    for t, n, v in hits:
        events.append((t - shift, mido.Message("note_on", channel=DRUM_CHANNEL, note=n, velocity=v)))
        events.append((t - shift + NOTE_MS, mido.Message("note_off", channel=DRUM_CHANNEL, note=n, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1].type == "note_on"))

    counts = Counter(SHORT.get(n, str(n)) for _, n, _ in hits)
    drums = "-".join(d for d, _ in counts.most_common())
    bpm = estimate_bpm([t for t, _, _ in hits]) if len(hits) >= 8 else ""
    if not name:
        stamp = path.stem.replace("take_", "")
        name = f"{drums}_{bpm:.0f}bpm_{stamp}" if bpm else f"{drums}_{stamp}"
    out = out_dir / f"{name}.mid"
    write_midi(events, out)

    seconds = (hits[-1][0] - hits[0][0]) / 1000
    extra = f", dropped {dict(dropped)}" if dropped else ""
    print(f"{path.name} -> {out.name}: {len(hits)} hits, {seconds:.1f} s, ~{bpm} BPM, {drums}{extra}")
    return {"file": out.name, "source": path.name, "bpm": bpm, "seconds": f"{seconds:.1f}",
            "hits": len(hits), "drums": drums, "note": note or ""}


def read_index(out_dir):
    index = out_dir / "index.csv"
    return list(csv.DictReader(open(index, newline=""))) if index.exists() else []


def used_elsewhere(paths, out_dir):
    """Takes already in another groove folder's index -> {take name: that folder}."""
    wanted = {p.name for p in paths}
    used = {}
    for index in GROOVES.glob("*/index.csv"):
        if index.parent.resolve() != out_dir.resolve():
            used.update({r["source"]: index.parent for r in read_index(index.parent) if r["source"] in wanted})
    return used


def update_index(rows, out_dir):
    new_files = {r["file"] for r in rows}
    merged = [r for r in read_index(out_dir) if r["file"] not in new_files] + rows
    with open(out_dir / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(sorted(merged, key=lambda r: r["file"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("takes", nargs="+", help="take .csv (preferred, exact times) or .mid files; wildcards work")
    ap.add_argument("--name", help="output name (one take only; default: drums, tempo and timestamp)")
    ap.add_argument("--note", help="free-text note for the index, e.g. what was played")
    ap.add_argument("--dir", type=Path, default=OUT_DIR, help=f"output folder (default {OUT_DIR}; grooves/heldout for tests)")
    args = ap.parse_args()

    paths = [Path(p) for pattern in args.takes for p in (sorted(glob.glob(pattern)) or [pattern])]
    if args.name and len(paths) > 1:
        sys.exit("--name only works with a single take")
    clash = used_elsewhere(paths, args.dir)
    if clash:
        sys.exit("Already in another groove folder (a take can't be both training and held-out): "
                 + ", ".join(f"{take} in {folder}" for take, folder in clash.items()))
    args.dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in (prep(p, args.name, args.note, args.dir) for p in paths) if r]
    if rows:
        update_index(rows, args.dir)
        print(f"\n{len(rows)} takes in {args.dir}; index at {args.dir / 'index.csv'}")


if __name__ == "__main__":
    main()

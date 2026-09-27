"""Turn music someone brings into what the fly plays: sheet music, a MIDI file, or a take played on the kit.

The result is songs/<name>.mid, in the same format as the training grooves (channel 10, TD-07 notes, the first hit
LEAD_MS in), plus songs/<name>.json, a report of what went in. Notes the fly has no drum for are dropped here and
listed, instead of being skipped silently by the fly's encoder. songs/ is gitignored.

Inputs:
    .musicxml / .xml / .mxl / .mid / .txt   through sheet_to_midi.convert (repeats, dynamics, GM -> TD-07 notes)
    .mscz                                   MuseScore's own format, exported to MusicXML with MuseScore's CLI
    .csv                                    a kit take from midi_capture.py, cleaned like prep_takes.py does

Usage:
    python human/to_fly.py my_score.musicxml
    python human/to_fly.py song.mscz --name intro --seconds 20
    python human/to_fly.py takes/take_20260927_013000.csv
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import mido

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from fly.drums import NOTE_TO_VOICE  # noqa: E402  (stdlib only: the fly's own list of what it can play)

from drum_map import DRUM_CHANNEL, write_midi  # noqa: E402
from prep_takes import LEAD_MS, load_hits  # noqa: E402
from sheet_to_midi import NOTE_MS, SHEET_EXTS, convert, load_xml, read_midi  # noqa: E402

SONGS = REPO / "songs"
GROOVE_DIRS = ("train", "heldout")  # a song that is one of these gets labelled
FLY_X_REALTIME = 15  # trained fly on the T5600's 3060 Ti: ~15 s of wall time per simulated second
FLY_STARTUP_S = 30   # wiring + building the brain and body
FLY_TAIL_MS = 700    # fly.loop simulates 200 ms of preroll and 500 ms after the last cue
LONG_S = 30          # past this the wait gets long (~8 min), so suggest --seconds
MUSESCORE = ("mscore", "mscore4", "MuseScore4", "musescore", "mscore3",
             r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe", r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
             "/Applications/MuseScore 4.app/Contents/MacOS/mscore")


class SongError(Exception):
    """The input can't be turned into a song (unsupported, empty, no MuseScore, ...)."""


def find_musescore(explicit=None):
    for cand in ([explicit] if explicit else MUSESCORE):
        exe = shutil.which(cand) or (cand if Path(cand).is_file() else None)
        if exe:
            return exe
    raise SongError("MuseScore not found: install MuseScore 4, pass --musescore PATH, "
                    "or export the score as MusicXML (File > Export)")


def mscz_to_musicxml(path, out, musescore=None):
    exe = find_musescore(musescore)
    env = {**os.environ, "QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM", "offscreen")}  # no window on Linux
    done = subprocess.run([exe, "-o", str(out), str(path)], capture_output=True, text=True, env=env)
    if done.returncode or not Path(out).exists():
        raise SongError(f"MuseScore couldn't export {path.name}: {(done.stderr or done.stdout).strip()[-300:]}")


def untagged_notes(path):
    """Score notes sheet_to_midi skips without a word: no <instrument>, or one with no MIDI drum number."""
    root = load_xml(path)
    known = {mi.get("id") for mi in root.iter("midi-instrument") if mi.find("midi-unpitched") is not None}
    count = 0
    for n in root.iter("note"):
        if n.find("rest") is not None or n.find("grace") is not None:
            continue
        inst = n.find("instrument")
        count += inst is None or inst.get("id") not in known
    return count


def load(path, bpm=None, repeat=1, musescore=None):
    """-> ([(t_ms, note, velocity)], notes) where notes lists anything the reader skipped or cleaned."""
    path = Path(path)
    ext = path.suffix.lower()
    notes = {}
    if ext == ".csv":
        hits, dropped = load_hits(path)
        if dropped:
            notes["cleaned"] = dict(dropped)  # crosstalk, kick bounces, stray touches, unknown notes
        return hits, notes
    if ext == ".mscz":
        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / f"{path.stem}.musicxml"
            mscz_to_musicxml(path, xml, musescore)
            return load(xml, bpm, repeat)
    if ext not in SHEET_EXTS + (".mid", ".midi"):
        raise SongError(f"unsupported file type {ext!r}: use MusicXML, .mscz, .mid, a .txt grid or a take .csv")
    try:
        events, _ = convert(path, bpm, repeat)
    except SystemExit as e:  # sheet_to_midi reports bad input by exiting
        raise SongError(str(e)) from None
    if ext in (".musicxml", ".xml", ".mxl"):
        skipped = untagged_notes(path)
        if skipped:
            notes["skipped_untagged"] = skipped
    return [(t, m.note, m.velocity) for t, m in events if m.type == "note_on"], notes


def prepare(hits, seconds=None):
    """Keep what the fly can play, one hit per drum per instant, the first hit at LEAD_MS, cropped to `seconds`.
    -> (hits, dropped Counter of note numbers)."""
    dropped = Counter(n for _, n, _ in hits if n not in NOTE_TO_VOICE)
    loudest = {}
    for t, n, v in hits:
        if n in NOTE_TO_VOICE:
            key = (round(t, 1), n)  # a note written in two parts, or a chord doubled, is still one hit
            loudest[key] = max(v, loudest.get(key, 0))
    kept = sorted((t, n, v) for (t, n), v in loudest.items())
    if not kept:
        raise SongError("no notes the fly can play" + (f" (only {sorted(dropped)})" if dropped else ""))
    shift = kept[0][0] - LEAD_MS
    kept = [(t - shift, n, v) for t, n, v in kept]
    if seconds:
        kept = [h for h in kept if h[0] < LEAD_MS + seconds * 1000]
    return kept, dropped


def known_groove(hits, grooves=REPO / "grooves"):
    """If these hits are one of the team's grooves -> "train/<file>" or "heldout/<file>", else None."""
    def shape(hs):  # (ms from the first hit, note), ordered the same way whatever order simultaneous hits came in
        t0 = min(t for t, _, _ in hs)
        return sorted((round(t - t0), n) for t, n, _ in hs)

    mine = shape(hits)
    for folder in GROOVE_DIRS:
        for mid in sorted((grooves / folder).glob("*.mid")):
            ref = read_midi(mid)[0]
            if len(ref) == len(mine) and all(rn == n and abs(rt - t) <= 2 for (rt, rn), (t, n) in zip(shape(ref), mine)):
                return f"{folder}/{mid.name}"
    return None


def fly_seconds(hits):
    """Rough wall time for a trained fly run on the T5600."""
    return round(FLY_X_REALTIME * (hits[-1][0] + FLY_TAIL_MS) / 1000 + FLY_STARTUP_S)


def slug(text):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_") or "song"


def make_song(path, name=None, bpm=None, repeat=1, seconds=None, musescore=None, out_dir=SONGS):
    """Input file -> songs/<name>.mid + .json. Returns the report (also written as the .json)."""
    path = Path(path)
    raw, notes = load(path, bpm, repeat, musescore)
    hits, dropped = prepare(raw, seconds)
    events = []
    for t, n, v in hits:
        events.append((t, mido.Message("note_on", channel=DRUM_CHANNEL, note=n, velocity=v)))
        events.append((t + NOTE_MS, mido.Message("note_off", channel=DRUM_CHANNEL, note=n, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1].type == "note_on"))  # note_offs first at equal times

    name = slug(name or path.stem)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = out_dir / f"{name}.mid"
    write_midi(events, mid)
    voices = Counter(NOTE_TO_VOICE[n].name for _, n, _ in hits)
    report = {
        "name": name, "mid": str(mid), "source": path.name, "hits": len(hits),
        "seconds": round((hits[-1][0] - hits[0][0]) / 1000, 1), "cropped_to": seconds,
        "voices": dict(voices.most_common()), "dropped_notes": {str(n): c for n, c in sorted(dropped.items())},
        **notes, "known_groove": known_groove(hits), "fly_seconds": fly_seconds(hits),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(report, indent=1))
    return report


def describe(r):
    lines = [f"{r['source']} -> {r['mid']}: {r['hits']} hits over {r['seconds']} s",
             "  " + ", ".join(f"{v} {c}" for v, c in r["voices"].items())]
    if r["dropped_notes"]:
        lines.append("  dropped (no drum the fly has): "
                     + ", ".join(f"note {n} x{c}" for n, c in r["dropped_notes"].items()))
    if r.get("skipped_untagged"):
        lines.append(f"  warning: {r['skipped_untagged']} score notes have no drum instrument and were skipped")
    if r.get("cleaned"):
        lines.append(f"  cleaned from the take: {r['cleaned']}")
    if r["known_groove"]:
        lines.append(f"  note: this is the groove grooves/{r['known_groove']}"
                     + (" (the fly trained on it)" if r["known_groove"].startswith("train/") else ""))
    lines.append(f"  fly time: ~{r['fly_seconds'] / 60:.0f} min on the T5600")
    if r["seconds"] > LONG_S and not r["cropped_to"]:
        lines.append(f"  warning: longer than {LONG_S} s; --seconds N plays only the first N seconds")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path, help="sheet music, .mid, .txt grid, .mscz, or a take .csv")
    ap.add_argument("--name", help="song name (default: the file name)")
    ap.add_argument("--bpm", type=float, help="tempo override for sheet music")
    ap.add_argument("--repeat", type=int, default=1, help="play the piece N times")
    ap.add_argument("--seconds", type=float, help="keep only the first N seconds (from the first hit)")
    ap.add_argument("--musescore", help="path to the MuseScore executable (for .mscz)")
    ap.add_argument("--out-dir", type=Path, default=SONGS, help=f"default {SONGS}")
    args = ap.parse_args()
    try:
        report = make_song(args.file, args.name, args.bpm, args.repeat, args.seconds, args.musescore, args.out_dir)
    except SongError as e:
        sys.exit(f"{args.file}: {e}")
    print(describe(report))


if __name__ == "__main__":
    main()

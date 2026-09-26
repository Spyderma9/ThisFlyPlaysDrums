"""Turn a kit take (or any drum .mid) into drum sheet music (MusicXML).

Hits are snapped to a 16th-note grid at the tempo you played to. Edges and rims fold onto their drum,
crosstalk, kick bounces and stray touches are dropped, and every note keeps its exact velocity. Loud hits are
marked as accents and soft snare hits as ghost notes. Open the result in MuseScore, or read it back
with sheet_to_midi.py.

Usage:
    python human/take_to_sheet.py takes/take_20260926_014618.mid --bpm 100
    python human/take_to_sheet.py take.mid --bpm 100 --start 2420 -o groove.musicxml
"""
import argparse
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from drum_map import DRUMS, SAME_DRUM, HitFilter
from sheet_to_midi import ACCENT_BOOST, GHOST_SCALE, read_midi

DIVISIONS = 4      # per quarter note, so one division is a 16th
BAR = 16           # 16ths in a 4/4 bar
ACCENT_MIN = 110   # velocity shown as an accent
GHOST_MAX = 60     # snare velocity shown as a ghost note

# TD-07 note -> (display step, octave, notehead, voice). Voice 1 is hands (stems up), 2 is feet (stems down).
NOTATION = {
    36: ("F", 4, None, 2),        # kick
    44: ("D", 4, "x", 2),         # hi-hat pedal
    38: ("C", 5, None, 1),        # snare
    37: ("C", 5, "circle-x", 1),  # cross-stick
    48: ("E", 5, None, 1),        # tom1
    45: ("D", 5, None, 1),        # tom2
    43: ("A", 4, None, 1),        # tom3
    42: ("G", 5, "x", 1),         # closed hat
    46: ("G", 5, "circle-x", 1),  # open hat
    49: ("A", 5, "x", 1),         # crash
    51: ("F", 5, "x", 1),         # ride
    53: ("F", 5, "diamond", 1),   # ride bell
}
# length in 16ths -> (note type, dots)
TYPES = {16: ("whole", 0), 12: ("half", 1), 8: ("half", 0), 6: ("quarter", 1), 4: ("quarter", 0),
         3: ("eighth", 1), 2: ("eighth", 0), 1: ("16th", 0)}


# ---------- reading and snapping ----------

def load_hits(path, raw):
    hits, _ = read_midi(path)
    hit_filter = HitFilter()
    out, dropped, unmapped = [], Counter(), set()
    for t, note, vel in hits:
        note = SAME_DRUM.get(note, note)
        reason = None if raw else hit_filter.check(t, note, vel)
        if reason:
            dropped[reason] += 1
            continue
        if note not in NOTATION:
            unmapped.add(note)
            continue
        out.append((t, note, vel))
    if unmapped:
        print(f"warning: no notation for notes {sorted(unmapped)}; left out")
    return out, dropped


def find_start(times, step_ms):
    """The grid phase that fits the hits best, placed at the grid point nearest the first hit."""
    def misfit(off):
        return sum(abs((t - off) / step_ms - round((t - off) / step_ms)) for t in times)

    phase = min((step_ms * i / 50 for i in range(50)), key=misfit)
    return phase + round((times[0] - phase) / step_ms) * step_ms


def snap(hits, start_ms, step_ms):
    """-> {(step, note): velocity}, list of timing errors in ms, hits before the start."""
    grid, errors, early = {}, [], 0
    for t, note, vel in hits:
        x = (t - start_ms) / step_ms
        step = round(x)
        if step < 0:
            early += 1
            continue
        errors.append(abs(x - step) * step_ms)
        grid[(step, note)] = max(vel, grid.get((step, note), 0))  # two hits in one 16th: keep the louder
    return grid, errors, early


# ---------- writing MusicXML ----------

def pieces(pos, length):
    """Split a span of 16ths into note/rest values that don't cross a beat or the half bar mid-note."""
    while length > 0:
        allowed = length
        if pos % 4:
            allowed = min(allowed, 4 - pos % 4)
        if pos % 8:
            allowed = min(allowed, 8 - pos % 8)
        dur = max(d for d in TYPES if d <= allowed)
        yield dur
        pos += dur
        length -= dur


def velocity_marks(note, vel):
    """Pick the accent/ghost mark to show, plus the dynamics % that makes sheet_to_midi read vel back exactly."""
    if vel >= ACCENT_MIN:
        return "accent", (vel - ACCENT_BOOST["accent"]) * 100 / 90
    if note == 38 and vel <= GHOST_MAX:
        return "ghost", vel / GHOST_SCALE * 100 / 90
    return None, vel * 100 / 90


def add_note(measure, dur, voice, note=None, vel=None, chord=False):
    el = ET.SubElement(measure, "note")
    if note is None:
        ET.SubElement(el, "rest")
    else:
        if chord:
            ET.SubElement(el, "chord")
        step, octave, head, _ = NOTATION[note]
        unp = ET.SubElement(el, "unpitched")
        ET.SubElement(unp, "display-step").text = step
        ET.SubElement(unp, "display-octave").text = str(octave)
    ET.SubElement(el, "duration").text = str(dur)
    if note is not None:
        ET.SubElement(el, "instrument", id=f"I{note}")
    ET.SubElement(el, "voice").text = str(voice)
    kind, dots = TYPES[dur]
    ET.SubElement(el, "type").text = kind
    for _ in range(dots):
        ET.SubElement(el, "dot")
    if note is None:
        return
    ET.SubElement(el, "stem").text = "up" if voice == 1 else "down"
    mark, percent = velocity_marks(note, vel)
    if head or mark == "ghost":
        nh = ET.SubElement(el, "notehead")
        nh.text = head or "normal"
        if mark == "ghost":
            nh.set("parentheses", "yes")
    if mark == "accent":
        ET.SubElement(ET.SubElement(ET.SubElement(el, "notations"), "articulations"), "accent")
    el.set("dynamics", f"{percent:.2f}")


def write_voice(measure, voice, onsets):
    """onsets: {position_in_bar: {note: velocity}} for one voice of one bar."""
    positions = sorted(onsets)
    cursor = 0
    for i, pos in enumerate(positions):
        for dur in pieces(cursor, pos - cursor):  # rests before the note
            add_note(measure, dur, voice)
            cursor += dur
        end = positions[i + 1] if i + 1 < len(positions) else BAR
        durs = list(pieces(pos, end - pos))
        for j, (note, vel) in enumerate(sorted(onsets[pos].items())):
            add_note(measure, durs[0], voice, note, vel, chord=j > 0)
        cursor = pos + durs[0]
        for dur in durs[1:]:  # the rest of the gap is silence
            add_note(measure, dur, voice)
            cursor += dur


def build_score(grid, bpm, title):
    used = sorted({note for _, note in grid})
    root = ET.Element("score-partwise", version="4.0")
    ET.SubElement(ET.SubElement(root, "work"), "work-title").text = title
    sp = ET.SubElement(ET.SubElement(root, "part-list"), "score-part", id="P1")
    ET.SubElement(sp, "part-name").text = "Drumset"
    for note in used:
        ET.SubElement(ET.SubElement(sp, "score-instrument", id=f"I{note}"), "instrument-name").text = DRUMS[note]
    for note in used:
        mi = ET.SubElement(sp, "midi-instrument", id=f"I{note}")
        ET.SubElement(mi, "midi-channel").text = "10"
        ET.SubElement(mi, "midi-unpitched").text = str(note + 1)  # 1-based

    part = ET.SubElement(root, "part", id="P1")
    bars = max(step for step, _ in grid) // BAR + 1
    for b in range(bars):
        measure = ET.SubElement(part, "measure", number=str(b + 1))
        if b == 0:
            attrs = ET.SubElement(measure, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
            ET.SubElement(ET.SubElement(attrs, "key"), "fifths").text = "0"
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = "4"
            ET.SubElement(time, "beat-type").text = "4"
            ET.SubElement(ET.SubElement(attrs, "clef"), "sign").text = "percussion"
            direction = ET.SubElement(measure, "direction", placement="above")
            metro = ET.SubElement(ET.SubElement(direction, "direction-type"), "metronome")
            ET.SubElement(metro, "beat-unit").text = "quarter"
            ET.SubElement(metro, "per-minute").text = f"{bpm:g}"
            ET.SubElement(direction, "sound", tempo=f"{bpm:g}")

        voices = {1: {}, 2: {}}
        for (step, note), vel in grid.items():
            if step // BAR == b:
                voices[NOTATION[note][3]].setdefault(step % BAR, {})[note] = vel
        if voices[1]:
            write_voice(measure, 1, voices[1])
        else:
            rest = ET.SubElement(measure, "note")
            ET.SubElement(rest, "rest", measure="yes")
            ET.SubElement(rest, "duration").text = str(BAR)
            ET.SubElement(rest, "voice").text = "1"
        if voices[2]:
            ET.SubElement(ET.SubElement(measure, "backup"), "duration").text = str(BAR)
            write_voice(measure, 2, voices[2])
    return ET.ElementTree(root), bars


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="a take or other drum .mid")
    ap.add_argument("--bpm", type=float, required=True, help="the tempo you played to")
    ap.add_argument("--start", type=float, metavar="MS", help="time of the first downbeat in ms (default: found from the hits)")
    ap.add_argument("--raw", action="store_true", help="keep crosstalk, kick bounces and stray touches")
    ap.add_argument("-o", "--out", help="output .musicxml (default: next to the input)")
    args = ap.parse_args()

    path = Path(args.file)
    hits, dropped = load_hits(path, args.raw)
    if not hits:
        sys.exit("No drum hits found.")
    step_ms = 60000 / args.bpm / 4
    start = args.start if args.start is not None else find_start([t for t, _, _ in hits], step_ms)
    grid, errors, early = snap(hits, start, step_ms)
    if not grid:
        sys.exit(f"No hits after the start ({start:.0f} ms).")

    tree, bars = build_score(grid, args.bpm, path.stem)
    ET.indent(tree)
    out = Path(args.out) if args.out else path.with_suffix(".musicxml")
    tree.write(out, encoding="UTF-8", xml_declaration=True)

    mean_err = sum(errors) / len(errors)
    print(f"Wrote {out}: {bars} bars, {len(grid)} notes at {args.bpm:g} BPM, first downbeat at {start:.0f} ms")
    print(f"Timing: {mean_err:.0f} ms off the 16th grid on average (max {max(errors):.0f} ms; a 16th is {step_ms:.0f} ms)")
    if mean_err > step_ms / 5:
        print("warning: that's a lot. Check --bpm is the tempo you played to, or set --start.")
    skipped = [f"{n} {reason}" for reason, n in dropped.items()] + ([f"{early} before the start"] if early else [])
    if skipped:
        print("Dropped: " + ", ".join(skipped))


if __name__ == "__main__":
    main()

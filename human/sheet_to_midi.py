"""Convert drum sheet music into a fly-ready .mid (channel 10, TD-07 note numbers).

Inputs:
    .musicxml / .xml / .mxl   MuseScore: File > Export > MusicXML (drumset staff)
    .mid                      any drum MIDI file (e.g. MuseScore: File > Export > MIDI)
    .txt                      text drum grid, see grooves/rock_beat.txt

Usage:
    python human/sheet_to_midi.py grooves/rock_beat.txt
    python human/sheet_to_midi.py my_score.musicxml --bpm 90 --repeat 2 -o out.mid
"""
import argparse
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import mido

from drum_map import DRUM_CHANNEL, GRID_NAMES, normalize, summarize, write_midi

NOTE_MS = 50  # length of each drum note; drums only care about the onset
GRID_VELOCITY = {"x": 90, "X": 120, "o": 50}  # normal, accent, ghost

# MusicXML loudness. With no dynamic marking, notes play at DEFAULT_VELOCITY.
DEFAULT_VELOCITY = 90
DYNAMIC_VELOCITY = {  # MuseScore's playback values
    "pppppp": 5, "ppppp": 10, "pppp": 12, "ppp": 16, "pp": 33, "p": 49, "mp": 64,
    "mf": 80, "f": 96, "ff": 112, "fff": 126, "ffff": 127, "fffff": 127, "ffffff": 127,
}
# One-shot marks: loud on the note they're attached to; fp/sfp then drop to piano.
SFORZANDO = {"sf": 112, "sfz": 112, "sffz": 127, "fz": 112, "rf": 104, "rfz": 104,
             "fp": 96, "sfp": 112, "sfpp": 112}
AFTER_SFORZANDO = {"fp": "p", "sfp": "p", "sfpp": "pp"}
ACCENT_BOOST = {"accent": 30, "strong-accent": 40}  # > and ^ (marcato)
GHOST_SCALE = 0.5  # parenthesised noteheads


# ---------- text drum grid ----------

def read_grid(path, bpm_override=None):
    """Rows in a block play together; blank lines separate blocks, which play one after another."""
    bpm, steps = 120.0, 16
    hits, block, block_start = [], {}, 0  # hits: (step_index, note, velocity)

    def flush():
        nonlocal block, block_start
        length = max((len(p) for p in block.values()), default=0)
        for note, pattern in block.items():
            for i, ch in enumerate(pattern):
                if ch in GRID_VELOCITY:
                    hits.append((block_start + i, note, GRID_VELOCITY[ch]))
                elif ch not in "-.":
                    sys.exit(f"{path}: unknown grid character {ch!r} (use x X o - .)")
        block_start += length
        block = {}

    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.split("#")[0].strip()
        if not line:
            flush()
            continue
        key, _, rest = line.partition(" ")
        if key.lower() == "bpm":
            bpm = float(rest)
        elif key.lower() == "steps":
            steps = int(rest)
        else:
            name = key.upper()
            note = int(name) if name.isdigit() else GRID_NAMES.get(name)
            if note is None:
                sys.exit(f"{path}:{lineno}: unknown row name {key!r}. Known: {', '.join(GRID_NAMES)}")
            pattern = rest.replace(" ", "").replace("|", "")
            block[note] = block.get(note, "") + pattern
    flush()

    bpm = bpm_override or bpm
    step_ms = 60000 / bpm * 4 / steps  # 4/4 bar split into `steps` steps
    return [(i * step_ms, note, vel) for i, note, vel in hits], block_start * step_ms


# ---------- MusicXML ----------

def load_xml(path):
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path) as z:
            container = ET.fromstring(z.read("META-INF/container.xml"))
            rootfile = container.find(".//rootfile").get("full-path")
            return ET.fromstring(z.read(rootfile))
    return ET.parse(path).getroot()


def read_musicxml(path, bpm_override):
    root = load_xml(path)
    if root.tag != "score-partwise":
        sys.exit(f"{path}: only score-partwise MusicXML is supported (MuseScore's default)")

    # instrument id -> MIDI note (midi-unpitched is 1-based)
    instruments = {}
    for mi in root.iter("midi-instrument"):
        unp = mi.find("midi-unpitched")
        if unp is not None:
            instruments[mi.get("id")] = int(unp.text) - 1

    # Parse each measure (merged across parts) into hits/tempos/dynamics relative to the measure start.
    # Dynamics can only be resolved in play order (repeats), so hits keep their modifiers until then.
    measures = []
    for pnum, part in enumerate(root.iter("part")):
        divisions = 1
        for idx, measure in enumerate(part.iter("measure")):
            if idx == len(measures):
                measures.append({"len": 0.0, "hits": [], "tempos": [], "dyn": [], "forward": False,
                                 "backward": 0, "endings": set()})
            m = measures[idx]
            pos, last_start, measure_len = 0, 0, 0
            for el in measure:
                if el.tag == "attributes" and el.find("divisions") is not None:
                    divisions = int(el.find("divisions").text)
                elif el.tag == "barline":
                    rep, end = el.find("repeat"), el.find("ending")
                    if rep is not None and rep.get("direction") == "forward":
                        m["forward"] = True
                    elif rep is not None:
                        m["backward"] = int(rep.get("times", 2))
                    if end is not None:
                        m["endings"] |= {int(n) for n in end.get("number", "").replace(",", " ").split() if n.isdigit()}
                elif el.tag in ("direction", "sound"):
                    snd = el if el.tag == "sound" else el.find(".//sound")
                    if snd is not None and snd.get("tempo"):
                        m["tempos"].append((pos / divisions, float(snd.get("tempo"))))
                    marks = [mk for d in el.iter("dynamics") for mk in dynamic_marks(d)]
                    if snd is not None and snd.get("dynamics"):  # exact value wins over the mark's default
                        marks = [mk for mk in marks if mk[0] == "once"] + [("level", percent_to_velocity(snd.get("dynamics")))]
                    m["dyn"] += [(pnum, pos / divisions, kind, vel) for kind, vel in marks]
                elif el.tag == "backup":
                    pos -= int(el.find("duration").text)
                elif el.tag == "forward":
                    pos += int(el.find("duration").text)
                elif el.tag == "note":
                    if el.find("grace") is not None:
                        continue
                    dur = int(el.findtext("duration", "0"))
                    start = last_start if el.find("chord") is not None else pos
                    if el.find("chord") is None:
                        last_start = pos
                        pos += dur
                    tie_stop = any(t.get("type") == "stop" for t in el.findall("tie"))
                    inst = el.find("instrument")
                    if el.find("rest") is None and not tie_stop and inst is not None:
                        note = instruments.get(inst.get("id"))
                        if note is not None:
                            rel = start / divisions
                            boost = sum(ACCENT_BOOST.get(a.tag, 0) for a in el.iterfind("notations/articulations/*"))
                            ghost = any(h.get("parentheses") == "yes" for h in el.iter("notehead"))
                            fixed = percent_to_velocity(el.get("dynamics")) if el.get("dynamics") else None
                            for d in el.iterfind("notations/dynamics"):
                                for kind, vel in dynamic_marks(d):
                                    if kind == "once":
                                        fixed = vel
                                    else:
                                        m["dyn"].append((pnum, rel, kind, vel))
                            m["hits"].append((pnum, rel, note, boost, ghost, fixed))
                measure_len = max(measure_len, pos)
            m["len"] = max(m["len"], measure_len / divisions)

    raw = []      # (quarter_offset, note, velocity)
    tempos = {}   # quarter_offset -> bpm
    levels = {}   # part -> current dynamic velocity
    q = 0.0
    for idx in play_order(measures):
        m = measures[idx]
        once = {}  # (part, rel) -> sforzando velocity
        # dynamics at a position apply to notes at that same position, so they sort first
        items = sorted([(d[1], 0, d) for d in m["dyn"]] + [(h[1], 1, h) for h in m["hits"]], key=lambda x: x[:2])
        for rel, is_hit, item in items:
            if not is_hit:
                pnum, _, kind, vel = item
                if kind == "level":
                    levels[pnum] = vel
                else:
                    once[(pnum, rel)] = vel
                continue
            pnum, _, note, boost, ghost, fixed = item
            vel = (fixed or once.get((pnum, rel)) or levels.get(pnum, DEFAULT_VELOCITY)) + boost
            if ghost:
                vel *= GHOST_SCALE
            raw.append((q + rel, note, max(1, min(127, round(vel)))))
        tempos.update({q + rel: bpm for rel, bpm in m["tempos"]})
        q += m["len"]

    if bpm_override:
        tempos = {0.0: bpm_override}
    tempos.setdefault(0.0, 120.0)
    return quarters_to_ms(raw, tempos), quarters_to_ms([(q, 0, 0)], tempos)[0][0]


def dynamic_marks(dynamics_el):
    """<dynamics><mf/></dynamics> -> [("level", 80)]; sforzandos give ("once", vel)."""
    marks = []
    for mark in dynamics_el:
        if mark.tag in DYNAMIC_VELOCITY:
            marks.append(("level", DYNAMIC_VELOCITY[mark.tag]))
        elif mark.tag in SFORZANDO:
            marks.append(("once", SFORZANDO[mark.tag]))
            if mark.tag in AFTER_SFORZANDO:
                marks.append(("level", DYNAMIC_VELOCITY[AFTER_SFORZANDO[mark.tag]]))
    return marks


def percent_to_velocity(value):
    """MusicXML dynamics attributes are a percentage of forte, which is velocity 90."""
    return max(1, min(127, round(float(value) * 90 / 100)))


def play_order(measures):
    """Expand repeat signs and 1st/2nd endings into the order measures are played."""
    order, i, start, rep_pass, jumps, jumped = [], 0, 0, 1, {}, False
    while i < len(measures):
        m = measures[i]
        if m["forward"] and not jumped:
            start, rep_pass = i, 1
        jumped = False
        if m["endings"] and rep_pass not in m["endings"]:
            i += 1  # volta for a different pass
            continue
        order.append(i)
        if m["backward"]:
            done = jumps.get(i, 1)
            if done < m["backward"]:
                jumps[i] = done + 1
                rep_pass = done + 1
                i, jumped = start, True
                continue
            # Like MuseScore, a later backward repeat with no forward sign of its own
            # goes back to the last forward sign (or the start of the piece).
        i += 1
    return order


def quarters_to_ms(raw, tempos):
    changes = sorted(tempos.items())

    def to_ms(qo):
        ms, prev_q, prev_bpm = 0.0, 0.0, changes[0][1]
        for cq, cbpm in changes:
            if cq >= qo:
                break
            ms += (cq - prev_q) * 60000 / prev_bpm
            prev_q, prev_bpm = cq, cbpm
        return ms + (qo - prev_q) * 60000 / prev_bpm

    return [(to_ms(qo), note, vel) for qo, note, vel in raw]


# ---------- MIDI ----------

def read_midi(path):
    hits, t = [], 0.0
    for msg in mido.MidiFile(path):  # iterating gives msg.time in seconds, tempo-aware
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            hits.append((t * 1000, msg.note, msg.velocity, msg.channel))
    drum_hits = [h for h in hits if h[3] == DRUM_CHANNEL]
    if hits and not drum_hits:
        print("warning: no notes on channel 10; treating every note as a drum hit")
        drum_hits = hits
    return [h[:3] for h in drum_hits], t * 1000


# ---------- main ----------

SHEET_EXTS = (".txt", ".musicxml", ".xml", ".mxl")


def convert(path, bpm=None, repeat=1):
    """Read any supported file -> ([(t_ms, mido.Message), ...] sorted, total length in ms)."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        hits, length_ms = read_grid(path, bpm)
    elif ext in (".musicxml", ".xml", ".mxl"):
        hits, length_ms = read_musicxml(path, bpm)
    elif ext in (".mid", ".midi"):
        hits, length_ms = read_midi(path)
    else:
        sys.exit(f"Unsupported file type {ext!r}")
    if not hits:
        sys.exit("No drum hits found.")

    unknown = set()
    events = []
    for rep in range(repeat):
        offset = rep * length_ms
        for t, note, vel in hits:
            note, known = normalize(note, sheet=ext != ".txt")  # grid rows already use TD-07 notes
            if not known:
                unknown.add(note)
            events.append((offset + t, mido.Message("note_on", channel=DRUM_CHANNEL, note=note, velocity=vel)))
            events.append((offset + t + NOTE_MS, mido.Message("note_off", channel=DRUM_CHANNEL, note=note, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1].type == "note_on"))  # note_offs first at equal times
    if unknown:
        print(f"warning: notes not in the TD-07 map: {sorted(unknown)}")
    return events, length_ms * repeat


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("-o", "--out", help="output .mid (default: next to the input)")
    ap.add_argument("--bpm", type=float, help="override the score's tempo (MusicXML and .txt grids)")
    ap.add_argument("--repeat", type=int, default=1, help="play the whole piece N times")
    args = ap.parse_args()

    path = Path(args.file)
    events, length_ms = convert(path, args.bpm, args.repeat)
    is_mid = path.suffix.lower() in (".mid", ".midi")
    out = Path(args.out) if args.out else path.with_name(path.stem + ("_fly.mid" if is_mid else ".mid"))
    write_midi(events, out)
    print(f"Wrote {out} ({sum(e[1].type == 'note_on' for e in events)} hits, {length_ms / 1000:.1f} s)")
    summarize(events)


if __name__ == "__main__":
    main()

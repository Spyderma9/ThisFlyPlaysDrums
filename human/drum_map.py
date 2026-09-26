"""Shared drum note map and MIDI helpers, so kit takes and sheet music use identical notes."""
from collections import defaultdict

import mido

DRUM_CHANNEL = 9  # MIDI channel 10 (0-based 9), the General MIDI drum channel
TICKS_PER_BEAT = 480
TEMPO_US = 500000  # 120 BPM; only affects how the .mid is displayed

# TD-07 default note map (close to General MIDI). Unknown notes print as "?".
DRUMS = {
    36: "kick",
    38: "snare", 40: "snare_rim", 37: "snare_xstick",
    48: "tom1", 50: "tom1_rim",
    45: "tom2", 47: "tom2_rim",
    43: "tom3", 58: "tom3_rim",
    42: "hat_closed", 22: "hat_closed_edge",
    46: "hat_open", 26: "hat_open_edge",
    44: "hat_pedal",
    49: "crash", 55: "crash_edge",
    51: "ride", 59: "ride_edge", 53: "ride_bell",
}

# General MIDI notes that notation apps use but the TD-07 map doesn't, folded onto the nearest TD-07 pad.
GM_TO_TD07 = {
    35: 36,          # acoustic bass drum -> kick
    39: 38,          # hand clap -> snare
    41: 43,          # low floor tom -> tom3
    52: 49, 57: 49,  # china, crash 2 -> crash
    54: 42,          # tambourine -> closed hat
    56: 53,          # cowbell -> ride bell
}

# Short row names for text drum grids.
GRID_NAMES = {
    "BD": 36, "K": 36, "KICK": 36,
    "SD": 38, "SN": 38, "SNARE": 38, "RIM": 40, "RS": 40, "XS": 37,
    "HH": 42, "HC": 42, "HO": 46, "HP": 44,
    "T1": 48, "T2": 45, "T3": 43, "FT": 43,
    "CR": 49, "CC": 49, "RD": 51, "RC": 51, "RB": 53,
}


def normalize(note):
    """Map a GM drum note onto the TD-07 map. Returns (note, known)."""
    note = GM_TO_TD07.get(note, note)
    return note, note in DRUMS


def write_midi(events, path):
    """Write [(t_ms, mido.Message), ...] (sorted by time) to a .mid file."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US, time=0))
    prev = 0.0
    for t, msg in events:
        delta = int(round(mido.second2tick((t - prev) / 1000, TICKS_PER_BEAT, TEMPO_US)))
        track.append(msg.copy(time=delta))
        prev = t
    mid.save(path)


def summarize(events):
    vels = defaultdict(list)
    for _, msg in events:
        if msg.type == "note_on" and msg.velocity > 0:
            vels[msg.note].append(msg.velocity)
    if not vels:
        print("No hits recorded.")
        return
    print(f"\n{'note':>4}  {'drum':<16}{'hits':>5}  velocity")
    for note in sorted(vels):
        v = vels[note]
        print(f"{note:>4}  {DRUMS.get(note, '?'):<16}{len(v):>5}  {min(v)}-{max(v)}")

"""TD-07 drum note map and shared MIDI helpers."""
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

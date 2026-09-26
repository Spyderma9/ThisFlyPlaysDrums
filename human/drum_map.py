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


# Sheet music uses GM drum names, where some numbers mean something else on the TD-07
# (GM 40 is "electric snare", the TD-07's snare rim; GM 47/50 are toms, the TD-07's tom rims).
# Only applied to sheet music, never to kit takes.
SHEET_GM_TO_TD07 = {
    **GM_TO_TD07,
    40: 38,          # electric snare -> snare
    50: 48, 47: 45,  # high tom -> tom1, low-mid tom -> tom2
    55: 49,          # splash -> crash
    59: 51,          # ride 2 -> ride
}


# Edge and rim zones count as the drum they're on, for notation and for comparing takes with scores.
SAME_DRUM = {
    22: 42, 26: 46,          # hat edges -> hat
    40: 38,                  # snare rim -> snare
    50: 48, 47: 45, 58: 43,  # tom rims -> toms
    55: 49, 59: 51,          # crash / ride edges -> crash / ride
}


def normalize(note, sheet=False):
    """Map a GM drum note onto the TD-07 map. Returns (note, known)."""
    note = (SHEET_GM_TO_TD07 if sheet else GM_TO_TD07).get(note, note)
    return note, note in DRUMS


# Hits the kit sends that nobody played. The kick pad is bolted to the rack, so hard hits on other
# pads shake it into quiet phantom kicks (velocity 7-18, 0-25 ms later), and hard kicks bounce the beater.
KICK = 36
STRAY_MAX = 11      # kicks this quiet are stray touches (other pads keep their soft hits)
CROSSTALK_MS = 40   # a quiet kick this soon after another pad's hit is crosstalk...
CROSSTALK_MAX = 30  # ...if it's at most this loud
BOUNCE_MS = 90      # a kick this soon after a kick...
BOUNCE_RATIO = 0.6  # ...and under 60% as loud is the beater bouncing


class HitFilter:
    """Feed hits in time order; check() returns why a hit is fake ("stray", "crosstalk", "bounce") or None."""

    def __init__(self):
        self.last_other = float("-inf")  # time of the last real non-kick hit
        self.last_kick = None            # (time, velocity) of the last real kick

    def check(self, t, note, vel):
        if note != KICK:
            self.last_other = t
            return None
        if vel <= STRAY_MAX:
            return "stray"
        if vel <= CROSSTALK_MAX and t - self.last_other < CROSSTALK_MS:
            return "crosstalk"
        if self.last_kick and t - self.last_kick[0] < BOUNCE_MS and vel < self.last_kick[1] * BOUNCE_RATIO:
            return "bounce"
        self.last_kick = (t, vel)
        return None


def write_midi(events, path):
    """Write [(t_ms, mido.Message), ...] (sorted by time) to a .mid file."""
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=TEMPO_US, time=0))
    prev_tick = 0
    for t, msg in events:
        # round absolute times, not gaps, so rounding errors don't add up over a long file
        tick = int(round(mido.second2tick(t / 1000, TICKS_PER_BEAT, TEMPO_US)))
        track.append(msg.copy(time=tick - prev_tick))
        prev_tick = tick
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

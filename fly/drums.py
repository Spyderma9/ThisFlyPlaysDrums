"""The drums the fly plays, keyed by TD-07 note number.

Each voice gets one cue-neuron group (encoder side) and one limb (decoder and body side).
Note numbers follow human/drum_map.py (DRUMS). Pad variants (hat edge/open, snare rim/cross-stick)
fold onto the same voice. Crash, toms and ride are dropped by the encoder until the fly's kit has
those pads. The fly side doesn't import human/, so the server doesn't need the laptop's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

DRUM_CHANNEL = 9  # MIDI channel 10, zero-based as in mido


@dataclass(frozen=True)
class Voice:
    name: str
    notes: frozenset[int]  # TD-07 note numbers that count as this drum
    out_note: int  # note written to hits.mid when the fly strikes this pad
    limb: str  # which leg plays it; the body model maps this to joints


VOICES: tuple[Voice, ...] = (
    # closed, closed edge, open, open edge. The hat pedal (44) is a foot and isn't included.
    Voice("hihat", frozenset({42, 22, 46, 26}), 42, "right_foreleg"),
    Voice("snare", frozenset({38, 40, 37}), 38, "left_foreleg"),  # head, rim, cross-stick
    Voice("kick", frozenset({36}), 36, "kick_leg"),
)

NOTE_TO_VOICE: dict[int, Voice] = {n: v for v in VOICES for n in v.notes}

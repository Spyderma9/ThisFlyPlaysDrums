"""The 12 drums the fly plays, keyed by TD-07 note number, and the legs that play each one.

The fly plays like a right-handed drummer on a kit placed within reach: the front legs hold the
sticks, the hind right leg works the kick pedal, the hind left leg the hi-hat pedal. Each drum gets one
cue-neuron group (encoder side). `limbs` lists every leg that moves to play it; the first is the one
that makes the sound. Edges and rims fold onto their drum, as in human/drum_map.py (SAME_DRUM).
The fly side doesn't import human/, so the server doesn't need the laptop's dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass

DRUM_CHANNEL = 9  # MIDI channel 10, zero-based as in mido
LEGS = ("front_left", "front_right", "hind_left", "hind_right")


@dataclass(frozen=True)
class Voice:
    name: str
    notes: frozenset[int]  # TD-07 note numbers that count as this drum
    out_note: int  # note written to hits.mid when the fly plays it
    limbs: tuple[str, ...]  # legs that move to play it, sounding leg first


VOICES: tuple[Voice, ...] = (
    Voice("kick", frozenset({36}), 36, ("hind_right",)),
    Voice("hat_pedal", frozenset({44}), 44, ("hind_left",)),
    Voice("snare", frozenset({38, 40}), 38, ("front_left",)),  # head, rim
    Voice("xstick", frozenset({37}), 37, ("front_left",)),
    Voice("hat_closed", frozenset({42, 22}), 42, ("front_right",)),  # bow, edge
    Voice("hat_open", frozenset({46, 26}), 46, ("front_right", "hind_left")),  # the pedal lifts to open it
    Voice("tom1", frozenset({48, 50}), 48, ("front_right", "front_left")),  # toms alternate sticks
    Voice("tom2", frozenset({45, 47}), 45, ("front_right", "front_left")),
    Voice("tom3", frozenset({43, 58}), 43, ("front_right", "front_left")),
    Voice("crash", frozenset({49, 55}), 49, ("front_right", "front_left")),  # left when the right is busy
    Voice("ride", frozenset({51, 59}), 51, ("front_right",)),
    Voice("ride_bell", frozenset({53}), 53, ("front_right",)),
)

NOTE_TO_VOICE: dict[int, Voice] = {n: v for v in VOICES for n in v.notes}

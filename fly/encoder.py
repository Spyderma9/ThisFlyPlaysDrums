"""Fixed encoder: MIDI -> per-drum cue-neuron firing rates.

Rule-based only. It translates what the fly senses and never sees a training target.

1. Keep channel-10 notes that appear in drums.NOTE_TO_VOICE, one timeline per voice.
2. Shift each onset `lookahead_ms` early so the fly can read ahead.
3. Put a burst of `burst_ms` on that voice, with rate scaled by velocity.

The output is a rate envelope in Hz. brain.py spreads each voice's rate over its cue group,
and fly-brain's Poisson generator turns it into spikes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from fly.drums import DRUM_CHANNEL, NOTE_TO_VOICE, VOICES

VOICE_NAMES = tuple(v.name for v in VOICES)


@dataclass
class Encoded:
    rates: np.ndarray  # [T, n_voices] firing rate in Hz, one row per dt step
    voices: tuple[str, ...]  # column order of `rates`
    dt_ms: float
    offset_ms: float  # sim time of score time 0; score time t_s is sim step (t_s*1000 + offset_ms) / dt_ms


def read_onsets(midi_path: Path) -> dict[str, list[tuple[float, int]]]:
    """Per-voice (onset seconds, velocity) lists from a channel-10 TD-07 MIDI file."""
    onsets: dict[str, list[tuple[float, int]]] = {name: [] for name in VOICE_NAMES}
    t = 0.0
    for msg in mido.MidiFile(midi_path):  # iterating a MidiFile yields delta times in seconds
        t += msg.time
        if msg.type == "note_on" and msg.velocity > 0 and msg.channel == DRUM_CHANNEL:
            voice = NOTE_TO_VOICE.get(msg.note)
            if voice is not None:
                onsets[voice.name].append((t, msg.velocity))
    return onsets


def encode_onsets(
    onsets: dict[str, list[tuple[float, int]]],
    lookahead_ms: float = 150.0,
    dt_ms: float = 1.0,
    burst_ms: float = 30.0,
    rate_min_hz: float = 50.0,
    rate_max_hz: float = 200.0,
    tail_ms: float = 500.0,
    preroll_ms: float = 0.0,
) -> Encoded:
    """Rate envelopes for already-parsed onsets. Bursts start `lookahead_ms` before each note.
    preroll_ms: extra silent sim time before score time 0 (e.g. to close the hi-hat before the music starts)."""
    offset_ms = lookahead_ms + preroll_ms  # so a note at score time 0 still gets its full early cue
    last_ms = max((t * 1000 for hits in onsets.values() for t, _ in hits), default=0.0)
    n_steps = int(np.ceil((last_ms + offset_ms + tail_ms) / dt_ms))
    rates = np.zeros((n_steps, len(VOICE_NAMES)), dtype=np.float32)
    burst_steps = max(1, int(round(burst_ms / dt_ms)))
    for col, name in enumerate(VOICE_NAMES):
        for t_s, velocity in onsets.get(name, ()):
            start = int(round((t_s * 1000 + offset_ms - lookahead_ms) / dt_ms))
            rate = rate_min_hz + (rate_max_hz - rate_min_hz) * velocity / 127
            seg = rates[start : start + burst_steps, col]
            np.maximum(seg, rate, out=seg)  # overlapping bursts keep the louder one
    return Encoded(rates, VOICE_NAMES, dt_ms, offset_ms)


MAX_RATE_HZ = 1000.0  # one spike per ms: the Poisson generator's ceiling at dt = 1 ms


def scale_cues(rates: np.ndarray, gain: float) -> np.ndarray:
    """Every cue rate times `gain` (capped at MAX_RATE_HZ): one fixed rule for every drum, stronger drive into the brain.
    A fly trained with a gain records it (checkpoint "cue_gain") and fly.loop plays it with the same one."""
    return rates if gain == 1 else np.minimum(rates * gain, MAX_RATE_HZ).astype(rates.dtype)


def encode(midi_path: Path, lookahead_ms: float = 150.0, dt_ms: float = 1.0, **kwargs) -> Encoded:
    """Cue rate envelopes for a whole MIDI file."""
    return encode_onsets(read_onsets(midi_path), lookahead_ms=lookahead_ms, dt_ms=dt_ms, **kwargs)

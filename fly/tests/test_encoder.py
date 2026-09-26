import mido
import numpy as np

from fly.drums import DRUM_CHANNEL
from fly.encoder import VOICE_NAMES, encode


def _write(path, hits, bpm=120):
    """hits: [(beat, note, velocity)] on the drum channel."""
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))
    now = 0
    for beat, note, vel in sorted(hits):
        tick = int(beat * 480)
        track.append(mido.Message("note_on", channel=DRUM_CHANNEL, note=note, velocity=vel, time=tick - now))
        track.append(mido.Message("note_off", channel=DRUM_CHANNEL, note=note, velocity=0, time=0))
        now = tick
    mid.save(path)


def test_burst_starts_lookahead_before_note(tmp_path):
    path = tmp_path / "g.mid"
    _write(path, [(0, 36, 127), (2, 38, 64)])  # kick at 0 s, snare at 1 s (120 bpm)
    enc = encode(path, lookahead_ms=150, burst_ms=30)
    kick, snare = VOICE_NAMES.index("kick"), VOICE_NAMES.index("snare")

    # The note at score time t is due at sim step t*1000 + offset; its cue starts lookahead earlier.
    snare_due = int(1000 + enc.offset_ms)
    active = np.flatnonzero(enc.rates[:, snare])
    assert active[0] == snare_due - 150 and len(active) == 30
    assert np.flatnonzero(enc.rates[:, kick])[0] == 0


def test_velocity_scales_rate_and_unmapped_notes_are_dropped(tmp_path):
    path = tmp_path / "g.mid"
    _write(path, [(0, 36, 127), (1, 36, 1), (1.5, 60, 127)])  # 60 isn't a TD-07 drum
    enc = encode(path, rate_min_hz=50, rate_max_hz=200)
    kick = enc.rates[:, VOICE_NAMES.index("kick")]
    assert kick.max() == 200
    assert 50 < kick[500:].max() < 52
    assert enc.rates.sum() == kick.sum()


def test_edges_fold_onto_their_drum(tmp_path):
    path = tmp_path / "g.mid"
    _write(path, [(0, 22, 100), (1, 26, 100), (2, 50, 100)])  # closed-hat edge, open-hat edge, tom 1 rim
    enc = encode(path)
    for name in ("hat_closed", "hat_open", "tom1"):
        assert enc.rates[:, VOICE_NAMES.index(name)].any()
    assert len(VOICE_NAMES) == 12

import csv
import json
from dataclasses import dataclass

import mido
import pytest

torch = pytest.importorskip("torch")

from fly.drums import DRUM_CHANNEL  # noqa: E402
from fly.encoder import encode_onsets  # noqa: E402
from fly.loop import score_time, simulate, write_run  # noqa: E402


@dataclass
class _Hit:  # same fields as fly.body.Hit
    t_ms: float
    pad: str
    voice: str
    note: int
    velocity: int
    contact_speed: float
    limb: str


class _Brain:
    def init_state(self):
        return (None, None, torch.zeros(1, 4), None, None)

    def step(self, state, rates, current=None, generator=None):
        return (None, None, (rates[:, :4] > 0).float(), None, None)


class _Decoder:
    def init_state(self):
        return torch.zeros(1, 4)

    def targets(self, r):
        return torch.zeros(1, 32)

    def __call__(self, r, spikes):
        return r, torch.zeros(1, 32)


class _Body:
    """Sounds a snare at the given sim times (ms)."""

    def __init__(self, at_ms):
        self.t, self.at = 0, list(at_ms)

    def step(self, targets):
        self.t += 1
        return [_Hit(self.t - 0.4, "snare", "snare", 38, 90, 2.2, "front_left")] if self.t in self.at else []


def test_hits_are_written_in_score_time(tmp_path):
    enc = encode_onsets({"snare": [(0.5, 100)]}, lookahead_ms=40)
    off = enc.offset_ms
    raw = simulate(enc.rates, _Brain(), _Decoder(), _Body([int(off) - 10, int(off) + 500, int(off) + 520]))
    assert len(raw) == 3
    hits = score_time(raw, off)
    assert [round(h["t_s"] * 1000, 1) for h in hits] == [499.6, 519.6]  # the one before score time 0 is dropped
    write_run(tmp_path, hits, {"offset_ms": off})

    rows = list(csv.DictReader(open(tmp_path / "hits.csv")))
    assert [(float(r["t_ms"]), int(r["note"]), int(r["velocity"])) for r in rows] == [(499.6, 38, 90), (519.6, 38, 90)]
    js = json.loads((tmp_path / "hits.json").read_text())
    assert set(js[0]) >= {"t_s", "note", "voice", "velocity", "contact_speed"}

    t, events = 0.0, []
    for msg in mido.MidiFile(tmp_path / "hits.mid"):
        t += msg.time
        if msg.type in ("note_on", "note_off"):
            assert msg.channel == DRUM_CHANNEL
            events.append((round(t * 1000, 1), msg.type, msg.note))
    # the first note is cut short by the repeat 20 ms later instead of overlapping it
    assert events == [(499.5, "note_on", 38), (519.5, "note_off", 38), (519.5, "note_on", 38), (569.5, "note_off", 38)]

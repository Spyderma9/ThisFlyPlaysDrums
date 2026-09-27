import mido
import pytest

import midi_capture
import record_song
from prep_takes import LEAD_MS
from sheet_to_midi import read_midi


class Clock:
    """Fake perf_counter: time only moves when the code under test sleeps."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, s):
        self.now += s


class Kit:
    """Fake TD-07 input: each (time_s, msg) comes out of iter_pending() once the clock passes it."""

    def __init__(self, clock, timed, interrupt_at=None):
        self.clock, self.queue, self.interrupt_at = clock, sorted(timed, key=lambda e: e[0]), interrupt_at

    def iter_pending(self):
        if self.interrupt_at is not None and self.clock() >= self.interrupt_at:
            raise KeyboardInterrupt
        while self.queue and self.queue[0][0] <= self.clock():
            yield self.queue.pop(0)[1]


class Out:
    def __init__(self, clock):
        self.clock, self.sent = clock, []

    def send(self, msg):
        self.sent.append((self.clock(), msg))


def hit(t, note, vel=90):
    return t, mido.Message("note_on", channel=9, note=note, velocity=vel)


def rec(timed, **kw):
    clock = Clock()
    kit = Kit(clock, timed, kw.pop("interrupt_at", None))
    events = record_song.record(kit, clock=clock, sleep=clock.sleep, echo=lambda *_: None, **kw)
    return events, clock.now


def test_stops_after_silence():
    events, stopped = rec([hit(0.5, 36), hit(1.0, 38), hit(1.5, 42)], silence_s=2.5)
    assert [m.note for _, m, _ in events] == [36, 38, 42]
    assert stopped == pytest.approx(4.0, abs=0.01)
    assert events[0][0] == pytest.approx(500, abs=1)  # ms from when recording began


def test_stops_at_max_seconds():
    events, stopped = rec([hit(1 + i * 0.25, 42) for i in range(200)], max_s=5)
    assert stopped == pytest.approx(6.0, abs=0.01)  # 5 s after the first hit
    assert len(events) == 21


def test_fake_kicks_are_marked_and_dont_keep_it_going():
    events, stopped = rec([hit(0.5, 36, 100), hit(0.55, 36, 40), hit(2.0, 36, 5)], silence_s=1.0)
    assert [r for _, _, r in events] == [None, "bounce"]  # stopped at 1.5 s, before the 2.0 s stray touch
    assert stopped == pytest.approx(1.5, abs=0.01)


def test_gives_up_when_nothing_is_played():
    events, stopped = rec([], wait_s=3)
    assert events == [] and stopped == pytest.approx(3, abs=0.01)


def test_ctrl_c_keeps_what_was_played():
    events, _ = rec([hit(0.5, 38), hit(0.9, 38)], interrupt_at=1.0, silence_s=10)
    assert len(events) == 2


def test_clock_messages_are_skipped_and_pedal_kept():
    pedal = mido.Message("control_change", channel=9, control=4, value=90)
    events, _ = rec([(0.2, mido.Message("clock")), (0.3, pedal), hit(0.5, 44)], silence_s=1)
    assert [m.type for _, m, _ in events] == ["control_change", "note_on"]


def test_count_in_clicks_then_starts_on_the_downbeat():
    clock = Clock()
    out = Out(clock)
    start = record_song.count_in(out, 4, 120, clock=clock, sleep=clock.sleep)
    ons = [(t, m.note) for t, m in out.sent if m.type == "note_on"]
    assert [n for _, n in ons] == [record_song.CLICK_NOTE] * 4
    assert [t for t, _ in ons] == pytest.approx([0, 0.5, 1.0, 1.5], abs=0.01)
    assert start == pytest.approx(2.0)
    # an echoed click during the count-in is dropped; the first beat lands at t = 0
    kit = Kit(clock, [hit(1.6, record_song.CLICK_NOTE), hit(2.0, 36), hit(2.5, 38)])
    events = record_song.record(kit, silence_s=1, start=start, clock=clock, sleep=clock.sleep, echo=lambda *_: None)
    assert [(round(t), m.note) for t, m, _ in events] == [(0, 36), (500, 38)]


def test_a_recording_becomes_a_song(tmp_path, monkeypatch):
    monkeypatch.setattr(midi_capture, "TAKES_DIR", tmp_path / "takes")
    clock = Clock()
    kit = Kit(clock, [hit(0.5, 36, 100), hit(0.55, 36, 40), hit(1.0, 38), hit(1.25, 42), hit(1.5, 36)])
    monkeypatch.setattr(record_song.mido, "get_input_names", lambda: ["TD-07 0"])
    monkeypatch.setattr(record_song.mido, "open_input", lambda name: _Ctx(kit))
    monkeypatch.setattr(record_song, "record", _with_clock(record_song.record, clock))
    report = record_song.record_song(name="groove", echo=lambda *_: None, out_dir=tmp_path / "songs")
    assert report["take"].endswith(".csv") and len(list((tmp_path / "takes").glob("take_*.csv"))) == 1
    assert [(round(t), n) for t, n, _ in read_midi(report["mid"])[0]] == \
        [(LEAD_MS, 36), (LEAD_MS + 500, 38), (LEAD_MS + 750, 42), (LEAD_MS + 1000, 36)]
    assert report["cleaned"] == {"bounce": 1} and report["name"] == "groove"


def test_nothing_played_is_an_error(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(record_song.mido, "get_input_names", lambda: ["TD-07 0"])
    monkeypatch.setattr(record_song.mido, "open_input", lambda name: _Ctx(Kit(clock, [])))
    monkeypatch.setattr(record_song, "record", _with_clock(record_song.record, clock))
    with pytest.raises(record_song.to_fly.SongError, match="nothing was played"):
        record_song.record_song(wait_s=1, echo=lambda *_: None)


def test_missing_kit_is_an_error(monkeypatch):
    monkeypatch.setattr(record_song.mido, "get_input_names", lambda: ["Midi Through"])
    with pytest.raises(record_song.to_fly.SongError, match="No TD-07 input"):
        record_song.record_song(echo=lambda *_: None)


class _Ctx:
    def __init__(self, port):
        self.port = port

    def __enter__(self):
        return self.port

    def __exit__(self, *exc):
        return False


def _with_clock(fn, clock):
    return lambda *a, **kw: fn(*a, **{**kw, "clock": clock, "sleep": clock.sleep})

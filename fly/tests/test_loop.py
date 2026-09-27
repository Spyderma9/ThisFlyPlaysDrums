import csv
import json
from dataclasses import dataclass

import mido
import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402

from fly.drums import DRUM_CHANNEL  # noqa: E402
from fly.encoder import encode_onsets  # noqa: E402
from fly.loop import (Recorder, SpikeRecorder, guided_current, listen, load_poses, load_spikes, play,  # noqa: E402
                      score_time, simulate, write_run)


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


class _PoseBody(_Body):
    """A _Body with a pose (qpos) that counts steps, and a snare that is touched on even steps."""

    pads = ["snare", "hat_pedal"]

    class _D:
        def __init__(self):
            self.qpos = np.zeros(3)

    class _M:
        nq = 3

    def __init__(self, at_ms):
        super().__init__(at_ms)
        self.m, self.d = self._M(), self._D()
        self.touching = np.zeros(2, dtype=bool)
        self.targets = []

    def step(self, targets):
        hits = super().step(targets)
        self.targets.append(np.asarray(targets))
        self.d.qpos[:] = self.t
        self.touching[:] = [self.t % 2 == 0, True]
        return hits


def test_recorder_keeps_every_step_pose_and_contacts(tmp_path):
    body = _PoseBody([5])
    rec = Recorder(body, 8)
    simulate(np.zeros((8, 12), np.float32), _Brain(), _Decoder(), body, on_step=rec)
    rec.save(tmp_path / "poses.npz", offset_ms=3.0)

    poses = load_poses(tmp_path)
    assert poses["qpos"].shape == (8, 3) and poses["qpos"].dtype == np.float32
    assert poses["qpos"][:, 0].tolist() == [1, 2, 3, 4, 5, 6, 7, 8]  # the pose after each 1 ms step
    assert poses["touching"][:, 0].tolist() == [False, True] * 4
    assert poses["pads"] == ["snare", "hat_pedal"]
    assert poses["offset_ms"] == 3.0


def test_play_drives_the_body_with_the_given_targets():
    body = _PoseBody([2, 4])
    q = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    rec = Recorder(body, 4)
    hits = play(q, body, on_step=rec)
    assert [h.t_ms for h in hits] == [1.6, 3.6]
    assert np.array_equal(np.stack(body.targets), q)
    assert rec.qpos[:, 0].tolist() == [1, 2, 3, 4]


def test_teacher_run_matches_the_ceiling_check(tmp_path):
    """--teacher drives the real body with q*, exactly as fly.strokes' ceiling check does."""
    pytest.importorskip("mujoco")
    from fly.connectome import REPO
    from fly.loop import teacher_run
    from fly.strokes import ceiling_run

    take = sorted((REPO / "grooves" / "train").glob("SD_90bpm_*.mid"))[0]
    ceiling = ceiling_run(str(take), str(tmp_path / "ceiling"), seconds=1.5)
    hits, meta = teacher_run(take, tmp_path / "teacher", seconds=1.5, poses=True)
    ref = json.loads((tmp_path / "ceiling" / take.stem / "hits.json").read_text())
    assert ceiling["hits"] > 0
    assert [(h["t_s"], h["voice"], h["velocity"]) for h in hits] == [(h["t_s"], h["voice"], h["velocity"]) for h in ref]
    assert meta["driver"] == "teacher"
    poses = load_poses(tmp_path / "teacher")
    assert poses["qpos"].shape == (meta["steps"], 102)


def test_spike_recorder_keeps_which_neurons_fired_each_step(tmp_path):
    rates = np.zeros((5, 12), np.float32)
    rates[1, 0] = rates[1, 2] = rates[3, 3] = 100.0  # _Brain fires neuron k when voice column k (< 4) is driven
    spk = SpikeRecorder(5)
    simulate(rates, _Brain(), _Decoder(), _Body([]), on_step=spk)
    spk.save(tmp_path / "spikes.npz")
    s = load_spikes(tmp_path)
    assert s["offsets"].tolist() == [0, 0, 2, 2, 3, 3]
    assert s["ids"].tolist() == [0, 2, 3]
    assert s["n_neurons"] == 4


class _CurrentDecoder:
    def current_for(self, q):
        return q.sum(dim=1, keepdim=True) * torch.ones(1, 3)


def test_guided_current_is_alpha_times_the_teachers_motor_current():
    qstar = np.arange(4 * 2, dtype=np.float32).reshape(4, 2)
    assert guided_current(qstar, _CurrentDecoder(), 0.0) is None  # the fly on its own
    cur = guided_current(qstar, _CurrentDecoder(), 0.5)
    assert torch.allclose(cur(2, None), torch.full((1, 3), 0.5 * (4 + 5)))


def test_listen_moves_the_body_with_qstar_while_the_brain_runs_alongside(tmp_path):
    """The visual branch's 'teacher plays, brain listens': q* drives the legs, the brain's firing is only recorded."""
    rates = np.zeros((4, 12), np.float32)
    rates[2, 1] = 100.0  # the cue for neuron 1 at step 2
    q = np.arange(4 * 32, dtype=np.float32).reshape(4, 32)
    body, spk, seen = _PoseBody([3]), SpikeRecorder(4), []
    current = lambda t, _q: seen.append(t) or torch.zeros(1, 4)  # noqa: E731
    hits = listen(rates, _Brain(), body, q, current=current, on_step=spk)
    assert np.array_equal(np.stack(body.targets), q)  # the legs follow q*, not the brain
    assert [h.t_ms for h in hits] == [2.6]
    assert seen == [0, 1, 2, 3]  # the teacher's current reaches the brain every step
    spk.save(tmp_path / "spikes.npz")
    assert load_spikes(tmp_path)["ids"].tolist() == [1]

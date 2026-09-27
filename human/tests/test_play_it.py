"""play_it + fly_remote against a fake server: the real shell commands run in bash on a temp folder, with stand-ins
for tmux, nvidia-smi and fx (which pretends to be the fly: it plays the song back perfectly)."""
import io
import json
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import fly_remote
import play_it
import to_fly
from fly_remote import Remote, RemoteError, job_command, parse_log, safe_extract

GROOVES = Path(__file__).resolve().parents[2] / "grooves"

FAKE_FLY = r'''
import json, shlex, sys
from pathlib import Path
sys.path.insert(0, {human!r})
from sheet_to_midi import read_midi
inner = sys.argv[1]
Path("fx_calls.txt").open("a").write(inner + "\n")
a = shlex.split(inner.split(" && ")[0])
groove, out = Path(a[a.index("--groove") + 1]), Path(a[a.index("--out") + 1])
if Path("fail_next").exists():
    print("Traceback: CUDA out of memory"); sys.exit(1)
hits, _ = read_midi(groove)
steps = int(hits[-1][0]) + 700
print(f"malecns-v1.0: {{steps}} steps on cuda, body dt 0.0002 s")
for t in range(1000, steps, 1000):
    print(f"  {{t:6d}} ms  {{t * 15 // 1000:6d}} s  {{sum(h[0] < t for h in hits)}} hits")
out.mkdir(parents=True, exist_ok=True)
(out / "hits.csv").write_text("t_ms,note,velocity\n" + "".join(f"{{t:.1f}},{{n}},{{v}}\n" for t, n, v in hits))
(out / "meta.json").write_text(json.dumps({{"groove": groove.name, "hits": len(hits)}}))
print(f"wrote {{out}}")
for name in ("scene.json", "hits.json", "hits.mid", "poses.bin", "spikes.bin", "brain.json", "poses.npz"):
    (out / name).write_text("x")
(out / "viewer.json").write_text(json.dumps({{"groove": str(groove), "driver": "fly", "alpha": 0,
                                              "hits": len(hits), "spikes": True}}))
Path("runs/model").mkdir(parents=True, exist_ok=True)
for name in ("fly.json", "fly.bin", "neurons.json", "neurons.bin"):
    Path("runs/model", name).write_text("model " + name)
'''


class FakeServer:
    """A temp folder standing in for the T5600's repo; `run` replaces subprocess.run for ssh."""

    def __init__(self, tmp):
        self.repo = tmp / "server" / "ThisFlyPlaysDrums"
        self.repo.mkdir(parents=True)
        self.bin = tmp / "server" / "bin"
        self.bin.mkdir()
        self.gpu = tmp / "server" / "gpu.txt"
        human = str(Path(play_it.__file__).parent)
        (tmp / "server" / "fake_fly.py").write_text(FAKE_FLY.format(human=human))
        self.fx = tmp / "server" / "fx"
        self._script(self.fx, f'cd {shlex.quote(str(self.repo))} && exec {shlex.quote(sys.executable)} '
                              f'{shlex.quote(str(tmp / "server" / "fake_fly.py"))} "$@"')
        self._script(self.bin / "tmux", 'if [ "$1" = has-session ]; then exit 1; fi\n'
                                        'if [ "$1" = new ]; then shift 4; bash -c "$1"; fi')  # runs the job at once
        self._script(self.bin / "nvidia-smi", f'cat {shlex.quote(str(self.gpu))} 2>/dev/null; true')
        self.calls = []

    def _script(self, path, body):
        path.write_text("#!/bin/bash\n" + body + "\n")
        path.chmod(0o755)

    def run(self, args, input=None, capture_output=True):
        assert args[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
        self.calls.append(args[-1])
        env = {"PATH": f"{self.bin}:/usr/bin:/bin"}
        return subprocess.run(["bash", "-c", args[-1]], input=input, capture_output=True, env=env)

    def remote(self):
        return Remote("root@fake", str(self.repo), str(self.fx), run=self.run)

    def fx_calls(self):
        return (self.repo / "fx_calls.txt").read_text().splitlines()


@pytest.fixture
def server(tmp_path):
    s = FakeServer(tmp_path)
    (s.repo / "runs" / "train" / "f3").mkdir(parents=True)
    (s.repo / "runs" / "train" / "f3" / "best.pt").write_text("weights")
    return s


@pytest.fixture
def song(tmp_path):
    return to_fly.make_song(GROOVES / "rock_beat.txt", out_dir=tmp_path / "songs")


def run_play(server, song, tmp_path, **kw):
    msgs = []
    kw = {"confirm": lambda jobs: False, "progress": msgs.append, "sleep": lambda s: None,
          "runs": tmp_path / "runs", "songs": tmp_path / "songs", **kw}
    result = play_it.play(song, server.remote(), "runs/train/f3/best.pt", **kw)
    return result, msgs


def test_a_song_goes_to_the_fly_and_comes_back(server, song, tmp_path):
    result, msgs = run_play(server, song, tmp_path)
    # the server got the song and ran the fly on it, on its own
    assert (server.repo / "songs" / "rock_beat.mid").read_bytes() == Path(song["mid"]).read_bytes()
    call = server.fx_calls()[0]
    assert "--alpha 0" in call and "--teacher" not in call and "--weights runs/train/f3/best.pt" in call
    assert "--poses --spikes" in call and call.endswith("python -m fly.viewer_export runs/songs/rock_beat")
    # the run came back without poses.npz, with the model, the song's report and an index entry
    run = tmp_path / "runs" / "songs" / "rock_beat"
    assert sorted(p.name for p in run.iterdir()) == sorted(fly_remote.RUN_FILES + ("song.json",))
    assert (tmp_path / "runs" / "model" / "fly.bin").read_text() == "model fly.bin"
    index = json.loads((tmp_path / "runs" / "viewer_index.json").read_text())
    assert index[0]["id"] == "songs/rock_beat" and index[0]["driver"] == "fly" and index[0]["brain"]
    # the fake fly played every note, so it scores perfectly
    assert result["score"]["f1"] == 1.0 and result["id"] == "songs/rock_beat"
    assert result["url"] == "http://localhost:8000/viewer/?run=songs/rock_beat"
    assert msgs[0].startswith("rock_beat: the fly is playing it") and msgs[-1] == "done"


def test_busy_gpu_asks_first(server, song, tmp_path):
    server.gpu.write_text("2224656, python, 820 MiB\n")
    asked = []
    with pytest.raises(RemoteError, match="GPU is busy"):
        run_play(server, song, tmp_path, confirm=lambda jobs: asked.append(jobs) or False)
    assert asked == [["2224656, python, 820 MiB"]]
    assert not (server.repo / "songs").exists()  # nothing was sent
    result, _ = run_play(server, song, tmp_path, confirm=lambda jobs: True)
    assert result["score"]["f1"] == 1.0


def test_missing_weights_stop_before_anything_runs(server, song, tmp_path):
    with pytest.raises(RemoteError, match="no weights"):
        play_it.play(song, server.remote(), "runs/train/nope/best.pt", confirm=lambda j: True,
                     progress=lambda m: None, sleep=lambda s: None, runs=tmp_path / "runs")
    assert not (server.repo / "fx_calls.txt").exists()


def test_a_failed_run_reports_the_log(server, song, tmp_path):
    (server.repo / "fail_next").write_text("")
    with pytest.raises(RemoteError, match="CUDA out of memory"):
        run_play(server, song, tmp_path)


def test_model_only_copied_when_it_changed(server, song, tmp_path):
    run_play(server, song, tmp_path)
    remote = server.remote()
    assert remote.sync_model(tmp_path / "runs" / "model") == []
    (server.repo / "runs" / "model" / "fly.bin").write_text("new mesh")
    assert remote.sync_model(tmp_path / "runs" / "model") == ["fly.bin"]


def test_fetch_a_finished_run_later(server, song, tmp_path):
    run_play(server, song, tmp_path)
    (tmp_path / "runs" / "songs" / "rock_beat" / "hits.csv").unlink()
    result = play_it.finish(server.remote(), "rock_beat", tmp_path / "runs", tmp_path / "songs")
    assert result["score"]["f1"] == 1.0


def test_unreachable_server_says_how_to_fix_it(tmp_path):
    def down(args, input=None, capture_output=True):
        return subprocess.CompletedProcess(args, 255, b"", b"ssh: connect to host: Connection timed out")
    with pytest.raises(RemoteError, match="Tailscale"):
        Remote("root@fake", "/repo", "/fx", run=down).check()


def test_a_run_that_never_started_is_reported(server):
    remote = server.remote()
    assert remote.status("ghost")["state"] == "missing"
    with pytest.raises(RemoteError, match="never appeared"):
        play_it.wait(remote, "ghost", progress=lambda m: None, sleep=lambda s: None)


def test_parse_log_stages():
    assert parse_log("")["state"] == "starting"
    log = "malecns-v1.0: 5700 steps on cuda, body dt 0.0002 s\n    2000 ms      31 s  7 hits\n"
    assert parse_log(log) | {} == {"state": "simulating", "steps": 5700, "t_ms": 2000, "wall_s": 31, "hits": 7}
    assert parse_log(log + "wrote runs/songs/x\n")["state"] == "exporting"
    assert parse_log(log + "wrote runs/songs/x\nDONE\n")["state"] == "done"
    assert parse_log(log + "Traceback\nFAIL\n")["state"] == "failed"
    assert "3 of 6 s" in play_it.progress_line({"state": "simulating", "steps": 5700, "t_ms": 3000, "wall_s": 45,
                                                "hits": 9})


def test_job_command_survives_odd_paths():
    cmd = job_command("/mnt/user/dev/Fly Drums", "/mnt/user/dev/fx", "x", "runs/train/my run/best.pt")
    fx, inner = shlex.split(cmd)[:2]
    assert fx == "/mnt/user/dev/fx"
    assert shlex.split(inner.split(" && ")[0])[-1] == "runs/train/my run/best.pt"


def test_bad_song_ids_are_refused(server):
    with pytest.raises(RemoteError, match="bad song id"):
        server.remote().start("x; rm -rf /", None)


def test_safe_extract_only_writes_expected_files(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name in ("hits.csv", "../evil.txt", "poses.npz"):
            info = tarfile.TarInfo(name)
            info.size = 2
            tar.addfile(info, io.BytesIO(b"ok"))
    got = safe_extract(buf.getvalue(), tmp_path / "run", ("hits.csv", "evil.txt"))
    assert got == ["hits.csv", "evil.txt"]  # ../evil.txt lands inside the run folder, poses.npz is skipped
    assert not (tmp_path / "evil.txt").exists() and not (tmp_path / "run" / "poses.npz").exists()

"""song_server over real HTTP, with a stand-in for play_it.play (no ssh, no fly)."""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import song_server
from fly_remote import RemoteError

GROOVES = Path(__file__).resolve().parents[2] / "grooves"


class FakeFly:
    """play_it.play's stand-in. `busy` makes it ask about the GPU first; `fail` makes the run fail."""

    def __init__(self, busy=False, fail=None):
        self.busy, self.fail, self.calls, self.release = busy, fail, [], threading.Event()
        self.release.set()

    def __call__(self, report, remote, weights, confirm, progress, poll_s, runs, songs, on_status):
        self.calls.append((report["name"], weights))
        if self.busy and not confirm(["2224656, python, 820 MiB"]):
            raise RemoteError("not started: the GPU is busy")
        progress("fly playing: 1 of 6 s")
        on_status({"state": "simulating", "steps": 6000, "t_ms": 1000, "wall_s": 15, "hits": 2})
        self.release.wait(5)
        if self.fail:
            raise RemoteError(self.fail)
        return {"id": f"songs/{report['name']}", "url": "u", "score": {"f1": 0.5}, "run": "r"}


@pytest.fixture
def serve(tmp_path):
    servers = []

    def start(fly, weights="runs/train/f3/best.pt"):
        songs = song_server.Songs(songs_dir=tmp_path / "songs", runs_dir=tmp_path / "runs", play=fly,
                                  remote=object(), weights=weights, poll_s=0)
        srv = song_server.make_server(0, songs)
        srv.RequestHandlerClass.takes_dir = tmp_path / "takes"
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return f"http://127.0.0.1:{srv.server_address[1]}"

    yield start
    for srv in servers:
        srv.shutdown()


def call(url, data=None, method=None):
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"))
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read()), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()), dict(e.headers)


def until(base, jid, stage, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        _, job, _ = call(f"{base}/api/jobs/{jid}")
        if job["stage"] == stage:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {jid} stuck at {job['stage']}: {job}")


def test_a_dropped_file_becomes_a_fly_run(serve):
    fly = FakeFly()
    base = serve(fly)
    assert call(f"{base}/api/health")[1] == {"ok": True, "weights": "runs/train/f3/best.pt"}
    code, job, _ = call(f"{base}/api/song?name=rock_beat.txt", (GROOVES / "rock_beat.txt").read_bytes())
    assert code == 200 and job["song"]["hits"] == 26 and len(job["notes"]) == 26 and job["notes"][0] == 1000
    done = until(base, job["id"], "done")
    assert done["result"] == {"id": "songs/rock_beat", "url": "u", "score": {"f1": 0.5}}
    assert done["status"]["t_ms"] == 1000 and fly.calls == [("rock_beat", "runs/train/f3/best.pt")]


def test_seconds_crop_and_mid_upload(serve):
    base = serve(FakeFly())
    heldout = (Path(GROOVES) / "heldout" / "heldout_1.mid").read_bytes()
    code, job, _ = call(f"{base}/api/song?name=my%20song.mid&seconds=10", heldout)
    assert code == 200 and job["song"]["name"] == "my_song" and job["song"]["seconds"] <= 10
    assert job["song"]["known_groove"] is None  # cropped, so no longer the whole held-out groove


def test_a_take_from_the_browser(serve, tmp_path):
    base = serve(FakeFly())
    events = [[100, [0x99, 36, 100]], [150, [0x99, 36, 40]],  # kick and its bounce
              [400, [0x99, 38, 90]], [450, [0x89, 38, 0]], [500, [0xF8]], [800, [0xB9, 4, 90]], [900, [0x99, 42, 70]]]
    code, job, _ = call(f"{base}/api/take", json.dumps({"events": events}).encode())
    assert code == 200, job
    assert job["song"]["hits"] == 3 and job["song"]["name"].startswith("take_")
    take = next((tmp_path / "takes").glob("take_*.csv")).read_text()
    assert "bounce" in take and "control_change" in take and "clock" not in take


def test_busy_gpu_waits_for_an_answer(serve):
    fly = FakeFly(busy=True)
    base = serve(fly)
    _, job, _ = call(f"{base}/api/song?name=rock_beat.txt", (GROOVES / "rock_beat.txt").read_bytes())
    asked = until(base, job["id"], "confirm")
    assert asked["gpu_jobs"] == ["2224656, python, 820 MiB"]
    call(f"{base}/api/jobs/{job['id']}/answer", json.dumps({"go": True}).encode())
    until(base, job["id"], "done")
    _, job2, _ = call(f"{base}/api/song?name=rock_beat.txt", (GROOVES / "rock_beat.txt").read_bytes())
    until(base, job2["id"], "confirm")
    call(f"{base}/api/jobs/{job2['id']}/answer", json.dumps({"go": False}).encode())
    assert until(base, job2["id"], "failed")["error"] == "not started: the GPU is busy"


def test_songs_wait_their_turn(serve):
    fly = FakeFly()
    fly.release.clear()
    base = serve(fly)
    body = (GROOVES / "rock_beat.txt").read_bytes()
    _, first, _ = call(f"{base}/api/song?name=a.txt", body)
    _, second, _ = call(f"{base}/api/song?name=b.txt", body)
    until(base, first["id"], "running")
    assert call(f"{base}/api/jobs/{second['id']}")[1]["stage"] == "queued"
    fly.release.set()
    until(base, second["id"], "done")
    assert [c[0] for c in fly.calls] == ["a", "b"]


def test_failures_are_reported(serve):
    base = serve(FakeFly(fail="the fly run failed on the server:\nCUDA out of memory"))
    _, job, _ = call(f"{base}/api/song?name=rock_beat.txt", (GROOVES / "rock_beat.txt").read_bytes())
    assert "CUDA out of memory" in until(base, job["id"], "failed")["error"]
    code, err, _ = call(f"{base}/api/song?name=notes.pdf", b"%PDF")
    assert code == 400 and "isn't sheet music" in err["error"]
    code, err, _ = call(f"{base}/api/take", json.dumps({"events": [[5, [0xF8]]]}).encode())
    assert code == 400 and err["error"] == "nothing was played"
    assert call(f"{base}/api/jobs/999")[0] == 404


def test_no_weights_no_runs(serve):
    base = serve(FakeFly(), weights="")  # None would mean "use fly_pipeline.json's"
    code, err, _ = call(f"{base}/api/song?name=rock_beat.txt", (GROOVES / "rock_beat.txt").read_bytes())
    assert code == 400 and "weights" in err["error"]


def test_serves_the_viewer_and_fresh_runs(serve):
    base = serve(FakeFly())
    with urllib.request.urlopen(f"{base}/") as r:  # redirected to the viewer
        assert r.url.endswith("/viewer/") and b"Fly Drums" in r.read()
    with urllib.request.urlopen(f"{base}/grooves/rock_beat.txt") as r:
        assert r.headers.get("Cache-Control") is None
    req = urllib.request.Request(f"{base}/runs/nothing.json")
    try:
        urllib.request.urlopen(req)
    except urllib.error.HTTPError as e:
        assert e.code == 404 and e.headers["Cache-Control"] == "no-cache"

"""Serve the viewer, and let it hand new songs to the fly.

Use it instead of `python -m http.server 8000`: it serves the repo the same way, and adds a small API that the
viewer's "New song" panel uses. A dropped file or a take recorded in the browser becomes a song (to_fly.py), the
fly plays it on the server (play_it.play, one song at a time), and the viewer loads the run when it's back.
Only this computer can reach it (127.0.0.1).

    python human/song_server.py          # then Chrome at http://localhost:8000/viewer/

API:
    GET  /api/health                   {"weights": ...}: the panel only shows when this answers
    POST /api/song?name=F&seconds=N    body: the file                                      -> a job
    POST /api/take?seconds=N           body: {"events": [[t_ms, [midi bytes]], ...]} (Web MIDI) -> a job
    GET  /api/jobs/<id>                stage (queued, running, confirm, done, failed), message, song, status, result
    POST /api/jobs/<id>/answer         {"go": true|false} when the server's GPU is busy
"""
import argparse
import itertools
import json
import queue
import threading
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import mido

import midi_capture
import play_it
import to_fly
from drum_map import HitFilter
from fly_remote import REPO, Remote, RemoteError, load_config
from sheet_to_midi import read_midi

MAX_UPLOAD = 20 * 2 ** 20
CONFIRM_TIMEOUT_S = 15 * 60  # a busy-GPU question nobody answers counts as "no"
SONG_EXTS = (".musicxml", ".xml", ".mxl", ".mid", ".midi", ".txt", ".mscz")


class Job:
    def __init__(self, jid, report):
        self.id, self.report = jid, report
        self.stage, self.message = "queued", "Waiting for the song before it"
        self.status = self.result = self.error = self.gpu_jobs = None
        self.notes = [round(t) for t, _, _ in read_midi(report["mid"])[0]]  # the strip the viewer draws
        self._answered, self._go = threading.Event(), False

    def say(self, message):
        prefix = f"{self.report['name']}: "  # the panel already shows the song's name
        self.message = message[len(prefix):] if message.startswith(prefix) else message

    def see(self, status):
        self.status = {k: status.get(k) for k in ("state", "steps", "t_ms", "wall_s", "hits")}

    def confirm(self, gpu_jobs):
        """play_it asks this when the server's GPU is busy; the viewer answers through /answer."""
        self.stage, self.gpu_jobs = "confirm", gpu_jobs
        self.message = "The server's GPU is busy"
        answered = self._answered.wait(CONFIRM_TIMEOUT_S)
        self.stage = "running"
        return answered and self._go

    def answer(self, go):
        self._go = bool(go)
        self._answered.set()

    def to_json(self):
        song = {k: self.report[k] for k in ("name", "hits", "seconds", "voices", "dropped_notes", "known_groove",
                                            "fly_seconds", "cropped_to")}
        song["skipped_untagged"] = self.report.get("skipped_untagged", 0)
        return {"id": self.id, "stage": self.stage, "message": self.message, "song": song, "notes": self.notes,
                "status": self.status, "gpu_jobs": self.gpu_jobs, "result": self.result, "error": self.error}


class Songs:
    """The jobs, and one worker that sends them to the fly in turn (one heavy job on the GPU at a time)."""

    def __init__(self, songs_dir=to_fly.SONGS, runs_dir=REPO / "runs", play=play_it.play, remote=None, weights=None,
                 poll_s=5):
        cfg = load_config()
        self.songs_dir, self.runs_dir, self._play, self.poll_s = Path(songs_dir), Path(runs_dir), play, poll_s
        self.remote = remote or Remote.from_config(cfg)
        self.weights = weights if weights is not None else cfg.get("weights")
        self.jobs, self._ids, self._queue = {}, itertools.count(1), queue.Queue()
        threading.Thread(target=self._work, daemon=True).start()

    def add(self, report):
        job = Job(str(next(self._ids)), report)
        self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def _work(self):
        while True:
            job = self._queue.get()
            job.stage, job.message = "running", "Sending the song to the server"
            try:
                result = self._play(job.report, self.remote, self.weights, confirm=job.confirm, progress=job.say,
                                    poll_s=self.poll_s, runs=self.runs_dir, songs=self.songs_dir, on_status=job.see)
                job.result = {k: result.get(k) for k in ("id", "url", "score")}
                job.stage, job.message = "done", "Done"
            except (RemoteError, to_fly.SongError) as e:
                job.stage, job.error = "failed", str(e)
            except Exception as e:  # keep serving; show what broke
                job.stage, job.error = "failed", f"{type(e).__name__}: {e}"

    def song_from_file(self, filename, data, seconds=None):
        name = Path(filename or "").name
        ext = Path(name).suffix.lower()
        if ext not in SONG_EXTS:
            raise to_fly.SongError(f"{name or 'that file'} isn't sheet music the fly can read: "
                                   "use MusicXML (.musicxml, .mxl), MuseScore (.mscz), MIDI (.mid) or a .txt grid")
        uploads = self.songs_dir / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        path = uploads / f"{to_fly.slug(Path(name).stem)}{ext}"
        path.write_bytes(data)
        return to_fly.make_song(path, seconds=seconds, out_dir=self.songs_dir)

    def song_from_take(self, raw_events, seconds=None, takes_dir=None):
        """Web MIDI messages from the kit -> a take saved like midi_capture.py's -> a song."""
        hit_filter, events = HitFilter(), []
        for t, data in sorted(raw_events, key=lambda e: e[0]):
            if not data or data[0] >= 0xF0:  # clock, active sensing, sysex
                continue
            try:
                msg = mido.Message.from_bytes(data)
            except ValueError:
                continue
            reason = None
            if msg.type == "note_on" and msg.velocity > 0:
                reason = hit_filter.check(float(t), msg.note, msg.velocity)
            events.append((float(t), msg, reason))
        if not any(m.type == "note_on" and m.velocity > 0 and not r for _, m, r in events):
            raise to_fly.SongError("nothing was played")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if takes_dir:
            midi_capture.TAKES_DIR = Path(takes_dir)
        csv_path, _ = midi_capture.save(events, stamp)
        return to_fly.make_song(csv_path, name=f"take_{stamp}", seconds=seconds, out_dir=self.songs_dir)


class Handler(SimpleHTTPRequestHandler):
    songs = None  # set by make_server
    takes_dir = REPO / "takes"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO), **kwargs)

    def end_headers(self):
        if self.path.startswith("/runs/") or self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")  # the run list changes under the viewer's feet
        super().end_headers()

    def log_message(self, fmt, *args):
        if self.path.startswith("/api/") and not self.path.startswith("/api/jobs/"):
            super().log_message(fmt, *args)

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise to_fly.SongError(f"that file is over {MAX_UPLOAD // 2 ** 20} MB")
        return self.rfile.read(n)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
        elif url.path == "/api/health":
            self._json(200, {"ok": True, "weights": self.songs.weights})
        elif url.path.startswith("/api/jobs/"):
            job = self.songs.jobs.get(url.path.rsplit("/", 1)[-1])
            self._json(200, job.to_json()) if job else self._json(404, {"error": "no such job"})
        else:
            super().do_GET()

    def do_POST(self):
        url = urlparse(self.path)
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path in ("/api/song", "/api/take") and not self.songs.weights:
            return self._json(400, {"error": "No trained weights: set \"weights\" in human/fly_pipeline.json"})
        try:
            seconds = float(query["seconds"]) if query.get("seconds") else None
            if url.path == "/api/song":
                report = self.songs.song_from_file(query.get("name"), self._body(), seconds)
            elif url.path == "/api/take":
                events = json.loads(self._body() or b"{}").get("events") or []
                report = self.songs.song_from_take(events, seconds, self.takes_dir)
            elif url.path.startswith("/api/jobs/") and url.path.endswith("/answer"):
                job = self.songs.jobs.get(url.path.split("/")[3])
                if not job:
                    return self._json(404, {"error": "no such job"})
                job.answer(json.loads(self._body() or b"{}").get("go"))
                return self._json(200, job.to_json())
            else:
                return self._json(404, {"error": "no such endpoint"})
        except (to_fly.SongError, ValueError) as e:
            return self._json(400, {"error": str(e)})
        self._json(200, self.songs.add(report).to_json())


def make_server(port=8000, songs=None, host="127.0.0.1"):
    handler = type("SongHandler", (Handler,), {"songs": songs or Songs()})
    return ThreadingHTTPServer((host, port), handler)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = make_server(args.port)
    print(f"Fly Drums viewer at http://localhost:{args.port}/viewer/  (weights: {server.RequestHandlerClass.songs.weights}"
          f"; Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

"""One command: bring music (a file, or play it on the kit), the fly plays it, the run comes back to the viewer.

    1. to_fly.py (or record_song.py) makes songs/<name>.mid
    2. the song goes to the T5600, where the trained fly plays it on its own (fly_remote.py; ~15x the song's
       length, so keep it short), then the run is exported for the viewer
    3. the run comes back to runs/songs/<name>/, is scored against the song, and opens in the viewer

Usage:
    python human/play_it.py my_score.musicxml
    python human/play_it.py --record --count-in 4 --bpm 90
    python human/play_it.py groove.mid --seconds 15 --weights runs/train/f5/best.pt
    python human/play_it.py --fetch my_score              # a run that finished while the laptop was away

Then view it: `python -m http.server 8000` in the repo root, and Chrome at the printed address.
"""
import argparse
import json
import sys
import time
import webbrowser
from pathlib import Path

import score as scoring
import to_fly
from fly_remote import REPO, Remote, RemoteError, load_config, update_index

VIEWER = "http://localhost:8000/viewer/?run={id}"


def ask(jobs):
    print("The server's GPU is already in use (probably Sam's training):\n  " + "\n  ".join(jobs))
    try:
        return input("Run the fly anyway? Both will slow down. [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def progress_line(st, eta_hint=None):
    if st["state"] == "starting":
        return "starting: wiring the connectome and building the body (~30 s)"
    if st["state"] == "simulating" and st["steps"]:
        t, steps = st["t_ms"], st["steps"]
        eta = st["wall_s"] / t * (steps - t) if t else eta_hint
        left = f", ~{eta / 60:.0f} min left" if eta else ""
        return f"fly playing: {t / 1000:.0f} of {steps / 1000:.0f} s{left} ({st['hits']} hits so far)"
    return {"exporting": "exporting the run for the viewer", "done": "done", "failed": "failed"}.get(st["state"], "")


def wait(remote, song_id, progress=print, poll_s=5, sleep=time.sleep, eta_hint=None, missing_s=60, on_status=None):
    """Poll the server until the run is done. Raises RemoteError (with the log's tail) if it failed.
    on_status(st) gets every poll's parsed log (song_server.py shows it in the viewer)."""
    last, missing = None, 0
    while True:
        st = remote.status(song_id)
        if on_status:
            on_status(st)
        if st["state"] == "missing":  # the job's log appears as soon as tmux starts it
            missing += 1
            if missing * poll_s >= missing_s:
                raise RemoteError(f"no run for {song_id} on the server (runs/songs/{song_id}.log never appeared)")
            sleep(poll_s)
            continue
        line = progress_line(st, eta_hint)
        if line != last:
            progress(line)
            last = line
        if st["state"] == "done":
            return st
        if st["state"] == "failed":
            raise RemoteError(f"the fly run failed on the server:\n{st['tail']}")
        sleep(poll_s)


def score_run(song_mid, hits_csv):
    ref, played = scoring.load_hits(song_mid), scoring.load_hits(hits_csv)
    return scoring.score(ref, played)  # no alignment: the fly's hits are on the song's own clock


def finish(remote, song_id, runs=REPO / "runs", songs=to_fly.SONGS):
    """Fetch a finished run, update the viewer's index, score it. -> result dict."""
    run = Path(runs) / "songs" / song_id
    got = remote.fetch_run(song_id, run)
    if "viewer.json" not in got:
        raise RemoteError(f"runs/songs/{song_id} on the server has no viewer export (got {got})")
    remote.sync_model(Path(runs) / "model")
    song_json = Path(songs) / f"{song_id}.json"
    if song_json.exists():  # keep what went in next to what came out
        (run / "song.json").write_text(song_json.read_text())
    update_index(runs)
    result = {"id": f"songs/{song_id}", "run": str(run), "url": VIEWER.format(id=f"songs/{song_id}")}
    song_mid = Path(songs) / f"{song_id}.mid"
    if song_mid.exists():
        result["score"], result["per_drum"] = score_run(song_mid, run / "hits.csv")
    return result


def play(report, remote, weights, confirm=ask, progress=print, poll_s=5, sleep=time.sleep, runs=REPO / "runs",
         songs=to_fly.SONGS, on_status=None):
    """A song made by to_fly -> the fly plays it on the server -> the run, fetched and scored."""
    song_id = report["name"]
    remote.check()
    if weights and not remote.exists(weights):
        raise RemoteError(f"no weights at {weights} on the server")
    jobs = remote.gpu_jobs()
    if jobs and not confirm(jobs):
        raise RemoteError("not started: the GPU is busy")
    remote.upload(report["mid"], f"songs/{song_id}.mid")
    remote.start(song_id, weights)
    progress(f"{song_id}: the fly is playing it on the server (~{report['fly_seconds'] / 60:.0f} min)")
    wait(remote, song_id, progress, poll_s, sleep, eta_hint=report["fly_seconds"], on_status=on_status)
    return finish(remote, song_id, runs, songs)


def describe_result(result):
    lines = [f"run: {result['run']}"]
    if "score" in result:
        s = result["score"]
        lines.append(f"the fly played {s['played']} hits for {s['ref']} notes: {s['hit']} right, {s['miss']} missed, "
                     f"{s['extra']} extra (F1 {s['f1']:.2f}, within {scoring.TOLERANCE_MS} ms)")
    lines.append(f"view: {result['url']}  (serve the repo root: python -m http.server 8000)")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, help="sheet music, .mid, .txt grid, .mscz or take .csv")
    ap.add_argument("--record", action="store_true", help="play it on the kit instead of giving a file")
    ap.add_argument("--fetch", metavar="NAME", help="fetch and score a finished run instead of starting one")
    ap.add_argument("--name", help="song name (default: the file name, or take_<time>)")
    ap.add_argument("--seconds", type=float, help="play only the first N seconds (from the first hit)")
    ap.add_argument("--bpm", type=float, help="tempo override for sheet music; count-in tempo with --record")
    ap.add_argument("--repeat", type=int, default=1, help="sheet music: play it N times")
    ap.add_argument("--musescore", help="path to MuseScore, for .mscz")
    ap.add_argument("--count-in", type=int, default=0, metavar="BEATS", help="--record: clicks on the kit first")
    ap.add_argument("--port", help="--record: TD-07 input port name")
    ap.add_argument("--weights", help="trained weights on the server (default: fly_pipeline.json's)")
    ap.add_argument("--untrained", action="store_true", help="run the untrained fly (no weights)")
    ap.add_argument("--yes", action="store_true", help="don't ask when the server's GPU is busy")
    ap.add_argument("--open", action="store_true", help="open the viewer in the browser at the end")
    args = ap.parse_args()

    cfg = load_config()
    remote = Remote.from_config(cfg)
    try:
        if args.fetch:
            result = finish(remote, to_fly.slug(args.fetch))
        else:
            weights = args.weights or cfg.get("weights")
            if not weights and not args.untrained:
                sys.exit("No trained weights: pass --weights runs/train/<id>/best.pt, set \"weights\" in "
                         "human/fly_pipeline.json, or use --untrained")
            if args.record:
                import record_song
                report = record_song.record_song(args.port, args.count_in, args.bpm or 100, max_s=args.seconds or 20,
                                                 name=args.name)
            elif args.file:
                report = to_fly.make_song(args.file, args.name, args.bpm, args.repeat, args.seconds, args.musescore)
            else:
                ap.error("give a file, --record or --fetch NAME")
            print(to_fly.describe(report))
            result = play(report, remote, None if args.untrained else weights,
                          confirm=(lambda jobs: True) if args.yes else ask)
    except (to_fly.SongError, RemoteError) as e:
        sys.exit(str(e))
    if "per_drum" in result:
        scoring.print_detail(result["score"], result["per_drum"])
    print(describe_result(result))
    if args.open:
        webbrowser.open(result["url"])


if __name__ == "__main__":
    main()

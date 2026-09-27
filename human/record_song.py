"""Play something on the TD-07 and turn it into a song for the fly.

Recording starts at the first hit (or after a count-in played on the kit) and stops by itself once you stop
playing (--silence seconds without a hit), at --max-seconds, or on Ctrl+C. The take is saved like midi_capture.py
saves one (takes/take_<stamp>.csv/.mid, fake kicks marked), then to_fly.py makes songs/<name>.mid from it.

Usage:
    python human/record_song.py                       # play, then stop for 2.5 s
    python human/record_song.py --count-in 4 --bpm 90 # four clicks on the kit first
    python human/record_song.py --name my_groove --max-seconds 12
    python human/record_song.py --list                # show MIDI ports
"""
import argparse
import sys
import time
from datetime import datetime

import mido

import midi_capture
import to_fly
from drum_map import DRUM_CHANNEL, DRUMS, HitFilter
from midi_capture import HAT_PEDAL_CC, SKIP_TYPES, find_port

midi_capture.TAKES_DIR = to_fly.REPO / "takes"  # her save() writes relative to the working folder
CLICK_NOTE = 37   # cross-stick: short and clearly not part of the groove
CLICK_MS = 30
POLL_S = 0.0005   # as midi_capture.py


def count_in(out_port, beats, bpm, clock=time.perf_counter, sleep=time.sleep):
    """Click `beats` times on the kit. Returns the clock time of the next downbeat, where the take starts."""
    beat_s = 60 / bpm
    start = clock()
    for i in range(beats):
        while clock() < start + i * beat_s:
            sleep(POLL_S)
        out_port.send(mido.Message("note_on", channel=DRUM_CHANNEL, note=CLICK_NOTE, velocity=100 if i == 0 else 70))
        sleep(CLICK_MS / 1000)
        out_port.send(mido.Message("note_off", channel=DRUM_CHANNEL, note=CLICK_NOTE, velocity=0))
    return start + beats * beat_s


def record(port, silence_s=2.5, max_s=20.0, wait_s=60.0, start=None, clock=time.perf_counter, sleep=time.sleep,
           echo=print):
    """Read `port` until the player stops. -> [(t_ms, msg, ignored_reason)] in midi_capture.save()'s format.

    Stops `silence_s` after the last real hit, `max_s` after the first one, after `wait_s` with nothing played,
    or on Ctrl+C. Times count from `start` (a count-in's downbeat) or from when recording began."""
    if start is not None:
        while clock() < start:  # the count-in: drop anything the kit sent meanwhile (e.g. echoed clicks)
            list(port.iter_pending())
            sleep(POLL_S)
    t0 = clock() if start is None else start
    events, hit_filter = [], HitFilter()
    first = last = None  # clock times of the first and latest real hit
    try:
        while True:
            now = clock()
            if first is None and now - t0 >= wait_s:
                echo(f"Nothing played in {wait_s:.0f} s; stopping.")
                break
            if last is not None and now - last >= silence_s:
                break
            if first is not None and now - first >= max_s:
                echo(f"Reached {max_s:.0f} s; stopping.")
                break
            for msg in port.iter_pending():
                if msg.type in SKIP_TYPES:
                    continue
                now = clock()
                t = (now - t0) * 1000
                reason = None
                if msg.type == "note_on" and msg.velocity > 0:
                    reason = hit_filter.check(t, msg.note, msg.velocity)
                    if not reason:
                        first = now if first is None else first
                        last = now
                    label = f"  ({reason}, ignored)" if reason else ""
                    echo(f"{t:10.1f} ms  HIT  {msg.note:>3} {DRUMS.get(msg.note, '?'):<16} vel {msg.velocity}{label}")
                elif msg.type == "control_change" and msg.control == HAT_PEDAL_CC:
                    echo(f"{t:10.1f} ms  PEDAL CC4 = {msg.value}")
                events.append((t, msg, reason))
            sleep(POLL_S)
    except KeyboardInterrupt:
        pass
    return events


def find_output(names):
    return next((o for o in names if "TD-07" in o.upper() or "TD07" in o.upper()), None)


def record_song(port=None, count=0, bpm=100.0, silence_s=2.5, max_s=20.0, wait_s=60.0, name=None, echo=print,
                out_dir=None):
    """Record from the kit, save the take, make the song. -> to_fly's report (with "take": the take .csv)."""
    try:
        in_name = find_port(mido.get_input_names(), port)
    except SystemExit as e:  # midi_capture reports a missing kit by exiting
        raise to_fly.SongError(str(e)) from None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with mido.open_input(in_name) as inp:
        start = None
        if count:
            out_name = find_output(mido.get_output_names())
            if not out_name:
                raise to_fly.SongError(f"no TD-07 output port for the count-in: {mido.get_output_names()}")
            echo(f"Count-in: {count} clicks at {bpm:g} BPM, then play.")
            with mido.open_output(out_name) as out:
                start = count_in(out, count, bpm)
        else:
            echo(f"Listening on {in_name}. Play; stop for {silence_s:g} s to finish.")
        events = record(inp, silence_s, max_s, wait_s, start, echo=echo)
    if not any(m.type == "note_on" and m.velocity > 0 and not r for _, m, r in events):
        raise to_fly.SongError("nothing was played")
    csv_path, _ = midi_capture.save(events, stamp)
    report = to_fly.make_song(csv_path, name=name or f"take_{stamp}", out_dir=out_dir or to_fly.SONGS)
    return {**report, "take": str(csv_path)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list MIDI ports and exit")
    ap.add_argument("--port", help="exact input port name (default: the first TD-07 input)")
    ap.add_argument("--count-in", type=int, default=0, metavar="BEATS", help="click this many beats on the kit first")
    ap.add_argument("--bpm", type=float, default=100, help="count-in tempo")
    ap.add_argument("--silence", type=float, default=2.5, help="stop after this many seconds without a hit")
    ap.add_argument("--max-seconds", type=float, default=20, help="stop this long after the first hit")
    ap.add_argument("--wait", type=float, default=60, help="give up if nothing is played for this long")
    ap.add_argument("--name", help="song name (default: take_<timestamp>)")
    args = ap.parse_args()
    if args.list:
        print("Inputs: ", mido.get_input_names())
        print("Outputs:", mido.get_output_names())
        return
    try:
        report = record_song(args.port, args.count_in, args.bpm, args.silence, args.max_seconds, args.wait, args.name)
    except to_fly.SongError as e:
        sys.exit(str(e))
    print(f"\nTake saved to {report['take']}")
    print(to_fly.describe(report))


if __name__ == "__main__":
    main()

"""Play drums through the TD-07 (or another MIDI output) so it sounds like the kit.

Accepts a .mid file, or sheet music (.musicxml / .mxl / .txt grid), which is converted on the fly.

Usage:
    python human/play_midi.py takes/take_20260926_120000.mid
    python human/play_midi.py songs/my_score.musicxml
    python human/play_midi.py grooves/rock_beat.txt --bpm 80
    python human/play_midi.py take.mid --port "Microsoft GS Wavetable Synth 0"
"""
import argparse
import sys
import time
from pathlib import Path

import mido

from sheet_to_midi import SHEET_EXTS, convert


def play_events(port, events):
    """Send [(t_ms, msg), ...] in real time."""
    t0 = time.perf_counter()
    for t, msg in events:
        wait = t / 1000 - (time.perf_counter() - t0)
        if wait > 0:
            time.sleep(wait)
        port.send(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help=".mid, .musicxml, .mxl or .txt file to play")
    ap.add_argument("--port", help="exact output port name (default: first TD-07 output)")
    ap.add_argument("--bpm", type=float, help="override the score's tempo (sheet music only)")
    args = ap.parse_args()

    outs = mido.get_output_names()
    name = args.port or next((o for o in outs if "TD-07" in o.upper() or "TD07" in o.upper()), None)
    if name not in outs:
        sys.exit(f"Output port not found. Available: {outs}")

    path = Path(args.file)
    if path.suffix.lower() in SHEET_EXTS:
        events, length_ms = convert(path, args.bpm)
    else:
        mid = mido.MidiFile(path)
        events, t = [], 0.0
        for msg in mid:  # msg.time is seconds since the previous message
            t += msg.time
            if not msg.is_meta:
                events.append((t * 1000, msg))
        length_ms = mid.length * 1000

    print(f"Playing {path} ({length_ms / 1000:.1f} s) on {name}. Ctrl+C to stop.")
    with mido.open_output(name) as port:
        try:
            play_events(port, events)
        except KeyboardInterrupt:
            port.reset()


if __name__ == "__main__":
    main()

"""Capture live MIDI from a Roland TD-07 kit and save each take as CSV + .mid.

Usage:
    python human/midi_capture.py --list          # show MIDI ports
    python human/midi_capture.py                 # capture from the first TD-07 input
    python human/midi_capture.py --port "NAME"   # capture from a specific input
    python human/midi_capture.py --echo          # send a test snare hit to the kit first
    python human/midi_capture.py --raw           # keep crosstalk, bounces and stray touches

Press Ctrl+C to stop; the take is saved to takes/take_<timestamp>.csv/.mid.
Fake hits (see HitFilter in drum_map.py) are left out of the .mid and marked in the CSV's "ignored" column.
"""
import argparse
import csv
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import mido

from drum_map import DRUMS, HitFilter, summarize, write_midi

HAT_PEDAL_CC = 4
SKIP_TYPES = {"active_sensing", "clock", "start", "stop", "continue"}
TAKES_DIR = Path("takes")


def find_port(names, wanted):
    if wanted:
        if wanted in names:
            return wanted
        sys.exit(f"Port {wanted!r} not found. Available: {names}")
    for n in names:
        if "TD-07" in n.upper() or "TD07" in n.upper():
            return n
    sys.exit(f"No TD-07 input found. Available inputs: {names or 'none'}\n"
             "Check the kit is switched on and the USB cable is plugged in.")


def echo_test(in_name):
    outs = mido.get_output_names()
    out = next((o for o in outs if "TD-07" in o.upper() or "TD07" in o.upper()), None)
    if not out:
        print(f"[echo] No TD-07 output port. Outputs: {outs}")
        return
    with mido.open_output(out) as port:
        print(f"[echo] Sending snare (38) to {out}")
        port.send(mido.Message("note_on", channel=9, note=38, velocity=100))
        time.sleep(0.1)
        port.send(mido.Message("note_off", channel=9, note=38, velocity=0))


def save(events, stamp):
    """events: [(t_ms, msg, ignored_reason)]. The CSV keeps everything; the .mid only real hits."""
    TAKES_DIR.mkdir(exist_ok=True)
    csv_path = TAKES_DIR / f"take_{stamp}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "type", "channel", "note", "drum", "velocity", "cc", "value", "ignored"])
        for t, msg, reason in events:
            note = getattr(msg, "note", "")
            w.writerow([f"{t:.2f}", msg.type, getattr(msg, "channel", ""), note,
                        DRUMS.get(note, "?") if note != "" else "",
                        getattr(msg, "velocity", ""), getattr(msg, "control", ""),
                        getattr(msg, "value", ""), reason or ""])

    mid_path = TAKES_DIR / f"take_{stamp}.mid"
    write_midi([(t, msg) for t, msg, reason in events if not reason], mid_path)
    return csv_path, mid_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list MIDI ports and exit")
    ap.add_argument("--port", help="exact input port name")
    ap.add_argument("--echo", action="store_true", help="send a test snare note to the kit")
    ap.add_argument("--raw", action="store_true", help="keep crosstalk, kick bounces and stray touches in the .mid")
    args = ap.parse_args()

    if args.list:
        print("Inputs: ", mido.get_input_names())
        print("Outputs:", mido.get_output_names())
        return

    name = find_port(mido.get_input_names(), args.port)
    if args.echo:
        echo_test(name)

    events = []  # (t_ms, msg, ignored_reason)
    hit_filter = HitFilter()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Listening on {name}. Hit some pads; Ctrl+C to stop.\n")
    with mido.open_input(name) as port:
        t0 = time.perf_counter()
        try:
            while True:
                for msg in port.iter_pending():
                    if msg.type in SKIP_TYPES:
                        continue
                    t = (time.perf_counter() - t0) * 1000
                    reason = None
                    if msg.type == "note_on" and msg.velocity > 0:
                        reason = None if args.raw else hit_filter.check(t, msg.note, msg.velocity)
                        label = f"  ({reason}, ignored)" if reason else ""
                        print(f"{t:10.1f} ms  HIT  {msg.note:>3} {DRUMS.get(msg.note, '?'):<16} vel {msg.velocity}{label}")
                    elif msg.type == "control_change" and msg.control == HAT_PEDAL_CC:
                        print(f"{t:10.1f} ms  PEDAL CC4 = {msg.value}")
                    elif msg.type not in ("note_off", "note_on"):
                        print(f"{t:10.1f} ms  {msg}")
                    events.append((t, msg, reason))
                time.sleep(0.0005)
        except KeyboardInterrupt:
            pass

    if events:
        csv_path, mid_path = save(events, stamp)
        print(f"\nSaved {len(events)} events to {csv_path} and {mid_path}")
    ignored = Counter(reason for _, _, reason in events if reason)
    if ignored:
        print("Ignored: " + ", ".join(f"{n} {reason}" for reason, n in ignored.items()))
    summarize([(t, msg) for t, msg, reason in events if not reason])


if __name__ == "__main__":
    main()

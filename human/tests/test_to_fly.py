import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import to_fly
from prep_takes import LEAD_MS
from sheet_to_midi import read_midi

REPO = Path(__file__).resolve().parents[2]
GROOVES = REPO / "grooves"

# One 4/4 bar at 120 BPM with an end-repeat (so it plays twice): kick, snare, a shaker the kit doesn't have,
# and a note with no <instrument> tag, which sheet_to_midi skips without saying so.
SCORE = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1">
 <part-list><score-part id="P1"><part-name>Drumset</part-name>
  <score-instrument id="P1-I36"><instrument-name>Bass Drum</instrument-name></score-instrument>
  <score-instrument id="P1-I39"><instrument-name>Snare</instrument-name></score-instrument>
  <score-instrument id="P1-I83"><instrument-name>Shaker</instrument-name></score-instrument>
  <midi-instrument id="P1-I36"><midi-channel>10</midi-channel><midi-unpitched>37</midi-unpitched></midi-instrument>
  <midi-instrument id="P1-I39"><midi-channel>10</midi-channel><midi-unpitched>39</midi-unpitched></midi-instrument>
  <midi-instrument id="P1-I83"><midi-channel>10</midi-channel><midi-unpitched>83</midi-unpitched></midi-instrument>
 </score-part></part-list>
 <part id="P1">
  <measure number="1">
   <attributes><divisions>1</divisions></attributes>
   <direction><sound tempo="120"/></direction>
   <note><unpitched/><duration>1</duration><instrument id="P1-I36"/></note>
   <note><unpitched/><duration>1</duration><instrument id="P1-I39"/></note>
   <note><unpitched/><duration>1</duration><instrument id="P1-I83"/></note>
   <note><unpitched/><duration>1</duration></note>
   <barline location="right"><repeat direction="backward"/></barline>
  </measure>
 </part>
</score-partwise>
"""


def hits_of(mid):
    return [(round(t), n, v) for t, n, v in read_midi(mid)[0]]


def test_grid_becomes_a_song(tmp_path):
    r = to_fly.make_song(GROOVES / "rock_beat.txt", out_dir=tmp_path)
    hits = hits_of(r["mid"])
    assert r["hits"] == len(hits) == 26
    assert hits[0][0] == LEAD_MS
    assert r["voices"] == {"hat_closed": 15, "kick": 6, "snare": 4, "crash": 1}
    assert r["dropped_notes"] == {} and r["known_groove"] is None
    assert json.loads((tmp_path / "rock_beat.json").read_text()) == r


def test_grid_keeps_ghosts_and_accents(tmp_path):
    r = to_fly.make_song(GROOVES / "funk_16ths.txt", out_dir=tmp_path)
    assert {50, 90, 120} <= {v for _, _, v in hits_of(r["mid"])}


def test_musicxml_repeats_drops_and_warns(tmp_path):
    score = tmp_path / "bar.musicxml"
    score.write_text(SCORE)
    r = to_fly.make_song(score, out_dir=tmp_path / "songs")
    # kick 0, snare 500 ms, then the repeat: kick 2000, snare 2500 -> shifted to start at LEAD_MS
    assert [(t, n) for t, n, _ in hits_of(r["mid"])] == [(1000, 36), (1500, 38), (3000, 36), (3500, 38)]
    assert r["dropped_notes"] == {"82": 2}  # the shaker, once per pass
    assert r["skipped_untagged"] == 1


def test_training_and_heldout_grooves_are_recognised(tmp_path):
    take = sorted((GROOVES / "train").glob("BD_112bpm_*.mid"))[0]
    assert to_fly.make_song(take, out_dir=tmp_path)["known_groove"] == f"train/{take.name}"
    assert to_fly.make_song(GROOVES / "heldout" / "heldout_1.mid", out_dir=tmp_path)["known_groove"] \
        == "heldout/heldout_1.mid"


def test_seconds_crops_from_the_first_hit(tmp_path):
    full = to_fly.make_song(GROOVES / "rock_beat.txt", out_dir=tmp_path)
    short = to_fly.make_song(GROOVES / "rock_beat.txt", name="short", seconds=2, out_dir=tmp_path)
    assert 0 < short["hits"] < full["hits"]
    assert max(t for t, _, _ in hits_of(short["mid"])) < LEAD_MS + 2000
    assert short["fly_seconds"] < full["fly_seconds"]


def test_kit_take_csv_is_cleaned(tmp_path):
    take = tmp_path / "take_20260927_010000.csv"
    rows = ["t_ms,type,note,velocity,ignored",
            "100,note_on,36,100,", "150,note_on,36,40,",  # a hard kick and its beater bounce
            "400,note_on,38,90,", "405,note_on,36,20,",   # a snare and the crosstalk kick it causes
            "800,note_on,42,70,"]
    take.write_text("\n".join(rows) + "\n")
    r = to_fly.make_song(take, out_dir=tmp_path)
    assert [(t, n) for t, n, _ in hits_of(r["mid"])] == [(1000, 36), (1300, 38), (1700, 42)]
    assert r["cleaned"] == {"bounce": 1, "crosstalk": 1}
    assert r["name"] == "take_20260927_010000"


def test_doubled_notes_become_one_hit():
    hits, dropped = to_fly.prepare([(0, 38, 60), (0, 38, 100), (500, 36, 90), (500, 99, 90)])
    assert hits == [(LEAD_MS, 38, 100), (LEAD_MS + 500, 36, 90)]
    assert dropped == {99: 1}


def test_bad_input_raises_song_error(tmp_path):
    (tmp_path / "notes.pdf").write_bytes(b"%PDF")
    with pytest.raises(to_fly.SongError, match="unsupported"):
        to_fly.make_song(tmp_path / "notes.pdf", out_dir=tmp_path)
    with pytest.raises(to_fly.SongError, match="no notes the fly can play"):
        to_fly.prepare([(0, 99, 90)])
    with pytest.raises(to_fly.SongError, match="MuseScore not found"):
        to_fly.find_musescore(str(tmp_path / "no-such-mscore"))


@pytest.mark.skipif(not os.environ.get("MUSESCORE"), reason="set MUSESCORE=<path to MuseScore> to test .mscz")
def test_mscz_round_trip(tmp_path):
    """A groove written out as sheet music, saved by MuseScore as .mscz, comes back note for note."""
    ms = os.environ["MUSESCORE"]
    xml, mscz = tmp_path / "funk.musicxml", tmp_path / "funk.mscz"
    subprocess.run([sys.executable, str(REPO / "human" / "take_to_sheet.py"), str(GROOVES / "funk_16ths.mid"),
                    "--bpm", "92", "-o", str(xml)], check=True, capture_output=True)
    subprocess.run([ms, "-o", str(mscz), str(xml)], check=True, capture_output=True,
                   env={**os.environ, "QT_QPA_PLATFORM": "offscreen"})
    r = to_fly.make_song(mscz, musescore=ms, out_dir=tmp_path / "songs")
    ref = to_fly.prepare(read_midi(GROOVES / "funk_16ths.mid")[0])[0]
    got = hits_of(r["mid"])
    assert [(n, v) for _, n, v in ref] == [(n, v) for _, n, v in got]
    assert max(abs(a[0] - b[0]) for a, b in zip(ref, got)) <= 2  # 92 BPM 16ths (163.04 ms) round to MIDI ticks


def test_names_are_safe_file_names():
    assert to_fly.slug("My Song (live) v2!") == "My_Song_live_v2"
    assert to_fly.slug("???") == "song"

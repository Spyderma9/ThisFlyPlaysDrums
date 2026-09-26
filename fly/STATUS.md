# Fly side: status and plan

As of Sat Sept 26, 04:00 EDT. Code freeze Sun 08:00, deadline Sun 11:00.

## Who builds what

| Piece | Who | Notes |
|---|---|---|
| Encoder (MIDI → cue-neuron input) | **Fly side, already built** (`fly/encoder.py`) | Please skip it. It already reads your `grooves/train/` format as-is. |
| Scoring (fly hits vs. the score) | **You** | Hits, misses, extra hits, timing and loudness per drum, plus your human-take baseline. The fly side only produces hits files. |
| Viewer (3D replay + Web MIDI to the kit) | Fly side (`viewer/`) | |
| Recording, cleaning, sheet music, playback | You | |

## Answers to your questions

1. **Rates, not spike times.** The encoder outputs a firing rate per drum over time (`[time, drums]` in Hz). The brain wrapper spreads each drum's rate over its cue neurons, and fly-brain's Poisson generator makes the spikes.
2. **Time step: yes, 1 ms.**
3. **Neuron IDs:** the cue neurons for each drum get picked by the probe in Phase 1 below. The whole connectome is simulated either way, so group size isn't a capacity limit. The IDs live in the brain wrapper, not the encoder, so you don't need placeholders.
4. **Benchmark:** not measured yet, because the server isn't reachable. The demo is a **recorded replay** either way: the viewer plays back a finished run and sends the hits to the kit in sync.

## The kit: all 12 drums, played like a person

The kit is modified so the fly can reach it.

| Fly leg | Plays |
|---|---|
| Front right leg (holds a stick) | hi-hat, ride, ride bell, crash; toms and fast repeats alternate with the left |
| Front left leg (holds a stick) | snare, cross-stick; toms and fast repeats alternate with the right |
| Back right leg | kick pedal (36) |
| Back left leg | hi-hat pedal (44). It holds the hat closed and lifts it for open hits, so a stick hit on the hat sounds 42 or 46 depending on the pedal, like a real kit. |

This uses your 12-note table exactly. Edge and rim folding (hat edge → hat, and so on) matches yours.

## What I need from you

1. **Push the training takes.** `origin/midi` still has only `.gitkeep` in `grooves/train/`. `prep_takes.py`'s output format is exactly what the encoder reads. Cross-stick and ride-bell takes are welcome when you have them.
2. **Held-out grooves: at least 2 in `grooves/heldout/` by Sun 02:00.** Write them yourself (no copyrighted transcriptions). The fly is scored on these and never trains on them, so don't put the same groove in `grooves/train/` too.
3. **Keep the note numbers stable.** The fly side copies them from your `human/drum_map.py`, so tell me if any change.
4. **Hits come back in three formats,** all in `runs/<id>/`:
   - `hits.mid`: channel 10, TD-07 notes, velocity from how hard the stick hit. `python human/play_midi.py hits.mid` plays it on the kit (the demo backup), so please keep that command working.
   - `hits.csv`: `t_ms, note, velocity`, as you proposed, for scoring.
   - `hits.json`: the same hits plus the contact speed (for the viewer).
5. **Ownership heads-up.** Besides `fly/`, I own `viewer/` (not created yet) and `.env.example` at the repo root. I don't touch `human/`, `grooves/` or the root `.gitignore`. Our branches share no edited files, so `server` and `midi` merge cleanly.

## How the fly side works

1. **Encoder:** your `.mid` becomes spike input. Each drum has a group of cue neurons that fires a short burst a little before each note (louder note, faster firing). It follows fixed rules and never sees what the fly should play.
2. **Brain:** a simulation of the real fruit-fly connectome (MaleCNS v1.0, brain and nerve cord, about 167,000 neurons), run with the eonsystems fly-brain neuron model.
3. **Decoder:** leg motor-neuron firing rates become joint angles for a simulated fly body (flybody in MuJoCo). Also fixed rules.
4. **Hits:** a note counts only when a stick or pedal actually makes contact in the physics simulation. Those contacts become the hits files.
5. **Training:** the fly first gets its legs guided through the right strokes, and the guidance fades to zero. Only a small set of synapses is trainable.
6. **Test:** with no guidance, on your held-out grooves, compared with a shuffled connectome. Your scoring code does the scoring.

## Done

- **Committed** (`server` branch, `6a4d687`):
  - the encoder
  - the brain wrapper (fly-brain's neuron model at 1 ms, only chosen synapses trainable)
  - the connectome container with a shuffled control
  - tests
- **Since then** (not committed yet):
  - **Switched the connectome to a direct download of MaleCNS v1.0.** neuPrint's API only hosts v0.9.
  - **The data is downloaded and checked:** 166,700 neurons, about 152 million connections. Leg motor neurons: 81 front, 116 middle and 122 hind.
- **All 7 tests pass.**

## Blocked right now

- **The server isn't on my tailnet yet.** The fly runs on an Unraid box with an RTX 3060 Ti. Everything in the plan runs there.

## Plan

Times are targets. Phase 0 is running about an hour late because of the server connection.

| Phase | When | What | What you'll see |
|---|---|---|---|
| 0 | now → ~05:00 | Server setup: CUDA container, Python env, benchmark, load the connectome, measure memory and speed | nothing yet |
| 1 | → Sat ~07:30 | **Probe:** does input reach the leg motor neurons, and how fast? This picks the cue neurons and the trainable synapses. | a short report |
| 2 | → Sat ~14:00 | Untrained loop end to end with the 12-drum kit: encoder → brain → body → hits | **Milestone 1:** the untrained fly's (wrong) hits play on the kit |
| 3 | → Sat ~17:00 | Stroke templates: stick and pedal motion for each drum, timed to your grooves | a check that the ideal strokes land on time |
| 4 | Sat 17:00 → Sun 02:00 | Training on `grooves/train/` | loss curves |
| 4b | Sat 18:00 → Sun 01:00 | The viewer (3D fly, neuron activity, Web MIDI), built while training runs | a replay in Chrome |
| 5 | Sun 02:00 → 06:00 | Held-out and shuffled-connectome runs; you score them | your scores table |
| 6 | Sun 06:00 → 08:00 | Record demo runs, test viewer and kit together, **freeze** | the demo |

**Fallback:** if training isn't clearly working by Sat 22:00, a simple fitted readout becomes the fallback. It will be labeled as a baseline.

## Times I'll need you

- **As soon as you can:** push the training takes.
- **Sat afternoon:** Milestone 1. Play a `hits.mid` on the kit and run your scorer on a `hits.csv`.
- **Sun 02:00:** held-out grooves pushed. Scoring ready.
- **Sun 06:00–08:00:** viewer and TD-07 rehearsal on the laptop with the kit.

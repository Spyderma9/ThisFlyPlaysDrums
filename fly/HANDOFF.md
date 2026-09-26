# Fly side handoff

Updated Sat Sept 26, ~08:00 EDT, at the end of Phase 1. Deadline Sun Sept 27, 11:00 EDT; code freeze Sun 08:00.

## Start here (for a new session)

You're continuing the fly side of Fly Drums. The previous session's context is gone; this file replaces it and is complete on its own.

1. **Read first:**
   - this file, in full
   - `/home/daniel/ThisFlyPlaysDrums/CLAUDE.md` (local, gitignored: scope, rules, architecture, data formats)
   - the plan at `/home/daniel/.claude/plans/i-own-the-fly-splendid-creek.md` (phases, timings, risks)
2. **This file wins on conflicts.** It's the newest record. Don't edit `CLAUDE.md`; the user moves permanent decisions into it themselves. **At the start of the session, tell the user about these conflicts:**
   - `CLAUDE.md` says "MaleCNS v1.0 via neuPrint". Superseded: bulk download.
   - `CLAUDE.md` says "flybody or NeuroMechFly". Superseded: flybody.
   - `CLAUDE.md` gives KC→MBON as the example trainable subset. Superseded: cue → descending-neuron synapses (D3), with the mushroom body's inputs cut (D8).
   - `CLAUDE.md` says the encoder shifts notes ~100–200 ms early. Measured cue → motor latency is 20–55 ms; lookahead = latency + stroke lead (Phase 3).
   - The kit is **12 drums played with four legs**. `CLAUDE.md` doesn't say so yet.
3. **Rules for every step:**
   - Never `git commit`, `git push` or amend. The user commits.
   - Edit only `fly/`, `viewer/` and `.env.example`. Never `human/`, `grooves/` or the root `.gitignore`.
   - Ask every open question **in one batch before starting**, after checking this file and `CLAUDE.md`.
   - **Stop at the end of each phase:** tell the user what was done and what the next phase is, then wait. Stop at every decision gate; don't choose for the user.
   - GPU: short runs (a few minutes) are pre-approved. **Stop and ask before training or any long run.** The GPU is shared with other Unraid containers.
   - Python 3.10-compatible code. Write like the surrounding code: dense, few comments.
4. **Where we are:** Phases 0 and 1 are done. **Next is Phase 2** (spec below).
   - Before starting, ask the open questions under "Open questions" in one batch.
   - Stop at Milestone 1 and report.

## State in one paragraph

**Git and working tree:**
- Branch `server` at `07e85c2`, which includes Sam's merged `midi` work: `human/`, grooves, and 21 training takes in `grooves/train/`.
- **Uncommitted:**
  - modified: `fly/HANDOFF.md`, `fly/connectome.py`, `fly/drums.py`, `fly/tests/test_encoder.py`
  - new: `fly/probe.py`, `fly/cues.json`, `fly/tests/test_probe.py`

**Server:**
- The server copy of `fly/` is rsynced from this machine and matches it.
- **The server doesn't have `grooves/` or `human/` yet.** Sync the whole repo before Phase 2 runs (see Commands).
- Nothing is running on the server, and no tmux sessions are open.

**Tests:** 10/10 pass locally.

## People and ownership

- **The user** owns `fly/` and `viewer/` (not created yet) on branch `server`. Also `.env.example`.
- **Sam** (partner, she/her) owns `human/`, `grooves/` and the root `.gitignore`. **Her work is done:**
  - capture, cleaning, sheet-music conversion, `play_midi.py`
  - **`human/score.py`** (scoring is hers: we only produce hits files)
  - 21 training takes (about 5 minutes)
  - She was asked to put ≥ 2 held-out grooves in `grooves/heldout/` by Sun 02:00. Not there yet.
- **`human/score.py`:**
  - Reads played hits from `hits.csv` (`t_ms,note,velocity`) or `.mid`. The reference can be `.mid`, MusicXML or a text grid.
  - Pairs hits within 60 ms, per drum, folding edges and rims.
  - **So our hits must be in score time.**
  - Usage: `python human/score.py <reference.mid> runs/<id>/hits.csv`.
- **Training takes:** channel-10 `.mid`, TD-07 notes, `note_on` followed by `note_off` 50 ms later, 1 s of silence before the first hit, fake hits already filtered. `grooves/train/index.csv` lists them; the encoder ignores it.
- The FlyWire fallback probe was skipped. The user got instructions to pass to Sam, if she wants to run it: free the GPU, then `/mnt/user/dev/fx "python -m fly.probe --connectome flywire"`.

## Machines and access

**Dev machine** (this one): Debian, `/home/daniel/ThisFlyPlaysDrums`.
- Python 3.13, no conda, no GPU, no tmux. Only ~1.2 GB of RAM free, so don't load full MaleCNS here.
- **Test venv** (it lives in `/tmp`, so recreate it if missing):
  ```bash
  python3 -m venv --without-pip /tmp/fdvenv
  pip3 --python /tmp/fdvenv/bin/python install -q numpy pandas pyarrow mido pytest scipy
  pip3 --python /tmp/fdvenv/bin/python install -q torch --index-url https://download.pytorch.org/whl/cpu
  cd /home/daniel/ThisFlyPlaysDrums && /tmp/fdvenv/bin/python -m pytest fly/tests -q -p no:cacheprovider
  ```
- `fly/vendor/fly-brain` here is a sparse clone (`code/` only), so the brain tests run but FlyWire data isn't here.
- Tailscale was installed from Tailscale's apt repo (tailscale.com itself is unreachable from this network). Tailnet `spyderma9.github`.

**Server:** Dell T5600 running Unraid 7.3.2. 2× Xeon E5-2670 (AVX, **no AVX2**), 64 GB RAM, RTX 3060 Ti 8 GB.
- **SSH:** `ssh tower`. `~/.ssh/config` maps it to `t5600.tail3495cd.ts.net`, user root, key `~/.ssh/id_ed25519`. The machine is shared into our tailnet from juan.borgesjr's account; the key was added in Unraid's Users → root.
- **Storage:**
  - Repo at `/mnt/user/dev/ThisFlyPlaysDrums`.
  - `/mnt/user/dev` is the ZFS dataset `/mnt/pool/dev`. There are no array disks.
  - `/root` is in RAM, so keep everything under `/mnt/user/dev`.
- **Container `flydrums`:**
  - Image `nvidia/cuda:12.6.3-base-ubuntu22.04`, `--runtime=nvidia --gpus all`, `-e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl`, `--shm-size=8g`, `sleep infinity`.
  - **Mounts `/mnt/pool/dev` at `/mnt/user/dev`.** This bypasses Unraid's FUSE layer (shfs), which made conda crawl.
  - apt packages (`git curl libgl1 libegl1 libosmesa6 libglib2.0-0`) live in the container layer. Reinstall them if you recreate the container.
- **Env:**
  - Miniforge at `/mnt/user/dev/miniforge3`, env `flydrums`, from `fly/environment.yml`: conda-forge plus pip torch cu126, numpy 1.26.4.
  - Versions: torch 2.14.0+cu126, mujoco 3.14.0, dm_control 1.0.47, flybody 0.1.0.
  - `LD_LIBRARY_PATH` is set as an env var through `conda env config vars`. Without it, pyarrow fails with GLIBCXX_3.4.31 when torch is imported first. Conda prints a harmless "overwriting variable" warning on every activation; filter it with `grep -vE "WARNING|overwriting"`.
- **Run anything in the env:** `ssh tower '/mnt/user/dev/fx "<command>"'`. `fx` does `docker exec` into the container with the env active, from the repo root.
  - For ad-hoc scripts outside the repo, add `PYTHONPATH=.`.
  - Nested quoting through ssh + fx + `python -c` breaks easily. Write the script to the scratchpad, `scp` it to `tower:/mnt/user/dev/`, and run it.
- **Long runs:** in host tmux, `ssh tower 'tmux new-session -d -s NAME "/mnt/user/dev/fx \"python -u -m fly.X ...\" > /mnt/user/dev/X.log 2>&1; echo DONE >> /mnt/user/dev/X.log"'`. Watch the log with the Monitor tool.
- **Sync code after editing:**
  - `fly/` only: `rsync -az --chown=root:root --exclude vendor/ --exclude __pycache__/ fly/ tower:/mnt/user/dev/ThisFlyPlaysDrums/fly/`
  - **whole repo** (needed once now, for `grooves/` and `human/`): `rsync -az --chown=root:root --exclude data/ --exclude fly/vendor/ --exclude runs/ --exclude __pycache__/ ./ tower:/mnt/user/dev/ThisFlyPlaysDrums/`
- **On the server only:**
  - fly-brain is a full clone with `data/` at `fly/vendor/fly-brain`.
  - MaleCNS is in `data/malecns/`.
  - Results are in `runs/bench/` and `runs/probe/`.
  - `setup_env.sh`, `dl_malecns.sh` and one-off analysis scripts are in `/mnt/user/dev/`.
- Claude Code isn't installed on the server; everything is driven over SSH from the dev machine.

## Code inventory

| File | State |
|---|---|
| `fly/drums.py` | **12 voices** in fixed order: kick, hat_pedal, snare, xstick, hat_closed, hat_open, tom1, tom2, tom3, crash, ride, ride_bell. Each has `notes` (edges and rims folded as in Sam's `SAME_DRUM`), `out_note`, and `limbs` (sounding leg first). `LEGS = (front_left, front_right, hind_left, hind_right)`. |
| `fly/encoder.py` | `encode(midi, lookahead_ms=150, dt_ms=1, burst_ms=30, rate_min_hz=50, rate_max_hz=200)` → `Encoded(rates[T, 12] Hz in VOICES order, voices, dt_ms, offset_ms)`. Score time t (s) is due at sim step `(t*1000 + offset_ms)/dt_ms`, and its burst starts `lookahead_ms` earlier; `offset_ms = lookahead_ms`. Rule-based and never sees a target. 3 tests. |
| `fly/brain.py` | `Brain(conn, cue_groups{name: idx}, plastic_mask=None, batch, device)`, `.init_state()`, `.step(state, voice_rates[B,V], current=None, generator=None)` → `(conductance, delay_buffer, spikes, v, refrac)`. Imports fly-brain's AlphaLIF/Poisson/MODEL_PARAMS from `fly/vendor/fly-brain/code/run_pytorch.py`. `_FrozenMatmul`: fixed CSR, gradients reach spikes only. `PlasticEdges`: trainable `nn.Parameter` on the masked edges, removed from the frozen matrix. `current` [B,N] is in mV, added to the Poisson drive (for α·I\*). **Every cue neuron gets fly-brain's no-refractory treatment.** Verified on CUDA. 3 tests. |
| `fly/connectome.py` | `Connectome(name, neurons, pre, post, weight)` (row = matrix index), `.to_torch(device, transpose)` → CSR `W[post, pre]`, `.shuffled(seed)` (permutes post endpoints; keeps degrees and presynaptic sign). `fetch_malecns()` downloads to `data/malecns/`, resumable. `load_malecns()` → 166,700 neurons (bodies with a superclass), 25,582,938 edges; sign from `consensus_nt` (gaba/glutamate/histamine = −1); soma x/y/z; columns in `NEURON_COLUMNS` including `rootSide`. `load_flywire(annotate=False)`; `annotate=True` merges the Schlegel et al. cell types. 1 test. |
| `fly/bench.py` | `python -m fly.bench sugar` (our 1 ms Brain vs fly-brain at 0.1 ms, FlyWire) and `python -m fly.bench speed [--min-synapses N]` (MaleCNS VRAM and speed). Writes `runs/bench/`. |
| `fly/probe.py` | Phase 1 probe: graph pass (hops) + GPU simulation pass + charts. Flags: `--connectome`, `--only`, `--no-graph`, `--graph-only`, `--fine-jo` (also writes `cues.json` via `pick_cues`), `--weight-scale`, `--drop-kc-kc`, `--drop-kc-in`, `--tag`, `--rate`, `--stim-ms`, `--trials`. `groups(conn, fine_jo)` gives candidates, readouts (leg MNs: ProLN/MetaLN × side, side = somaSide else rootSide) and KC/MBON. 3 tests. |
| `fly/cues.json` | **D2 result:** for each of the 12 drums, the JO group name, `bodyIds`, `limbs`, and per-leg latency/peak. Use the bodyIds, not the group names. |
| `fly/environment.yml` | As above. |
| `fly/STATUS.md` | Partner-facing status **as of 04:00, now stale** (it predates Phase 0/1). Published at https://claude.ai/artifact/Tsq1wJP5gZdwH7VS8ZLnFG from a scratchpad file that no longer exists. Update only if the user asks. |
| **Stubs** (docstring only) | `fly/decoder.py`, `fly/body.py`, `fly/train.py`, `fly/evaluate.py` |
| **Not created** | `fly/loop.py`, `fly/strokes.py`, `viewer/` |

## Decisions (all user-approved)

| ID | Decision |
|---|---|
| D1 | Simulator: fly-brain's LIF, imported from `fly/vendor/`, at dt = 1 ms, with gradients only on the trainable subset. **Verified:** r = 0.998 against fly-brain's own 0.1 ms run. |
| Data | MaleCNS v1.0 by **bulk download** (neuPrint only hosts v0.9). FlyWire v783 is the fallback. |
| D6 | Body: **flybody**, Python 3.10. |
| Kit | **12 TD-07 drums, played like a person** on a kit placed within the fly's reach. **Front left and right legs hold the sticks. Hind right = kick pedal (36). Hind left = hi-hat pedal (44)**, which also opens and closes the hat. Middle legs don't play. |
| Sticking | Teacher strokes are right-handed. Right stick: hat, ride, bell, crash. Left stick: snare, cross-stick. Toms and fast repeats alternate R-L. |
| Hi-hat | Like a real kit. The hind-left leg holds the pedal down (closed) and lifts it for 46. A stick hit on the hat pad sounds **42 or 46 depending on the pedal at contact**. A pedal press alone sends 44. |
| **D8** | **Runaway fix:** stock weights (wScale 0.275, no rescale) **plus every edge onto Kenyon cells removed** (the probe's `--drop-kc-in`). The shuffled control gets the same cut. |
| **D2** | Cue groups = one JO fine subtype per drum, from `fly/cues.json` (below). |
| **D3** | Trainable = **synapses from the 12 cue groups onto descending neurons** (`superclass == "descending_neuron"`). About 7,469 edges for all JO; fewer for the 12 chosen groups. |
| **D5** | Motor neurons → joints by **muscle annotation** (the `type` column). Joint target = rest + gain·(flexor-type rate − extensor-type rate). Mapping below. |
| D4 | Hits out: `runs/<id>/hits.mid` (channel 10, the voice's `out_note`, velocity from contact speed), **`hits.csv` (`t_ms,note,velocity`) for Sam's `score.py`**, and `hits.json` (`{t_s, note, voice, velocity, contact_speed}`), **all in score time**. |
| α | Teacher forcing enters the leg MNs as current α·I\* (I\* from the decoder's pseudo-inverse), with α annealed from 1 to 0. The loss is always on the fly's decoded joint angles (MuJoCo isn't differentiable). |
| Teacher | Any training `.mid` (kit takes, sheet music, grids). **Never held-out grooves.** Strokes come from per-drum IK templates. |
| Encoder | Fly side (`fly/encoder.py`). Rule-based and never sees the target (the demo rule). |
| Scoring | Sam's `human/score.py`. `evaluate.py` only runs the α = 0 held-out, shuffled and untrained runs and writes their hits files. |
| Viewer | Fly side, `viewer/`: Three.js in Chrome replays a recorded run and sends hits to the TD-07 through Web MIDI. `play_midi.py hits.mid` is the backup. Timing is undecided; Sun 02:00–06:00 was recommended. |
| Gates left | D7 at Sat 22:00: switch to the scikit-learn readout baseline if the loss hasn't dropped. |

**Cue groups (`fly/cues.json`):**

| Drum | Group (n) | Drum | Group (n) |
|---|---|---|---|
| kick | JO-A1_L (4) | tom1 | JO-ED2_b_R (10) |
| hat_pedal | JO-EV1_R (19) | tom2 | JO-EV6_L (14) |
| snare | JO-FV_L (36) | tom3 | JO-EV1_L (31) |
| xstick | JO-EV5_L (13) | crash | JO-CM_L (16) |
| hat_closed | JO-ED2_a_L (23) | ride | JO-ED2_b_L (11) |
| hat_open | JO-EV3_L (22) | ride_bell | JO-ED1_L (13) |

Picked by `python -m fly.probe --fine-jo --drop-kc-in --no-graph --tag fine_nokcin`. Score = the weakest peak over the drum's legs, counting only legs reached ≤ 60 ms. Greedy, most constrained drum first. **Every group drives all four legs** (20–65 ms, 5–24 Hz), so leg selectivity has to be learned.

## Measured facts

**Speed and memory:**
- MaleCNS load: 44 s, 6.2 GB RAM. Brain build: 18 s. VRAM: 0.78 GB (1.07 GB with a 150-step gradient window).
- **Brain: 4.9 s wall per simulated second** at batch 1. Batch 8 ≈ 1.3 s per trial-second.
- 150-step backprop window: 2.1 s.
- **MuJoCo flybody at its 1e-4 s timestep: 22 s wall per simulated second.** It's single-threaded, and these Xeons are old. This is now the slowest part of the loop.

**Network (stock gain, KC inputs cut):**
- No spontaneous activity.
- JO → descending neurons (DNp10, DNb05, DNg15, pIP1, DNp18, DNg50, DNg35, ...) → leg MNs: **2 hops**.
- Cue → MN latency 20–55 ms. Lingering MN activity 1–3 Hz for ~250 ms after a cue.
- The MBONs keep 8–15 Hz of lingering activity even with KC inputs cut. That's harmless for now.

**Right-side JO:** JO-A/B/ED1/ED2_c/DP barely connect in v1.0 (for example 880 vs 16,869 output synapses, JO-A R vs L).

**Leg MNs:**
- Counts: ProLN 41 L / 40 R, MetaLN 62 L / 60 R.
- Types, front: Acc. ti flexor 10, Ti flexor 5, Ta depressor 5, Fe reductor 4, ltm 4, Acc. tr flexor 3, Ti extensor 2, ltm2-femur 2, Ta levator 3 (R).
- Types, hind: Ti flexor 8–9, Acc. tr flexor 4–8, Acc. ti flexor 8, Tr flexor 5, Sternal posterior rotator 4, ltm 3–4, Sternotrochanter 3, Pleural remotor/abductor 2, plus unnamed (`?`) and `MNhl59`.
- **Mostly extensor-type MNs respond at first** (Fe reductor, Ti extensor, Ta levator; hind: sternal rotators, Tr flexor). 59 JO-driven DNs also reach flexor-type MNs.

**flybody `fruitfly.xml`**
(at `…/site-packages/flybody/fruitfly/assets/fruitfly.xml` in the env):
- **Units:** cm, gravity −981. Timestep 1e-4 s.
- **Root:** the joint `free` is a **free joint**. Pin the thorax to the world for drumming.
- **Leg actuators are position servos:** `ctrl` = target angle in rad, kp 0.8 (coxa/femur) or 0.4 (tibia/tarsus), ctrlrange = joint range.
- **Front-left joint ranges:** coxa_abduct [−1, 0.7], coxa_twist [−0.8, 0.8], coxa [−0.2, 1.7], femur_twist [−1, 1], femur [−0.15, 2], tibia [−1.35, 1.3], tarsus [−0.7, 1.2]. `tarsus2_*` is **tendon-driven** [−0.9, 0.9].
- **32 leg actuators on the four playing legs**, in model order:
  - `{coxa_abduct, coxa_twist, coxa, femur_twist, femur, tibia, tarsus, tarsus2}_{T1_left, T1_right, T3_left, T3_right}`
  - plus `adhere_claw_*` (keep at 0) and wing/abdomen/head actuators (hold still).
- **Leg bodies:** `coxa_T1_left, femur_T1_left, tibia_T1_left, tarsus_T1_left, tarsus2–4_T1_left, claw_T1_left`, and the same for the other legs.

## Phase 2 spec: untrained loop, end to end → Milestone 1

Goal: `python -m fly.loop --groove <x>.mid --alpha 0 --out runs/<id>` runs encoder → brain → decoder → body. It writes `hits.mid`, `hits.csv` and `hits.json` in score time. The untrained fly's legs move and produce some (wrong) contacts.

1. **Wiring** (`fly/wiring.py`, new): one function builds everything the brain needs, so `loop`, `train` and `evaluate` can't drift apart.
   - `load_malecns()` → **remove every edge whose post is a Kenyon cell** (`type` starts with "KC") (D8).
   - Optional `.shuffled(seed)` for the control. Apply the KC cut the same way, and recompute the masks on the shuffled edges.
   - Cue groups: map `cues.json` bodyIds → row indices, in `drums.VOICES` order.
   - Plastic mask (D3): `pre ∈ cue neurons & post ∈ descending neurons`.
   - Leg MN indices per leg with their `type` (as in `probe.groups`).
2. **Encoder:** already 12-voice. For Phase 2, set `lookahead_ms` to a placeholder of about 40 (≈ latency). The real value comes after Phase 3's stroke lead.
   - The probe drove 200 Hz for 50 ms. Consider `burst_ms=50` so cues match what was measured.
3. **Decoder** (`fly/decoder.py`, D5): fixed and differentiable (torch), and it never sees a target.
   - **Rates:** exponential filter on each MN's spikes (τ ≈ 20 ms).
   - **Per leg and joint:** target = rest + gain·(mean rate of that joint's + types − mean rate of its − types), clipped to ctrlrange. Joints with no MNs hold rest.
   - **Proposed mapping** ("+" = flexion or the named direction):

     | Joint | + types | − types |
     |---|---|---|
     | tibia | Ti flexor, Acc. ti flexor | Ti extensor |
     | femur | Tr flexor, Acc. tr flexor, Sternotrochanter | Tr extensor |
     | femur_twist | Fe reductor | — |
     | coxa_twist | Sternal anterior rotator | Sternal posterior rotator |
     | coxa | Pleural promotor | Pleural remotor, Pleural remotor/abductor |
     | coxa_abduct | Sternal adductor | abductor types |
     | tarsus | Ta depressor | Ta levator |
     | tarsus2 | ltm, ltm1-tibia, ltm2-femur | — |

     Unmapped types (`MNhl59`, `?`, and so on) are ignored. List them when you build it.
   - **Verify each joint's sign against flybody's axis** before trusting the table: move the joint +0.3 rad and check that the stick or tarsus tip moves the expected way. Keep a per-joint ±1 table.
   - **Gains:** about 20 Hz of difference → about half the joint's range. Set by hand once.
   - **Also expose the decoder's pseudo-inverse**, which maps joint targets back to MN currents I\*. Phase 4 needs it for α.
4. **Body** (`fly/body.py`):
   - Load `fruitfly.xml` with `dm_control.mjcf` and **pin the thorax**: delete the free joint so the root body is fixed to the world.
   - **Sticks:** capsule geoms on each front leg's tibia or tarsus. Size them to the fly (it's about 0.25 cm long).
   - **Kit:** 10 stick pads plus 2 pedals (hind right = kick, hind left = hat).
     - Place them **inside each leg's reachable workspace**: sample random joint configurations, collect the stick-tip and tarsus positions, and put pads at reachable points spread apart.
     - Pads for right-stick drums go on the right-stick side, snare and cross-stick on the left, toms reachable by both.
   - **Control:** `Body.step(targets[32])` sets `ctrl` for the 32 leg actuators (others hold 0) and runs 1 ms of physics (10 substeps at 1e-4).
   - **Contacts:**
     - The first contact of a stick or leg with a pad records a hit, followed by a refractory window (~30 ms).
     - `contact_speed` = relative normal speed at contact. `velocity = clip(round(k·speed), 1, 127)`.
     - Hat pad note = 42 if the hind-left pedal is pressed, else 46. Pedal presses send 36 or 44.
   - **Speed:** 22 s per simulated second at 1e-4. Test whether 2e-4 s (5 substeps) is stable with the thorax pinned, and **ask the user before changing the timestep**.
5. **Loop** (`fly/loop.py`):
   - CLI: `--groove`, `--alpha 0`, `--out`, `--seconds` (truncate), `--shuffled SEED`, `--device`.
   - **Each 1 ms step:** encoder rates[t] → `brain.step` → decoder(spikes) → `body.step` → contacts.
   - **Score time:** `t_score_ms = t_sim_ms − offset_ms`. Drop hits with t < 0.
   - **Writes:**
     - `hits.mid`: mido, channel 9, tempo meta, `note_on` + `note_off` 50 ms later, like Sam's files.
     - `hits.csv`: header `t_ms,note,velocity`.
     - `hits.json`: `[{t_s, note, voice, velocity, contact_speed}]`.
     - `meta.json`: groove, α, seed, offset, and the wiring summary.
   - Viewer files (spikes, poses) can wait for Phase 4b, but keep the loop easy to extend.
6. **Tests** (in `fly/tests/`):
   - decoder shapes and mapping signs
   - body: a scripted stick driven into a pad gives exactly one hit with the right note; the hat note follows the pedal
   - loop writes the files in score time (use a stub brain so the test is fast)
7. **Milestone 1:**
   - Sync the whole repo to the server first (see Commands).
   - Run a short training groove, for example `grooves/train/HH-BD-SD-HO-HP_94bpm_20260926_031104.mid` with `--seconds 5` (≈ 2–3 minutes of wall-clock at current speeds).
   - Check that the legs move, some contacts happen, the files exist, and `python human/score.py <groove> runs/<id>/hits.csv` runs on the server (the env has mido).
   - The user plays `hits.mid` on the kit from the laptop: `python human/play_midi.py hits.mid`.
   - **Then stop and report.** Phase 3 (strokes via IK, the ceiling check) is next.

## Open questions (ask in one batch before starting Phase 2)

1. **Kick cue group.** Kick got JO-A1_L, only 4 neurons (11.7 Hz on its leg). Option: swap so kick = JO-EV1_L (24 Hz on hind right) and tom3 = JO-EV6_R (5 Hz on the front legs). The user hasn't answered.
2. **MuJoCo speed.** May the body use a 2e-4 s timestep (5 substeps per ms) if it's stable? That roughly halves the 22 s per simulated second.
3. **Commit.** The Phase 1 changes are uncommitted. The user commits when they choose; just remind them.

## Tried, failed or ruled out

- **FlyGym 2:** needs Python ≥ 3.12.
- **`male-cns:v1.0` on neuPrint:** doesn't exist (only v0.9).
- **The `pytorch` conda channel:** stale at 2.5, and clashes with newer MKL. Replaced by pip cu126 wheels.
- **conda on `/mnt/user` (shfs):** crawled. Fixed by mounting `/mnt/pool/dev`.
- **tailscale.com install script:** that host times out from this network. The apt repo at pkgs.tailscale.com works.
- **Runaway fixes that didn't work well:**
  - Dropping KC→KC edges: only halves the runaway.
  - Weight scale 0.53: no runaway, but weak legs.
  - Scales 0.6–0.8: KC activity is bistable, either off or runaway.
- **pandas `groupby(...).groups`** returns NaN-key groups. This caused a bad first probe run (a "JO-nan" group of ~80k neurons). Use explicit masks.
- **Account skills sync** (`avoid-ai-writing-tells`, `docx`, and others): still pending, and can't be forced.

## Files that matter

| Path | Notes |
|---|---|
| `CLAUDE.md` | Local and gitignored. Scope, architecture, data formats. **Has the conflicts listed in Start here.** |
| `~/.claude/plans/i-own-the-fly-splendid-creek.md` | Approved plan: phases, timings, stops, risks |
| `~/.claude/projects/-home-daniel-ThisFlyPlaysDrums/memory/` | Feedback memories: ask every question in one batch, and stop at the end of each phase |
| `fly/` | All fly-side code (see Code inventory) |
| `data/` | Gitignored. `data/malecns/` (on both machines; the dev copy is the full 1 GB download), `data/flywire/` |
| `runs/` | Gitignored. On the server: `runs/bench/`, `runs/probe/malecns*/` (`report.json`, `hops.png`, `response.png`, `timecourse.png`). Locally: `runs/probe/malecns_fine_nokcin/` |
| `human/drum_map.py` | Sam's note map (read-only). Keep `fly/drums.py` in step with it. |
| `grooves/train/*.mid` | 21 training takes. `grooves/heldout/` is still empty. |

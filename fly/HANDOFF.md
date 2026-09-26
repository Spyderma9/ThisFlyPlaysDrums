# Fly side handoff

Written Sat Sept 26, 03:00 EDT. Deadline Sun Sept 27, 11:00 EDT, with a code freeze at 08:00.

## Start here (for the new session)

You're continuing work on the fly side of Fly Drums. The previous session's context is gone, and this file replaces it.

1. **Read three files first:**
   - this file, in full
   - `/home/daniel/ThisFlyPlaysDrums/CLAUDE.md` (local, gitignored: scope, rules, architecture, data formats)
   - the approved plan at `/home/daniel/.claude/plans/i-own-the-fly-splendid-creek.md` (phases, timings, stops, risks)
2. **This file wins on conflicts.** It's the newest record. Known conflict: `CLAUDE.md` still says "MaleCNS v1.0 via neuPrint" and "flybody or NeuroMechFly". Both are superseded: MaleCNS comes by **bulk download**, and the body is **flybody**. Don't edit `CLAUDE.md` yourself. The user moves permanent decisions into it; tell them about the conflict at the start of the session.
3. **Rules that apply to every step:**
   - Never `git commit`, `git push` or amend.
   - Edit only `fly/`, `viewer/` and `.env.example`.
   - Ask every open question in one batch before starting, and check `CLAUDE.md` and this file first before asking.
   - Stop at each gate below and ask; don't choose for the user.
4. **Then do the next step:** finish the MaleCNS v1.0 bulk switch (see "In progress"). The first command is under "Next concrete step" at the end of this file. When the switch is done and tested, **stop and tell the user before continuing.**

## Updates from session 2 (Sat Sept 26, 03:00–04:00). These win over anything below.

**Done:**
- **MaleCNS bulk switch:** code done, 7/7 tests pass.
  - `connectome.py`: `fetch_malecns()` downloads three files, resumable through `.part` files. The body-stats file isn't needed; soma positions are in the annotations.
  - `load_malecns()`: a neuron is a body with a `superclass` (166,700 of 211,577). Sign comes from `consensus_nt`, with unclear or missing counted as excitatory. Soma is `somaLocation`, else `tosomaLocation`.
  - Test: `fly/tests/test_connectome.py`.
  - `neuprint-python` and `python-dotenv` are removed from `environment.yml`. `.env.example` now says the token is optional.
- **Real data checked:** leg MNs by `exitNerve`: ProLN 81, MesoLN 116, MetaLN 122.
- **Weights file:** downloaded to `data/malecns/` on the dev machine. About **152M edges** (2,318 batches × 65,536 rows), sorted by weight descending.
  - Too big for the dev machine's RAM (1.2 GB free).
  - The GPU CSR may not fit in 8 GB, so a minimum-synapse threshold may be needed. Measure in Phase 0.
- **Tailscale:** installed on the dev machine from the apt repo (tailscale.com itself is unreachable from this network).
  - Tailnet `spyderma9.github`.
  - `tower` is **not** in it yet, so the user needs to join the Unraid box. The only peer is `N1`, shared from another tailnet, with its SSH port closed.
- **`fly/STATUS.md`:** partner-facing status. Also published at https://claude.ai/artifact/Tsq1wJP5gZdwH7VS8ZLnFG from the scratchpad file `fly-status.html`. Republish it when STATUS.md changes.

**New decisions (user-approved):**

| Decision | Choice |
|---|---|
| Encoder | Fly side (`fly/encoder.py`). Sam (the partner) doesn't build one. |
| Scoring | **Sam builds it** on the laptop (hits, misses, extras, timing, loudness per drum, plus a human-take baseline). `evaluate.py` only runs held-out and shuffled runs and writes hits files. |
| Hits out | `hits.mid` + `hits.json` + **`hits.csv` (`t_ms, note, velocity`)** for Sam |
| Drum scope | **All 12 TD-07 drums** (36, 44, 38, 37, 42, 46, 48, 45, 43, 49, 51, 53), played like a person on a kit modified so the fly can reach. Supersedes the 3-voice `drums.py`. |
| Limbs | Front left and right legs hold sticks (the hands). **Back right leg = kick pedal (36).** **Back left leg = hi-hat pedal (44)** and hi-hat open/closed. Middle legs don't play. |
| Sticking (teacher) | Right-handed default. Right stick: hi-hat, ride, bell, crash. Left stick: snare, cross-stick. Toms and fast repeats alternate R-L. |
| Hi-hat | Like a real kit. The back-left leg holds the pedal down (closed) and lifts it for 46. A stick hit on the hat sounds 42 or 46 depending on the pedal at contact. A pedal press alone sends 44. |

**To do after Phase 0** (not started): rewrite `drums.py` for 12 drums with allowed limbs per drum. The encoder gets 12 cue groups. The body gets 10 stick pads, a kick pedal and a hi-hat pedal. Strokes need per-stick IK templates for each pad. D2 and D5 now cover the forelegs and both hind legs.

**Partner facts:**
- Sam's `human/prep_takes.py` writes cleaned takes to `grooves/train/` (resolves open question 1).
- About 5 minutes of takes are said to be ready but **not pushed yet** (`origin/midi` at `db0c48e`).
- Sam's note on `fly/environment.yml` being on `midi` doesn't matter: it's only in the shared base commit, so no conflict.

### Phase 0: done (Sat ~05:20 EDT). Stopped before Phase 1; waiting for the user's go-ahead.

**Server access:**
- The box is `t5600.tail3495cd.ts.net`, shared into our tailnet from juan.borgesjr's account. `~/.ssh/config` aliases it as `tower` (user root, key `~/.ssh/id_ed25519`, added in Unraid's Users → root).
- Unraid 7.3.2, no array disks. Everything lives on the ZFS pool `pool`: `/mnt/user/dev` is the same data as `/mnt/pool/dev`.

**Container:**
- `flydrums`, image `nvidia/cuda:12.6.3-base-ubuntu22.04`, `--runtime=nvidia --gpus all`, env `MUJOCO_GL=egl`, `sleep infinity`.
- It mounts **`/mnt/pool/dev` at `/mnt/user/dev`**. It bypasses shfs deliberately: shfs made conda crawl.
- apt packages (git, curl, libgl1, libegl1, libosmesa6, libglib2.0-0) are in the container layer. Reinstall them if the container is recreated.

**Env and helpers:**
- Miniforge is at `/mnt/user/dev/miniforge3`, env `flydrums`, built from `fly/environment.yml`, which now uses **conda-forge + pip torch cu126**. The `pytorch` channel is gone. `python-dotenv` and `neuprint-python` are removed.
- Env var `LD_LIBRARY_PATH` = env lib + `/usr/local/nvidia/lib{,64}`, set with `conda env config vars`. Without it, torch loads the system libstdc++ and pyarrow fails with GLIBCXX_3.4.31. Conda prints a harmless "overwriting variable" warning on every activation.
- `/mnt/user/dev/fx "<cmd>"` runs a command in the container with the env active, from the repo root.
- `setup_env.sh` and `dl_malecns.sh` are in `/mnt/user/dev`.
- The repo got to the server by **rsync from the dev machine** (not git clone), with the uncommitted changes. After an edit, resync with: `rsync -az --chown=root:root --exclude vendor/ --exclude __pycache__/ fly/ tower:/mnt/user/dev/ThisFlyPlaysDrums/fly/`.
- fly-brain is a full clone with `data/` in `fly/vendor/fly-brain`.
- MaleCNS is in `data/malecns/` on the server, downloaded directly.

**Not done from Step 0:** Claude Code isn't installed on the host, and the plan, skills and `/root/.claude` weren't copied. Everything was driven over SSH from the dev machine instead.

**Measured:**
- **Imports on AVX-only Xeons:** torch 2.14.0+cu126 (CPU capability DEFAULT), mujoco 3.14.0, dm_control 1.0.47, flybody 0.1.0, numpy 1.26.4. No illegal instructions.
  - CUDA and sparse CSR on the GPU work.
  - flybody `fruitfly.xml`: nq 109, nu 78, timestep 1e-4. It steps.
- **pytest on the server:** 7/7.
- **fly-brain benchmark** (`main.py --pytorch --t_run 1 --n_run 1`, FlyWire, 0.1 ms): 44.6 s per simulated second, 0.38 GB VRAM, 391 active, 17,311 spikes.
- **`python -m fly.bench sugar`** (1 s, 8 trials): our Brain at 1 ms vs fly-brain's TorchModel at 0.1 ms.
  - Pearson r = 0.998 over active neurons. Median rate ratio 1.02 over 314 neurons ≥ 5 Hz. Top responders are 3–7% lower in ours. **D1 holds.**
  - Wall-clock per simulated second: 103 s (fly-brain) vs 10.3 s (ours), batch 8.
- **`python -m fly.bench speed`** (MaleCNS):
  - 166,700 neurons, **25,582,938 edges** (152M raw rows, mostly fragments). No threshold needed. 61,210 KC→MBON edges.
  - Load 44 s, 6.2 GB peak RAM. Build 18 s. VRAM 0.78 GB.
  - **4.93 s per simulated second** at batch 1.
  - 150-step gradient window: 2.08 s, 1.07 GB. Gradients reach KC→MBON.
- **Annotations:**
  - Leg MN types name their muscle (Ti flexor 37, Acc. ti flexor 47, Ti extensor 12, Tr flexor 22, ...). By side: ProLN 41 L / 40 R, MesoLN 58/58, MetaLN 62 L / 60 R.
  - JO subtypes: 672 neurons, types JO-A/B/CA/CL/CM/DA/DP/ED/EV/FD/FV/mz.
  - 1,314 DNs, 4,064 KCs, 97 MBONs.
- Results are in `runs/bench/*.json` on the server.

## State in one paragraph

**Git:** branch `server`, last commit `6a4d687` "foundations for serverside", pushed, working tree clean.

**Done:** code for Phase 0 exists and is tested on the dev machine's CPU with toy data.

**Never run on the server.** The server isn't reachable from the dev machine yet.

**In progress:** switching the connectome source to **MaleCNS v1.0 via bulk download**. None of that code is written yet.

**Stop point:** the user asked to be told once the switch is complete, before any work continues past it.

## People and ownership

| Owner | Area | Branch |
|---|---|---|
| User | `fly/` (server side) and `viewer/` (not created yet) | `server` |
| Partner | `human/`, `grooves/`, the root `.gitignore` | `midi` (at `c628f47`) |

- Never edit `human/`, `grooves/` or the root `.gitignore`. Put fly-only ignores in `fly/.gitignore`.
- `.env.example` at the repo root is ours (the user asked for it).
- **Checked at `c628f47`:** no files overlap between the two branches, so they merge cleanly.
- The partner "has begun training on his side". Nothing training-related is on `midi`. **Unresolved:** whether that means recording kit takes or writing training code. Kit takes save to `takes/`, which is gitignored and so never reaches the server through git.
- Never `git commit`, `git push` or amend. The user commits.

## Machines

**Dev machine:** Debian, hostname `debian`, at `/home/daniel/ThisFlyPlaysDrums`.
- Python 3.13, with no conda, no GPU and no tmux.
- No `claude` CLI on PATH (Claude Code runs as the VS Code extension).
- **Tailscale is not installed,** so `tower` doesn't resolve.

**Server:** Unraid host `root@tower`, over Tailscale MagicDNS. **Unreachable from the dev machine so far.**
- Hardware: Dell T5600, 2× Xeon E5-2670 (AVX, no AVX2), 64 GB RAM, RTX 3060 Ti 8 GB.
- The repo goes to `/mnt/user/dev/ThisFlyPlaysDrums`. It isn't cloned yet.
- Claude Code will run on the Unraid host itself.
- The CUDA container isn't built, the conda env isn't created, and nothing is benchmarked.
- `/root` lives in RAM on Unraid, so anything persistent (Claude config, conda, repo) must live on `/mnt/user`.
- The GPU is shared with other containers. **Before every long run, stop and ask the user to free it.**

## Built and working

| File | What it does | Tested? |
|---|---|---|
| `fly/drums.py` | Three voices keyed by TD-07 note, each with a note written back to `hits.mid` and a leg. hihat {42, 22, 46, 26} → 42, right foreleg. snare {38, 40, 37} → 38, left foreleg. kick {36} → 36, kick leg. Crash, toms and ride are dropped. Numbers checked against the partner's `human/drum_map.py` on `midi`. | imports |
| `fly/encoder.py` | `encode(midi_path, lookahead_ms=150, dt_ms=1, burst_ms=30, rate_min_hz=50, rate_max_hz=200)` → `Encoded(rates[T, 3] Hz, voices, dt_ms, offset_ms)`. Score time t is due at sim step `(t*1000 + offset_ms)/dt_ms`, and its cue burst starts `lookahead_ms` earlier. `offset_ms = lookahead_ms`. Velocity scales the rate linearly. Overlapping bursts keep the higher rate. | 3 tests pass |
| `fly/brain.py` | `Brain(conn, cue_groups, plastic_mask=None, batch=1, device="cuda")`, one 1 ms step via `step(state, voice_rates[B, V], current=None, generator=None)`. Imports fly-brain's `AlphaLIF`, `PoissonSpikeGenerator` and `MODEL_PARAMS` from `fly/vendor/fly-brain/code/run_pytorch.py` (imported, never copied). The frozen weights use our own `_FrozenMatmul`, a custom autograd function with fixed CSR `W` and `W.T`, so gradients reach the spikes but never the weights. The trainable edges are a `PlasticEdges` `nn.Parameter` over the edges in `plastic_mask`, and they are removed from the frozen matrix. `current` is added to the Poisson drive in mV (for α·I\*). | 2 tests pass |
| `fly/connectome.py` | `Connectome` dataclass (neurons DataFrame, where row = matrix index; `pre`, `post`, signed `weight`), `.to_torch(device, transpose)` giving CSR `W[post, pre]`, and `.shuffled(seed)` which permutes post endpoints (keeps every in/out degree and the presynaptic sign). The sign rule: gaba, glutamate and histamine are −1, everything else +1. `load_flywire()` reads fly-brain's `data/` files. `fetch_malecns()` / `load_malecns()` are **the neuPrint version, now obsolete** (see Known bugs). | shuffle: 1 test passes |
| `fly/environment.yml` | Conda env `flydrums`, Python 3.10. Adds matplotlib, python-dotenv, pytest, and via pip dm_control and flybody (`git+https://github.com/TuragaLab/flybody.git`). | not built |
| `fly/tests/` | `test_encoder.py` (3 tests), `test_brain.py` (3 tests, skipped if `fly/vendor/fly-brain/code` is missing) | 6/6 pass |
| `.env` (gitignored) | `NEUPRINT_TOKEN=<64-char token>`, checked working against neuPrint | — |
| `.env.example` | Template, committed | — |

**Stubs** (docstring only): `fly/decoder.py`, `fly/body.py`, `fly/train.py`, `fly/evaluate.py`.

**Not created yet:** `fly/probe.py`, `fly/strokes.py`, `fly/loop.py`, `viewer/`.

**Local only:** fly-brain is sparse-cloned into `fly/vendor/fly-brain` (gitignored, `code/` only). Its `data/` folder is not fetched, so `load_flywire()` can't run on the dev machine.

## In progress: switching to MaleCNS v1.0 bulk download (0% of code done)

Files at `https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/`:

| File | Size | Needed? |
|---|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | 14.5 MB | yes |
| `body-neurotransmitters-male-cns-v1.0.feather` | 43.3 MB | yes (for the sign) |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | 1,051 MB | yes |
| `body-stats-male-cns-v1.0-minconf-0.5.feather` | 778 MB | only if soma positions aren't in the annotations |

**Column names are unverified.** From third-party code (github.com/seeton/fly):
- weights: `body_pre, body_post, weight`
- annotations: `bodyId, type, instance, superclass, somaSide, ...`
- neurotransmitters: column names unknown

**Next step:** read the real schemas. Download the two small files and read the headers of the two big ones with an HTTP range request (Arrow IPC puts the schema right after the 8-byte `ARROW1` magic). Then:
1. Rewrite `connectome.py`:
   - `fetch_malecns()` downloads the four files to `data/malecns/`, resumable with `.part` files.
   - `load_malecns()` joins the annotations and neurotransmitter tables, keeps only real neurons (decide the filter from the columns), maps bodyIds to indices, and signs the weights.
   - Keep `load_flywire()` and `shuffled()` as they are.
2. Remove `_client()`, the neuPrint imports, and `DATASET = "male-cns:v1.0"`.
3. Decide whether `neuprint-python` stays in `environment.yml`.
4. Update the `CLAUDE.md` line that says "MaleCNS v1.0 via neuPrint" to say bulk download. The user asked to be told first; see the decisions list at the end.
5. Add a loader test built on a tiny synthetic feather file.
6. **Then stop and tell the user the switch is done.**

## Decisions approved by the user (and why)

| Decision | Choice | Why |
|---|---|---|
| D1 simulator | fly-brain's LIF, imported from `fly/vendor/`, stepped at dt = 1 ms, with gradients only on the trainable subset. **Reopen and ask** if it can't do 1 ms steps or gradients. The fallback is our own LIF with the same parameters. | CLAUDE.md names fly-brain. It's GPL, so import it, never copy it. |
| D6 body | flybody, Python 3.10 | FlyGym 2 needs Python ≥ 3.12, and the partner is on 3.10 |
| Connectome | MaleCNS **v1.0 via bulk download** (changed from "via neuPrint") | neuPrint only hosts `male-cns:v0.9`. Bulk is one download, similar total time, and doesn't depend on the API during the event. |
| D4 hits out | `hits.mid`: channel 10, the voice's note written back to `hits.mid` (from `drums.py`), velocity from contact speed, same format as the input, so the partner's `play_midi.py` plays it on the TD-07. Plus `hits.json` holding `{t_s, note, voice, velocity, contact_speed}`. | The user wants every simulated stick–pad contact heard on the real kit, with the fly's own velocity. |
| Viewer | Owned by the user (fly side). It lives in `viewer/`: Three.js in Chrome, replaying a recorded run, sending hits to the TD-07 through Web MIDI in sync. `play_midi.py hits.mid` is the backup. | The simulation is slower than real time. Run files pass only between `fly/` and `viewer/`, so the fly side sets their format. |
| α | Teacher forcing injected into leg motor neurons as current α·I\* (I\* from the decoder's pseudo-inverse). α anneals from 1 to 0. The loss is always on the fly's decoded joint angles. | The body sends no feedback to the brain, so α only affects learning if it enters the brain. |
| Loss space | Joint angles, since MuJoCo isn't differentiable | Stated in the user's brief |
| Teacher source | Any training `.mid` (kit takes, sheet music, grids). **Never held-out grooves.** | Approved as recommended |
| Teacher trajectories | Per-drum inverse-kinematics stroke templates timed to the hits | Kit takes carry only times and velocities |
| Drum folding | Pad variants fold onto three voices; crash, toms and ride are dropped | The fly has three pads |
| Server | Claude Code on the Unraid host, `root@tower`, repo at `/mnt/user/dev/ThisFlyPlaysDrums` | User's choice |
| GPU sharing | The user pauses other containers themselves. Stop and ask before every long run. | User's choice |
| Grooves | The user asks the partner. Code uses whatever is in `grooves/train` and `grooves/heldout`. Synthetic development grooves go only in `data/dev_grooves/` (gitignored, never used for evaluation). | Don't block on the partner |
| Python | 3.10 everywhere | Matches the teammate |

**Gates still pending** (stop and ask the user):
- D2: cue neuron groups
- D3: trainable synapses
- D5: which motor neurons drive which joints
- D7 at Sat 22:00: switch to the scikit-learn fallback readout if the loss hasn't dropped

D2, D3 and D5 come after the Phase 1 probe, with its numbers. The probe checks whether cue input reaches the leg motor neurons, how fast, and whether KC→MBON is on the path. The encoder's lookahead comes from the measured latency.

## Tried, failed or ruled out

- **FlyGym 2:** ruled out, needs Python ≥ 3.12.
- **`male-cns:v1.0` on neuPrint:** doesn't exist. The public datasets are `hemibrain:v1.2.1`, `male-cns:v0.9`, `manc:v1.0`, `manc:v1.2.1`, `manc:v1.2.3`, `mushroombody`, `optic-lobe:v1.0.1` and `optic-lobe:v1.1`.
- **neuPrint token format:** not a JWT. neuPrint moved to a new auth system; 64 alphanumeric characters is expected. A fake token gets HTTP 401 with "invalid or expired token".
- **`GET https://neuprint.janelia.org/api/profile`:** timed out (HTTP 000). Irrelevant: authenticated Cypher queries work.
- **Our own LIF:** not chosen (it's the fallback for D1).
- **`python3 -m venv` on the dev machine:** fails (no ensurepip, and it needs sudo). The workaround is below.
- **SSH to `root@tower`:** "Could not resolve hostname". Tailscale isn't installed on the dev machine, and there's no ssh config entry.
- **Account skills sync:** `~/.claude/skills/synced/*/manifest.json` still has `lastUpdated: 0`, with pending `avoid-ai-writing-tells`, `docs`, `import-memory`, `morning`, `skill-creator`, `xlsx`, `pptx`, `pdf` and `docx`. It can't be forced; it may sync after a Claude Code restart.

## Measured numbers

- **neuPrint `male-cns:v0.9`:** 176,571 `:Neuron` nodes (Cypher count through the user's token, HTTP 200).
- **MaleCNS v1.0 file sizes:** see the table above.
- **fly-brain `MODEL_PARAMS`:** tauSyn 5 ms, tDelay 1.8 ms, v0 = vReset = vRest = −52 mV, vThreshold −45 mV, tauMem 20 ms, tRefrac 2.2 ms, scalePoisson 250, wScale 0.275. Its native DT is 0.1 ms.
- **At dt = 1 ms:** the synaptic delay becomes 1 step and the refractory period 2 steps. A spike's total effect on voltage is ≈ 0.25·wScale·w at both 0.1 ms and 1 ms, worked out by hand, not measured.
- **Not measured yet:**
  - cue → motor-neuron latency
  - VRAM use
  - wall-clock speed per simulated second
  - fly-brain benchmark
  - any score

## Commands

**Tests on the dev machine (CPU):**
- The scratchpad venv is session-specific, so recreate it:

  ```bash
  python3 -m venv --without-pip /tmp/fdvenv
  pip3 --python /tmp/fdvenv/bin/python install -q numpy pandas pyarrow mido pytest
  pip3 --python /tmp/fdvenv/bin/python install -q torch --index-url https://download.pytorch.org/whl/cpu
  ```
- If `fly/vendor/fly-brain` is missing:

  ```bash
  git clone --depth 1 --filter=blob:none --sparse https://github.com/eonsystemspbc/fly-brain fly/vendor/fly-brain
  git -C fly/vendor/fly-brain sparse-checkout set code
  ```
- Run the tests:

  ```bash
  cd /home/daniel/ThisFlyPlaysDrums && /tmp/fdvenv/bin/python -m pytest fly/tests -q -p no:cacheprovider
  ```

**Server (planned, not yet run).** Run from the repo root in the CUDA container:

```bash
conda env create -f fly/environment.yml && conda activate flydrums
git clone https://github.com/eonsystemspbc/fly-brain fly/vendor/fly-brain      # full clone, with data/
python -m fly.connectome                   # fetch MaleCNS into data/malecns/ (being rewritten for bulk)
python -m fly.connectome --summary         # counts by superclass and exitNerve
python -m fly.connectome --flywire         # FlyWire fallback summary
pytest fly/tests
```

**tmux:** no sessions exist and nothing is running. tmux isn't installed on the dev machine; long runs go in tmux on the server.

## Step 0 handoff to the server (not done, blocked)

1. **Blocked:** the dev machine needs a route to `root@tower`. Either the user installs Tailscale (`curl -fsSL https://tailscale.com/install.sh | sh`, then `sudo tailscale up`) or gives a LAN IP.
2. `ssh root@tower`, then clone the `server` branch into `/mnt/user/dev/ThisFlyPlaysDrums`.
3. scp to the server:
   - `CLAUDE.md` into the checkout
   - the plan into the server's Claude plans folder
   - `.env`
   - `~/.claude/skills/{skill-creator,webapp-testing,frontend-design,pdf}`

   Point `/root/.claude` at `/mnt/user` so it survives a reboot.
4. The user starts Claude Code on the Unraid host. Phase 0 continues there: build the CUDA 12.x container with `--runtime=nvidia` and `/mnt/user/dev` mounted, install Miniconda, create the env, test that `import mujoco, torch` work (no AVX2), run the fly-brain benchmark, load MaleCNS, and compare our 1 ms rates with fly-brain's 0.1 ms "sugar" run.

## Known bugs and unverified assumptions

- **`connectome.py`, `fetch_malecns()` / `load_malecns()`:** use `DATASET = "male-cns:v1.0"` through neuPrint, which doesn't exist. **Broken as committed.** They're being replaced by the bulk loader.
- **`load_flywire()`:** assumes the first CSV column is the FlyWire ID. It's untested, because the data isn't on the dev machine.
- **`brain.py`:**
  - `_FrozenMatmul` and `PlasticEdges` are tested on CPU only, not CUDA.
  - `exc_indices` is a CPU tensor indexing fly-brain's `refrac_steps`, which will be on the GPU. That indexing is untested there.
  - fly-brain's code is at `fly/vendor/fly-brain/code`; `run_pytorch.py` imports from `benchmark.py` in that same folder.
- **`encoder.py`:** `lookahead_ms=150` is a placeholder until the Phase 1 latency is measured.
- **`environment.yml`:** the `pytorch` conda channel with `pytorch-cuda=12.1` may be stale (PyTorch stopped publishing conda packages after 2.5). flybody's pip install may pull a MuJoCo build that differs from conda's.
- **`fly/vendor/`** is gitignored, so the server needs its own full clone.
- **Scratchpad files from this session** won't exist in a new session: the complete annotations feather and a partial neurotransmitters feather.

## Files that matter

| Path | Notes |
|---|---|
| `CLAUDE.md` | Local and gitignored. Holds scope, the architecture, decided data formats and skills. **Still says MaleCNS "via neuPrint".** |
| `~/.claude/plans/i-own-the-fly-splendid-creek.md` | Approved phased plan with timings, stops, risks and skills |
| `~/.claude/projects/-home-daniel-ThisFlyPlaysDrums/memory/` | Holds one feedback memory: ask every question in one batch before starting, and check CLAUDE.md or the docs before asking |
| `~/.claude/skills/{skill-creator,webapp-testing,frontend-design,pdf}` | Installed from github.com/anthropics/skills at commit 3337550 |
| `fly/` | All fly-side code |
| `data/` | Gitignored. Connectome caches (`data/malecns/`) and `data/dev_grooves/` |
| `runs/` | Gitignored. Run outputs: `runs/<id>/`, `runs/probe/`, `runs/eval/` |
| `human/drum_map.py` on `origin/midi` | The partner's note map (read-only for us). Keep `fly/drums.py` in step with it. |
| `grooves/rock_beat.*`, `grooves/funk_16ths.*` on `origin/midi` | The only grooves so far. Nothing is in `grooves/train` or `grooves/heldout`. |

## Open questions for the user

1. What does "training on his side" mean: kit takes, or training code? If takes, how do they get to the server?
2. How many training and held-out grooves will the partner write, and when? Evaluation needs at least 2 held-out grooves by Sun 02:00.
3. Does the partner know that `viewer/` and `.env.example` belong to the fly side?
4. How does the dev machine reach `root@tower`: Tailscale or a LAN IP?

## Next concrete step

Finish the MaleCNS v1.0 bulk switch (see "In progress"), then stop and tell the user before starting Step 0 or Phase 0.

1. **Recreate the test environment** if `/tmp/fdvenv` is missing (see "Commands").
2. **Read the real v1.0 schemas.** This downloads about 58 MB to `/tmp/mcns`; for the two big files it reads only the header. Run it exactly as written; the heredoc needs `EOF` at the start of its line.

```bash
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
mkdir -p /tmp/mcns && cd /tmp/mcns
for f in body-annotations-male-cns-v1.0-minconf-0.5.feather body-neurotransmitters-male-cns-v1.0.feather; do curl -s -o $f $B/$f; done
for f in connectome-weights-male-cns-v1.0-minconf-0.5.feather body-stats-male-cns-v1.0-minconf-0.5.feather; do curl -s -r 0-262143 -o head_$f $B/$f; done
/tmp/fdvenv/bin/python - <<'EOF'
import pyarrow as pa, pyarrow.feather as ft, pyarrow.ipc as ipc
for f in ["body-annotations-male-cns-v1.0-minconf-0.5.feather", "body-neurotransmitters-male-cns-v1.0.feather"]:
    t = ft.read_table(f); print(f"\n== {f}: {t.num_rows:,} rows\n{t.schema}")
    print(t.slice(0, 3).to_pandas().T)
for f in ["connectome-weights-male-cns-v1.0-minconf-0.5.feather", "body-stats-male-cns-v1.0-minconf-0.5.feather"]:
    print(f"\n== {f}\n{ipc.read_schema(pa.py_buffer(open('head_' + f, 'rb').read()[8:]))}")
EOF
```

   If the header schema parse fails, the file may use a different layout. Download the file's footer instead (the last 64 KB, via a range request) or ask the user.
3. **Check which columns exist:** neuron type, class and superclass, `exitNerve` or leg-nerve labels, soma position, neurotransmitter, and a way to tell real neurons from fragments. Write the loader from what you find.
4. **Rewrite `fly/connectome.py`'s MaleCNS parts** as described under "In progress". Keep `Connectome`, `to_torch`, `shuffled` and `load_flywire`. Add a test in `fly/tests/` built on tiny synthetic feather files, and run the whole suite.
5. **Tell the user** the switch is done. Give them the proposed `CLAUDE.md` wording ("MaleCNS v1.0 via bulk download"; the neuPrint token is optional) and the open questions below. Wait for their go-ahead.

// Fly Drums viewer: replays a recorded run (fly.loop --poses, exported by fly.viewer_export) in 3D, with a free
// camera, and sends the fly's hits to the TD-07 over Web MIDI in time with playback.
// Serve the repo root (python -m http.server 8000) and open http://localhost:8000/viewer/?run=<id>.
// Units in the files are cm with z up (MuJoCo); `root` turns that into three.js's y-up and scales it by S.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { BrainPanel } from "./brain.js";
import { initSongPanel } from "./song.js";

const RUNS = new URL("../runs/", location.href);
const S = 10; // scene units per cm
const LANE_ORDER = ["crash", "ride", "hat", "tom1", "tom2", "tom3", "snare", "kick", "hat_pedal"];
const PAD_NAMES = { snare: "Snare", hat: "Hi-hat", crash: "Crash", ride: "Ride", tom1: "Tom 1", tom2: "Tom 2",
  tom3: "Tom 3", kick: "Kick", hat_pedal: "Hi-hat pedal" };
const VOICE_NAMES = { snare: "snare", xstick: "snare rim", hat_closed: "hi-hat", hat_open: "open hi-hat",
  hat_pedal: "hi-hat pedal", kick: "kick", tom1: "tom 1", tom2: "tom 2", tom3: "tom 3", crash: "crash",
  ride: "ride", ride_bell: "ride bell" };
// MuJoCo free-camera presets (lookat cm, distance cm, azimuth, elevation), as in fly/clip.py
const VIEWS = {
  front: [[0.0, 0.0, -0.17], 0.62, 200, -22],
  side: [[-0.02, 0.0, -0.17], 0.75, 270, -12],
  three_quarter: [[0.0, 0.0, -0.16], 0.72, 235, -18],
  top: [[0.0, 0.0, -0.18], 0.7, 180, -84],
  behind: [[0.03, 0.0, -0.17], 0.8, 25, -32],
};
const MIDI_LOOKAHEAD_MS = 150; // hits are queued this far ahead, on their own timer (not the render loop)
const MIDI_TICK_MS = 20;
const MIDI_LATE_MS = 80; // a hit this late still sounds (a hitch in the page); later ones are dropped
const NOTE_MS = 50;
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = (id) => document.getElementById(id);
const ui = {
  loading: $("loading"), runPicker: $("run-picker"), runInfo: $("run-info"), recent: $("recent"),
  play: $("play"), playIcon: $("play-icon"), now: $("now"), total: $("total"),
  lanes: $("lanes"), lanesCanvas: $("lanes-canvas"), orbit: $("orbit"),
  midiConnect: $("midi-connect"), midiPick: $("midi-pick"), midiOutput: $("midi-output"), midiStatus: $("midi-status"),
};

// ---------- three.js stage ----------

const renderer = new THREE.WebGLRenderer({ canvas: $("stage"), antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.05;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 0.4;
controls.maxDistance = 25;
controls.autoRotateSpeed = 0.6;
controls.maxPolarAngle = Math.PI * 0.55; // the free camera can dip a little below the fly, never under the floor
controls.addEventListener("start", () => { // you grabbed the camera: stop any glide, and it's the free camera now
  camTween = null;
  if (currentView !== "free") markView("free");
});

const root = new THREE.Group(); // MuJoCo frame: z up, cm
root.rotation.x = -Math.PI / 2;
root.scale.setScalar(S);
scene.add(root);

scene.add(new THREE.HemisphereLight(0xfff1e0, 0x2a1a22, 0.9));
const key = new THREE.DirectionalLight(0xfff4e8, 2.4);
key.position.set(3, 6, 2.5);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
Object.assign(key.shadow.camera, { left: -4, right: 4, top: 4, bottom: -4, near: 0.5, far: 20 });
key.shadow.bias = -0.0004;
scene.add(key);
const rim = new THREE.DirectionalLight(0xff9a7a, 1.1);
rim.position.set(-4, 3, -3);
scene.add(rim);

function mjToThree([x, y, z]) {
  return new THREE.Vector3(x * S, z * S, -y * S);
}

function viewPose(name) {
  const [lookat, dist, az, el] = VIEWS[name];
  const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
  const fwd = [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)];
  const eye = lookat.map((v, i) => v - dist * fwd[i]);
  return { target: mjToThree(lookat), position: mjToThree(eye) };
}

const VIEW_KEYS = [...Object.keys(VIEWS), "free"]; // keys 1-6; "free" = wherever you drag the camera
let camTween = null;
let currentView = null;

function markView(name) { // highlight the button, remember it in the URL, show the drag hint for the free camera
  currentView = name;
  document.querySelectorAll(".views [data-view]").forEach((b) => b.setAttribute("aria-pressed", b.dataset.view === name));
  $("free-hint").hidden = name !== "free";
  const u = new URL(location.href);
  u.searchParams.set("view", name);
  history.replaceState(null, "", u);
}

function setView(name) {
  markView(name);
  if (name === "free") { // the camera stays put; you move it
    camTween = null;
    return;
  }
  const to = viewPose(name);
  if (reducedMotion) {
    camera.position.copy(to.position);
    controls.target.copy(to.target);
    return;
  }
  camTween = { from: { p: camera.position.clone(), t: controls.target.clone() }, to, start: performance.now(), ms: 700 };
}

function stepCamera(now) {
  if (!camTween) return;
  const k = Math.min(1, (now - camTween.start) / camTween.ms);
  const e = k < 0.5 ? 2 * k * k : 1 - (-2 * k + 2) ** 2 / 2;
  camera.position.lerpVectors(camTween.from.p, camTween.to.position, e);
  controls.target.lerpVectors(camTween.from.t, camTween.to.target, e);
  if (k >= 1) camTween = null;
}

// ---------- geometry ----------

function primGeometry(type, size) {
  let g;
  if (type === "cylinder") {
    g = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 48);
    g.rotateX(Math.PI / 2);
  } else if (type === "capsule") {
    g = new THREE.CapsuleGeometry(size[0], 2 * size[1], 6, 16);
    g.rotateX(Math.PI / 2);
  } else if (type === "ellipsoid") {
    g = new THREE.SphereGeometry(1, 24, 16);
    g.scale(size[0], size[1], size[2]);
  } else if (type === "box") {
    g = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
  } else {
    throw new Error(`unknown primitive ${type}`);
  }
  return g;
}

function setQuat(obj, q) { // MuJoCo w x y z
  obj.quaternion.set(q[1], q[2], q[3], q[0]);
}

function material(rgba, shine) {
  const m = new THREE.MeshStandardMaterial({
    color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
    roughness: 0.85 - 0.6 * shine,
    metalness: shine > 0.75 ? 0.55 : 0.05,
  });
  if (rgba[3] < 1) Object.assign(m, { transparent: true, opacity: rgba[3], depthWrite: false, side: THREE.DoubleSide });
  return m;
}

// ---------- the fly (runs/model/fly.json + fly.bin) ----------

let fly = null; // {info, bodies: [Group]}

async function loadFly() {
  if (fly) return fly;
  ui.loading.textContent = "Loading the fly (12\u00a0MB)…";
  const info = await getJSON("model/fly.json");
  const buf = await getBuffer("model/fly.bin");
  const bodies = info.bodies.map(() => {
    const g = new THREE.Group();
    root.add(g);
    return g;
  });
  const meshes = info.meshes.map((m) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, m.position_offset, m.count * 3), 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(buf, m.normal_offset, m.count * 3), 3, true));
    return geo;
  });
  for (const g of info.geoms) {
    const geo = g.type === "mesh" ? meshes[g.mesh] : primGeometry(g.type, g.size);
    const mesh = new THREE.Mesh(geo, material(g.rgba, Math.min(1, g.shininess * g.specular * 1.6)));
    mesh.position.fromArray(g.pos);
    setQuat(mesh, g.quat);
    mesh.castShadow = g.rgba[3] >= 1;
    bodies[g.body].add(mesh);
  }
  fly = { info, bodies };
  return fly;
}

// ---------- the kit (runs/<id>/scene.json, from fly/scene.py) ----------

let kit = null; // {group, prims: [{p, mesh, base}], ...}

function buildKit(sc) {
  if (kit) {
    root.remove(kit.group);
    kit.group.traverse((o) => { o.geometry?.dispose(); o.material?.dispose(); });
  }
  const group = new THREE.Group();
  root.add(group);
  const prims = sc.prims.map((p) => {
    const mesh = new THREE.Mesh(primGeometry(p.type, p.size), material(p.rgba, p.shine));
    mesh.position.fromArray(p.pos);
    setQuat(mesh, p.quat);
    mesh.castShadow = p.part !== "floor";
    mesh.receiveShadow = true;
    group.add(mesh);
    return { p, mesh, color: mesh.material.color.clone(), pos: mesh.position.clone(), quat: mesh.quaternion.clone() };
  });
  const flashColor = {};
  for (const [pad, hex] of Object.entries(sc.voice_colors)) {
    const c = new THREE.Color(hex);
    flashColor[pad] = new THREE.Color().setRGB(0.65 * c.r + 0.35, 0.65 * c.g + 0.35, 0.65 * c.b + 0.35);
  }
  kit = { group, prims, anim: sc.anim, pads: sc.pads, touching: sc.touching, flashColor };
}

function eased(pad, row, ms) { // fraction of the last `ms` sim ms the pad was touched (fly/scene.py _eased)
  const w = Math.max(1, Math.round(ms));
  const a = row - w + 1, b = row + 1;
  let on = 0;
  for (const [s, e] of kit.touching[pad] || []) {
    if (e <= a) continue;
    if (s >= b) break;
    on += Math.min(e, b) - Math.max(s, a);
  }
  return on / w;
}

const _q = new THREE.Quaternion(), _v = new THREE.Vector3(), _pivot = new THREE.Vector3(), _axis = new THREE.Vector3();

function animateKit(simMs) { // fly/scene.py frame_state
  const a = kit.anim;
  const row = Math.floor(simMs);
  const hatOpen = 1 - eased("hat_pedal", row, a.hat_ease_ms);
  const kickDown = eased("kick", row, a.pedal_ease_ms);
  for (const k of kit.prims) {
    const { p, mesh } = k;
    const hit = lastHit(p.pad, simMs);
    const dt = hit ? simMs - hit.sim : Infinity;
    mesh.material.color.copy(k.color);
    mesh.material.emissive.setRGB(0, 0, 0);
    if (p.flash && dt < a.flash_ms) {
      const amount = (hit.velocity / 127) ** 0.7 * (1 - dt / a.flash_ms);
      mesh.material.color.lerp(kit.flashColor[p.pad], amount);
      mesh.material.emissive.copy(kit.flashColor[p.pad]).multiplyScalar(0.35 * amount);
    }
    mesh.position.copy(k.pos);
    mesh.quaternion.copy(k.quat);
    let angle = 0;
    if (p.anim === "cymbal" && isFinite(dt)) {
      angle = THREE.MathUtils.degToRad(a.wobble_deg) * hit.velocity / 127 * Math.exp(-dt / a.wobble_tau_ms)
        * Math.sin(2 * Math.PI * dt / a.wobble_period_ms);
    } else if (p.anim === "beater") {
      angle = THREE.MathUtils.degToRad(a.beater_hit_deg - a.beater_rest_deg) * kickDown;
    } else if (p.anim === "hat_bottom") {
      mesh.position.z -= a.hat_open * hatOpen;
    }
    if (angle) {
      _q.setFromAxisAngle(_axis.fromArray(p.axis).normalize(), angle);
      _pivot.fromArray(p.pivot);
      mesh.position.copy(_v.copy(k.pos).sub(_pivot).applyQuaternion(_q).add(_pivot));
      mesh.quaternion.premultiply(_q);
    }
  }
}

// ---------- the run ----------

let run = null; // {id, v, hits, poses, byPad, lanes, endS}

let brain = null; // BrainPanel, or null when runs/model/neurons.* is missing

async function setBrain(id, v, voiceColors) {
  if (!brain) {
    try {
      brain = new BrainPanel(document.getElementById("brain"), { runsUrl: RUNS, voiceColors });
      await brain.load();
    } catch (err) {
      console.warn("brain panel off:", err.message);
      brain = null;
      document.getElementById("brain").classList.add("no-brain");
      document.querySelector("#brain .brain-empty").hidden = false;
      return;
    }
  }
  if (!v.spikes) return brain.setRun(null);
  const [spikes, b] = await Promise.all([getBuffer(`${id}/spikes.bin`), getJSON(`${id}/brain.json`)]);
  brain.setRun({ spikes, brain: b, viewer: v });
}

async function loadRun(id) {
  ui.loading.hidden = false;
  ui.loading.textContent = "Loading the run…";
  stopPlayback();
  await loadFly();
  const v = await getJSON(`${id}/viewer.json`);
  if (v.n_bodies !== fly.info.bodies.length) throw new Error(`${id} was exported with a different fly model`);
  const [sc, hits, poses] = await Promise.all([
    getJSON(`${id}/scene.json`), getJSON(`${id}/hits.json`), getBuffer(`${id}/poses.bin`),
  ]);
  buildKit(sc);
  await setBrain(id, v, sc.voice_colors);
  hits.sort((x, y) => x.t_s - y.t_s);
  const byPad = {};
  for (const h of hits) {
    h.sim = h.t_s * 1000 + v.offset_ms;
    (byPad[h.pad] ||= []).push(h);
  }
  const lanes = LANE_ORDER.filter((p) => byPad[p]);
  const endS = Math.max(0, (v.t0_ms + (v.n_frames - 1) * v.frame_ms - v.offset_ms) / 1000);
  run = { id, v, hits, poses: new Float32Array(poses), byPad, lanes, endS, midiNext: 0, voiceColors: sc.voice_colors };
  ui.total.textContent = endS.toFixed(2);
  ui.lanes.setAttribute("aria-valuemax", endS.toFixed(2));
  const who = v.driver === "teacher" && v.spikes ? "Teacher plays, brain listens"
    : v.driver === "teacher" ? "Teacher's strokes played into the body"
    : v.alpha > 0 ? `The fly's brain, guided by the teacher (α = ${v.alpha})`
    : v.weights ? "The trained fly, on its own" : "The untrained fly, on its own";
  const groove = (v.groove || "").split("/").pop();
  ui.runInfo.textContent = `${who}. ${groove}, ${hits.length} hits.`;
  ui.runPicker.value = id;
  seek(0);
  ui.loading.hidden = true;
}

function lastHit(pad, simMs) { // the latest hit on `pad` at or before simMs
  const list = run?.byPad[pad];
  if (!list || !list.length || list[0].sim > simMs) return null;
  let lo = 0, hi = list.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (list[mid].sim <= simMs) lo = mid; else hi = mid - 1;
  }
  return list[lo];
}

const _qa = new THREE.Quaternion(), _qb = new THREE.Quaternion();

function poseFly(simMs) {
  const { v, poses } = run;
  const f = THREE.MathUtils.clamp((simMs - v.t0_ms) / v.frame_ms, 0, v.n_frames - 1);
  const i = Math.floor(f), j = Math.min(i + 1, v.n_frames - 1), t = f - i;
  const nb = v.n_bodies;
  for (let b = 0; b < nb; b++) {
    const A = (i * nb + b) * 7, B = (j * nb + b) * 7;
    const g = fly.bodies[b];
    g.position.set(
      poses[A] + t * (poses[B] - poses[A]),
      poses[A + 1] + t * (poses[B + 1] - poses[A + 1]),
      poses[A + 2] + t * (poses[B + 2] - poses[A + 2]),
    );
    _qa.set(poses[A + 4], poses[A + 5], poses[A + 6], poses[A + 3]);
    _qb.set(poses[B + 4], poses[B + 5], poses[B + 6], poses[B + 3]);
    g.quaternion.slerpQuaternions(_qa, _qb, t);
  }
}

// ---------- playback clock (score seconds) ----------

const clock = { playing: false, speed: 1, anchorS: 0, anchorPerf: 0, s: 0 };

function nowS(perf = performance.now()) {
  return clock.playing ? clock.anchorS + (perf - clock.anchorPerf) / 1000 * clock.speed : clock.s;
}

function seek(s) {
  clock.s = THREE.MathUtils.clamp(s, 0, run ? run.endS : 0);
  clock.anchorS = clock.s;
  clock.anchorPerf = performance.now();
  resetMidi();
}

function play() {
  if (!run) return;
  if (clock.s >= run.endS - 1e-3) clock.s = 0;
  clock.anchorS = clock.s;
  clock.anchorPerf = performance.now();
  clock.playing = true;
  resetMidi();
  setPlayIcon();
}

function stopPlayback() {
  if (clock.playing) clock.s = nowS();
  clock.playing = false;
  resetMidi();
  setPlayIcon();
}

function setSpeed(speed) {
  const s = nowS();
  clock.speed = speed;
  clock.s = s;
  clock.anchorS = s;
  clock.anchorPerf = performance.now();
  resetMidi();
  document.querySelectorAll(".speed button").forEach((b) => b.setAttribute("aria-checked", +b.dataset.speed === speed));
}

function setPlayIcon() {
  ui.playIcon.setAttribute("d", clock.playing ? "M6.5 4.5h4v15h-4zM13.5 4.5h4v15h-4z" : "M7 4.5v15l12.5-7.5z");
  ui.play.setAttribute("aria-label", clock.playing ? "Pause" : "Play");
}

// ---------- Web MIDI: the fly's hits to the TD-07 ----------

const midi = { access: null, out: null };

async function midiAccess() { // shared by playback (outputs) and the New song panel's recording (inputs)
  if (midi.access) return midi.access;
  if (!navigator.requestMIDIAccess) throw new Error("This browser has no Web MIDI. Use Chrome.");
  try {
    midi.access = await navigator.requestMIDIAccess();
  } catch (e) {
    throw new Error("MIDI access was blocked. Allow it in the address bar, then try again.");
  }
  midi.access.onstatechange = listOutputs;
  listOutputs();
  return midi.access;
}

async function connectMidi() {
  try {
    await midiAccess();
  } catch (e) {
    ui.midiStatus.textContent = e.message;
  }
}

function listOutputs() {
  const outs = [...midi.access.outputs.values()];
  ui.midiOutput.replaceChildren(...outs.map((o) => new Option(o.name, o.id)));
  const td07 = outs.find((o) => /TD-07 1/i.test(o.name)) || outs.find((o) => /TD-07/i.test(o.name));
  const pick = outs.find((o) => o.id === midi.out?.id) || td07 || outs[0] || null;
  midi.out = pick;
  ui.midiConnect.hidden = true;
  ui.midiPick.hidden = !outs.length;
  if (pick) ui.midiOutput.value = pick.id;
  ui.midiStatus.textContent = !outs.length ? "No MIDI outputs found. Is the TD-07 switched on and plugged in by USB?"
    : td07 ? "" : "No TD-07 found, so hits go to the output above.";
}

function resetMidi() {
  midi.out?.clear?.();
  if (!run) return;
  const s = nowS();
  let i = 0;
  while (i < run.hits.length && run.hits[i].t_s < s) i++;
  run.midiNext = i;
}

function scheduleMidi(perfNow) {
  if (!midi.out || !clock.playing) return;
  const horizonS = nowS(perfNow + MIDI_LOOKAHEAD_MS);
  while (run.midiNext < run.hits.length && run.hits[run.midiNext].t_s <= horizonS) {
    const h = run.hits[run.midiNext++];
    const at = clock.anchorPerf + (h.t_s - clock.anchorS) / clock.speed * 1000;
    if (at < perfNow - MIDI_LATE_MS) continue;
    const when = Math.max(at, perfNow);
    midi.out.send([0x99, h.note, h.velocity], when);
    midi.out.send([0x89, h.note, 0], when + NOTE_MS);
  }
}

// ---------- drum lanes ----------

const lanesCtx = ui.lanesCanvas.getContext("2d");
const lanesSize = { w: 0, h: 0 }; // kept by a ResizeObserver, so drawing never reads layout
new ResizeObserver(([e]) => { lanesSize.w = e.contentRect.width; lanesSize.h = e.contentRect.height; }).observe(ui.lanesCanvas);
const LABEL_W = 96;

function drawLanes(s) {
  const c = ui.lanesCanvas, dpr = Math.min(devicePixelRatio, 2);
  const { w, h } = lanesSize;
  if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(w * dpr);
    c.height = Math.round(h * dpr);
  }
  const ctx = lanesCtx;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!run || !run.lanes.length) return;
  const rows = run.lanes.length, rowH = h / rows, trackW = w - LABEL_W;
  const x = (t) => LABEL_W + (t / Math.max(run.endS, 1e-3)) * trackW;
  ctx.font = "500 15px 'Barlow Condensed', sans-serif";
  ctx.textBaseline = "middle";
  // second marks
  ctx.fillStyle = "rgba(239,230,220,0.07)";
  for (let t = 0; t <= run.endS; t += 1) ctx.fillRect(Math.round(x(t)), 0, 1, h);
  const colors = kit.flashColor;
  run.lanes.forEach((pad, r) => {
    const y0 = r * rowH, mid = y0 + rowH / 2;
    ctx.fillStyle = "rgba(239,230,220,0.05)";
    ctx.fillRect(LABEL_W, mid - 0.5, trackW, 1);
    ctx.fillStyle = "#a3968c";
    ctx.fillText(PAD_NAMES[pad] || pad, 0, mid);
    const hex = "#" + new THREE.Color(sceneVoiceColor(pad)).getHexString();
    for (const hit of run.byPad[pad]) {
      const past = hit.t_s <= s;
      const fresh = past && s - hit.t_s < 0.12;
      const len = Math.max(3, (rowH - 4) * (0.35 + 0.65 * hit.velocity / 127));
      ctx.globalAlpha = fresh ? 1 : past ? 0.85 : 0.32;
      ctx.fillStyle = fresh ? "#" + colors[pad].getHexString() : hex;
      ctx.fillRect(Math.round(x(hit.t_s)) - (fresh ? 1.5 : 1), mid - len / 2, fresh ? 3 : 2, len);
    }
    ctx.globalAlpha = 1;
  });
  ctx.fillStyle = "#efe6dc";
  ctx.fillRect(Math.round(x(s)) - 1, 0, 2, h);
}

function sceneVoiceColor(pad) {
  return run.voiceColors?.[pad] || "#cccccc";
}

function laneSeek(ev) {
  const r = ui.lanesCanvas.getBoundingClientRect();
  const f = THREE.MathUtils.clamp((ev.clientX - r.left - LABEL_W) / (r.width - LABEL_W), 0, 1);
  const wasPlaying = clock.playing;
  clock.playing = false;
  seek(f * run.endS);
  if (wasPlaying) play();
}

// ---------- recent hits ----------

let recentKey = "";
function drawRecent(s) {
  const list = run.hits.filter((h) => h.t_s <= s && s - h.t_s < 0.6 / Math.max(clock.speed, 0.25)).slice(-5);
  const keyNow = list.map((h) => h.t_s).join();
  if (keyNow === recentKey) return;
  recentKey = keyNow;
  ui.recent.replaceChildren(...list.reverse().map((h) => {
    const li = document.createElement("li");
    const dot = document.createElement("i");
    dot.style.background = sceneVoiceColor(h.pad);
    const side = h.limb.startsWith("hind") ? "foot" : h.limb.replace("front_", "");
    const small = document.createElement("small");
    small.textContent = `${side}, velocity ${h.velocity}`;
    li.append(dot, `${VOICE_NAMES[h.voice] || h.voice} `, small);
    return li;
  }));
}

// ---------- loop ----------

function resize() { // the stage fills its grid column, not the window
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

const fps = { last: 0, value: 0 };

function frame(perf) {
  requestAnimationFrame(frame);
  if (fps.last) fps.value = 0.9 * fps.value + 0.1 * (1000 / Math.max(1, perf - fps.last));
  fps.last = perf;
  stepCamera(perf);
  controls.update();
  if (run) {
    let s = nowS(perf);
    if (clock.playing && s >= run.endS) {
      clock.s = run.endS;
      stopPlayback();
      s = run.endS;
    }
    const sim = s * 1000 + run.v.offset_ms;
    poseFly(sim);
    animateKit(sim);
    ui.now.textContent = s.toFixed(2);
    ui.lanes.setAttribute("aria-valuenow", s.toFixed(2));
    ui.lanes.setAttribute("aria-valuetext", `${s.toFixed(2)} seconds`);
    drawLanes(s);
    drawRecent(s);
    brain?.update(sim);
  }
  renderer.render(scene, camera);
  brain?.render();
}

// ---------- files ----------

async function getJSON(path) {
  const r = await fetch(new URL(path, RUNS));
  if (!r.ok) throw new Error(`${r.status} for runs/${path}`);
  return r.json();
}

async function getBuffer(path) {
  const r = await fetch(new URL(path, RUNS));
  if (!r.ok) throw new Error(`${r.status} for runs/${path}`);
  return r.arrayBuffer();
}

// ---------- wiring ----------

function wire() {
  ui.play.addEventListener("click", () => (clock.playing ? stopPlayback() : play()));
  document.querySelectorAll(".speed button").forEach((b) => b.addEventListener("click", () => setSpeed(+b.dataset.speed)));
  document.querySelectorAll(".views [data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
  ui.orbit.addEventListener("click", () => {
    controls.autoRotate = !controls.autoRotate;
    ui.orbit.setAttribute("aria-pressed", controls.autoRotate);
  });
  ui.midiConnect.addEventListener("click", connectMidi);
  ui.midiOutput.addEventListener("change", () => {
    midi.out = midi.access.outputs.get(ui.midiOutput.value) || null;
    resetMidi();
  });
  ui.runPicker.addEventListener("change", () => {
    const u = new URL(location.href);
    u.searchParams.set("run", ui.runPicker.value);
    history.replaceState(null, "", u);
    loadRun(ui.runPicker.value).catch(showError);
  });
  let dragging = false;
  ui.lanes.addEventListener("pointerdown", (e) => {
    dragging = true;
    ui.lanes.setPointerCapture(e.pointerId);
    laneSeek(e);
  });
  ui.lanes.addEventListener("pointermove", (e) => dragging && laneSeek(e));
  ui.lanes.addEventListener("pointerup", () => { dragging = false; });
  ui.lanes.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const step = (e.shiftKey ? 1 : 0.1) * (e.key === "ArrowLeft" ? -1 : 1);
    const wasPlaying = clock.playing;
    clock.s = nowS();
    clock.playing = false;
    seek(clock.s + step);
    if (wasPlaying) play();
  });
  addEventListener("keydown", (e) => {
    if (e.target.closest("select") || e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === " " && !e.target.closest("button")) {
      e.preventDefault();
      clock.playing ? stopPlayback() : play();
    }
    if (/^[1-6]$/.test(e.key)) setView(VIEW_KEYS[+e.key - 1]);
  });
  new ResizeObserver(resize).observe(renderer.domElement);
}

function showError(err) {
  console.error(err);
  ui.loading.hidden = false;
  ui.loading.innerHTML = "";
  const p = document.createElement("p");
  p.textContent = `Couldn't load the run: ${err.message}.`;
  const hint = document.createElement("p");
  hint.innerHTML = "Serve the repo root with <code>python -m http.server 8000</code> and open " +
    "<code>localhost:8000/viewer/</code>. Runs need <code>python -m fly.viewer_export runs/&lt;id&gt;</code> first.";
  ui.loading.append(p, hint);
}

async function main() {
  wire();
  resize();
  const u = new URL(location.href);
  const view = VIEW_KEYS.includes(u.searchParams.get("view")) ? u.searchParams.get("view") : "three_quarter";
  const start = viewPose(VIEWS[view] ? view : "three_quarter"); // the free camera starts from three-quarter
  camera.position.copy(start.position);
  controls.target.copy(start.target);
  markView(view);
  requestAnimationFrame(frame);
  setInterval(() => run && scheduleMidi(performance.now()), MIDI_TICK_MS);
  initSongPanel({ loadRun: showNewRun, stopPlayback, midiAccess }).catch(console.error);
  const index = await refreshRuns();
  if (!index.length) throw new Error("no exported runs in runs/viewer_index.json");
  const id = u.searchParams.get("run") || index[0].id;
  await loadRun(id);
}

async function refreshRuns() {
  const index = await getJSON("viewer_index.json");
  ui.runPicker.replaceChildren(...index.map((r) => {
    const who = r.driver === "teacher" ? "teacher" : r.alpha > 0 ? `guided α ${r.alpha}` : "fly on its own";
    return new Option(`${r.groove || r.id} (${who}${r.brain ? ", brain" : ""}, ${r.hits} hits)`, r.id);
  }));
  return index;
}

async function showNewRun(id) { // a run the New song panel just brought back
  await refreshRuns();
  const u = new URL(location.href);
  u.searchParams.set("run", id);
  history.replaceState(null, "", u);
  await loadRun(id);
}

main().catch(showError);

// exposed for tests (Playwright) and the console
window.flyDrums = {
  get run() { return run; }, get kit() { return kit; }, get brain() { return brain; }, clock, seek, play, pause: stopPlayback, setSpeed, setView, nowS,
  midi, scheduleMidi, resetMidi, fps, get view() { return currentView; }, camera, controls,
};

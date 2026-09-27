// The brain panel: every neuron of the male fly's CNS (MaleCNS v1.0) with a mapped cell body, at its real position,
// flaring as it fires. Hearing (cue) neurons flare in their drum's colour; everything else goes amber to white.
// Data from fly.viewer_export: runs/model/neurons.bin (+ .json) once, and each brain run's spikes.bin + brain.json.

import * as THREE from "three";
import { OrbitControls } from "three/addons/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const DECAY_MS = 30; // a spike's flare fades with this time constant: short, so single spikes read as blinks
const REPLAY_MS = 300; // after a jump, rebuild the flares from this much history
const METER_WINDOW_MS = 50; // meters show the mean firing rate per neuron over this window
const SPARK_MS = 2000;
const METER_MAX_HZ = { hearing: 150, descending: 40, motor: 80 };
const REGION_BASE = { 0: "#3a3438", 1: "#2c3a4d", 2: "#4a3a36", 3: "#3a3550" }; // other, optic lobe, central brain, nerve cord
const HOT = new THREE.Color("#ffd29a");
const LEG_LABEL = { front_left: "Front left", front_right: "Front right", hind_left: "Hind left", hind_right: "Hind right" };
const VOICE_LABEL = { kick: "Kick", hat_pedal: "Hat pedal", snare: "Snare", xstick: "Snare rim", hat_closed: "Hi-hat",
  hat_open: "Open hat", tom1: "Tom 1", tom2: "Tom 2", tom3: "Tom 3", crash: "Crash", ride: "Ride", ride_bell: "Ride bell" };

const VERTEX = /* glsl */ `
  attribute float aAct;
  attribute vec3 aBase;
  attribute vec3 aHot;
  attribute float aBoost; // 1 hearing, 0.6 leg motor, 0.3 descending, 0 the rest: the groups the story is about
  uniform float uSize;
  uniform float uScale;
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float a = clamp(aAct, 0.0, 1.0);
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = uSize * uScale * (1.0 + (0.6 + 2.2 * aBoost) * a) / -mv.z;
    // Each neuron stays a point of light, not a lamp: thousands fire together in the central brain, and with additive
    // blending full-strength flares sum to a white blob. Resting neurons draw a faint map; firing ones brighten gently.
    vColor = mix(aBase, aHot, a) * (0.8 + (1.0 + 0.8 * aBoost) * a);
    vAlpha = 0.1 + (0.4 + 0.5 * aBoost) * a;
  }`;

const FRAGMENT = /* glsl */ `
  varying vec3 vColor;
  varying float vAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float soft = smoothstep(0.5, 0.0, d);
    gl_FragColor = vec4(vColor * soft, vAlpha * soft);
  }`;

export class BrainPanel {
  constructor(root, { runsUrl, voiceColors }) {
    this.root = root;
    this.runsUrl = runsUrl;
    this.voiceColors = voiceColors;
    this.canvas = root.querySelector(".brain-canvas");
    this.empty = root.querySelector(".brain-empty");
    this.mode = root.querySelector(".brain-mode");
    this.stats = root.querySelector(".brain-stats");
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    // the panel's own colour (--lab). As scene.background it survives the bloom composer's colour conversion;
    // a clear colour gets converted twice and comes out grey.
    this.scene.background = new THREE.Color(0x110d10);
    this.camera = new THREE.PerspectiveCamera(30, 1, 1, 20000);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.4, 0, 0.7); // a soft touch, not a flood
    // keep the glow tight around firing neurons: the coarse (most blurred) levels would haze the whole panel
    this.bloom.compositeMaterial.uniforms.bloomFactors.value = [1.0, 0.55, 0.16, 0.03, 0.0];
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.run = null;
    this.lastMs = null;
    new ResizeObserver(() => this.resize()).observe(this.canvas);
    root.querySelector(".brain-reset").addEventListener("click", () => this.resetView());

  }

  async load() {
    const [info, buf] = await Promise.all([
      fetch(new URL("model/neurons.json", this.runsUrl)).then((r) => { if (!r.ok) throw new Error("runs/model/neurons.json"); return r.json(); }),
      fetch(new URL("model/neurons.bin", this.runsUrl)).then((r) => { if (!r.ok) throw new Error("runs/model/neurons.bin"); return r.arrayBuffer(); }),
    ]);
    this.info = info;
    const n = info.count;
    const xyz = new Float32Array(buf, 0, n * 3);
    this.codes = new Uint8Array(buf, n * 12, n);
    const groups = info.groups;
    // points: only neurons with a soma position; point[k] is neuron order[k]
    const order = [];
    for (let i = 0; i < n; i++) if (!Number.isNaN(xyz[3 * i])) order.push(i);
    this.pointOf = new Int32Array(n).fill(-1);
    order.forEach((i, k) => { this.pointOf[i] = k; });
    const mid = info.min.map((v, k) => (v + info.max[k]) / 2);
    const pos = new Float32Array(order.length * 3), base = new Float32Array(order.length * 3), hot = new Float32Array(order.length * 3);
    const boost = new Float32Array(order.length);
    const BOOST = { hearing: 1.0, motor: 0.6, descending: 0.3 };
    const c = new THREE.Color();
    order.forEach((i, k) => {
      // MaleCNS axes: x across the body, z from the brain down the nerve cord, y front to back. Brain on top.
      pos[3 * k] = xyz[3 * i] - mid[0];
      pos[3 * k + 1] = -(xyz[3 * i + 2] - mid[2]);
      pos[3 * k + 2] = -(xyz[3 * i + 1] - mid[1]);
      const g = groups[this.codes[i]];
      boost[k] = BOOST[g.kind] || 0;
      if (g.kind === "hearing") {
        c.set(this.voiceColors[g.voice === "hat_closed" || g.voice === "hat_open" ? "hat" : g.voice.replace("xstick", "snare").replace("ride_bell", "ride")] || "#ffffff");
        c.toArray(hot, 3 * k);
        c.multiplyScalar(0.55).toArray(base, 3 * k);
      } else if (g.kind === "descending") {
        c.set(g.name.includes("learning") ? "#f5c26b" : "#c9925a").multiplyScalar(0.45).toArray(base, 3 * k);
        HOT.toArray(hot, 3 * k);
      } else if (g.kind === "motor") {
        c.set("#8fd3c8").multiplyScalar(0.4).toArray(base, 3 * k);
        c.set("#e9fff9").toArray(hot, 3 * k);
      } else {
        c.set(REGION_BASE[this.codes[i]] || REGION_BASE[0]).toArray(base, 3 * k);
        HOT.toArray(hot, 3 * k);
      }
    });
    this.act = new Float32Array(order.length);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("aBase", new THREE.BufferAttribute(base, 3));
    geo.setAttribute("aHot", new THREE.BufferAttribute(hot, 3));
    geo.setAttribute("aBoost", new THREE.BufferAttribute(boost, 1));
    this.actAttr = new THREE.BufferAttribute(this.act, 1);
    this.actAttr.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute("aAct", this.actAttr);
    geo.computeBoundingSphere();
    this.material = new THREE.ShaderMaterial({
      vertexShader: VERTEX, fragmentShader: FRAGMENT,
      uniforms: { uSize: { value: 5.0 }, uScale: { value: 300 } },
      transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    this.points = new THREE.Points(geo, this.material);
    this.scene.add(this.points);
    this.extent = info.max.map((v, k) => v - info.min[k]);
    this.groupIdx = this.#groupIndex();
    this.#buildLegend();
    this.resetView();
    this.resize();
  }

  #groupIndex() { // group code -> neuron count, and which legend row each code feeds
    const count = {};
    for (const [code, g] of Object.entries(this.info.groups)) count[code] = g.count;
    return count;
  }

  #buildLegend() {
    const hear = this.root.querySelector(".brain-hearing");
    this.chips = {};
    for (const g of Object.values(this.info.groups).filter((g) => g.kind === "hearing")) {
      const chip = document.createElement("li");
      chip.className = "chip";
      const pad = g.voice === "hat_closed" || g.voice === "hat_open" ? "hat" : g.voice.replace("xstick", "snare").replace("ride_bell", "ride");
      chip.style.setProperty("--c", this.voiceColors[pad] || "#fff");
      chip.innerHTML = `<i></i><span>${VOICE_LABEL[g.voice] || g.voice}</span>`;
      hear.append(chip);
      this.chips[g.voice] = chip;
    }
    const meters = this.root.querySelector(".brain-meters");
    const row = (key, label, kind) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="m-label">${label}</span><span class="m-bar"><b></b></span><span class="m-val"></span>` +
        `<canvas class="m-spark" width="96" height="22"></canvas>`;
      meters.append(li);
      return { el: li, bar: li.querySelector("b"), val: li.querySelector(".m-val"), spark: li.querySelector("canvas"), kind, key };
    };
    this.meters = [
      row("descending", "Descending", "descending"),
      ...["front_left", "front_right", "hind_left", "hind_right"].map((leg) => row(leg, LEG_LABEL[leg], "motor")),
    ];
    const codes = Object.entries(this.info.groups);
    this.meterCodes = {
      descending: codes.filter(([, g]) => g.kind === "descending").map(([c]) => c),
      ...Object.fromEntries(["front_left", "front_right", "hind_left", "hind_right"].map((leg) =>
        [leg, codes.filter(([, g]) => g.leg === leg).map(([c]) => c)])),
    };
    this.hearCodes = Object.fromEntries(codes.filter(([, g]) => g.kind === "hearing").map(([c, g]) => [g.voice, c]));
  }

  resetView() {
    if (!this.extent) return;
    const tall = this.extent[2], wide = this.extent[0];
    const aspect = Math.max(0.2, this.canvas.clientWidth / Math.max(1, this.canvas.clientHeight));
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fit = Math.max(tall, wide / aspect) * 0.55 / Math.tan(fov / 2);
    this.camera.position.set(0, 0, fit);
    this.controls.target.set(0, 0, 0);
    this.camera.near = fit / 50;
    this.camera.far = fit * 10;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    if (this.material) this.material.uniforms.uScale.value = h * this.renderer.getPixelRatio() * 0.9;
  }

  // run: null (no brain) or { spikes: ArrayBuffer, brain: {bin_ms, bins}, viewer: viewer.json }
  setRun(run) {
    this.run = null;
    this.lastMs = null;
    if (this.act) this.act.fill(0);
    if (this.actAttr) this.actAttr.needsUpdate = true;
    const brainRun = run && run.spikes;
    this.root.classList.toggle("no-brain", !brainRun);
    this.empty.hidden = Boolean(brainRun);
    if (!brainRun) {
      this.mode.textContent = "No brain in this run";
      return;
    }
    const steps = run.viewer.spikes.steps;
    this.run = {
      offsets: new Uint32Array(run.spikes, 0, steps + 1),
      ids: new Uint32Array(run.spikes, (steps + 1) * 4),
      steps, bins: run.brain.bins, binMs: run.brain.bin_ms,
    };
    const a = run.viewer.alpha;
    this.mode.textContent = run.viewer.driver === "teacher" ? "Listening while the teacher plays"
      : a > 0 ? `Teacher-guided (α = ${a})` : "On its own (α = 0)";
  }

  #addSpikes(fromStep, toStep, weightAt) { // steps [fromStep, toStep): add each spike's flare
    const { offsets, ids } = this.run;
    const a = Math.max(0, fromStep), b = Math.min(this.run.steps, toStep);
    for (let t = a; t < b; t++) {
      const w = weightAt(t);
      for (let j = offsets[t]; j < offsets[t + 1]; j++) {
        const k = this.pointOf[ids[j]];
        if (k >= 0) this.act[k] += w;
      }
    }
  }

  update(simMs) {
    if (!this.run) return;
    const step = Math.floor(simMs) - 1; // spikes of step t happen at sim time t + 1 ms
    const last = this.lastMs;
    if (last === null || simMs < last || simMs - last > REPLAY_MS) { // a jump: rebuild from recent history
      this.act.fill(0);
      this.#addSpikes(step - REPLAY_MS + 1, step + 1, (t) => Math.exp(-(step - t) / DECAY_MS));
    } else if (simMs !== last) {
      const k = Math.exp(-(simMs - last) / DECAY_MS);
      for (let i = 0; i < this.act.length; i++) this.act[i] *= k;
      const lastStep = Math.floor(last) - 1;
      this.#addSpikes(lastStep + 1, step + 1, (t) => Math.exp(-(step - t) / DECAY_MS));
    }
    this.lastMs = simMs;
    this.actAttr.needsUpdate = true;
    this.#updateLegend(simMs);
  }

  #rate(codes, simMs, windowMs) { // spikes per neuron per second over the window ending at simMs
    const { bins, binMs } = this.run;
    const end = Math.floor((simMs - 1) / binMs), n = Math.max(1, Math.round(windowMs / binMs));
    let spikes = 0, neurons = 0;
    for (const c of codes) {
      neurons += this.groupIdx[c] || 0;
      const arr = bins[c];
      if (!arr) continue;
      for (let b = Math.max(0, end - n + 1); b <= end && b < arr.length; b++) spikes += arr[b];
    }
    return neurons ? spikes / neurons / (n * binMs / 1000) : 0;
  }

  #updateLegend(simMs) {
    let total = 0;
    for (const [voice, code] of Object.entries(this.hearCodes)) {
      const hz = this.#rate([code], simMs, METER_WINDOW_MS);
      const on = Math.min(1, hz / METER_MAX_HZ.hearing);
      this.chips[voice].style.setProperty("--on", on.toFixed(3));
    }
    for (const m of this.meters) {
      const codes = this.meterCodes[m.key];
      const hz = this.#rate(codes, simMs, METER_WINDOW_MS);
      m.bar.style.transform = `scaleX(${Math.min(1, hz / METER_MAX_HZ[m.kind]).toFixed(3)})`;
      m.val.textContent = `${hz < 10 ? hz.toFixed(1) : Math.round(hz)} Hz`;
      this.#spark(m, codes, simMs);
    }
    const s = Math.max(0, Math.min(this.run.steps - 1, Math.floor(simMs) - 1));
    for (let t = Math.max(0, s - 49); t <= s; t++) total += this.run.offsets[t + 1] - this.run.offsets[t];
    this.stats.textContent = `${(total * 20).toLocaleString("en-US")} spikes/s right now`;
  }

  #spark(m, codes, simMs) {
    const ctx = m.spark.getContext("2d"), w = m.spark.width, h = m.spark.height;
    const { binMs } = this.run;
    const n = Math.round(SPARK_MS / binMs / 4); // 20 ms points
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = m.kind === "motor" ? "#8fd3c8" : "#f5c26b";
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const t = simMs - (n - 1 - i) * binMs * 4;
      const v = Math.min(1, this.#rate(codes, t, binMs * 4) / METER_MAX_HZ[m.kind]);
      const x = (i / (n - 1)) * (w - 1), y = h - 1 - v * (h - 2);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }

  render() {
    if (!this.points) return;
    this.controls.update();
    this.composer.render();
  }
}

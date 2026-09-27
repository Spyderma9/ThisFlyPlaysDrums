// "New song": hand the fly new music, from a file or played on the TD-07, and load the run when it's back.
// Only works when the viewer is served by human/song_server.py (it answers /api/health); with plain
// `python -m http.server` the button stays hidden and nothing else changes.

const API = new URL("../api/", location.href);
const SILENCE_MS = 2500; // recording stops after this long without a hit, as in human/record_song.py
const WAIT_MS = 60000;   // ... or if nothing is played at all
const POLL_MS = 1500;
const TAIL_MS = 700;     // the fly simulates this past the last note

export async function initSongPanel({ loadRun, stopPlayback, midiAccess }) {
  let health;
  try {
    const r = await fetch(new URL("health", API), { cache: "no-store" });
    health = r.ok ? await r.json() : null;
  } catch { health = null; }
  if (!health?.ok) return;

  const $ = (sel) => panel.querySelector(sel);
  const panel = document.getElementById("song");
  const open = document.getElementById("song-open");
  const ui = {
    pick: $(".song-pick"), drop: $(".song-drop"), file: $(".song-file"), record: $(".song-record"),
    seconds: $(".song-seconds"), rec: $(".song-rec"), recCount: $(".song-rec-count"),
    run: $(".song-run"), name: $(".song-name"), strip: $(".song-strip"), msg: $(".song-msg"),
    ask: $(".song-ask"), askText: $(".song-ask-text"), result: $(".song-result"), again: $(".song-again"),
    error: $(".song-error"),
  };
  const trained = (health.weights || "").split("/").slice(-2, -1)[0];
  if (trained) $(".song-note").textContent += ` The fly trained in run ${trained} plays it.`;
  let job = null, recording = null, polling = null;

  open.hidden = false;
  open.addEventListener("click", () => show(panel.hidden));
  $(".song-close").addEventListener("click", () => show(false));
  addEventListener("keydown", (e) => { if (e.key === "Escape" && !panel.hidden && !recording) show(false); });

  function show(on) {
    panel.hidden = !on;
    open.setAttribute("aria-expanded", on);
    if (on) (ui.pick.hidden ? panel.querySelector(".song-close") : ui.file).focus();
  }

  function view(which) {
    ui.pick.hidden = which !== "pick";
    ui.rec.hidden = which !== "rec";
    ui.run.hidden = which !== "run";
    ui.error.textContent = "";
  }

  function seconds() {
    const s = parseFloat(ui.seconds.value);
    return s > 0 ? s : null;
  }

  // ---------- a file ----------

  ui.file.addEventListener("change", () => ui.file.files[0] && send(ui.file.files[0]));
  ui.drop.addEventListener("dragover", (e) => { e.preventDefault(); ui.drop.classList.add("over"); });
  ui.drop.addEventListener("dragleave", () => ui.drop.classList.remove("over"));
  ui.drop.addEventListener("drop", (e) => {
    e.preventDefault();
    ui.drop.classList.remove("over");
    if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]);
  });

  async function send(file) {
    const q = new URLSearchParams({ name: file.name });
    if (seconds()) q.set("seconds", seconds());
    await start(fetch(new URL(`song?${q}`, API), { method: "POST", body: file }));
    ui.file.value = "";
  }

  // ---------- played on the kit ----------

  ui.record.addEventListener("click", startRecording);
  $(".song-rec-done").addEventListener("click", () => finishRecording(true));
  $(".song-rec-cancel").addEventListener("click", () => finishRecording(false));

  async function startRecording() {
    let access;
    try { access = await midiAccess(); } catch (e) { ui.error.textContent = e.message; return; }
    const inputs = [...access.inputs.values()];
    const input = inputs.find((i) => /TD-07 0/i.test(i.name)) || inputs.find((i) => /TD-07/i.test(i.name)) || inputs[0];
    if (!input) {
      ui.error.textContent = "No kit found. Switch the TD-07 on, plug it in by USB, then try again.";
      return;
    }
    stopPlayback(); // the fly's hits going out to the kit mustn't come back in as yours
    const t0 = performance.now();
    recording = { input, t0, events: [], hits: 0, first: null, last: null };
    input.onmidimessage = (e) => {
      if (e.data[0] >= 0xf8) return; // clock, active sensing
      const t = e.timeStamp - t0;
      recording.events.push([t, [...e.data]]);
      if ((e.data[0] & 0xf0) === 0x90 && e.data[2] > 0) {
        recording.hits++;
        recording.first ??= t;
        recording.last = t;
      }
    };
    recording.timer = setInterval(() => {
      const r = recording, now = performance.now() - r.t0;
      ui.recCount.textContent = r.hits ? `${r.hits} hit${r.hits === 1 ? "" : "s"}, ${(now / 1000).toFixed(1)} s`
        : "Listening to the kit";
      const max = (seconds() || 20) * 1000;
      if (r.last !== null && (now - r.last >= SILENCE_MS || now - r.first >= max)) finishRecording(true);
      else if (r.first === null && now >= WAIT_MS) {
        finishRecording(false);
        ui.error.textContent = "Nothing was played in a minute, so recording stopped.";
      }
    }, 100);
    view("rec");
    $(".song-rec-done").focus();
  }

  async function finishRecording(keep) {
    if (!recording) return;
    const r = recording;
    recording = null;
    clearInterval(r.timer);
    r.input.onmidimessage = null;
    if (!keep) return view("pick");
    const q = new URLSearchParams();
    if (seconds()) q.set("seconds", seconds());
    await start(fetch(new URL(`take?${q}`, API), {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ events: r.events }),
    }));
  }

  // ---------- the fly plays it ----------

  async function start(request) {
    let r, body;
    try {
      r = await request;
      body = await r.json();
    } catch (e) {
      view("pick");
      ui.error.textContent = `The song server didn't answer (${e.message}). Is human/song_server.py still running?`;
      return;
    }
    if (!r.ok) {
      view("pick");
      ui.error.textContent = body.error;
      return;
    }
    job = body;
    const s = job.song;
    const dropped = Object.values(s.dropped_notes).reduce((a, b) => a + b, 0);
    ui.name.textContent = `${s.name}: ${s.hits} notes over ${s.seconds} s` +
      (dropped ? `, ${dropped} left out (no drum for them)` : "") +
      (s.known_groove?.startsWith("train/") ? ". The fly learned on this one." : "");
    ui.result.textContent = "";
    ui.again.hidden = true;
    view("run");
    render(job);
    open.textContent = "Fly playing…";
    clearInterval(polling);
    polling = setInterval(poll, POLL_MS);
  }

  async function poll() {
    try {
      const r = await fetch(new URL(`jobs/${job.id}`, API), { cache: "no-store" });
      if (r.ok) render(job = await r.json());
    } catch { /* the next poll tries again */ }
  }

  function render(j) {
    drawStrip(j);
    ui.ask.hidden = j.stage !== "confirm";
    if (j.stage === "confirm") {
      const n = j.gpu_jobs.length;
      ui.askText.textContent = `The server's GPU is already running ${n} other job${n === 1 ? "" : "s"}, ` +
        "probably Sam's training. Running the fly now slows both down.";
    }
    ui.msg.textContent = j.stage === "queued" ? "Waiting for the song before it."
      : j.stage === "failed" || j.stage === "confirm" ? "" // the question (or the error) says it
      : j.stage === "done" ? "Loading the run…" : `${capital(j.message)}.`;
    if (j.stage === "failed") {
      clearInterval(polling);
      open.textContent = "New song";
      ui.error.textContent = j.error;
      ui.again.hidden = false;
    }
    if (j.stage === "done") {
      clearInterval(polling);
      open.textContent = "New song";
      finished(j);
    }
  }

  async function finished(j) {
    const s = j.result.score;
    ui.result.textContent = s ? `The fly played ${s.played} hits for ${s.ref} notes and got ${s.hit} right ` +
      `(F1 ${s.f1.toFixed(2)}).` : "";
    try {
      await loadRun(j.result.id);
      ui.msg.textContent = "Now showing the fly's run. Press play.";
    } catch (e) {
      ui.error.textContent = `The run came back but didn't load: ${e.message}`;
    }
    ui.again.hidden = false;
  }

  ui.again.addEventListener("click", () => view("pick"));
  $(".song-go").addEventListener("click", () => answer(true));
  $(".song-nogo").addEventListener("click", () => answer(false));

  async function answer(go) {
    ui.ask.hidden = true;
    await fetch(new URL(`jobs/${job.id}/answer`, API), { method: "POST", body: JSON.stringify({ go }) });
    poll();
  }

  // ---------- the strip: the song's notes, and how far the fly has got ----------

  function drawStrip(j) {
    const c = ui.strip, dpr = Math.min(devicePixelRatio, 2);
    const w = c.clientWidth, h = c.clientHeight;
    if (!w) return;
    c.width = w * dpr;
    c.height = h * dpr;
    const g = c.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(panel);
    const amber = css.getPropertyValue("--amber").trim(), muted = css.getPropertyValue("--line").trim();
    const total = Math.max(j.status?.steps || 0, (j.notes.at(-1) || 0) + TAIL_MS);
    const done = j.stage === "done" ? total : j.status?.t_ms || 0;
    const x = (t) => 1 + (w - 2) * t / total;
    g.clearRect(0, 0, w, h);
    g.fillStyle = "rgba(239, 230, 220, 0.05)";
    g.fillRect(0, h / 2 - 1, w, 2);
    for (const t of j.notes) {
      g.fillStyle = t <= done ? amber : muted;
      g.fillRect(Math.round(x(t)) - 1, 4, 2, h - 8);
    }
    if (done > 0 && done < total) {
      g.fillStyle = css.getPropertyValue("--text").trim();
      g.fillRect(Math.round(x(done)) - 1, 0, 2, h);
    }
  }
  new ResizeObserver(() => job && drawStrip(job)).observe(ui.strip);
}

function capital(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

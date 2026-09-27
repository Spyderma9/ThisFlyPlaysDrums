"""Run the fly on a song on the T5600 from the laptop, and bring the run back for the viewer.

Everything goes over plain `ssh` (files too, as tar streams): the Windows laptop has OpenSSH but no rsync, and
scp trips over drive letters. On the server a song run is a host tmux job, so it survives a dropped connection:

    fx "python -u -m fly.loop --groove songs/<id>.mid --out runs/songs/<id> --alpha 0 --weights W --poses --spikes
        && python -m fly.viewer_export runs/songs/<id>"   > runs/songs/<id>.log, then DONE or FAIL

--alpha 0 and no --teacher, always: the fly plays on its own, nothing else sees the song (the demo rule).
Settings (host, repo, fx, weights) come from human/fly_pipeline.json.
"""
import hashlib
import io
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path

HUMAN = Path(__file__).resolve().parent
REPO = HUMAN.parent
CONFIG = HUMAN / "fly_pipeline.json"
DEFAULTS = {
    "host": "root@t5600.tail3495cd.ts.net",  # the T5600 over Tailscale
    "repo": "/mnt/user/dev/ThisFlyPlaysDrums",
    "fx": "/mnt/user/dev/fx",                # runs a command in the flydrums container, from the repo root
    "weights": None,                         # e.g. runs/train/<id>/best.pt on the server
}
RUN_FILES = ("viewer.json", "scene.json", "hits.json", "hits.csv", "hits.mid", "poses.bin", "spikes.bin",
             "brain.json", "meta.json")  # what the viewer and score.py need; never poses.npz (large)
MODEL_FILES = ("fly.json", "fly.bin", "neurons.json", "neurons.bin")
STEPS_RE = re.compile(r": (\d+) steps on (\w+)")
PROGRESS_RE = re.compile(r"^\s*(\d+) ms\s+(\d+) s\s+(\d+) hits", re.M)
SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


class RemoteError(Exception):
    pass


def load_config(path=CONFIG):
    cfg = dict(DEFAULTS)
    if Path(path).exists():
        cfg.update(json.loads(Path(path).read_text()))
    return cfg


def parse_log(text):
    """A song run's log -> {"state", "steps", "t_ms", "wall_s", "hits"}. state: starting, simulating, exporting,
    done or failed."""
    lines = text.strip().splitlines()
    steps = STEPS_RE.search(text)
    progress = PROGRESS_RE.findall(text)
    info = {"steps": int(steps.group(1)) if steps else None, "t_ms": 0, "wall_s": 0, "hits": 0}
    if progress:
        info["t_ms"], info["wall_s"], info["hits"] = map(int, progress[-1])
    if lines and lines[-1].strip() == "DONE":
        info["state"] = "done"
    elif lines and lines[-1].strip() == "FAIL":
        info["state"] = "failed"
    elif re.search(r"^wrote runs/", text, re.M):
        info["state"] = "exporting"
    elif steps:
        info["state"] = "simulating"
    else:
        info["state"] = "starting"
    return info


def job_command(repo, fx, song_id, weights):
    """The shell command the server's tmux runs for one song."""
    q = shlex.quote
    run = f"runs/songs/{song_id}"
    inner = (f"python -u -m fly.loop --groove songs/{song_id}.mid --out {run} --alpha 0 --poses --spikes"
             + (f" --weights {q(weights)}" if weights else "")
             + f" && python -m fly.viewer_export {run}")
    log = q(f"{repo}/{run}.log")
    return f"{q(fx)} {q(inner)} > {log} 2>&1 && echo DONE >> {log} || echo FAIL >> {log}"


def safe_extract(data, dest, allowed):
    """Unpack a tar stream, keeping only plain files named in `allowed` (nothing else can be written)."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    got = []
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            name = Path(m.name).name
            if m.isfile() and name in allowed:
                (dest / name).write_bytes(tar.extractfile(m).read())
                got.append(name)
    return got


class Remote:
    def __init__(self, host, repo, fx, run=subprocess.run):
        self.host, self.repo, self.fx, self._run = host, repo.rstrip("/"), fx, run

    @classmethod
    def from_config(cls, cfg=None, **kw):
        cfg = cfg or load_config()
        return cls(cfg["host"], cfg["repo"], cfg["fx"], **kw)

    def ssh(self, command, data=None, check=True):
        """Run `command` in the server's shell. -> stdout bytes."""
        args = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", self.host, command]
        done = self._run(args, input=data, capture_output=True)
        if check and done.returncode:
            err = done.stderr.decode(errors="replace").strip()
            raise RemoteError(f"ssh {self.host}: {err or f'exit {done.returncode}'}")
        return done.stdout

    def path(self, rel):
        return f"{self.repo}/{rel}"

    def check(self):
        """Reachable, repo and fx present. Raises RemoteError with a hint."""
        try:
            out = self.ssh(f"test -d {shlex.quote(self.repo)} && test -x {shlex.quote(self.fx)} && echo ok", check=False)
        except FileNotFoundError:
            raise RemoteError("no `ssh` command: on Windows, enable OpenSSH Client (Settings > Optional features)")
        if out.strip() != b"ok":
            raise RemoteError(f"can't reach {self.repo} and {self.fx} on {self.host}: check Tailscale, and that "
                              f"`ssh {self.host}` works without a password")

    def exists(self, rel):
        return self.ssh(f"test -e {shlex.quote(self.path(rel))} && echo yes", check=False).strip() == b"yes"

    def gpu_jobs(self):
        """Processes using the server's GPU (Sam's training, other containers) -> ["pid, name, MiB", ...]."""
        out = self.ssh("nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader",
                       check=False)
        return [line.strip() for line in out.decode(errors="replace").splitlines() if line.strip()]

    def upload(self, local, rel):
        target = shlex.quote(self.path(rel))
        self.ssh(f"mkdir -p $(dirname {target}) && cat > {target}", data=Path(local).read_bytes())

    def start(self, song_id, weights):
        if not SAFE_ID.match(song_id):
            raise RemoteError(f"bad song id {song_id!r}")
        q = shlex.quote
        session = f"song_{song_id}"
        if self.ssh(f"tmux has-session -t {q(session)} 2>/dev/null && echo yes", check=False).strip() == b"yes":
            raise RemoteError(f"{song_id} is already running on the server (tmux session {session})")
        log = self.path(f"runs/songs/{song_id}.log")
        self.ssh(f"mkdir -p {q(self.path('runs/songs'))} && rm -f {q(log)} && "
                 f"tmux new -d -s {q(session)} {q(job_command(self.repo, self.fx, song_id, weights))}")
        return session

    def status(self, song_id):
        log = shlex.quote(self.path(f"runs/songs/{song_id}.log"))
        text = self.ssh(f"test -e {log} || echo NO_LOG; grep -m1 ' steps on ' {log}; tail -n 8 {log}",
                        check=False).decode(errors="replace")
        if text.startswith("NO_LOG"):
            return {"state": "missing", "steps": None, "t_ms": 0, "wall_s": 0, "hits": 0, "tail": ""}
        info = parse_log(text)
        info["tail"] = "\n".join(text.strip().splitlines()[-8:])
        return info

    def fetch_run(self, song_id, dest):
        """Copy the viewer's files for runs/songs/<id> into `dest`. -> file names copied."""
        run = shlex.quote(self.path(f"runs/songs/{song_id}"))
        names = " ".join(RUN_FILES)
        data = self.ssh(f"cd {run} && tar cf - $(ls {names} 2>/dev/null)")
        return safe_extract(data, dest, RUN_FILES)

    def sync_model(self, dest):
        """Bring runs/model/ (the fly's meshes, neuron positions) over if it differs. -> file names copied."""
        model = shlex.quote(self.path("runs/model"))
        listing = self.ssh(f"cd {model} && sha1sum {' '.join(MODEL_FILES)} 2>/dev/null", check=False).decode()
        remote = {name: digest for digest, name in (line.split() for line in listing.splitlines() if line.strip())}
        dest = Path(dest)
        stale = [n for n, d in remote.items()
                 if not (dest / n).exists() or hashlib.sha1((dest / n).read_bytes()).hexdigest() != d]
        if not stale:
            return []
        return safe_extract(self.ssh(f"cd {model} && tar cf - {' '.join(stale)}"), dest, MODEL_FILES)


def update_index(runs_root=REPO / "runs"):
    """runs/viewer_index.json, exactly as fly.viewer_export.update_index writes it (that module needs numpy)."""
    runs_root = Path(runs_root)
    entries = []
    for v in runs_root.rglob("viewer.json"):
        info = json.loads(v.read_text())
        entries.append({"id": v.parent.relative_to(runs_root).as_posix(), "groove": Path(info.get("groove") or "").name,
                        "driver": info.get("driver"), "alpha": info.get("alpha"), "brain": bool(info.get("spikes")),
                        "hits": info.get("hits"), "mtime": v.stat().st_mtime})
    entries.sort(key=lambda e: -e["mtime"])
    (runs_root / "viewer_index.json").write_text(json.dumps(entries, indent=1))
    return entries

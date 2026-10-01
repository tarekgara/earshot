#!/usr/bin/env python3
"""Build the Earshot sound library.

Reads data/library.yaml, downloads only the recordings it references (cached in
.cache/), and writes:

  audio/<sound>/<key>.mp3    loudness-matched clips
  audio/<sound>/<key>.webp   spectrograms (transparent background)
  js/data.js                 everything the app needs, as a plain script
  data/provenance.json       where every clip came from
  SOURCES.md                 credits and licenses

Sample entries in the YAML (under `samples:`, one list per sound):
  - { src: commons, file: "File:Name.ogg", label: "...", start: 3.0, dur: 4 }
  - { src: url, url: "https://...", page: "...", license: "CC0", author: "...", label: "..." }
  - { src: zip, url: "https://.../pack.zip", inner: "path/in/zip.ogg", page: "...", license: "CC0", author: "..." }
  - { src: generated, preset: siren-wail, label: "..." }
`start` and `dur` are optional; without `start` the build picks the most
representative stretch of the recording.

Usage:
  python tools/build.py              build everything (cached per sound)
  python tools/build.py --only cough,laugh
  python tools/build.py --force
"""

import argparse
import concurrent.futures as cf
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pyloudnorm
import yaml
from PIL import Image
from scipy import signal

sys.path.insert(0, str(Path(__file__).parent))
import signals  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
AUDIO = ROOT / "audio"
SR = 44100
BUILD_VERSION = 1
UA = "EarshotBuild/1.0 (open-source sound reference; sample sourcing)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
DEFAULT_DUR = {"once": 4.0, "repeats": 6.0, "steady": 6.0, "irregular": 6.0}


def log(*a):
    print(*a, flush=True)


def find_ffmpeg():
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not exe:
        hits = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\*\bin\ffmpeg.exe"))
        exe = hits[0] if hits else None
    if not exe:
        sys.exit("ffmpeg not found. Install it or set FFMPEG=/path/to/ffmpeg")
    return exe


FFMPEG = find_ffmpeg()


def http_get(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f)
            tmp.replace(dest)
            return dest
        except Exception as e:  # network hiccups and rate limits: back off and retry
            if attempt == 5:
                raise RuntimeError(f"download failed: {url}: {e}")
            wait = 5 * (attempt + 1)
            ra = getattr(e, "headers", None) and e.headers.get("Retry-After")
            if ra and str(ra).isdigit():
                wait = max(wait, int(ra))
            elif getattr(e, "code", None) == 429:
                wait = 30 * (attempt + 1)
            time.sleep(wait)


def short_hash(s):
    return hashlib.sha1(s.encode()).hexdigest()[:8]


# ------------------------------------------------------------------ sources

def commons_key(title):
    name = title.replace("File:", "")
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("_")[:60]
    return f"{stem or 'file'}-{short_hash(title)}{Path(name).suffix.lower()}"


def fetch_commons(title):
    key = commons_key(title)
    meta_path = CACHE / "commons" / (key + ".json")
    if not meta_path.exists():
        q = urllib.parse.urlencode({"action": "query", "titles": title, "prop": "imageinfo",
                                    "iiprop": "url|extmetadata", "format": "json", "formatversion": "2"})
        http_get(f"{COMMONS_API}?{q}", meta_path)
    page = json.loads(meta_path.read_text(encoding="utf-8"))["query"]["pages"][0]
    if "imageinfo" not in page:
        raise RuntimeError(f"not found on Commons: {title}")
    ii = page["imageinfo"][0]
    em = ii.get("extmetadata", {})
    strip = lambda v: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", v or "")).strip()
    meta = {"source": "commons", "file": title, "url": ii["descriptionurl"],
            "license": strip(em.get("LicenseShortName", {}).get("value")),
            "author": strip(em.get("Artist", {}).get("value")) or "Unknown author"}
    audio = CACHE / "commons" / key
    if not audio.exists():
        http_get(ii["url"].split("?")[0], audio)
    return audio, meta


def fetch_url(spec):
    url = spec["url"]
    ext = Path(urllib.parse.urlparse(url).path).suffix.lower() or ".bin"
    dest = CACHE / "url" / f"{short_hash(url)}{ext}"
    if not dest.exists():
        http_get(url, dest)
    return dest, {"source": "url", "file": Path(urllib.parse.urlparse(url).path).name, "url": spec.get("page") or url,
                  "license": spec["license"], "author": spec.get("author", "Unknown author")}


def fetch_zip(spec):
    url = spec["url"]
    archive = CACHE / "zips" / f"{short_hash(url)}.zip"
    if not archive.exists():
        http_get(url, archive)
    inner = spec["inner"]
    if any(ch in inner for ch in "*?["):
        # A pattern plus `pick` (index into the sorted matches) selects one file from a dataset.
        listing = CACHE / "zips" / f"{short_hash(url)}.names.json"
        if not listing.exists():
            with zipfile.ZipFile(archive) as z:
                listing.write_text(json.dumps(z.namelist()), encoding="utf-8")
        import fnmatch
        matches = sorted(n for n in json.loads(listing.read_text(encoding="utf-8")) if fnmatch.fnmatch(n.lower(), inner.lower()))
        if not matches:
            raise RuntimeError(f"nothing matches {inner} in {url}")
        inner = matches[spec.get("pick", 0) % len(matches)]
    out = CACHE / "zips" / short_hash(url) / inner
    if not out.exists():
        with zipfile.ZipFile(archive) as z:
            names = {n.lower(): n for n in z.namelist()}
            member = names.get(inner.lower())
            if not member:  # packs often wrap everything in a top-level folder
                tail = "/" + inner.lower().lstrip("/")
                member = next((n for low, n in names.items() if low.endswith(tail)), None)
            if not member:
                raise RuntimeError(f"{inner} not in {url}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(z.read(member))
    return out, {"source": "zip", "file": inner, "url": spec.get("page") or url,
                 "license": spec["license"], "author": spec.get("author", "Unknown author")}


def fetch(spec):
    src = spec["src"]
    if src == "commons":
        return fetch_commons(spec["file"])
    if src == "url":
        return fetch_url(spec)
    if src == "zip":
        return fetch_zip(spec)
    raise ValueError(f"unknown source {src}")


# ------------------------------------------------------------------ audio io and dsp

def load(path):
    raw = subprocess.run([FFMPEG, "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def encode(x, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
                    "-c:a", "libmp3lame", "-q:a", "4", str(dest)],
                   input=np.ascontiguousarray(x, dtype=np.float32).tobytes(), check=True)


METER = pyloudnorm.Meter(SR)


def normalize(x, target=-20.0):
    if len(x) == 0:
        return x
    probe = x if len(x) >= int(0.5 * SR) else np.pad(x, (0, int(0.5 * SR) - len(x)))
    try:
        lufs = METER.integrated_loudness(probe.astype(np.float64))
    except Exception:
        lufs = float("-inf")
    y = x * 10 ** ((target - lufs) / 20) if np.isfinite(lufs) else x / (np.max(np.abs(x)) or 1) * 0.5
    peak = np.max(np.abs(y))
    if peak > 0.89:
        y *= 0.89 / peak
    return y.astype(np.float32)


def frame_levels(x, hop):
    n = max(1, len(x) // hop)
    f = x[: n * hop].reshape(n, hop)
    return 20 * np.log10(np.sqrt(np.mean(f ** 2, axis=1)) + 1e-9)


def pick_window(x, dur, shape):
    """Start time of the clip. One-off events: the loudest stretch, starting just
    before its onset. Ongoing sounds: the steadiest, most active stretch."""
    if len(x) <= dur * SR:
        return 0.0
    hop = int(0.05 * SR)
    lv = frame_levels(x, hop)
    w = int(dur / 0.05)
    if len(lv) <= w:
        return 0.0
    if shape == "once":
        energy = np.convolve(10 ** (lv / 10), np.ones(w), mode="valid")
        peak_i = int(np.argmax(energy))
        # The loudest window may start mid-event: step back to just before the onset.
        floor = np.max(lv) - 30
        i = peak_i
        while i > 0 and lv[i] > floor and peak_i - i < int(1.0 / 0.05):
            i -= 1
        return max(0, i - 2) * 0.05
    lo = min(int(1.0 / 0.05), len(lv) - w)
    hi = max(lo, len(lv) - w - int(1.0 / 0.05))
    best, best_i = -1e9, lo
    for i in range(lo, hi + 1):
        seg = lv[i:i + w]
        score = np.median(seg) - 0.6 * np.std(seg) - 2.0 * np.mean(seg < np.max(lv) - 35)
        if score > best:
            best, best_i = score, i
    return best_i * 0.05


def tidy(x, fade_in=0.01, fade_out=0.25):
    """Trim leading/trailing silence, fade the edges."""
    if len(x) == 0:
        return x
    peak = np.max(np.abs(x)) or 1
    idx = np.nonzero(np.abs(x) > peak * 10 ** (-45 / 20))[0]
    if len(idx):
        x = x[max(0, idx[0] - int(0.02 * SR)): idx[-1] + int(0.08 * SR)]
    x = x.copy()
    fi, fo = min(len(x), int(fade_in * SR)), min(len(x), int(fade_out * SR))
    x[:fi] *= np.linspace(0, 1, fi)
    x[len(x) - fo:] *= np.linspace(1, 0, fo) ** 2
    return x


def peaks(x, n=180):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    if len(x) == 0:
        return "A" * n
    edges = np.linspace(0, len(x), n + 1).astype(int)
    v = np.array([np.max(np.abs(x[edges[i]:max(edges[i + 1], edges[i] + 1)])) for i in range(n)])
    v = v / (np.max(v) or 1)
    return "".join(alphabet[int(round(c * 63))] for c in v)


SPEC_STOPS = np.array([
    [0.00, 30, 8, 20, 0], [0.22, 90, 20, 45, 120], [0.45, 170, 45, 40, 200],
    [0.68, 232, 110, 45, 245], [0.86, 250, 185, 90, 255], [1.00, 255, 244, 214, 255],
])


def spectrogram(x, dest, w=360, h=128, fmin=40.0, fmax=16000.0):
    nfft = 2048
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    starts = np.linspace(0, len(x) - nfft, w).astype(int)
    win = np.hanning(nfft)
    mag = np.abs(np.fft.rfft(np.stack([x[s:s + nfft] * win for s in starts]), axis=1))
    bins = np.fft.rfftfreq(nfft, 1 / SR)
    rows = np.geomspace(fmin, fmax, h)
    img = np.stack([np.interp(rows, bins, m) for m in mag], axis=1)
    level = 20 * np.log10(img + 1e-9)
    level -= level.max()
    v = np.clip((level + 72) / 72, 0, 1)[::-1]
    rgba = np.stack([np.interp(v, SPEC_STOPS[:, 0], SPEC_STOPS[:, c]) for c in range(1, 5)], axis=-1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba.astype(np.uint8), "RGBA").save(dest, "WEBP", quality=80, method=6)


# ------------------------------------------------------------------ context renders

def reverb(x, secs=1.2, wet=0.3, seed=3):
    rng = np.random.default_rng(seed)
    n = int(secs * SR)
    t = np.arange(n) / SR
    ir = rng.standard_normal(n) * np.exp(-t / (secs / 5))
    ir = signal.sosfilt(signal.butter(2, 5000, "lowpass", fs=SR, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    tail = signal.fftconvolve(x, ir)[: len(x) + n]
    return np.pad(x, (0, len(tail) - len(x))) * (1 - wet) + tail * wet


def speech_noise(n, seed=5):
    """Pink-ish noise band-limited to speech and chopped into syllables."""
    rng = np.random.default_rng(seed)
    pink = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                          [1, -2.494956002, 2.017265875, -0.522189400], rng.standard_normal(n))
    band = signal.sosfilt(signal.butter(3, [220, 3400], "bandpass", fs=SR, output="sos"), pink)
    t = np.arange(n) / SR
    env = np.clip(np.sin(2 * np.pi * 4.1 * t + 2 * np.sin(2 * np.pi * 0.7 * t)), 0, None) ** 1.5
    env = signal.sosfilt(signal.butter(2, 18, "lowpass", fs=SR, output="sos"), env)
    return band * env


def under_speech(x):
    n = len(x) + int(0.3 * SR)
    a = np.pad(normalize(x, -20), (0, n - len(x)))
    return normalize(a + normalize(speech_noise(n), -21), -18)


def far_away(x):
    y = signal.sosfilt(signal.butter(2, 2200, "lowpass", fs=SR, output="sos"), x)
    y = reverb(y, secs=2.0, wet=0.65)
    rng = np.random.default_rng(9)
    bed = signal.sosfilt(signal.butter(2, 600, "lowpass", fs=SR, output="sos"), rng.standard_normal(len(y)))
    return normalize(normalize(y, -20) + normalize(bed, -38), -22)


def small_speaker(x):
    y = signal.sosfilt(signal.butter(4, [320, 3600], "bandpass", fs=SR, output="sos"), x)
    y = np.tanh(2.2 * y / (np.max(np.abs(y)) or 1)) * 0.8
    return normalize(y, -20)


# ------------------------------------------------------------------ per sound

def primary(v):
    return v[0] if isinstance(v, list) else v


def build_sound(snd):
    sid = snd["id"]
    outdir = AUDIO / sid
    if outdir.exists():
        shutil.rmtree(outdir)
    shape = primary(snd["traits"].get("shape", "once"))
    samples = []

    def emit(key, label, group, x, origin):
        x = normalize(x)
        encode(x, outdir / f"{key}.mp3")
        spectrogram(x, outdir / f"{key}.webp")
        samples.append({"key": key, "label": label, "group": group, "dur": round(len(x) / SR, 2),
                        "peaks": peaks(x), "origin": origin if isinstance(origin, list) else [origin]})

    first = None
    for k, spec in enumerate(snd.get("samples", []), 1):
        if spec["src"] == "generated":
            x = signals.render(spec["preset"])
            origin = {"source": "generated", "preset": spec["preset"]}
        else:
            path, origin = fetch(spec)
            full = load(path)
            dur = spec.get("dur", DEFAULT_DUR.get(shape, 5.0))
            start = spec["start"] if "start" in spec else pick_window(full, dur, shape)
            x = full[int(start * SR): int((start + dur) * SR)]
            origin = {**origin, "excerpt": f"{start:.1f}-{start + len(x) / SR:.1f}s"}
        x = tidy(x)
        emit(f"ex-{k}", spec.get("label") or f"Recording {k}", "example", x, origin)
        if first is None:
            first = (x, origin)

    if first is not None:
        x, origin = first
        emit("speech", "Under speech", "context", under_speech(x),
             [origin, {"source": "generated", "preset": "speech-shaped noise"}])
        emit("far", "Far away", "context", far_away(x), [origin, {"source": "generated", "preset": "distance and room"}])
        emit("phone", "Through a phone", "context", small_speaker(x), origin)
    return samples


def spec_hash(snd):
    blob = json.dumps({"v": BUILD_VERSION, "samples": snd.get("samples"), "shape": snd["traits"].get("shape"),
                       "signals": Path(signals.__file__).stat().st_mtime
                       if any(s.get("src") == "generated" for s in snd.get("samples", [])) else 0},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def worker(snd, force):
    cache = CACHE / "built" / f"{snd['id']}.json"
    h = spec_hash(snd)
    if not force and cache.exists():
        prev = json.loads(cache.read_text(encoding="utf-8"))
        if prev["hash"] == h and all((AUDIO / snd["id"] / f"{s['key']}.mp3").exists() for s in prev["samples"]):
            return snd["id"], prev["samples"], "cached"
    t0 = time.time()
    samples = build_sound(snd)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"hash": h, "samples": samples}), encoding="utf-8")
    return snd["id"], samples, f"{len(samples)} clips in {time.time() - t0:.0f}s"


# ------------------------------------------------------------------ output

def credit_of(o):
    if o["source"] == "generated":
        return "CC0-1.0", "Earshot (generated)", None
    tag = {"commons": " (Wikimedia Commons)"}.get(o["source"], "")
    return o["license"], f"{o['author']}{tag}", o.get("url")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--cached-only", action="store_true",
                    help="skip Wikimedia Commons files that aren't downloaded yet (publish what's ready; rerun later for the rest)")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    lib = yaml.safe_load((ROOT / "data" / "library.yaml").read_text(encoding="utf-8"))
    sounds = lib["sounds"]
    ids = {s["id"] for s in sounds}
    fams = {f["id"] for f in lib["families"]}
    for s in sounds:
        assert s["family"] in fams, f"{s['id']}: unknown family {s['family']}"
        for c in s.get("confusions", []):
            assert c["id"] in ids, f"{s['id']}: confusion target {c['id']} not defined"

    todo = [s for s in sounds if s.get("samples")]
    if args.only:
        only = set(args.only.split(","))
        todo = [s for s in todo if s["id"] in only]

    if args.cached_only:
        def available(sp):
            return sp["src"] != "commons" or (CACHE / "commons" / commons_key(sp["file"])).exists()
        todo = [dict(s, samples=[sp for sp in s["samples"] if available(sp)]) for s in todo]
        todo = [s for s in todo if s["samples"]]

    # Remote hosts rate-limit parallel downloads, so fetch everything one at a time first.
    remote = [sp for s in todo for sp in s["samples"] if sp["src"] != "generated"]
    for n, sp in enumerate(remote, 1):
        try:
            before = time.time()
            fetch(sp)
            if time.time() - before > 0.5:  # a real download, not a cache hit
                log(f"  fetched {n}/{len(remote)}: {sp.get('file') or sp.get('inner') or sp.get('url')}")
                time.sleep(2)
        except Exception as e:
            log(f"  ! fetch failed: {sp.get('file') or sp.get('url')}: {e}")

    built = {f.stem: json.loads(f.read_text(encoding="utf-8"))["samples"] for f in (CACHE / "built").glob("*.json")}
    failures = []
    with cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(worker, s, args.force): s["id"] for s in todo}
        for fut in cf.as_completed(futs):
            sid = futs[fut]
            try:
                _, samples, msg = fut.result()
                built[sid] = samples
                if msg != "cached":
                    log(f"  {sid:24s} {msg}")
            except Exception as e:
                failures.append(sid)
                log(f"  ! {sid:22s} FAILED: {e}")

    conf = {s["id"]: {c["id"]: c["cue"] for c in s.get("confusions", [])} for s in sounds}
    for a in list(conf):
        for b, cue in list(conf[a].items()):
            conf[b].setdefault(a, cue)

    provenance, out = {}, []
    built_ids = {s["id"] for s in todo} if args.cached_only else None
    for s in sounds:
        samples = built.get(s["id"], []) if s.get("samples") else []
        if built_ids is not None and s["id"] not in built_ids and not args.only:
            samples = built.get(s["id"], []) if (CACHE / "built" / f"{s['id']}.json").exists() else []
        slim = []
        for smp in samples:
            creds = [credit_of(o) for o in smp["origin"]]
            provenance[f"{s['id']}/{smp['key']}"] = [{**o, "license": c[0], "credit": c[1]} for o, c in zip(smp["origin"], creds)]
            item = {k: v for k, v in smp.items() if k != "origin"}
            item.update(src=f"audio/{s['id']}/{smp['key']}.mp3", spec=f"audio/{s['id']}/{smp['key']}.webp",
                        license=" + ".join(sorted({c[0] for c in creds})), credit=sorted({c[1] for c in creds}))
            link = next((c[2] for c in creds if c[2]), None)
            if link:
                item["link"] = link
            slim.append(item)
        out.append({
            "id": s["id"], "name": s["name"], "family": s["family"], "sub": s.get("sub"),
            "aliases": s.get("aliases", []), "status": "ready" if slim else "wanted",
            "traits": {k: (v if isinstance(v, list) else [v]) for k, v in s["traits"].items()},
            "summary": s.get("summary", ""), "listen": s.get("listen", []), "note": s.get("note"),
            "confusions": [{"id": k, "cue": v} for k, v in conf[s["id"]].items()],
            "range": None, "samples": slim,
        })

    data = {"version": time.strftime("%Y-%m-%d"), "families": lib["families"], "questions": lib["questions"],
            "sources": lib["sources"], "sounds": out}
    (ROOT / "js" / "data.js").write_text("/* generated by tools/build.py, do not edit */\nwindow.EARSHOT = "
                                         + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
    (ROOT / "data" / "provenance.json").write_text(json.dumps(provenance, indent=1, ensure_ascii=False), encoding="utf-8")
    write_sources_md(provenance)
    ready = sum(1 for s in out if s["samples"])
    log(f"\n{ready} sounds with audio, {sum(len(s['samples']) for s in out)} clips, {len(out) - ready} listed without audio.")
    if failures:
        log(f"{len(failures)} failed: " + ", ".join(failures))
        sys.exit(1)


def write_sources_md(provenance):
    rows = {}
    for ref, entries in provenance.items():
        for o in entries:
            if o["source"] in ("commons", "url", "zip"):
                rows.setdefault((o["file"], o["url"]), (ref.split("/")[0], o))
    lines = [
        "# Sources and licenses",
        "",
        "Every clip in `audio/` is an excerpt of an openly licensed recording, or was generated in code",
        "(`tools/signals.py`, CC0). Excerpts are trimmed, faded, mixed to mono and loudness-normalized to",
        "about -20 LUFS. \"Under speech\", \"Far away\" and \"Through a phone\" versions add generated noise,",
        "filtering and reverb. Per-clip detail is in [`data/provenance.json`](data/provenance.json).",
        "",
        "| Sound | File | Author | License |",
        "| --- | --- | --- | --- |",
    ]
    for (name, url), (sid, o) in sorted(rows.items(), key=lambda kv: kv[1][0]):
        nm = name.replace("File:", "").replace("|", "\\|")
        lines.append(f"| {sid} | [{nm}]({url}) | {o['author'].replace('|', '/')} | {o['license']} |")
    (ROOT / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

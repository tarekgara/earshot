#!/usr/bin/env python3
"""Turn sources/candidates.json into `samples:` entries in data/library.yaml.

Keeps hand-written sample lists (sounds that already have `samples:` are left
alone unless --replace is given), skips anything whose license isn't CC0, public
domain, CC BY or CC BY-SA, and takes up to --per recordings per sound.
"""
import argparse
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

# candidate ids that differ from library ids
ALIAS = {
    "whistle-human": "whistling", "crowd-chatter": "crowd", "dog-bark": "bark", "dog-growl": "growl",
    "dog-whine": "whine", "bee": "bees", "wolf-howl": "wolf", "lion-roar": "lion", "stream": "brook",
    "fire-crackle": "fire", "door-open-close": "door", "keys-jingle": "keys", "faucet": "tap-water",
    "microwave-beep": "microwave", "kettle-whistle": "kettle", "vacuum-cleaner": "vacuum",
    "car-engine-idle": "car-idle", "car-pass-by": "car-pass", "airplane-jet": "airplane",
    "propeller-plane": "airplane", "fire-alarm": "smoke-alarm", "fan-hum": "fan",
}

VOCALSOUND = {
    "url": "https://www.dropbox.com/s/c5ace70qh1vbyzb/vs_release_16k.zip?dl=1",
    "page": "https://github.com/YuanGongND/vocalsound",
    "license": "CC BY-SA 4.0",
    "author": "VocalSound dataset (Gong, Yu and Glass, MIT)",
}
VS_CLASS = {"cough": "cough", "throat-clear": "throatclearing", "sneeze": "sneeze", "sniff": "sniff",
            "laugh": "laughter", "sigh": "sigh"}


# Files whose provenance the sourcing pass flagged as doubtful (commercial libraries re-uploaded, unclear origin).
DENY = re.compile(r"soundsnap|SBssa|mike[-_ ]?koenig|soundbible|soundgoats|freesoundslibrary|wilhelm", re.I)


def ok_license(lic):
    l = (lic or "").lower()
    if any(bad in l for bad in ("nc", "nd", "non-commercial", "noncommercial", "no deriv")):
        return False
    if "gfdl" in l and "cc" not in l:
        return False
    return any(good in l for good in ("cc0", "public domain", "pd", "cc by", "cc-by", "attribution"))


def window(notes):
    m = re.search(r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*s\b", notes or "")
    if not m:
        return {}
    a, b = float(m.group(1)), float(m.group(2))
    return {"start": a, "dur": round(min(b - a, 8.0), 2)} if b > a else {}


def to_samples(c, label):
    src = c["source"]
    if src == "commons":
        return [{"src": "commons", "file": c["title"], "label": label, **window(c.get("notes"))}]
    if src == "opengameart" and c.get("file_url") and c["file_url"].lower().endswith(".zip"):
        inners = c.get("inner") or []
        inners = inners if isinstance(inners, list) else [inners]
        return [{"src": "zip", "url": c["file_url"], "inner": i, "page": c.get("page_url"), "license": c["license"],
                 "author": c.get("author") or "Unknown author", "label": label} for i in inners]
    if src == "opengameart" and c.get("file_url"):
        return [{"src": "url", "url": c["file_url"], "page": c.get("page_url"), "license": c["license"],
                 "author": c.get("author") or "Unknown author", "label": label, **window(c.get("notes"))}]
    if src == "kenney" and c.get("file_url") and c.get("inner"):
        inners = c["inner"] if isinstance(c["inner"], list) else [c["inner"]]
        return [{"src": "zip", "url": c["file_url"], "inner": i, "page": c.get("page_url"), "license": "CC0",
                 "author": "Kenney (kenney.nl)", "label": Path(i).stem.replace("_", " ")} for i in inners]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=3)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--ids", help="comma-separated sound ids to (re)import; implies --replace for them")
    args = ap.parse_args()

    cands = json.loads((ROOT / "sources" / "candidates.json").read_text(encoding="utf-8"))
    pool = {}
    for cid, lst in cands.items():
        if cid.startswith("_"):
            continue
        pool.setdefault(ALIAS.get(cid, cid), []).extend(lst)

    path = ROOT / "data" / "library.yaml"
    text = path.read_text(encoding="utf-8")
    lib = yaml.safe_load(text)
    added = 0
    for snd in lib["sounds"]:
        sid = snd["id"]
        only = set(args.ids.split(",")) if args.ids else None
        if only is not None and sid not in only:
            continue
        replace = args.replace or only is not None
        if snd.get("samples") and not replace:
            continue
        if replace and any(x.get("src") == "generated" for x in snd.get("samples", [])):
            continue  # hand-made signal lists stay as they are
        picks = []
        if sid in VS_CLASS:  # two different speakers from VocalSound, one female, one male
            for sex, n in (("f", 7), ("m", 11)):
                picks.append({"src": "zip", "url": VOCALSOUND["url"], "inner": f"audio_16k/{sex}*_{VS_CLASS[sid]}.wav",
                              "pick": n, "page": VOCALSOUND["page"], "license": VOCALSOUND["license"],
                              "author": VOCALSOUND["author"], "label": "Woman" if sex == "f" else "Man"})
        ranked = sorted(pool.get(sid, []), key=lambda c: (c.get("isolated") is False, c["source"] == "opengameart"))
        for c in ranked:
            if len(picks) >= args.per:
                break
            if c["source"] in ("dataset", "freesound_lead") or not ok_license(c.get("license")):
                continue
            if DENY.search(" ".join(str(c.get(k) or "") for k in ("title", "author", "notes", "file_url"))):
                continue
            for smp in to_samples(c, f"Recording {len(picks) + 1}"):
                if len(picks) < args.per and smp not in picks:
                    picks.append(smp)
        if not picks:
            continue
        for k, smp in enumerate(picks, 1):
            if smp["label"].startswith("Recording"):
                smp["label"] = f"Recording {k}"
        block = "    samples:\n" + "".join(
            "      - " + json.dumps({k: v for k, v in smp.items() if v is not None}, ensure_ascii=False) + "\n" for smp in picks)
        i = text.index(f"  - id: {sid}\n")
        j = text.find("\n  - id: ", i + 5)
        j = len(text) if j < 0 else j
        entry = text[i:j]
        entry = re.sub(r"\n    samples:\n(      - .*\n?)+", "\n", entry) if replace else entry
        entry = entry.replace("    status: wanted\n", "")
        text = text[:i] + entry.rstrip("\n") + "\n" + block + text[j:]
        added += 1
    path.write_text(text, encoding="utf-8")
    yaml.safe_load(text)  # still valid
    print(f"added samples to {added} sounds")


if __name__ == "__main__":
    main()

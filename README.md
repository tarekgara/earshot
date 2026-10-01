# Earshot

Identify everyday sounds by ear. By Tarek Gara.

A cough or a throat clear? Rain or applause? A drill or a blender? Earshot is a reference library of real recordings, built to help you tell look-alike sounds apart.

- **Library**: sounds in nine families (voice and mouth, body and actions, animals, nature, home, vehicles, alarms and signals, tools and machines, impacts). Each has several different recordings, plus versions under speech, far away, and through a phone. Every clip is loudness-matched.
- **Not sure?**: a few questions answered by ear (what made it, how it plays out over time, its texture), each illustrated with sounds that still fit your earlier answers.
- **Quiz**: hear a sound, pick from look-alikes, learn how to tell them apart. Sounds you miss come back more often.
- **Compare**: put sounds side by side and flip between them with the number keys. You can drop in a recording of your own, too; it never leaves your device.

A sibling of [Instrumedia](https://github.com/tarekgara/instrumedia), which does the same for musical instruments.

## Use it

**Online:** the GitHub Pages site. The download icon in the sidebar saves everything for offline use.

**Offline:** download this repository as a ZIP, unzip it, and open `index.html`.

## How it's built

A static site with no front-end build step. `tools/build.py` turns [`data/library.yaml`](data/library.yaml) into `audio/` and `js/data.js`:

```
pip install numpy scipy pyyaml pillow pyloudnorm
python tools/build.py
```

It needs `ffmpeg` on PATH (or `FFMPEG=` set). Recordings are downloaded once, one at a time to respect rate limits, and cached in `.cache/`. Alarms, sirens and phone tones that follow published patterns are generated in [`tools/signals.py`](tools/signals.py).

### Adding a sound

Add an entry to `data/library.yaml` with its family, traits, listening cues and look-alikes, then list recordings under `samples:` (Wikimedia Commons titles, direct URLs, or files inside a zip). `python tools/commons_search.py "cough"` lists openly licensed Commons recordings. Sounds without recordings are kept as `status: wanted` so they still show up in the confusion map.

## Sources and licenses

Only openly licensed audio is used: CC0, public domain, CC BY, and CC BY-SA. Every file is credited in [SOURCES.md](SOURCES.md) and per clip in [`data/provenance.json`](data/provenance.json).

Code © Tarek Gara, MIT licensed (see `LICENSE`). Each recording keeps the license of its source.

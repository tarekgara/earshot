"""Generated reference signals.

These sounds are electronic by nature and follow published patterns (siren
sweeps, the T3 smoke-alarm cadence, telephone tone plans), so generating them
is both accurate and license-free.
"""

import numpy as np
from scipy import signal

SR = 44100
RNG = np.random.default_rng(11)


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def _sweep(freq):
    """Sine-rich tone from an instantaneous frequency curve (sirens are square-ish)."""
    ph = 2 * np.pi * np.cumsum(freq) / SR
    return np.sin(ph) + 0.3 * np.sin(3 * ph) + 0.12 * np.sin(5 * ph)


def _gate(x, on, off, ramp=0.004):
    """Switch a signal on/off in a repeating on/off cadence."""
    t = np.arange(len(x)) / SR
    period = on + off
    g = ((t % period) < on).astype(float)
    k = max(1, int(ramp * SR))
    g = np.convolve(g, np.ones(k) / k, mode="same")
    return x * g


def siren_wail(dur=6.0):
    """Slow up-down sweep, about 4 s per cycle (US 'wail')."""
    t = _t(dur)
    f = 650 + 650 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 4.2))
    return _sweep(f)


def siren_yelp(dur=4.0):
    """Fast sweep, about 3 cycles per second ('yelp')."""
    t = _t(dur)
    f = 700 + 700 * (0.5 - 0.5 * np.cos(2 * np.pi * t * 3.2))
    return _sweep(f)


def siren_hilo(dur=5.0):
    """Two alternating tones, the European 'hi-lo' / 'nee-naw'."""
    t = _t(dur)
    f = np.where((t % 1.2) < 0.6, 440 * 2 ** (7 / 12) * 1.2, 440 * 1.2)
    return _sweep(signal.savgol_filter(f, 401, 2))


def siren_passby(dur=8.0):
    """A wail heard from a passing vehicle: Doppler shift plus distance."""
    t = _t(dur)
    base = 650 + 650 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 4.2))
    v, c, d0 = 20.0, 343.0, 12.0  # 20 m/s past a listener 12 m from the road
    x = v * (t - dur / 2)
    radial = v * x / np.sqrt(x ** 2 + d0 ** 2)
    f = base * c / (c + radial)
    amp = 1 / np.sqrt(1 + (x / d0) ** 2)
    return _sweep(f) * amp


def smoke_alarm_t3(dur=7.0):
    """ISO 8201 / NFPA 72 'temporal three': three 0.5 s beeps, then 1.5 s silence."""
    t = _t(dur)
    x = np.sign(np.sin(2 * np.pi * 3150 * t)) * 0.6
    cycle = t % 4.0
    on = (cycle < 2.5) & ((cycle % 1.0) < 0.5)
    k = int(0.003 * SR)
    g = np.convolve(on.astype(float), np.ones(k) / k, mode="same")
    return signal.sosfilt(signal.butter(2, 6000, "lowpass", fs=SR, output="sos"), x * g)


def smoke_alarm_chirp(dur=3.0):
    """The single short chirp of a low battery."""
    t = _t(dur)
    x = np.sign(np.sin(2 * np.pi * 3150 * t)) * 0.6
    x[(t < 0.4) | (t > 0.45)] = 0
    return x


def reversing_beeper(dur=6.0):
    """Typical 1 kHz backup alarm, about one beep per second."""
    t = _t(dur)
    x = np.sign(np.sin(2 * np.pi * 1100 * t)) * 0.5
    x = signal.sosfilt(signal.butter(2, 4000, "lowpass", fs=SR, output="sos"), x)
    return _gate(x, 0.5, 0.5)


def buzzer(dur=1.6):
    t = _t(dur)
    x = signal.sawtooth(2 * np.pi * 110 * t) * 0.5 + np.sign(np.sin(2 * np.pi * 220 * t)) * 0.3
    x[t > 1.2] = 0
    return x


def dial_tone_us(dur=4.0):
    t = _t(dur)
    return np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 440 * t)


def dial_tone_uk(dur=4.0):
    t = _t(dur)
    return np.sin(2 * np.pi * 350 * t) + np.sin(2 * np.pi * 450 * t)


def busy_us(dur=4.0):
    t = _t(dur)
    return _gate(np.sin(2 * np.pi * 480 * t) + np.sin(2 * np.pi * 620 * t), 0.5, 0.5)


def ringback_us(dur=8.0):
    t = _t(dur)
    return _gate(np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 480 * t), 2.0, 4.0)


DTMF = {"1": (697, 1209), "2": (697, 1336), "3": (697, 1477), "4": (770, 1209), "5": (770, 1336),
        "6": (770, 1477), "7": (852, 1209), "8": (852, 1336), "9": (852, 1477), "0": (941, 1336)}


def dtmf(number="5551234", tone=0.12, gap=0.08):
    parts = []
    for d in number:
        lo, hi = DTMF[d]
        t = _t(tone)
        parts += [np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t), np.zeros(int(gap * SR))]
    return np.concatenate(parts)


def mains_hum(freq=50, dur=4.0):
    """Mains hum: the fundamental, strong odd harmonics, a little buzz."""
    t = _t(dur)
    x = np.zeros_like(t)
    for k, a in ((1, 1.0), (2, 0.35), (3, 0.6), (5, 0.3), (7, 0.15), (9, 0.08)):
        x += a * np.sin(2 * np.pi * freq * k * t + RNG.uniform(0, 6.28))
    x += 0.02 * RNG.standard_normal(len(t))
    return x


def chime(notes=(1318.5, 1760.0), step=0.13, decay=0.35):
    """A short two-note electronic notification chime."""
    out = np.zeros(int((step * len(notes) + 1.2) * SR))
    for k, f in enumerate(notes):
        t = _t(1.0)
        tone = (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-t / decay)
        i = int(k * step * SR)
        out[i:i + len(tone)] += tone
    return out


def ping(f=2093.0):
    t = _t(1.2)
    return (np.sin(2 * np.pi * f * t) + 0.15 * np.sin(2 * np.pi * 2.76 * f * t) * np.exp(-t / 0.05)) * np.exp(-t / 0.3)


def metronome(bpm=100, beats=8, accent_every=4):
    """Electronic metronome: short woodblock-like clicks, the first of each bar higher."""
    period = 60 / bpm
    out = np.zeros(int((period * beats + 0.3) * SR))
    t = _t(0.05)
    for b in range(beats):
        f = 1800 if b % accent_every == 0 else 1200
        click = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.008)
        i = int(b * period * SR)
        out[i:i + len(click)] += click
    return out


PRESETS = {
    "chime": chime, "chime-3": lambda: chime((1046.5, 1318.5, 1568.0)), "ping": ping,
    "metronome": metronome, "metronome-fast": lambda: metronome(160, 12),
    "siren-wail": siren_wail, "siren-yelp": siren_yelp, "siren-hilo": siren_hilo, "siren-passby": siren_passby,
    "smoke-t3": smoke_alarm_t3, "smoke-chirp": smoke_alarm_chirp, "reversing": reversing_beeper,
    "buzzer": buzzer, "dial-us": dial_tone_us, "dial-uk": dial_tone_uk, "busy-us": busy_us,
    "ringback-us": ringback_us, "dtmf": dtmf, "hum-50": lambda: mains_hum(50), "hum-60": lambda: mains_hum(60),
}


def render(preset):
    x = PRESETS[preset]()
    peak = np.max(np.abs(x)) or 1.0
    return (x / peak * 0.9).astype(np.float32)

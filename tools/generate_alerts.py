"""
Pre-generates every alert clip as a WAV, plus a short attention chime.

Run once on your Windows machine, then copy the output WAVs into
    app/src/main/res/raw/

Why pre-generate: calling a TTS engine at alert time costs 300-800ms, which is
larger than the entire end-to-end latency budget for a collision warning. These
files get loaded into SoundPool at startup and play in under ~30ms.

Install:
    pip install gTTS pydub numpy
    (pydub needs ffmpeg on PATH: https://ffmpeg.org/download.html)

Usage:
    python generate_alerts.py
"""

import os
import numpy as np
from gtts import gTTS
from pydub import AudioSegment

OUT = os.path.join(os.path.dirname(__file__), "raw")
os.makedirs(OUT, exist_ok=True)

# audio_key -> (english phrase, urdu phrase)
# Keep every phrase SHORT. Under ~800ms. "Brake" not "there is a vehicle ahead".
PHRASES = {
    "brake":         ("Brake",              "بریک"),
    "vehicle_close": ("Object ahead",       "آگے رکاوٹ"),
    "lane":          ("Solid line",         "لائن"),
    "slow_down":     ("Slow down",          "رفتار کم کریں"),
    "obstacle":      ("Obstacle",           "رکاوٹ"),
}

TARGET_DBFS = -6.0   # loud enough to cut through road noise


def normalise(seg: AudioSegment) -> AudioSegment:
    seg = seg.set_frame_rate(44100).set_channels(1)
    return seg.apply_gain(TARGET_DBFS - seg.dBFS)


def make_speech(text: str, lang: str, path: str, speed_up: float = 1.15):
    tmp = path + ".mp3"
    gTTS(text=text, lang=lang, slow=False).save(tmp)
    seg = AudioSegment.from_mp3(tmp)
    os.remove(tmp)

    # Speed up slightly: urgent cues read better fast, and it shortens playback.
    seg = seg._spawn(seg.raw_data, overrides={
        "frame_rate": int(seg.frame_rate * speed_up)
    }).set_frame_rate(44100)

    # padding MUST be < silence_len or pydub raises InvalidDuration. Its default
    # is 100ms, which is larger than the 60ms window we want, so set it here.
    # A little padding is kept deliberately: trimming hard to the waveform clips
    # the attack off plosives like the "B" in "Brake".
    seg = normalise(seg).strip_silence(silence_len=60, silence_thresh=-45,
                                       padding=20)
    seg.export(path, format="wav")
    print(f"  {os.path.basename(path):28s} {len(seg):5d} ms")
    if len(seg) > 1100:
        print("    ^ WARNING: over 1.1s. Shorten this phrase.")


def make_chime(path: str, ms: int = 110, freq: int = 1180):
    """Short rising sine burst. Perceived faster than speech onset."""
    sr = 44100
    n = int(sr * ms / 1000)
    t = np.linspace(0, ms / 1000, n, endpoint=False)
    # slight upward sweep reads as 'attention' rather than 'error'
    f = np.linspace(freq, freq * 1.25, n)
    wave = np.sin(2 * np.pi * np.cumsum(f) / sr)
    env = np.minimum(1.0, np.linspace(0, 6, n)) * np.linspace(1.0, 0.0, n) ** 0.6
    pcm = (wave * env * 26000).astype(np.int16)
    AudioSegment(pcm.tobytes(), frame_rate=sr, sample_width=2, channels=1) \
        .export(path, format="wav")
    print(f"  {os.path.basename(path):28s} {ms:5d} ms")


if __name__ == "__main__":
    print("Generating alert audio ->", OUT)
    make_chime(os.path.join(OUT, "chime.wav"))
    for key, (en, ur) in PHRASES.items():
        make_speech(en, "en", os.path.join(OUT, f"en_{key}.wav"))
        make_speech(ur, "ur", os.path.join(OUT, f"ur_{key}.wav"))
    print("\nDone. Copy the WAVs into app/src/main/res/raw/")
    print("Filenames must stay lowercase with underscores only.")

"""
trip_summary.py — end-of-trip driving score + AI coaching notes.

Deterministic score (0-10, 1dp) is computed locally from warning counts, so
it's always available and reproducible even offline. Gemini is only used to
turn the already-computed numbers into a few short coaching bullets — it
never invents or overrides the counts or the score.

Calls the Gemini REST API directly with `requests` (not the google-
generativeai SDK) because the SDK's Gemini-capable versions require Python
3.9+, and carlaenv is pinned to 3.8 for the CARLA wheel.

    pip install requests python-dotenv

Set GEMINI_API_KEY in your environment, or put it in a .env file next to
this script (see .env.example). Without a key, or if the request fails or
times out, recommendations fall back to a small canned tip list instead of
failing the trip summary.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from adas_core import AlertType

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

HISTORY_PATH = Path(__file__).with_name("trip_history.json")
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_URL = (f"https://generativelanguage.googleapis.com/v1beta/models/"
              f"{GEMINI_MODEL}:generateContent")
GEMINI_TIMEOUT_S = 10

# Points deducted per event, before duration normalization.
_PENALTY_URGENT = 1.0
_PENALTY_COLLISION = 0.5
# Speeding sits between a lane violation and a collision warning: it is a
# deliberate, sustained choice rather than a momentary drift, but it is not
# yet an imminent-impact event.
_PENALTY_SPEED_LIMIT = 0.4
_PENALTY_SOLID_LINE = 0.3
_PENALTY_OTHER = 0.1

# Penalties are normalized against this trip length so a long trip isn't
# unfairly punished for simply lasting longer than a short one.
_REFERENCE_MINUTES = 5.0


@dataclass
class TripStats:
    start_time: float = field(default_factory=time.time)
    total_warnings: int = 0
    solid_line_warnings: int = 0
    collision_warnings: int = 0
    urgent_collision_warnings: int = 0
    speed_limit_warnings: int = 0

    def record(self, alert_type: AlertType) -> None:
        self.total_warnings += 1
        if alert_type is AlertType.LANE_SOLID:
            self.solid_line_warnings += 1
        elif alert_type is AlertType.COLLISION_WARN:
            self.collision_warnings += 1
        elif alert_type is AlertType.COLLISION_URGENT:
            self.urgent_collision_warnings += 1
        elif alert_type is AlertType.SPEED_LIMIT:
            self.speed_limit_warnings += 1

    def other_warnings(self) -> int:
        return (self.total_warnings - self.solid_line_warnings
                - self.collision_warnings - self.urgent_collision_warnings
                - self.speed_limit_warnings)

    def duration_s(self) -> float:
        return max(time.time() - self.start_time, 1.0)


def compute_score(stats: TripStats) -> float:
    minutes = stats.duration_s() / 60.0
    blocks = max(minutes / _REFERENCE_MINUTES, 1.0)

    raw_penalty = (
        stats.urgent_collision_warnings * _PENALTY_URGENT
        + stats.collision_warnings * _PENALTY_COLLISION
        + stats.speed_limit_warnings * _PENALTY_SPEED_LIMIT
        + stats.solid_line_warnings * _PENALTY_SOLID_LINE
        + stats.other_warnings() * _PENALTY_OTHER
    )
    return round(max(0.0, 10.0 - raw_penalty / blocks), 1)


def _fallback_recommendations(stats: TripStats) -> list[str]:
    tips = []
    if stats.urgent_collision_warnings:
        tips.append("You had urgent collision warnings - increase following "
                     "distance so you have more time to react.")
    if stats.collision_warnings:
        tips.append("Keep more space from the vehicle ahead to cut down on "
                     "collision warnings.")
    if stats.speed_limit_warnings:
        tips.append("You exceeded the posted speed limit - ease off and check "
                     "the limit whenever you pass a speed limit sign.")
    if stats.solid_line_warnings:
        tips.append("Stay centered in your lane - you crossed solid lane "
                     "markings during this trip.")
    if not tips:
        tips.append("Good trip - keep up the safe following distance, steady "
                     "speed, and lane discipline.")
    return tips


def _build_prompt(stats: TripStats, score: float) -> str:
    return (
        "You are a driving coach reviewing telemetry from one trip in a "
        "driver-assistance system. The score below is already computed — "
        "do not restate, recompute, or contradict it.\n"
        f"Score (out of 10): {score}\n"
        f"Total warnings: {stats.total_warnings}\n"
        f"Solid line crossing warnings: {stats.solid_line_warnings}\n"
        f"Collision warnings: {stats.collision_warnings}\n"
        f"Urgent collision warnings: {stats.urgent_collision_warnings}\n"
        f"Speed limit (over-speed) warnings: {stats.speed_limit_warnings}\n\n"
        "Give 2-4 short, specific, actionable driving recommendations as a "
        "plain bullet list — one per line, no numbering, no preamble, no "
        "closing remarks."
    )


def generate_recommendations(stats: TripStats, score: float) -> list[str]:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return _fallback_recommendations(stats)

    payload = {
        "contents": [{"parts": [{"text": _build_prompt(stats, score)}]}]
    }
    try:
        # Key goes in a header, not the URL/query string, so it never ends
        # up in exception messages, proxy logs, or requests' own repr().
        resp = requests.post(
            GEMINI_URL,
            headers={"x-goog-api-key": api_key},
            json=payload,
            timeout=GEMINI_TIMEOUT_S,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
        bullets = [line.strip("-* ").strip()
                   for line in text.splitlines() if line.strip()]
        return bullets[:4] or _fallback_recommendations(stats)
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "?"
        print(f"[trip_summary] Gemini call failed (HTTP {status}), "
              "using fallback tips")
        return _fallback_recommendations(stats)
    except Exception as exc:
        print(f"[trip_summary] Gemini call failed ({type(exc).__name__}), "
              "using fallback tips")
        return _fallback_recommendations(stats)


def build_report(stats: TripStats) -> dict:
    score = compute_score(stats)
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_s": round(stats.duration_s(), 1),
        "total_warnings": stats.total_warnings,
        "solid_line_warnings": stats.solid_line_warnings,
        "collision_warnings": stats.collision_warnings,
        "urgent_collision_warnings": stats.urgent_collision_warnings,
        "speed_limit_warnings": stats.speed_limit_warnings,
        "score": score,
        "recommendations": generate_recommendations(stats, score),
    }


def print_report(report: dict) -> None:
    print("\n===== TRIP SUMMARY =====")
    print(f"Duration: {report['duration_s']:.0f}s")
    print(f"Total warnings: {report['total_warnings']}")
    print(f"  Solid line:       {report['solid_line_warnings']}")
    print(f"  Collision:        {report['collision_warnings']}")
    print(f"  Urgent collision: {report['urgent_collision_warnings']}")
    # .get: history entries written before speed limit tracking lack this key.
    print(f"  Speed limit:      {report.get('speed_limit_warnings', 0)}")
    print(f"Driving score: {report['score']}/10")
    print("Recommendations:")
    for tip in report["recommendations"]:
        print(f"  - {tip}")
    print("=========================\n")


def save_report(report: dict, path: Path = HISTORY_PATH) -> None:
    history = []
    if path.exists():
        try:
            history = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            history = []
    history.append(report)
    path.write_text(json.dumps(history, indent=2))

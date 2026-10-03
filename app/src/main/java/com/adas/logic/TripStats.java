package com.adas.logic;

import android.os.SystemClock;

import java.util.ArrayList;
import java.util.List;

/**
 * End-of-trip driving score and coaching notes.
 *
 * MIRRORS tools/trip_summary.py. The penalty weights, the 5-minute reference
 * block and the score formula must stay identical to that file, so a trip scored
 * on the phone and the same trip scored at a desk agree.
 *
 * WHY THE SCORE IS COMPUTED LOCALLY
 *     It is deterministic arithmetic over warning counts, so it is always
 *     available — offline, mid-drive, with no key and no network. The Python
 *     version uses Gemini only to REPHRASE the already-computed numbers into
 *     coaching bullets; it never lets the model invent or override a count.
 *
 * WHY THERE IS NO GEMINI CALL HERE
 *     Reaching Gemini from the phone would mean shipping an API key inside the
 *     APK, where anyone can extract it, and depending on mobile data in a moving
 *     car for a screen that appears the moment you park. The Python's own offline
 *     fallback tips are keyed off exactly the same counts, so the advice is the
 *     same advice. Port that, not the network call.
 */
public class TripStats {

    // Points deducted per event, before duration normalisation.
    private static final float PENALTY_URGENT = 1.0f;
    private static final float PENALTY_COLLISION = 0.5f;
    private static final float PENALTY_SPEED_LIMIT = 0.4f;
    private static final float PENALTY_SOLID_LINE = 0.3f;
    private static final float PENALTY_OTHER = 0.1f;

    /**
     * Penalties are normalised against this trip length, so a long drive is not
     * punished simply for lasting longer than a short one.
     */
    private static final float REFERENCE_MINUTES = 5.0f;

    private final long startedAtMs = SystemClock.elapsedRealtime();
    private final long startedWallMs = System.currentTimeMillis();

    public int totalWarnings = 0;
    public int solidLineWarnings = 0;
    public int collisionWarnings = 0;
    public int urgentCollisionWarnings = 0;
    public int speedLimitWarnings = 0;

    public synchronized void record(AlertType type) {
        totalWarnings++;
        if (type == AlertType.LANE_SOLID) solidLineWarnings++;
        else if (type == AlertType.COLLISION_WARN) collisionWarnings++;
        else if (type == AlertType.COLLISION_URGENT) urgentCollisionWarnings++;
        else if (type == AlertType.SPEED_LIMIT) speedLimitWarnings++;
    }

    public synchronized int otherWarnings() {
        return totalWarnings - solidLineWarnings - collisionWarnings
                - urgentCollisionWarnings - speedLimitWarnings;
    }

    /** Elapsed wall time, floored at 1 s so a zero-length trip cannot divide by zero. */
    public float durationS() {
        return Math.max((SystemClock.elapsedRealtime() - startedAtMs) / 1000f, 1f);
    }

    public long startedWallMs() { return startedWallMs; }

    /** 0.0 to 10.0, one decimal place. */
    public synchronized float score() {
        float minutes = durationS() / 60f;
        float blocks = Math.max(minutes / REFERENCE_MINUTES, 1f);

        float rawPenalty =
                  urgentCollisionWarnings * PENALTY_URGENT
                + collisionWarnings * PENALTY_COLLISION
                + speedLimitWarnings * PENALTY_SPEED_LIMIT
                + solidLineWarnings * PENALTY_SOLID_LINE
                + otherWarnings() * PENALTY_OTHER;

        float s = Math.max(0f, 10f - rawPenalty / blocks);
        return Math.round(s * 10f) / 10f;
    }

    /**
     * Coaching notes keyed off the counts. Ported from
     * trip_summary.py :: _fallback_recommendations.
     */
    public synchronized List<String> recommendations() {
        List<String> tips = new ArrayList<>();
        if (urgentCollisionWarnings > 0) {
            tips.add("You had urgent collision warnings - increase following "
                    + "distance so you have more time to react.");
        }
        if (collisionWarnings > 0) {
            tips.add("Keep more space from the vehicle ahead to cut down on "
                    + "collision warnings.");
        }
        if (speedLimitWarnings > 0) {
            tips.add("You exceeded the posted speed limit - ease off and check "
                    + "the limit whenever you pass a speed limit sign.");
        }
        if (solidLineWarnings > 0) {
            tips.add("Stay centered in your lane - you crossed solid lane "
                    + "markings during this trip.");
        }
        if (tips.isEmpty()) {
            tips.add("Good trip - keep up the safe following distance, steady "
                    + "speed, and lane discipline.");
        }
        return tips;
    }

    /** A one-word verdict to headline the summary. */
    public String grade() {
        float s = score();
        if (s >= 9f) return "EXCELLENT";
        if (s >= 7.5f) return "GOOD";
        if (s >= 6f) return "FAIR";
        return "NEEDS WORK";
    }
}

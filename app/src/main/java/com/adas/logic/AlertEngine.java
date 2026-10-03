package com.adas.logic;

import com.adas.track.Track;

import java.util.ArrayList;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * The part of this project that actually determines whether it is usable.
 *
 * ============================================================================
 * MIRRORS tools/adas_core.py :: AlertEngine — LINE FOR LINE.
 *
 * Every constant, every threshold and every branch order below has a twin in
 * that file. Tune in Python against CARLA ground truth, then copy the numbers
 * here. If the two ever diverge, the simulator results stop saying anything
 * about the app that actually ships.
 *
 * There is a parity test that enforces this: tools/gen_parity_golden.py writes
 * a golden trace from the Python engine, and AlertEngineParityTest replays the
 * same inputs through this class and asserts the alerts match frame for frame.
 * Run it after touching either side.
 * ============================================================================
 *
 * Raw threshold crossing produces an unusable system: boxes jitter, detections
 * flicker, and the driver is alerted five times a kilometre for nothing. Four
 * mechanisms guard against that:
 *
 *   1. TEMPORAL CONFIRMATION - a condition must hold for N consecutive
 *      evaluations before it can fire.
 *   2. HYSTERESIS            - the clear threshold is looser than the set
 *      threshold, so a value sitting on the boundary does not chatter.
 *   3. COOLDOWN              - after firing, that alert is suppressed for a
 *      fixed period regardless of conditions.
 *   4. PRIORITY ARBITRATION  - at most one alert sounds; a higher-priority
 *      alert pre-empts and cancels lower ones.
 *
 * Plus context gating: below MIN_SPEED_MS nothing VEHICLE-related fires, because
 * stop-start traffic would otherwise alert continuously. Obstacles get their own,
 * far lower gate — see OBSTACLE_MIN_SPEED_MS.
 */
public class AlertEngine {

    // ---- tunables (start here when reducing false alarms) ----
    public static float TTC_URGENT_SET = 1.6f;
    public static float TTC_URGENT_CLEAR = 2.2f;
    public static float TTC_WARN_SET = 2.8f;
    public static float TTC_WARN_CLEAR = 3.6f;

    public static float SPEED_OVER_SET = 1.10f;   // 10% over limit
    public static float SPEED_OVER_CLEAR = 1.03f;

    /** ~15 km/h. Gate for VEHICLE collision alerts only. */
    public static float MIN_SPEED_MS = 4.2f;

    /**
     * ~3 km/h. Obstacles get a far lower gate than vehicles.
     *
     * The 15 km/h floor exists to stop vehicle warnings firing continuously in
     * stop-start traffic, where you are always a few metres from the car in
     * front. A tree does not behave that way: if you are closing on one at
     * walking pace you still want to be told, and the corridor test already
     * excludes anything at the roadside.
     */
    public static float OBSTACLE_MIN_SPEED_MS = 0.9f;

    // --- obstacle trigger: STOPPING DISTANCE, not time-to-contact ------------
    //
    // Time-to-contact for an obstacle was computed by differentiating the depth
    // measurement. Depth is noisy, and the derivative of a noisy signal is far
    // noisier still, so TTC flickered across the threshold and the warning
    // re-fired over and over. That was the real fault, not the threshold value.
    //
    // An obstacle is STATIC. A tree is not going anywhere, so the closing speed
    // simply IS our own speed - a number we measure directly and accurately.
    // No differentiation, no noise.
    //
    //     warn_distance = v * t_reaction  +  v^2 / (2 * decel)
    public static float REACT_WARN_S = 1.2f;
    public static float DECEL_WARN_MS2 = 4.5f;     // comfortable braking
    public static float REACT_URGENT_S = 0.6f;
    public static float DECEL_URGENT_MS2 = 7.0f;   // hard braking

    // Floors, so the speed formula never delays a warning past a sensible
    // distance. At 5 km/h the formula alone asks for barely 4 m, which means an
    // object six metres ahead is silently ignored while you roll towards it.
    public static float WARN_FLOOR_M = 12.0f;
    public static float URGENT_FLOOR_M = 6.0f;

    public static int OBSTACLE_CONFIRM = 3;          // frames below threshold before speaking
    public static int OBSTACLE_ABSENT_FRAMES = 12;   // frames with no obstacle before the latch clears
    public static float OBSTACLE_RELEASE_M = 6.0f;   // backing off this far re-arms the same object

    // Approach is judged over a WINDOW, not frame to frame. Depth jitters by
    // about a metre, so a single-frame comparison decides you are retreating
    // several times a second and resets the confirmation counters each time.
    public static float OBSTACLE_APPROACH_M = 0.8f;      // net closing over the window
    public static int OBSTACLE_APPROACH_WINDOW = 15;     // frames (~0.75 s)

    public static int CONFIRM_FRAMES = 3;
    public static int SPEED_CONFIRM_FRAMES = 25;  // speeding must persist ~1.5s
    /** Frames of history before a closing-speed estimate means anything. */
    public static int MIN_HITS_FOR_TTC = 6;
    public static int OBSTACLE_CONFIRM_FRAMES = 5;

    private final Map<AlertType, Integer> consecutive = new EnumMap<>(AlertType.class);
    private final Map<AlertType, Boolean> latched = new EnumMap<>(AlertType.class);
    private final Map<AlertType, Long> lastFiredMs = new EnumMap<>(AlertType.class);

    private long suppressUntilMs = 0L;   // global gate, e.g. during a maps-instructed turn

    // Per-obstacle latch. Stage 0 = nothing said, 1 = warned, 2 = urgent.
    // We only ever escalate; the same tree is never announced twice.
    private int obsStage = 0;
    private int obsAbsent = 0;
    private float obsMinDist = Float.POSITIVE_INFINITY;
    private int obsWarnCount = 0;
    private int obsUrgentCount = 0;
    private float obsSmooth = Float.NaN;
    private final List<Float> obsHistory = new ArrayList<>();
    private String obsReason = "";

    /**
     * Populated every frame so the caller can show exactly why the obstacle path
     * did or did not speak. Silence has many possible causes, and guessing
     * between them wastes far more time than reporting them.
     */
    public final ObstacleDebug obsDebug = new ObstacleDebug();

    public static class ObstacleDebug {
        public float distM = Float.NaN;
        public float speedMs = 0f;
        public int stage = 0;
        public boolean approaching = false;
        public float closedM = 0f;
        public float warnD = 0f;
        public float urgentD = 0f;
        public int warnCount = 0;
        public int urgentCount = 0;
        public String reason = "";
    }

    public AlertEngine() {
        for (AlertType t : AlertType.values()) {
            consecutive.put(t, 0);
            latched.put(t, false);
            lastFiredMs.put(t, 0L);
        }
    }

    /** Result of one evaluation cycle. */
    public static class Decision {
        public final AlertType fire;        // null = stay silent
        public final Track subject;         // the track that triggered it, may be null
        public final String reason;         // for the session log

        Decision(AlertType fire, Track subject, String reason) {
            this.fire = fire; this.subject = subject; this.reason = reason;
        }
    }

    private static final Decision SILENT = new Decision(null, null, "silent");
    private static final Decision SUPPRESSED = new Decision(null, null, "suppressed");

    /**
     * Backwards-compatible overload for callers that have no obstacle or lane
     * input yet (ReplayRunner on legacy recordings). Equivalent to passing no
     * obstacle and no lane crossing.
     */
    public Decision evaluate(List<Track> tracks,
                             int frameWidth,
                             float egoSpeedMs,
                             float speedLimitMs,
                             float laneOffsetUnused,
                             long nowMs) {
        return evaluate(tracks, frameWidth, egoSpeedMs, speedLimitMs, nowMs,
                Float.NaN, null);
    }

    /**
     * @param tracks        current tracks
     * @param frameWidth    analysed frame width in px
     * @param egoSpeedMs    own speed from GPS, m/s
     * @param speedLimitMs  current speed limit, m/s (&lt;=0 disables the check)
     * @param nowMs         SystemClock.elapsedRealtime()
     * @param obstacleDistM distance to the nearest unnamed obstacle in the
     *                      corridor, or Float.NaN when nothing credible is there
     * @param laneCross     "left"/"right" on the single frame a solid line is
     *                      crossed, otherwise null
     */
    public Decision evaluate(List<Track> tracks,
                             int frameWidth,
                             float egoSpeedMs,
                             float speedLimitMs,
                             long nowMs,
                             float obstacleDistM,
                             String laneCross) {

        if (nowMs < suppressUntilMs) {
            decayAll();
            return SUPPRESSED;
        }

        Track worst = null;
        float worstTtc = Float.POSITIVE_INFINITY;

        boolean moving = egoSpeedMs >= MIN_SPEED_MS;

        if (moving) {
            for (Track t : tracks) {
                if (!t.isConfirmed()) continue;
                if (Float.isNaN(t.distanceM)) continue;
                if (!DistanceEstimator.inEgoCorridor(t.box, frameWidth, t.distanceM)) continue;
                // Ignore traffic coming the other way. We will not rear-end a
                // vehicle that is in the opposite lane heading towards us.
                if (DistanceEstimator.isOncoming(t.closingSpeedMs, egoSpeedMs)) continue;
                // A closing-speed estimate needs a few frames of history before
                // it means anything. Without this, a car that has just appeared
                // produces a wild TTC on its second frame.
                if (t.hits < MIN_HITS_FOR_TTC) continue;
                if (t.ttcS < worstTtc) { worstTtc = t.ttcS; worst = t; }
            }
        }

        // NOTE (mirrors the Python): OBSTACLE_AHEAD's counter is cleared every
        // frame, so that type can never win the priority loop below. Obstacles
        // speak through the COLLISION_* early-return path instead, which is what
        // gives them the per-object latch. Kept identical to adas_core.py on
        // purpose — do not "fix" one side alone.
        reset(AlertType.OBSTACLE_AHEAD);

        // Obstacles are judged on stopping distance and latched per object.
        AlertType obsAlert = obstacleAlert(obstacleDistM, egoSpeedMs);
        obsReason = "";
        if (obsAlert != null) {
            obsReason = String.format(Locale.US,
                    "obstacle at %.1f m, stopping needs %.1f m at %.0f km/h",
                    obstacleDistM, warnDistance(egoSpeedMs), egoSpeedMs * 3.6f);
        }

        // Short-circuit is load-bearing: when not moving, hyst() must NOT be
        // called, so the latch keeps its previous state. Python relies on the
        // same `and` short-circuit.
        boolean urgent = moving && hyst(AlertType.COLLISION_URGENT,
                worstTtc, TTC_URGENT_SET, TTC_URGENT_CLEAR);
        boolean warn = moving && hyst(AlertType.COLLISION_WARN,
                worstTtc, TTC_WARN_SET, TTC_WARN_CLEAR);

        // Crossing a solid line is a single EVENT, not a sustained condition, so
        // it bypasses the frame-confirmation counter - by the time you confirmed
        // it over five frames the wheels are already in the next lane. The
        // detector does its own multi-frame voting before reporting a crossing.
        boolean lane = (laneCross != null && !laneCross.isEmpty()) && moving;
        if (lane) {
            consecutive.put(AlertType.LANE_SOLID, CONFIRM_FRAMES);
        } else {
            reset(AlertType.LANE_SOLID);
        }

        boolean speeding = false;
        if (speedLimitMs > 0f && egoSpeedMs > 0f) {
            speeding = hystAbove(AlertType.SPEED_LIMIT, egoSpeedMs / speedLimitMs,
                    SPEED_OVER_SET, SPEED_OVER_CLEAR);
        } else {
            reset(AlertType.SPEED_LIMIT);
        }

        tick(AlertType.COLLISION_URGENT, urgent);
        tick(AlertType.COLLISION_WARN, warn && !urgent);
        tick(AlertType.SPEED_LIMIT, speeding);

        // An obstacle alert bypasses the frame counters and cooldowns. The latch
        // has already guaranteed it speaks at most once per object, so a cooldown
        // on top could only ever silence a genuine SECOND obstacle.
        // A vehicle URGENT still outranks an obstacle WARN.
        if (obsAlert == AlertType.COLLISION_URGENT) {
            lastFiredMs.put(obsAlert, nowMs);
            return new Decision(obsAlert, null, obsReason);
        }
        if (obsAlert == AlertType.COLLISION_WARN) {
            boolean vehUrgentReady =
                    consecutive.get(AlertType.COLLISION_URGENT) >= CONFIRM_FRAMES
                    && (nowMs - lastFiredMs.get(AlertType.COLLISION_URGENT))
                        >= AlertType.COLLISION_URGENT.cooldownMs;
            if (!vehUrgentReady) {
                lastFiredMs.put(obsAlert, nowMs);
                return new Decision(obsAlert, null, obsReason);
            }
        }

        // Priority arbitration: first ready alert in ordinal order wins.
        // Declaration order in AlertType IS priority order (see that file).
        for (AlertType t : AlertType.values()) {
            int need;
            if (t == AlertType.SPEED_LIMIT) need = SPEED_CONFIRM_FRAMES;
            else if (t == AlertType.OBSTACLE_AHEAD) need = OBSTACLE_CONFIRM_FRAMES;
            else need = CONFIRM_FRAMES;

            if (consecutive.get(t) >= need && offCooldown(t, nowMs)) {
                lastFiredMs.put(t, nowMs);
                consecutive.put(t, 0);
                return new Decision(t, worst,
                        describe(t, worst, worstTtc, egoSpeedMs, speedLimitMs, laneCross));
            }
        }
        return SILENT;
    }

    /** Distance needed to stop comfortably, plus reaction time. */
    public static float warnDistance(float speedMs) {
        float d = speedMs * REACT_WARN_S + (speedMs * speedMs) / (2f * DECEL_WARN_MS2);
        return Math.max(WARN_FLOOR_M, d);
    }

    /** Distance needed to stop under hard braking. Below this you are late. */
    public static float urgentDistance(float speedMs) {
        float d = speedMs * REACT_URGENT_S + (speedMs * speedMs) / (2f * DECEL_URGENT_MS2);
        return Math.max(URGENT_FLOOR_M, d);
    }

    /**
     * Decide what to say about an unnamed obstacle, and say it ONCE.
     *
     * The latch is the important part. A time-based cooldown re-fires while you
     * are still approaching the same object, which is what made the old behaviour
     * so noisy. Instead we remember what we have already said about THIS obstacle
     * and only ever escalate - warn, then urgent, then nothing more until the
     * object is gone or we have backed away from it.
     *
     * @param obstacleDistM Float.NaN means "no obstacle detected this frame"
     * @return COLLISION_URGENT, COLLISION_WARN, or null
     */
    private AlertType obstacleAlert(float obstacleDistM, float egoSpeedMs) {
        obsDebug.distM = obstacleDistM;
        obsDebug.speedMs = egoSpeedMs;
        obsDebug.stage = obsStage;
        obsDebug.reason = "";

        if (Float.isNaN(obstacleDistM)) {
            obsDebug.reason = "no obstacle detected";
            obsAbsent++;
            if (obsAbsent >= OBSTACLE_ABSENT_FRAMES) {
                obsStage = 0;
                obsMinDist = Float.POSITIVE_INFINITY;
                obsWarnCount = 0;
                obsUrgentCount = 0;
                obsSmooth = Float.NaN;
                obsHistory.clear();
            }
            return null;
        }
        obsAbsent = 0;

        // Moving well clear of it means the next approach is a fresh event -
        // whether it is a different object or the same one on a second pass.
        if (obstacleDistM > obsMinDist + OBSTACLE_RELEASE_M) {
            obsStage = 0;
            obsMinDist = obstacleDistM;
            obsWarnCount = 0;
            obsUrgentCount = 0;
            obsHistory.clear();
        }
        obsMinDist = Math.min(obsMinDist, obstacleDistM);

        if (egoSpeedMs < OBSTACLE_MIN_SPEED_MS) {
            obsDebug.reason = String.format(Locale.US, "too slow (%.1f < %.1f km/h)",
                    egoSpeedMs * 3.6f, OBSTACLE_MIN_SPEED_MS * 3.6f);
            return null;
        }

        // Only warn about something we are actually CLOSING ON. Without this,
        // reversing away from a wall re-arms the latch and then immediately fires
        // again as the distance passes back through the threshold - a warning
        // about an object we are escaping.
        //
        // Judged on a smoothed distance, because raw depth jitters by tens of
        // centimetres and a single-frame comparison would call that a retreat.
        obsSmooth = Float.isNaN(obsSmooth)
                ? obstacleDistM
                : 0.4f * obstacleDistM + 0.6f * obsSmooth;
        obsHistory.add(obsSmooth);
        if (obsHistory.size() > OBSTACLE_APPROACH_WINDOW) obsHistory.remove(0);

        boolean approaching = obsHistory.size() >= 3
                && obsSmooth < obsHistory.get(0) - OBSTACLE_APPROACH_M;
        obsDebug.approaching = approaching;
        obsDebug.closedM = obsHistory.isEmpty() ? 0f : (obsHistory.get(0) - obsSmooth);

        if (!approaching) {
            obsDebug.reason = "not closing (distance steady or increasing)";
            obsWarnCount = 0;
            obsUrgentCount = 0;
            return null;
        }

        float dWarn = warnDistance(egoSpeedMs);
        float dUrgent = urgentDistance(egoSpeedMs);

        // A short confirmation still applies, so one bad depth frame cannot
        // trigger anything on its own.
        obsUrgentCount = (obstacleDistM <= dUrgent) ? obsUrgentCount + 1 : 0;
        obsWarnCount = (obstacleDistM <= dWarn) ? obsWarnCount + 1 : 0;

        obsDebug.warnD = dWarn;
        obsDebug.urgentD = dUrgent;
        obsDebug.warnCount = obsWarnCount;
        obsDebug.urgentCount = obsUrgentCount;

        if (obstacleDistM > dWarn) {
            obsDebug.reason = String.format(Locale.US,
                    "still %.1f m beyond the warn distance (%.1f m at %.0f km/h)",
                    obstacleDistM - dWarn, dWarn, egoSpeedMs * 3.6f);
        } else if (obsStage >= 1 && obstacleDistM > dUrgent) {
            obsDebug.reason = "already warned about this object";
        } else if (obsStage >= 2) {
            obsDebug.reason = "already at urgent for this object";
        }

        if (obsStage < 2 && obsUrgentCount >= OBSTACLE_CONFIRM) {
            obsStage = 2;
            return AlertType.COLLISION_URGENT;
        }
        if (obsStage < 1 && obsWarnCount >= OBSTACLE_CONFIRM) {
            obsStage = 1;
            return AlertType.COLLISION_WARN;
        }
        return null;
    }

    /** Call when the navigation layer reports an imminent instructed turn. */
    public void suppressFor(long ms, long nowMs) {
        suppressUntilMs = nowMs + ms;
    }

    // ---- internals ----

    /** True while value is BELOW the set threshold; releases only above clear. */
    private boolean hyst(AlertType t, float value, float set, float clear) {
        boolean was = Boolean.TRUE.equals(latched.get(t));
        boolean now = was ? (value < clear) : (value < set);
        latched.put(t, now);
        return now;
    }

    /** True while value is ABOVE the set threshold; releases only below clear. */
    private boolean hystAbove(AlertType t, float value, float set, float clear) {
        boolean was = Boolean.TRUE.equals(latched.get(t));
        boolean now = was ? (value > clear) : (value > set);
        latched.put(t, now);
        return now;
    }

    private void tick(AlertType t, boolean active) {
        consecutive.put(t, active ? consecutive.get(t) + 1 : 0);
    }

    private void reset(AlertType t) {
        consecutive.put(t, 0);
        latched.put(t, false);
    }

    private void decayAll() {
        for (AlertType t : AlertType.values()) consecutive.put(t, 0);
    }

    private boolean offCooldown(AlertType t, long nowMs) {
        return nowMs - lastFiredMs.get(t) >= t.cooldownMs;
    }

    private static String describe(AlertType t, Track s, float ttc,
                                   float ego, float limit, String laneCross) {
        switch (t) {
            case COLLISION_URGENT:
            case COLLISION_WARN:
                if (s == null) {
                    return String.format(Locale.US,
                            "ttc=%.2f OBSTACLE (unnamed object in path)", ttc);
                }
                return String.format(Locale.US, "ttc=%.2f d=%.1f close=%.1f id=%d",
                        ttc, s.distanceM, s.closingSpeedMs, s.id);
            case LANE_SOLID:
                return "crossed solid line on " + laneCross;
            case SPEED_LIMIT:
                return String.format(Locale.US, "ego=%.1f limit=%.1f", ego, limit);
            default:
                return "";
        }
    }
}

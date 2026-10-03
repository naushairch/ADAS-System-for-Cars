package com.adas.road;

import android.content.Context;

/**
 * Turns a stream of GPS fixes into a stable speed limit, or into no opinion
 * at all.
 *
 * WHY THIS IS NOT JUST "call RoadIndex.lookup() and use the answer"
 *   A fix lands somewhere different every half second, and near a junction,
 *   a slip road, or a service road running alongside a carriageway, the
 *   nearest segment genuinely flips between roads with very different limits.
 *   Adopting each answer as it arrives would swing the limit between 120 and
 *   30 while the car drives dead straight, and every swing downward is a
 *   "slow down" warning at a perfectly legal speed.
 *
 *   So a new limit has to be confirmed by CONFIRM_FIXES consecutive fixes
 *   before it is adopted. At the app's 500 ms location interval that is about
 *   a second and a half of agreement - fast enough that a motorway exit is
 *   picked up well before the ramp, slow enough that a single stray fix
 *   changes nothing.
 *
 * WHY IT CAN RETURN NOTHING
 *   No GPS fix, no index, or nothing within RoadIndex's snap distance all
 *   produce 0, meaning "no opinion". The caller keeps whatever limit it
 *   already had. Silence beats a guess here for the same reason it does in
 *   the lane module: the cost of a wrong limit is a false alert, and false
 *   alerts are what teach a driver to ignore the alerts that matter.
 */
public class SpeedLimitProvider {

    private static final int CONFIRM_FIXES = 3;

    private final RoadIndex index;

    private int currentKph = 0;
    private String currentClass = "-";
    private double lastDistM = Double.NaN;

    private int pendingKph = 0;
    private String pendingClass = "-";
    private int pendingCount = 0;

    private int lookups = 0, hits = 0;

    /** @throws Exception when the asset is missing or malformed; the caller is
     *  expected to treat that as "feature not installed", not as fatal. */
    public SpeedLimitProvider(Context ctx, String asset) throws Exception {
        index = new RoadIndex(ctx, asset);
    }

    public int segmentCount() { return index.segmentCount(); }

    /**
     * @return speed limit in km/h, or 0 for "no opinion - keep what you have".
     */
    public int update(double lat, double lon) {
        if (lat == 0.0 && lon == 0.0) return currentKph;   // no fix yet
        lookups++;

        RoadIndex.Match m = index.lookup(lat, lon);
        if (m == null) {
            // Off-road, or too far from anything we know about. Deliberately
            // does NOT clear the current limit - driving through an unmapped
            // car park should not discard the limit of the road you just left.
            pendingCount = 0;
            lastDistM = Double.NaN;
            return currentKph;
        }
        hits++;
        lastDistM = m.distanceM;

        int kph = m.speedKph;
        String cls = m.className();

        if (kph == currentKph) {
            pendingCount = 0;
            return currentKph;
        }
        if (kph == pendingKph) {
            if (++pendingCount >= CONFIRM_FIXES) {
                currentKph = pendingKph;
                currentClass = pendingClass;
                pendingCount = 0;
            }
        } else {
            pendingKph = kph;
            pendingClass = cls;
            pendingCount = 1;
        }
        return currentKph;
    }

    public String describe() {
        if (currentKph == 0) return "no fix";
        String d = Double.isNaN(lastDistM) ? "?" : String.format("%.0fm", lastDistM);
        return currentClass + " " + currentKph + " (" + d + ")";
    }

    /** Fraction of fixes that matched a known road - a coverage figure for the
     *  area actually driven, which is the only one that matters. */
    public float hitRate() {
        return lookups == 0 ? 0f : (float) hits / lookups;
    }
}

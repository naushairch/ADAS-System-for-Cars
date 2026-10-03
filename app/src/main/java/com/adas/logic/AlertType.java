package com.adas.logic;

/**
 * Alert categories. Lower ordinal = higher priority.
 * Exactly one alert may sound at a time; see AlertEngine.
 *
 * MIRRORS tools/adas_core.py :: AlertType. Priorities, audio keys and cooldowns
 * must match that file value-for-value. AlertEngine iterates values() and relies
 * on DECLARATION ORDER being priority order, exactly as the Python sorts by
 * .priority — so do not reorder these without reordering the Python too.
 *
 * OBSTACLE_AHEAD sits between the two collision levels on purpose: an unnamed
 * object you are about to hit outranks a vehicle you merely might, but never
 * outranks an imminent vehicle impact.
 */
public enum AlertType {
    COLLISION_URGENT(0, "brake", 5000),
    OBSTACLE_AHEAD(1, "obstacle", 4500),
    COLLISION_WARN(2, "vehicle_close", 3500),
    LANE_SOLID(3, "lane", 5000),
    SPEED_LIMIT(4, "slow_down", 12000);

    public final int priority;
    public final String audioKey;   // maps to res/raw/<lang>_<audioKey>
    public final long cooldownMs;

    AlertType(int priority, String audioKey, long cooldownMs) {
        this.priority = priority;
        this.audioKey = audioKey;
        this.cooldownMs = cooldownMs;
    }
}

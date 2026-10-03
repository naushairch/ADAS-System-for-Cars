package com.adas.audio;

import android.content.Context;
import android.media.AudioAttributes;
import android.media.SoundPool;

import com.adas.logic.AlertType;

import java.util.EnumMap;
import java.util.Map;

/**
 * Pre-loaded, low-latency alert playback.
 *
 * TTS is NOT used at alert time. Synthesising speech on demand adds 300-800ms,
 * which alone would blow the entire end-to-end latency budget. All clips are
 * generated ahead of time (see tools/generate_alerts.py) and held in SoundPool,
 * giving playback latency under ~30ms.
 */
public class AlertPlayer {

    public enum Lang { EN, UR }

    private final SoundPool pool;
    private final Map<AlertType, Integer> en = new EnumMap<>(AlertType.class);
    private final Map<AlertType, Integer> ur = new EnumMap<>(AlertType.class);
    private int chimeId = 0;

    private Lang lang = Lang.EN;
    private int activeStream = 0;
    private volatile boolean loaded = false;

    public AlertPlayer(Context ctx) {
        AudioAttributes attrs = new AudioAttributes.Builder()
                // ASSISTANCE_NAVIGATION_GUIDANCE ducks music rather than stopping it,
                // and routes correctly over Bluetooth car audio.
                .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
                .build();

        pool = new SoundPool.Builder()
                .setMaxStreams(2)
                .setAudioAttributes(attrs)
                .build();

        int res;
        res = id(ctx, "chime");            if (res != 0) chimeId = pool.load(ctx, res, 1);

        for (AlertType t : AlertType.values()) {
            int rEn = id(ctx, "en_" + t.audioKey);
            int rUr = id(ctx, "ur_" + t.audioKey);
            if (rEn != 0) en.put(t, pool.load(ctx, rEn, 1));
            if (rUr != 0) ur.put(t, pool.load(ctx, rUr, 1));
        }
        pool.setOnLoadCompleteListener((sp, sampleId, status) -> loaded = true);
    }

    private static int id(Context ctx, String name) {
        return ctx.getResources().getIdentifier(name, "raw", ctx.getPackageName());
    }

    public void setLanguage(Lang l) { this.lang = l; }

    public Lang getLanguage() { return lang; }

    /**
     * Plays a 100ms attention chime followed by the spoken cue. The chime is
     * perceived faster than speech onset, so the driver's attention is already
     * captured by the time the word arrives.
     */
    public void play(AlertType type) {
        if (!loaded) return;
        stop();
        if (chimeId != 0) pool.play(chimeId, 1f, 1f, 2, 0, 1f);

        Integer sid = (lang == Lang.UR ? ur : en).get(type);
        if (sid == null) sid = (lang == Lang.UR ? en : ur).get(type); // fall back
        if (sid != null) {
            activeStream = pool.play(sid, 1f, 1f, 1, 0, 1f);
        }
    }

    /** Higher-priority alerts pre-empt whatever is currently speaking. */
    public void stop() {
        if (activeStream != 0) {
            pool.stop(activeStream);
            activeStream = 0;
        }
    }

    public void release() { pool.release(); }
}

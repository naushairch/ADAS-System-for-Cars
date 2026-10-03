package com.adas.track;

import android.graphics.RectF;

import com.adas.detect.Detection;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * Greedy IoU association tracker.
 *
 * Deliberately simple: at 20fps on a forward-facing camera, frame-to-frame IoU
 * overlap is high and a Kalman filter buys little. If you later find ID switches
 * when vehicles cross, swap this for ByteTrack — the interface stays the same.
 */
public class IouTracker {

    private static final float MATCH_IOU = 0.30f;
    private static final int MAX_MISSED = 6;   // ~0.3s at 20fps

    private final List<Track> tracks = new ArrayList<>();

    public List<Track> update(List<Detection> detections) {
        boolean[] usedDet = new boolean[detections.size()];

        // Prefer matching to older, better-established tracks first.
        List<Track> ordered = new ArrayList<>(tracks);
        Collections.sort(ordered, (a, b) -> Integer.compare(b.hits, a.hits));

        for (Track t : ordered) {
            int bestIdx = -1;
            float bestIou = MATCH_IOU;
            for (int i = 0; i < detections.size(); i++) {
                if (usedDet[i]) continue;
                Detection d = detections.get(i);
                if (!classCompatible(t.classId, d.classId)) continue;
                float v = iou(t.box, d.box);
                if (v > bestIou) { bestIou = v; bestIdx = i; }
            }
            if (bestIdx >= 0) {
                Detection d = detections.get(bestIdx);
                t.update(d.box, d.classId, d.score);
                usedDet[bestIdx] = true;
            } else {
                t.missed++;
            }
            t.age++;
        }

        for (int i = 0; i < detections.size(); i++) {
            if (!usedDet[i]) {
                Detection d = detections.get(i);
                tracks.add(new Track(d.box, d.classId, d.score));
            }
        }

        Iterator<Track> it = tracks.iterator();
        while (it.hasNext()) {
            if (it.next().missed > MAX_MISSED) it.remove();
        }
        return tracks;
    }

    /** Vehicle classes get confused with each other constantly; allow swaps between them. */
    private static boolean classCompatible(int a, int b) {
        if (a == b) return true;
        return isVehicle(a) && isVehicle(b);
    }

    private static boolean isVehicle(int c) {
        return c == 2 || c == 3 || c == 5 || c == 7; // car, motorcycle, bus, truck
    }

    private static float iou(RectF a, RectF b) {
        float ix = Math.max(0, Math.min(a.right, b.right) - Math.max(a.left, b.left));
        float iy = Math.max(0, Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top));
        float inter = ix * iy;
        float uni = a.width() * a.height() + b.width() * b.height() - inter;
        return uni <= 0 ? 0 : inter / uni;
    }

    public void reset() { tracks.clear(); }
}

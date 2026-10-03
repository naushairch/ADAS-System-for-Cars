package com.adas.detect;

import android.graphics.RectF;

/** One detected object in a single frame, in input-image pixel coordinates. */
public class Detection {
    public final RectF box;
    public final int classId;
    public final float score;

    public Detection(RectF box, int classId, float score) {
        this.box = box;
        this.classId = classId;
        this.score = score;
    }

    public float cx() { return (box.left + box.right) * 0.5f; }
    public float widthPx() { return box.width(); }
}

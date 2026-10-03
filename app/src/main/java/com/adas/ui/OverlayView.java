package com.adas.ui;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.view.View;

import com.adas.logic.DistanceEstimator;
import com.adas.track.Track;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public class OverlayView extends View {

    private final Paint box = new Paint();
    private final Paint boxHot = new Paint();
    private final Paint text = new Paint();
    private final Paint hud = new Paint();
    private final Paint corridor = new Paint();

    private List<Track> tracks = new ArrayList<>();
    private int srcW = 1, srcH = 1;
    private String bannerText = null;
    private long bannerUntil = 0;
    /** 1.0 = no bonnet in frame. Below that, where the bonnet starts. */
    private float bonnetTopFrac = 1f;

    /**
     * Analysis-frame -> screen mapping, recomputed every draw.
     *
     * THIS MUST MATCH PreviewView'S SCALE TYPE EXACTLY, which MainActivity pins
     * to FILL_CENTER. It did not, and that was the whole bug: this view used to
     * scale x and y independently (getWidth()/srcW, getHeight()/srcH), i.e. it
     * squashed the frame to fit the screen precisely, while the preview
     * underneath scaled UNIFORMLY and cropped the overflow.
     *
     * The two agree only on a screen whose aspect ratio happens to equal the
     * analysis frame's — never true on a modern phone. Everywhere else the error
     * is zero at the centre of the screen and grows toward the edges, so boxes
     * sat low on objects near the top of the frame (speed limit signs, exactly
     * where it was noticed) and high on objects near the bottom, and every box
     * was the wrong size in one axis. A box that is not tight on the object also
     * feeds a wrong width into DistanceEstimator, so this was quietly corrupting
     * distance and TTC as well as looking wrong.
     */
    private float mapScale = 1f, mapDx = 0f, mapDy = 0f;

    /** FILL_CENTER: scale uniformly until the frame covers the view, centre it,
     *  and let the overflow fall off both edges (so the offsets are <= 0). */
    private void computeMapping() {
        mapScale = Math.max((float) getWidth() / srcW, (float) getHeight() / srcH);
        mapDx = (getWidth() - srcW * mapScale) * 0.5f;
        mapDy = (getHeight() - srcH * mapScale) * 0.5f;
    }

    private float mapX(float x) { return x * mapScale + mapDx; }

    private float mapY(float y) { return y * mapScale + mapDy; }

    public OverlayView(Context c, AttributeSet a) {
        super(c, a);
        box.setStyle(Paint.Style.STROKE); box.setStrokeWidth(3f); box.setColor(Color.GREEN);
        boxHot.setStyle(Paint.Style.STROKE); boxHot.setStrokeWidth(6f); boxHot.setColor(Color.RED);
        text.setColor(Color.WHITE); text.setTextSize(30f); text.setFakeBoldText(true);
        hud.setColor(Color.YELLOW); hud.setTextSize(34f); hud.setFakeBoldText(true);
        corridor.setStyle(Paint.Style.STROKE); corridor.setStrokeWidth(2f);
        corridor.setColor(Color.argb(120, 0, 200, 255));
    }

    /**
     * The diagnostics line is deliberately NOT drawn here. It used to be, at the
     * same time as MainActivity wrote it into the status TextView sitting on top
     * of this view — so it rendered twice, very slightly offset, which looked like
     * a rendering fault. The TextView owns it now, and it is hidden by default:
     * on a dashboard the driver wants boxes and a voice, not telemetry.
     */
    public void submit(List<Track> t, int sourceW, int sourceH) {
        this.tracks = new ArrayList<>(t);
        this.srcW = Math.max(1, sourceW);
        this.srcH = Math.max(1, sourceH);
        postInvalidate();
    }

    /** Draws the bonnet cut-off so it can be set by eye during calibration. */
    public void setBonnetTopFrac(float frac) {
        this.bonnetTopFrac = frac;
    }

    public void banner(String s, long untilMs) {
        this.bannerText = s;
        this.bannerUntil = untilMs;
        postInvalidate();
    }

    @Override
    protected void onDraw(Canvas c) {
        super.onDraw(c);
        computeMapping();

        // Ego corridor at a nominal 20m. Takes the half-width from
        // DistanceEstimator rather than repeating the number, so the guide on
        // screen always matches the corridor actually used for decisions. It was
        // hardcoded to the old 1.9m and kept drawing that after the constant was
        // corrected — a guide that disagrees with the logic is worse than none.
        //
        // Both ends are now expressed in ANALYSIS-frame coordinates and mapped,
        // where the y values used to be raw view pixels (getHeight()) against x
        // values in frame pixels. Half of this guide lived in one coordinate
        // space and half in the other, so it could not have been right in both.
        float halfPx = (DistanceEstimator.CORRIDOR_HALF_M
                * DistanceEstimator.FOCAL_PX) / 20f;
        float cxv = srcW * 0.5f;
        float yNear = srcH;
        float yFar = srcH * 0.55f;
        c.drawLine(mapX(cxv - halfPx), mapY(yNear),
                mapX(cxv - halfPx * 0.25f), mapY(yFar), corridor);
        c.drawLine(mapX(cxv + halfPx), mapY(yNear),
                mapX(cxv + halfPx * 0.25f), mapY(yFar), corridor);

        for (Track t : tracks) {
            if (!t.isConfirmed()) continue;
            RectF r = new RectF(mapX(t.box.left), mapY(t.box.top),
                    mapX(t.box.right), mapY(t.box.bottom));
            boolean hot = t.ttcS < 3.0f
                    && DistanceEstimator.inEgoCorridor(t.box, srcW, t.distanceM);
            c.drawRect(r, hot ? boxHot : box);
            String label = Float.isNaN(t.distanceM)
                    ? ("#" + t.id)
                    : String.format(Locale.US, "#%d %.0fm %.1fs", t.id, t.distanceM,
                        Float.isInfinite(t.ttcS) ? -1f : t.ttcS);
            c.drawText(label, r.left, Math.max(34f, r.top - 8f), text);
        }

        if (bonnetTopFrac < 1f) {
            // A fraction of the ANALYSIS FRAME height, not of the screen —
            // MainActivity applies it as bonnetFrac * bmp.getHeight() when it
            // discards detections. Drawing it at a fraction of the view put the
            // orange line somewhere the filter does not actually cut, and the
            // driver sets this value BY EYE against that line. So the old code
            // did not merely draw the guide wrong, it taught the driver to store
            // a wrong number, which then threw away real detections.
            float y = mapY(srcH * bonnetTopFrac);
            Paint p = new Paint(corridor);
            p.setColor(Color.argb(160, 255, 140, 0));
            p.setStrokeWidth(3f);
            c.drawLine(0, y, getWidth(), y, p);
            Paint lbl = new Paint(text);
            lbl.setTextSize(24f);
            lbl.setColor(Color.argb(200, 255, 140, 0));
            c.drawText("bonnet - ignored below", 24f, y - 10f, lbl);
        }

        // The alert banner is the one thing a driver may legitimately glance at,
        // so it is centred and large rather than tucked in a corner.
        if (bannerText != null && android.os.SystemClock.elapsedRealtime() < bannerUntil) {
            Paint p = new Paint(hud);
            p.setColor(Color.RED);
            p.setTextSize(96f);
            p.setTextAlign(Paint.Align.CENTER);
            // Mid-screen, NOT near the bottom: the control row lives bottom-right
            // and was covering the word. An alert the driver cannot read is the
            // one thing this view exists to prevent.
            c.drawText(bannerText, getWidth() * 0.5f, getHeight() * 0.55f, p);
        }
    }
}

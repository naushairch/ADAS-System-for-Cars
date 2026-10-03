package android.graphics;

/**
 * Desktop stand-in for android.graphics.RectF, used ONLY by the offline parity
 * harness so the real com.adas logic classes can run on a plain JVM.
 *
 * The framework class is a plain data holder; these are the only members the
 * ADAS logic touches. Not shipped, not in the app source tree.
 */
public class RectF {
    public float left, top, right, bottom;

    public RectF() { }

    public RectF(float left, float top, float right, float bottom) {
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
    }

    public final float width() { return right - left; }

    public final float height() { return bottom - top; }

    @Override
    public String toString() {
        return "RectF(" + left + ", " + top + ", " + right + ", " + bottom + ")";
    }
}

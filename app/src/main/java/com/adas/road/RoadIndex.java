package com.adas.road;

import android.content.Context;
import android.content.res.AssetFileDescriptor;

import java.io.FileInputStream;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

/**
 * Offline road-class lookup, built by tools/build_road_index.py.
 *
 * WHAT IT ANSWERS
 *   "Given this GPS fix, what kind of road am I on, and what is the speed
 *   limit there?" - from an OpenStreetMap extract carried in the APK. No
 *   network, no API key, no per-query cost, and it keeps working in a tunnel
 *   or with no signal, which is where a live lookup would fail.
 *
 * WHY IT CAN SAY "I DON'T KNOW"
 *   Same principle as the lane module. Snapping a fix to a road is genuinely
 *   ambiguous next to a service road, a parallel carriageway, or an overpass,
 *   and a 15 m lateral error can hand back the wrong road with total
 *   confidence. So a match further than MAX_SNAP_M is not a match, and the app
 *   keeps whatever limit it had rather than adopting a guess. A wrong limit is
 *   worse than a stale one: it fires "slow down" at a legal speed, and those
 *   are the alerts that teach a driver to ignore the device.
 *
 * FILE LAYOUT (little-endian, see build_road_index.py)
 *   magic "ADASROAD" | version i32 | tileDeg f32 | minLat f32 | minLon f32
 *   nTilesLat i32 | nTilesLon i32 | nSeg i32
 *   tiles:    nTilesLat*nTilesLon * (offset i32, count i32)
 *   segments: nSeg * (lat1 i32, lon1 i32, lat2 i32, lon2 i32, class u8, kph u8)
 *   Coordinates are microdegrees. A segment sits in the tile of its midpoint
 *   and is never longer than a tile, so searching 3x3 tiles around a fix
 *   cannot miss the nearest one.
 */
public class RoadIndex {

    public static final String[] CLASS_NAMES = {
            "motorway", "motorway_link", "trunk", "trunk_link",
            "primary", "primary_link", "secondary", "secondary_link",
            "tertiary", "tertiary_link", "unclassified", "residential",
            "living_street",
    };

    /**
     * Beyond this, we are not confident which road the fix belongs to.
     *
     * 25 m is about a dual carriageway's half-width plus a civilian GPS error.
     * Tighter and a legitimate fix on a wide road stops matching; looser and a
     * fix on a service road starts matching the motorway beside it, which is
     * a 120-vs-30 error in the direction that produces false warnings.
     */
    private static final double MAX_SNAP_M = 25.0;

    private static final int HEADER_BYTES = 8 + 4 + 4 + 4 + 4 + 4 + 4 + 4;
    private static final int TILE_BYTES = 8;
    private static final int SEG_COORD_BYTES = 16;
    private static final int SEG_META_BYTES = 2;

    private static final double M_PER_DEG_LAT = 111320.0;

    private final MappedByteBuffer buf;
    private final float tileDeg, minLat, minLon;
    private final int nTilesLat, nTilesLon, nSeg;
    private final int tilesStart, coordsStart, metaStart;

    public static final class Match {
        public int classId;
        public int speedKph;
        public double distanceM;
        public String className() {
            return (classId >= 0 && classId < CLASS_NAMES.length)
                    ? CLASS_NAMES[classId] : "?";
        }
    }

    private final Match scratch = new Match();

    public RoadIndex(Context ctx, String asset) throws Exception {
        try (AssetFileDescriptor afd = ctx.getAssets().openFd(asset);
             FileInputStream fis = new FileInputStream(afd.getFileDescriptor())) {
            buf = fis.getChannel().map(FileChannel.MapMode.READ_ONLY,
                    afd.getStartOffset(), afd.getDeclaredLength());
        }
        buf.order(ByteOrder.LITTLE_ENDIAN);

        byte[] magic = new byte[8];
        buf.position(0);
        buf.get(magic);
        if (!new String(magic, "US-ASCII").equals("ADASROAD")) {
            throw new IllegalStateException("not a road index (bad magic)");
        }
        int version = buf.getInt();
        if (version != 1) {
            throw new IllegalStateException("road index version " + version
                    + ", expected 1 - rebuild with build_road_index.py");
        }
        tileDeg = buf.getFloat();
        minLat = buf.getFloat();
        minLon = buf.getFloat();
        nTilesLat = buf.getInt();
        nTilesLon = buf.getInt();
        nSeg = buf.getInt();

        tilesStart = HEADER_BYTES;
        coordsStart = tilesStart + nTilesLat * nTilesLon * TILE_BYTES;
        metaStart = coordsStart + nSeg * SEG_COORD_BYTES;

        long expect = (long) metaStart + (long) nSeg * SEG_META_BYTES;
        if (buf.capacity() < expect) {
            throw new IllegalStateException("road index truncated: have "
                    + buf.capacity() + " bytes, need " + expect);
        }
    }

    public int segmentCount() { return nSeg; }

    /**
     * @return the nearest road within MAX_SNAP_M, or null when the fix cannot
     *         be attributed confidently. The returned object is reused - read
     *         it before the next call.
     */
    public Match lookup(double lat, double lon) {
        int ti = (int) Math.floor((lat - minLat) / tileDeg);
        int tj = (int) Math.floor((lon - minLon) / tileDeg);

        double cosLat = Math.cos(Math.toRadians(lat));
        double best = MAX_SNAP_M * MAX_SNAP_M;
        int bestIdx = -1;

        for (int di = -1; di <= 1; di++) {
            int i = ti + di;
            if (i < 0 || i >= nTilesLat) continue;
            for (int dj = -1; dj <= 1; dj++) {
                int j = tj + dj;
                if (j < 0 || j >= nTilesLon) continue;

                int tile = i * nTilesLon + j;
                int p = tilesStart + tile * TILE_BYTES;
                int off = buf.getInt(p);
                int cnt = buf.getInt(p + 4);

                for (int k = 0; k < cnt; k++) {
                    int s = off + k;
                    int c = coordsStart + s * SEG_COORD_BYTES;
                    double la1 = buf.getInt(c) / 1e6;
                    double lo1 = buf.getInt(c + 4) / 1e6;
                    double la2 = buf.getInt(c + 8) / 1e6;
                    double lo2 = buf.getInt(c + 12) / 1e6;

                    double d2 = pointSegDist2M(lat, lon, la1, lo1, la2, lo2, cosLat);
                    if (d2 < best) {
                        best = d2;
                        bestIdx = s;
                    }
                }
            }
        }

        if (bestIdx < 0) return null;
        int m = metaStart + bestIdx * SEG_META_BYTES;
        scratch.classId = buf.get(m) & 0xFF;
        scratch.speedKph = buf.get(m + 1) & 0xFF;
        scratch.distanceM = Math.sqrt(best);
        return scratch;
    }

    /**
     * Squared distance in metres from a point to a segment, using a local
     * equirectangular projection. Over the ~2 km a tile spans, that is accurate
     * to well under a metre - far inside the GPS error it is compared against,
     * and enormously cheaper than a proper geodesic for something evaluated
     * against thousands of segments per fix.
     */
    private static double pointSegDist2M(double plat, double plon,
                                         double la1, double lo1,
                                         double la2, double lo2,
                                         double cosLat) {
        double px = (plon - lo1) * M_PER_DEG_LAT * cosLat;
        double py = (plat - la1) * M_PER_DEG_LAT;
        double vx = (lo2 - lo1) * M_PER_DEG_LAT * cosLat;
        double vy = (la2 - la1) * M_PER_DEG_LAT;

        double len2 = vx * vx + vy * vy;
        double t = (len2 <= 1e-9) ? 0.0 : (px * vx + py * vy) / len2;
        if (t < 0) t = 0;
        else if (t > 1) t = 1;

        double dx = px - t * vx;
        double dy = py - t * vy;
        return dx * dx + dy * dy;
    }
}

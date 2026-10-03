"""
run_parity.py — prove the Java engine still matches the Python one. No phone,
no emulator, no Gradle.

WHAT IT DOES
    1. regenerates the golden trace from adas_core.py  (gen_parity_golden.py)
    2. compiles the REAL app sources — AlertEngine, DistanceEstimator,
       IouTracker, Track, Detection — against a tiny stand-in for RectF
    3. replays all scenarios through the compiled Java
    4. compares alert type and frame index against the Python

WHY A STAND-IN RectF
    Those five classes touch exactly one Android API: android.graphics.RectF, a
    plain data holder. Substituting a 20-line equivalent lets the genuine
    shipping logic run on a desktop JVM. Nothing else is stubbed, mocked or
    reimplemented — if this passes, the code in app/src/main/java is correct.

    The on-device AlertEngineParityTest asserts the same thing against the real
    framework class. Run that too before a road test; run this one constantly.

USAGE
    python run_parity.py
    python run_parity.py --skip-golden     # reuse the existing golden file

EXIT CODE
    0 = engines agree, non-zero = they diverged (or the build failed).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
APP_SRC = os.path.join(ROOT, "app", "src", "main", "java")
GOLDEN = os.path.join(ROOT, "app", "src", "androidTest", "assets",
                      "parity_golden.json")
PARITY_DIR = os.path.join(HERE, "parity")

# The only app classes the harness needs. Detector/MainActivity are excluded on
# purpose: they pull in TFLite and the Android framework, and they contain no
# decision logic worth asserting.
APP_CLASSES = [
    "com/adas/detect/Detection.java",
    "com/adas/track/Track.java",
    "com/adas/track/IouTracker.java",
    "com/adas/logic/AlertType.java",
    "com/adas/logic/DistanceEstimator.java",
    "com/adas/logic/AlertEngine.java",
]


def find_jdk():
    """Locate javac/java: JAVA_HOME, then Android Studio's bundled JBR, then PATH."""
    candidates = []
    if os.environ.get("JAVA_HOME"):
        candidates.append(os.path.join(os.environ["JAVA_HOME"], "bin"))
    candidates += [
        r"C:\Program Files\Android\Android Studio\jbr\bin",
        r"C:\Program Files\Android\Android Studio Preview\jbr\bin",
        os.path.expanduser(r"~\AppData\Local\Programs\Android Studio\jbr\bin"),
    ]
    for b in candidates:
        javac = os.path.join(b, "javac.exe")
        java = os.path.join(b, "java.exe")
        if os.path.exists(javac) and os.path.exists(java):
            return javac, java
        javac, java = os.path.join(b, "javac"), os.path.join(b, "java")
        if os.path.exists(javac) and os.path.exists(java):
            return javac, java

    javac, java = shutil.which("javac"), shutil.which("java")
    if javac and java:
        return javac, java

    sys.exit("No JDK found. Install Android Studio (it bundles one) or set JAVA_HOME.")


def flatten(golden_path, out_path):
    """JSON -> flat text, so Harness.java needs no JSON library."""
    d = json.load(open(golden_path))
    lines = [
        "FOCAL %r" % d["focal_px"],
        "FRAMEW %d" % d["frame_width"],
        "STEP %d" % d["step_ms"],
        "START %d" % d["start_ms"],
    ]
    for s in d["scenarios"]:
        lines.append("SCEN " + s["name"])
        for f in s["frames"]:
            obs = "NaN" if f["obstacle_dist"] is None else repr(f["obstacle_dist"])
            lane = "-" if f["lane_cross"] is None else f["lane_cross"]
            lines.append("F %r %r %s %s %d" % (
                f["ego_speed_ms"], f["speed_limit_ms"], obs, lane, len(f["dets"])))
            for det in f["dets"]:
                b = det["box"]
                lines.append("D %r %r %r %r %d %r" % (
                    b[0], b[1], b[2], b[3], det["class_id"], det["score"]))
        for e in s["expected"]:
            lines.append("E %d %s" % (e["frame"], e["alert"]))
        lines.append("ENDSCEN")
    open(out_path, "w").write("\n".join(lines) + "\n")
    return len(d["scenarios"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-golden", action="store_true",
                    help="reuse the existing golden file instead of regenerating")
    ap.add_argument("--keep", action="store_true", help="keep the build directory")
    args = ap.parse_args()

    if not args.skip_golden:
        print("regenerating golden trace from adas_core.py ...")
        r = subprocess.run([sys.executable,
                            os.path.join(HERE, "gen_parity_golden.py")],
                           cwd=HERE)
        if r.returncode != 0:
            sys.exit("golden generation failed")
        print()

    if not os.path.exists(GOLDEN):
        sys.exit("no golden file at %s - run gen_parity_golden.py" % GOLDEN)

    javac, java = find_jdk()
    workdir = tempfile.mkdtemp(prefix="adas_parity_")
    outdir = os.path.join(workdir, "out")

    try:
        flat = os.path.join(workdir, "golden.txt")
        n_scen = flatten(GOLDEN, flat)

        sources = [os.path.join(PARITY_DIR, "android", "graphics", "RectF.java"),
                   os.path.join(PARITY_DIR, "Harness.java")]
        sources += [os.path.join(APP_SRC, p.replace("/", os.sep))
                    for p in APP_CLASSES]

        missing = [s for s in sources if not os.path.exists(s)]
        if missing:
            sys.exit("missing source(s):\n  " + "\n  ".join(missing))

        print("compiling %d sources with %s" % (len(sources), javac))
        r = subprocess.run([javac, "-nowarn", "-d", outdir] + sources,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out = r.stdout.decode("utf-8", "replace")
        if r.returncode != 0:
            print(out)
            sys.exit("COMPILE FAILED - the Java does not build")
        if out.strip():
            print(out)

        print("running %d scenarios\n" % n_scen)
        r = subprocess.run([java, "-cp", outdir, "Harness", flat])
        sys.exit(r.returncode)

    finally:
        if args.keep:
            print("\nbuild dir kept: " + workdir)
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()

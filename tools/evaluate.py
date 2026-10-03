"""
Compares what the Android app estimated (app_output.csv) against what was
actually true in the simulator (ground_truth.csv).

Usage:
    python evaluate.py out/s2_emergency_brake

Produces the numbers that go straight into your report:
  - distance estimation error vs range (MAE, bias, %)
  - TTC error at the moment each alert fired
  - alert timing: how long before impact-course each warning arrived
  - false positives on the silence scenario
  - a plot: estimated vs true distance over time
"""

import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def f(row, key, default=float("nan")):
    try:
        v = float(row[key])
        return default if v < 0 and key.endswith(("_m", "_s")) else v
    except (KeyError, ValueError, TypeError):
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session_dir")
    args = ap.parse_args()

    gt_path = os.path.join(args.session_dir, "ground_truth.csv")
    app_path = os.path.join(args.session_dir, "app_output.csv")
    for p in (gt_path, app_path):
        if not os.path.exists(p):
            sys.exit(f"missing: {p}")

    gt = load_csv(gt_path)
    app = load_csv(app_path)
    n = min(len(gt), len(app))
    if len(gt) != len(app):
        print(f"NOTE: frame count mismatch ({len(gt)} truth vs {len(app)} app); "
              f"comparing first {n}")

    t, true_d, est_d, true_ttc, est_ttc, in_lane = [], [], [], [], [], []
    alerts = []

    for i in range(n):
        g, a = gt[i], app[i]
        ts = float(g["t_s"])
        td = float(g["true_dist_m"])
        ed = float(a["est_dist_m"])
        tt = float(g["true_ttc_s"])
        et = float(a["est_ttc_s"])

        t.append(ts)
        true_d.append(td if td < 900 else np.nan)
        est_d.append(ed if ed > 0 else np.nan)
        true_ttc.append(tt if tt > 0 else np.nan)
        est_ttc.append(et if et > 0 else np.nan)
        in_lane.append(int(g["in_ego_lane"]))

        if a["alert"]:
            alerts.append(dict(frame=i, t=ts, type=a["alert"],
                               true_dist=td, true_ttc=tt,
                               est_dist=ed, est_ttc=et,
                               in_lane=int(g["in_ego_lane"])))

    t = np.array(t)
    true_d = np.array(true_d, dtype=float)
    est_d = np.array(est_d, dtype=float)

    name = os.path.basename(args.session_dir.rstrip("/\\"))
    print(f"\n=== {name} ===")
    print(f"frames: {n}   duration: {t[-1]:.1f}s")

    # ---------- distance estimation accuracy ----------
    valid = ~np.isnan(true_d) & ~np.isnan(est_d)
    if valid.sum() > 0:
        err = est_d[valid] - true_d[valid]
        print("\n-- distance estimation --")
        print(f"  samples      : {valid.sum()}")
        print(f"  MAE          : {np.mean(np.abs(err)):.2f} m")
        print(f"  bias         : {np.mean(err):+.2f} m  "
              f"({'over' if np.mean(err) > 0 else 'under'}-estimating)")
        print(f"  MAPE         : {np.mean(np.abs(err) / true_d[valid]) * 100:.1f} %")

        # Error grows with range; report it banded, not just as one number.
        print("\n  by range:")
        for lo, hi in [(0, 15), (15, 30), (30, 50), (50, 100)]:
            m = valid.copy()
            m[valid] &= (true_d[valid] >= lo) & (true_d[valid] < hi)
            k = m.sum()
            if k > 3:
                e = est_d[m] - true_d[m]
                print(f"    {lo:3d}-{hi:3d} m : n={k:4d}  MAE={np.mean(np.abs(e)):5.2f} m  "
                      f"bias={np.mean(e):+5.2f} m")

    # ---------- alerts ----------
    print(f"\n-- alerts fired: {len(alerts)} --")
    for a in alerts:
        tag = "" if a["in_lane"] else "   <-- FALSE POSITIVE (nothing in ego lane)"
        ttc_str = f"{a['true_ttc']:.2f}" if a["true_ttc"] > 0 else "n/a"
        print(f"  t={a['t']:6.2f}s  {a['type']:17s} "
              f"true_dist={a['true_dist']:6.1f}m  true_ttc={ttc_str:>5}s{tag}")

    fp = [a for a in alerts if not a["in_lane"]]
    if "roadside" in name or "parked" in name:
        print(f"\n  SILENCE TEST: {len(alerts)} alert(s) fired. "
              f"{'PASS' if len(alerts) == 0 else 'FAIL - tune corridor / CONFIRM_FRAMES'}")
    elif fp:
        print(f"\n  {len(fp)} false positive(s) - object not in ego lane.")

    # ---------- alert lead time ----------
    # How much warning did the driver actually get before true TTC hit 1.0s?
    tt = np.array(true_ttc, dtype=float)
    crit = np.where(tt < 1.0)[0]
    if len(crit) and alerts:
        crit_t = t[crit[0]]
        print(f"\n-- lead time (true TTC crosses 1.0s at t={crit_t:.2f}s) --")
        for a in alerts:
            print(f"  {a['type']:17s} fired {crit_t - a['t']:+.2f}s before critical")

    # ---------- plot ----------
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t, true_d, label="true distance", lw=2)
    ax[0].plot(t, est_d, label="app estimate", lw=1.4, alpha=0.85)
    ax[0].set_ylabel("distance (m)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[0].set_title(name)

    ax[1].plot(t, true_ttc, label="true TTC", lw=2)
    ax[1].plot(t, est_ttc, label="app TTC", lw=1.4, alpha=0.85)
    ax[1].axhline(2.8, ls="--", c="orange", lw=1, label="WARN threshold")
    ax[1].axhline(1.6, ls="--", c="red", lw=1, label="URGENT threshold")
    ax[1].set_ylabel("TTC (s)"); ax[1].set_xlabel("time (s)")
    ax[1].set_ylim(0, 12)
    ax[1].legend(); ax[1].grid(alpha=0.3)

    for a in alerts:
        c = "red" if "URGENT" in a["type"] else "orange"
        for axis in ax:
            axis.axvline(a["t"], color=c, alpha=0.6, lw=1.2)

    out = os.path.join(args.session_dir, "evaluation.png")
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nplot -> {out}\n")


if __name__ == "__main__":
    main()

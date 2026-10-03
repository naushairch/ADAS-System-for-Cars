"""
list_signs.py — cheapest possible check before touching a camera at all.

Just connects, counts the real traffic.speed_limit.* actors CARLA has
already placed on each map, and tells you whether SIGN_CLASSES in
carla_collect_signs.py actually matches what's there. No ego vehicle, no
camera, no driving - this should return in a couple of seconds per town.

Run this FIRST, before carla_collect_signs.py --preview, before anything
else. If the class list is wrong, everything downstream is wrong too.

USAGE
    python list_signs.py --towns Town01,Town02,Town03,Town04,Town05,Town10HD
"""

import argparse
from collections import Counter

import carla

from carla_collect_signs import SIGN_CLASSES, sign_speed_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--towns", default="",
                    help="comma-separated, e.g. Town01,Town03,Town05 - "
                    "empty checks whatever is already loaded. NOTE: this uses "
                    "client.load_world(), which has been observed to crash "
                    "CarlaUE4 outright on some machines. If it crashes, omit "
                    "--towns and instead launch CarlaUE4.exe <TownName> "
                    "directly, one town per launch.")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    print("Maps this CARLA server actually has available:")
    for m in sorted(client.get_available_maps()):
        print(f"  {m}")
    print()

    towns = [t.strip() for t in args.towns.split(",") if t.strip()]
    load = bool(towns)
    towns = towns or ["current"]

    print(f"SIGN_CLASSES currently set to: {SIGN_CLASSES}\n")

    all_unknown = Counter()
    total_signs = 0
    for town in towns:
        world = client.load_world(town) if load else client.get_world()
        actual_name = world.get_map().name
        signs = list(world.get_actors().filter("traffic.speed_limit.*"))
        total_signs += len(signs)
        counts = Counter(sign_speed_value(a.type_id) for a in signs)
        known = {v: n for v, n in counts.items() if v in SIGN_CLASSES}
        unknown = {v: n for v, n in counts.items() if v not in SIGN_CLASSES}
        all_unknown.update(unknown)

        print(f"[{town}] map={actual_name}  total signs={len(signs)}")
        print(f"    in SIGN_CLASSES : {dict(sorted(known.items()))}")
        if unknown:
            print(f"    NOT in SIGN_CLASSES : {dict(sorted(unknown.items()))}  <-- "
                  f"these will be silently skipped by the collector")
        print()

    if total_signs == 0:
        print("NO speed-limit sign actors found on any checked town. Either "
              "the wrong map is loaded, or this map genuinely has none "
              "(some small/demo maps like Town10HD don't). Load a different "
              "town and try again before assuming SIGN_CLASSES is fine.")
    elif all_unknown:
        print(f"Speed values seen but not in SIGN_CLASSES: {dict(all_unknown)}")
        print("Add them to SIGN_CLASSES in carla_collect_signs.py if you want "
              "them detected - do this BEFORE your first real collection run, "
              "changing it after means re-collecting (class indices shift).")
    else:
        print(f"Every sign on every checked town is covered by SIGN_CLASSES "
              f"({total_signs} sign(s) total).")


if __name__ == "__main__":
    main()

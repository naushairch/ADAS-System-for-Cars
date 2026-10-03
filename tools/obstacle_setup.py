"""
obstacle_setup.py — puts things in the road so you can test the obstacle warning.

WHY THIS EXISTS
    The obstacle detector exists to catch what YOLO cannot name: barriers,
    barrels, containers, poles, debris. CARLA's roads are bare by default, so
    there is nothing to drive at. This spawns some.

    Everything it spawns is a `static.prop.*` - none of these are in YOLO's COCO
    classes, which is exactly the point. If the warning fires on these, it is
    the optical-flow path doing the work, not the vehicle detector.

--------------------------------------------------------------------------------
COMMANDS

    python obstacle_setup.py list
        Show which props this CARLA build has.

    python obstacle_setup.py place --every 90
        Scatter obstacles along the roads, one cluster every 90 m, in the middle
        of a driving lane so you will meet them head-on.

    python obstacle_setup.py wall --every 200
        Spawn barriers spanning the full lane width - a solid obstacle you
        cannot drive around. The clearest test case.

    python obstacle_setup.py clear
        Remove everything this script spawned.

--------------------------------------------------------------------------------
"""

import argparse
import json
import os

import carla

PLACED_FILE = os.path.join(os.path.dirname(__file__), "placed_obstacles.json")

# Large, textured props. Size and surface detail both matter: optical flow needs
# corners to track, so a plain smooth box is harder than a barrier with slats.
PREFERRED = [
    "static.prop.streetbarrier",
    "static.prop.container",
    "static.prop.trafficwarning",
    "static.prop.warningconstruction",
    "static.prop.constructioncone",
    "static.prop.barrel",
    "static.prop.vendingmachine",
    "static.prop.busstop",
    "static.prop.warningaccident",
    "static.prop.box03",
    "static.prop.trafficcone01",
]


def available_props(world, only_preferred=True):
    have = {bp.id: bp for bp in world.get_blueprint_library().filter("static.prop.*")}
    if only_preferred:
        picked = [have[i] for i in PREFERRED if i in have]
        if picked:
            return picked
    return list(have.values())


def cmd_list(client, args):
    world = client.get_world()
    all_props = available_props(world, only_preferred=False)
    good = available_props(world, only_preferred=True)
    print(f"map: {world.get_map().name}")
    print(f"static props in this build: {len(all_props)}\n")
    print("recommended for obstacle testing (big and textured):")
    for bp in good:
        print(f"  {bp.id}")
    if not good:
        print("  none of the preferred set found; place will use whatever exists")


def _lane_spawn_points(world, every):
    """Waypoints down the centre of driving lanes, spaced out."""
    pts = []
    for wp in world.get_map().generate_waypoints(every):
        if wp.lane_type == carla.LaneType.Driving:
            pts.append(wp)
    return pts


def _record(placed, world):
    with open(PLACED_FILE, "w") as f:
        json.dump({"map": world.get_map().name, "ids": placed}, f)
    print(f"\nspawned {len(placed)} objects")
    print(f"record saved to {PLACED_FILE}")
    print("\nremove them later with:  python obstacle_setup.py clear")


def cmd_place(client, args):
    world = client.get_world()
    props = available_props(world)
    if not props:
        print("no static props in this build")
        return

    placed = []
    for i, wp in enumerate(_lane_spawn_points(world, args.every)):
        if args.count and len(placed) >= args.count:
            break
        bp = props[i % len(props)]
        tf = wp.transform
        loc = carla.Location(x=tf.location.x, y=tf.location.y, z=tf.location.z + 0.4)
        try:
            a = world.spawn_actor(bp, carla.Transform(loc, tf.rotation))
            placed.append(a.id)
        except Exception:
            pass      # spot blocked, skip

    _record(placed, world)
    print("\nThese sit in the middle of a lane. Drive straight at one.")


def cmd_wall(client, args):
    """
    A row of barriers spanning the lane. Unlike a single barrel you cannot
    steer around it, which makes it the cleanest test: either the warning
    fires or it does not.
    """
    world = client.get_world()
    lib = world.get_blueprint_library()
    bp = None
    for candidate in ("static.prop.streetbarrier", "static.prop.container",
                      "static.prop.constructioncone"):
        found = lib.filter(candidate)
        if found:
            bp = found[0]
            break
    if bp is None:
        print("no suitable barrier prop found; try 'place' instead")
        return

    placed = []
    for i, wp in enumerate(_lane_spawn_points(world, args.every)):
        if args.count and len(placed) >= args.count:
            break
        tf = wp.transform
        right = tf.get_right_vector()
        # Three across covers a standard lane width.
        for offset in (-1.1, 0.0, 1.1):
            loc = carla.Location(
                x=tf.location.x + right.x * offset,
                y=tf.location.y + right.y * offset,
                z=tf.location.z + 0.4)
            try:
                a = world.spawn_actor(bp, carla.Transform(loc, tf.rotation))
                placed.append(a.id)
            except Exception:
                pass

    _record(placed, world)
    print(f"\nBarrier walls every {args.every} m, spanning the lane.")
    print("Drive straight into one at 40-60 km/h.")


def cmd_clear(client, args):
    if not os.path.exists(PLACED_FILE):
        print("nothing recorded as placed")
        return
    world = client.get_world()
    with open(PLACED_FILE) as f:
        data = json.load(f)
    removed = 0
    for aid in data.get("ids", []):
        a = world.get_actor(aid)
        if a is not None:
            try:
                a.destroy()
                removed += 1
            except Exception:
                pass
    os.remove(PLACED_FILE)
    print(f"removed {removed} objects")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["list", "place", "wall", "clear"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--every", type=float, default=120.0,
                    help="metres between obstacles")
    ap.add_argument("--count", type=int, default=0,
                    help="stop after this many objects (0 = no cap)")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)
    {"list": cmd_list, "place": cmd_place,
     "wall": cmd_wall, "clear": cmd_clear}[args.command](client, args)


if __name__ == "__main__":
    main()

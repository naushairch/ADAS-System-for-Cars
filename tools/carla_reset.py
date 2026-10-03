"""
carla_reset.py — un-wedge a CARLA server left stuck in synchronous mode.

WHY THIS EXISTS
    In synchronous mode the SERVER advances only when a client ticks it. If a
    collection script dies, is killed, or returns early without restoring
    asynchronous mode, the server sits there waiting for ticks that will never
    come. Everything afterwards fails like this:

        RuntimeError: time-out of 60000ms while waiting for the simulator,
        make sure the simulator is ready and connected to 127.0.0.1:2000

    which looks like CARLA has crashed or the port is wrong. It has not and it
    is not - the server is alive and merely waiting. Ticking it a few times
    and putting it back into asynchronous mode fixes it, no restart needed.

    Also destroys leftover vehicles and sensors from the dead run. Those keep
    driving around and rendering, and a leaked camera in particular shows up
    as "sensor object went out of the scope but the sensor is still alive".

USAGE
    python carla_reset.py            # after any crashed / killed collection run
    python carla_reset.py --keep     # restore async mode but leave actors alone

If this script ITSELF times out, the server is genuinely gone or hung at the
engine level - restart CarlaUE4.exe.
"""

import argparse

import carla


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2000)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument("--keep", action="store_true",
                    help="do not destroy leftover vehicles/sensors")
    args = ap.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    world = client.get_world()
    settings = world.get_settings()
    print(f"connected. map={world.get_map().name}")
    print(f"  synchronous_mode    = {settings.synchronous_mode}")
    print(f"  fixed_delta_seconds = {settings.fixed_delta_seconds}")

    if settings.synchronous_mode:
        # Tick it ourselves first. A wedged server is mid-frame waiting for a
        # tick, and some RPCs (apply_settings among them) can block behind
        # that until it gets one.
        print("\nserver is in synchronous mode - ticking it loose...")
        for _ in range(5):
            try:
                world.tick()
            except RuntimeError as e:
                print(f"  tick failed: {e}")
                break

    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("asynchronous mode restored")

    try:
        tm = client.get_trafficmanager()
        tm.set_synchronous_mode(False)
        print("traffic manager set to asynchronous")
    except RuntimeError as e:
        print(f"traffic manager reset skipped: {e}")

    if not args.keep:
        vehicles = list(world.get_actors().filter("vehicle.*"))
        sensors = list(world.get_actors().filter("sensor.*"))
        # Sensors first: a camera attached to a vehicle whose parent has
        # already been destroyed is exactly the "still alive in the
        # simulation" leak this is here to clear.
        killed = 0
        for a in sensors + vehicles:
            try:
                a.destroy()
                killed += 1
            except Exception:
                pass
        print(f"destroyed {killed} leftover actor(s) "
              f"({len(sensors)} sensor, {len(vehicles)} vehicle)")

    print("\nserver is clean - re-run your collection script.")


if __name__ == "__main__":
    main()

"""Monster spawns for a generated cave: packs placed room by room on level dry floor.

    "spawns": [{"type": "gigas", "room": "hub", "count": 2}, {"type": "genbu", "room": 2, "spot": "centre"}]

room: "entrance", "hub", "lair" or a room number. spot "centre" puts the pack on the middle of the room's
first lobe (e.g. a lake's platform); otherwise packs spread over the room's lobes. Positions come back in
the cave frame (x east, y north, z up) with a facing toward the lobe's centre.
"""
import math
import random

import numpy as np


def _room(net, key):
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        return net.rooms[int(key)]
    return next(r for r in net.rooms if r.name == key)


def place(net, spawns, types, seed):
    rng = random.Random(seed * 17 + 3)
    out = []
    taken = []

    def wet(x, y, room):
        return bool(room.lava) and not getattr(room, "frozen", True) and \
            net.channel(room, np.array([x]), np.array([y]))[0] > 0.2

    def level(room, x, y, reach):
        z = net.ground(x, y, room.floor + 3)
        if z is None:
            return None
        for k in range(6):
            a = k * math.pi / 3
            pz = net.ground(x + reach * math.cos(a), y + reach * math.sin(a), z + 2)
            if pz is None or abs(pz - z) > 1.0:
                return None
        return z

    def free(room, x, y, reach):
        if wet(x, y, room):
            return False
        if any(math.hypot(x - tx, y - ty) < reach + tr for tx, ty, tr in taken):
            return False
        if any(math.hypot(x - px, y - py) < pr + reach for px, py, pr in room.pillars):
            return False
        if any(sp[0] != "tite" and math.hypot(x - sp[1], y - sp[2]) < sp[5] + reach for sp in room.spikes):
            return False
        return True

    per_room = {}
    for s in spawns:
        per_room.setdefault(str(s["room"]), []).append(s)
    for key, packs in per_room.items():
        room = _room(net, key)
        for k, s in enumerate(packs):
            count = s.get("count", 1)
            size = types[s["type"]].get("model", {}).get("scale", 1.0)
            gap = 3.0 + 2.5 * size                      # bigger monsters stand further apart
            placed = False
            for attempt in range(80):
                if s.get("spot") == "centre":
                    lb = room.lobes[0]
                    cx, cy = lb.x, lb.y
                else:
                    lb = room.lobes[(k + attempt) % len(room.lobes)]
                    a = rng.uniform(0, 2 * math.pi)
                    d = rng.uniform(0.15, 0.6) * lb.r
                    cx, cy = lb.x + d * math.cos(a), lb.y + d * math.sin(a)
                spots = [(cx, cy)] + [(cx + gap * math.cos(2 * math.pi * i / max(1, count - 1)),
                                       cy + gap * math.sin(2 * math.pi * i / max(1, count - 1)))
                                      for i in range(count - 1)]
                zs = [level(room, x, y, 1.5) if free(room, x, y, gap / 2) else None for x, y in spots]
                if all(z is not None for z in zs):
                    for (x, y), z in zip(spots, zs):
                        out.append({"type": s["type"], "pos": (x, y, z),
                                    "facing": math.atan2(lb.y - y, lb.x - x) if (x, y) != (lb.x, lb.y) else 0.0})
                        taken.append((x, y, gap / 2))
                    placed = True
                    break
            if not placed:
                raise SystemExit(f"no room for {count} {s['type']} in {room.name}")
    return out

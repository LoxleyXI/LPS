"""Props from another zone's kit, placed in a generated cave by rules from the JSON.

    "props": {"from": 111, "rules": [
        {"where": "floor", "models": ["gar_c03_m", "gar_c02_m"], "per_lobe": [2, 4], "scale": [3, 6], "solid": true},
        {"where": "edge",  "models": ["gar_iwa01_m"], "per_lobe": [0, 2], "scale": [0.8, 1.2], "solid": true},
        {"where": "mouth", "models": ["mado_turo"], "scale": [3, 3]}]}

sink: bed the model into the ground by this fraction of its height (models like roots are made to be
partly buried). where: floor (dry level floor anywhere in a lobe), edge (dry level floor near the walls), mouth (at each
tunnel entrance, facing along the tunnel). Every prop needs its whole footprint on level dry ground,
clear of water, bridges, pillars, stalagmites and other props. solid: its triangles join the cave's
collision (you can't walk through it). Models are the source zone's own meshes, copied with their
textures at build time: nothing is shipped.

Frames: the cave frame is x east, y north, z up; FFXI's is x east, y down, z north (zonekit.writer.to_ffxi).
A placement's FFXI rotation is (0, yaw, 0): world = translate . rotateZYX . scale, so a model's local
+z faces (sin yaw, 0, cos yaw) in FFXI, i.e. (sin yaw, cos yaw) east/north in the cave frame.
"""
import math
import random

import numpy as np

from zonekit import writer as fz
from zonekit.borrow import load_kit  # noqa: F401  (caves.props.load_kit)


def _rot_y(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])       # rotateZYX with only a Y angle


def place(net, geometry, spec, kit, seed):
    """Placements for the rules in spec["rules"]: [{mesh, pos (cave), yaw_ffxi, scale, solid}]."""
    rng = random.Random(seed * 31 + 7)
    out = []
    taken = []                                          # (x, y, radius) already used

    def wet(x, y):
        for r in net.rooms:
            if r.lava and not getattr(r, "frozen", True) and r.inside(np.array([x]), np.array([y]))[0]:
                if net.channel(r, np.array([x]), np.array([y]))[0] > 0.2:
                    return True
        return False

    def level(room, x, y, reach):
        z = net.ground(x, y, room.floor + 3)
        if z is None:
            return None
        if room.lava and not getattr(room, "frozen", True) and z < room.lava_level + 0.3:
            return None                                     # at or under the water line (a bank)
        for k in range(8):
            a = k * math.pi / 4
            px, py = x + reach * math.cos(a), y + reach * math.sin(a)
            pz = net.ground(px, py, z + 2)
            if pz is None or abs(pz - z) > 0.8 or wet(px, py):
                return None
        return z

    def clear(room, x, y, reach):
        if wet(x, y):
            return False
        if any(math.hypot(x - tx, y - ty) < reach + tr + 1.5 for tx, ty, tr in taken):
            return False
        if any(fz_seg(x, y, b) < b[4] + reach + 1 for b in room.bridges):
            return False
        if any(math.hypot(x - px, y - py) < pr + reach + 1 for px, py, pr in room.pillars):
            return False
        if any(sp[0] != "tite" and math.hypot(x - sp[1], y - sp[2]) < sp[5] + reach + 1 for sp in room.spikes):
            return False
        return True

    def footprint(name, scale):
        k = kit[name]
        return max(k["hi"][0] - k["lo"][0], k["hi"][2] - k["lo"][2]) / 2 * scale

    for rule in spec["rules"]:
        lo_s, hi_s = rule.get("scale", [1.0, 1.0])
        solid = bool(rule.get("solid", False))
        where = rule["where"]
        for room in net.rooms:
            if where == "mouth":
                for lb in room.lobes:
                    for mx, my in net.mouths(room, lb):
                        name = rng.choice(rule["models"])
                        scale = rng.uniform(lo_s, hi_s)
                        z = net.ground(mx, my, room.floor + 6)
                        if z is None:
                            continue
                        ux, uy = mx - lb.x, my - lb.y
                        out.append({"mesh": name, "pos": (mx, my, z), "yaw": math.atan2(ux, uy),
                                    "scale": scale, "solid": solid})
                        taken.append((mx, my, footprint(name, scale)))
                continue
            for lb in room.lobes:
                want = int(round(rng.uniform(*rule.get("per_lobe", [1, 2])) * lb.r / 40.0))
                placed, tries = 0, 0
                while placed < want and tries < want * 40 + 40:
                    tries += 1
                    a = rng.uniform(0, 2 * math.pi)
                    rho = rng.uniform(0.6, 0.9) if where == "edge" else math.sqrt(rng.uniform(0, 0.85))
                    x, y = lb.x + rho * lb.r * math.cos(a), lb.y + rho * lb.r * math.sin(a)
                    name = rng.choice(rule["models"])
                    scale = rng.uniform(lo_s, hi_s)
                    reach = footprint(name, scale)
                    if not clear(room, x, y, reach):
                        continue
                    z = level(room, x, y, reach)
                    if z is None:
                        continue
                    k = kit[name]
                    z -= rule.get("sink", 0.0) * (k["hi"][1] - k["lo"][1]) * scale   # bedded in, not perched
                    out.append({"mesh": name, "pos": (x, y, z), "yaw": rng.uniform(0, 2 * math.pi),
                                "scale": scale, "solid": solid})
                    taken.append((x, y, reach))
                    placed += 1
    return out


def fz_seg(x, y, b):
    from .network import seg_dist
    return float(seg_dist(x, y, b[0], b[1], b[2], b[3]))


def object_pos(kit, p):
    """The FFXI object position that puts the model's bottom-centre on the placement's spot. Models
    aren't all centred on their origin (the stone archway sits 9-16 units off it), and FFXI's y is
    down, so a model's base is its largest local y."""
    k = kit[p["mesh"]]
    anchor = np.array([(k["lo"][0] + k["hi"][0]) / 2, k["hi"][1], (k["lo"][2] + k["hi"][2]) / 2])
    return fz.to_ffxi(np.array([p["pos"]]))[0] - _rot_y(p["yaw"]) @ (anchor * p["scale"])


def world_triangles(kit, p):
    """A placement's triangles in the cave frame, wound our way ((b-a)x(c-a) into the air)."""
    k = kit[p["mesh"]]
    if not len(k["tris"]):
        return np.zeros((0, 3, 3))
    local = k["tris"].reshape(-1, 3) * p["scale"]
    world_ffxi = local @ _rot_y(p["yaw"]).T + object_pos(kit, p)
    cave = fz.from_ffxi(world_ffxi).reshape(-1, 3, 3)
    return cave[:, [0, 2, 1]]                          # retail winding -> ours


def for_writer(kit, textures, placements, lod=(10.0, 150.0, 400.0)):
    """The writer's geometry["props"] block."""
    objs = []
    for p in placements:
        k = kit[p["mesh"]]
        corners = np.array([[x, y, z] for x in (k["lo"][0], k["hi"][0]) for y in (k["lo"][1], k["hi"][1])
                            for z in (k["lo"][2], k["hi"][2])]) * p["scale"]
        pos = object_pos(kit, p)
        w = corners @ _rot_y(p["yaw"]).T + pos
        objs.append({"name": k["name16"], "pos": pos, "rot": (0.0, p["yaw"], 0.0),
                     "scale": (p["scale"],) * 3, "lo": w.min(0), "hi": w.max(0), "lod": lod})
    used = {p["mesh"] for p in placements}
    return {"sections": [kit[n]["section"] for n in sorted(used)], "textures": textures, "objects": objs}

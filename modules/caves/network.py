"""Cave networks: rooms joined by tunnels, descending, from a seed (shared with the Solace WoW generator).

Measured from Molten Core (Sep 24): 12 spaces, 7 caverns (ceilings 33-65 yd, floors 70-120 yd
across) and 5 tunnels (17-27 wide, 14-20 high); a tree with one loop and a hub; ~150 yd of descent
in 15-50 yd steps, all ramps; lava rivers; walls near vertical, slightly overhanging.

How it's built (the same algorithm as the three.js demo):
  1. room graph from the seed: a tree grown from the entrance, one extra tunnel closing a loop;
     every hop goes down; the deepest room is the lair (lava moat round an island)
  2. a density field: rooms are ellipsoids, tunnels capsules, both with flat floors; a smooth
     minimum joins them; noise makes walls and ceilings lumpy (little on floors); pillars in halls
  3. surface nets turns the field into triangles (one vertex per cell, a quad per crossing edge),
     every face turned to look into the air (the client's collision is one-sided)
  4. lava: channels cut into some floors, one magma liquid grid with a height per room

Units are yards, server coordinates. Density > 0 is rock, < 0 is air.
"""
import math
import random

import numpy as np
from scipy import ndimage

SMOOTH = 7.0          # smooth-min blend where tunnels meet rooms (yd)
LAVA_BED = 5.0        # channels are cut this far below the floor
LAVA_DEPTH = 1.5      # the lava surface sits this far above the bed: you wade in it


def seg_dist(x, y, x0, y0, x1, y1):
    """Horizontal distance from points to a segment (works on arrays and plain numbers)."""
    dx, dy = x1 - x0, y1 - y0
    ll = dx * dx + dy * dy
    t = 0.0 if ll == 0 else np.clip(((x - x0) * dx + (y - y0) * dy) / ll, 0, 1)
    return np.hypot(x - (x0 + t * dx), y - (y0 + t * dy))


class Lobe:
    def __init__(self, x, y, r, h):
        self.x, self.y, self.r, self.h = x, y, r, h


class Room:
    """A space with one floor height, made of overlapping lobes (a hall is 2-3 in a line, a bend or an L)."""

    def __init__(self, i, kind, floor, lobes):
        self.i, self.kind, self.floor, self.lobes = i, kind, floor, lobes
        self.hall = kind != "room"
        self.name = f"room {i}"
        self.lava = []             # ("moat", lobe, inner, outer), ("river", lobe, angle, offset, half width),
                                   # ("rim", lobe, inner) or ("lake", lobe)
        self.lava_depth = LAVA_BED  # how far the lava's rock bed is below the floor
        self.bridges = []          # (x0, y0, x1, y1, half width): rock left standing over the lava
        self.pillars = []          # (x, y, radius)
        self.spikes = []           # ("mite" | "tite", x, y, base z, length, radius) and ("boulder", x, y, z, r, r)
        self.raises = []           # (x, y, radius, height, ramp angle, ramp length, ramp half width): a dais
        self.feature = kind
        self.degree = 0

    @property
    def x(self):
        return self.lobes[0].x

    @property
    def y(self):
        return self.lobes[0].y

    @property
    def r(self):
        return self.lobes[0].r

    @property
    def lava_level(self):
        return self.floor - self.lava_depth + LAVA_DEPTH

    def inside(self, x, y, frac=0.97):
        out = np.zeros(np.shape(x), dtype=bool)
        for lb in self.lobes:
            out |= np.hypot(x - lb.x, y - lb.y) < frac * lb.r
        return out

    def extent(self):
        xs = [lb.x for lb in self.lobes]
        ys = [lb.y for lb in self.lobes]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) + 2 * max(lb.r for lb in self.lobes)


class Network:
    def __init__(self, c):
        self.c = c
        self.rng = rng = random.Random(c["seed"])
        n = c.get("rooms", 9)
        self.rooms, self.tunnels = [], []

        def lobes_for(kind, x, y, ang):
            """Lobes of a new room whose first lobe is at (x, y), growing away along ang."""
            if kind == "room":
                return [Lobe(x, y, rng.uniform(22, 32), rng.uniform(14, 22))]
            if kind == "lair":
                return [Lobe(x, y, rng.uniform(60, 68), rng.uniform(46, 56))]
            r, h = rng.uniform(38, 52), rng.uniform(28, 42)
            count = rng.choice((2, 3, 3))
            shape = rng.choice(("line", "bend", "L"))
            turn = {"line": 0.0, "bend": rng.choice((-1, 1)) * 0.6, "L": rng.choice((-1, 1)) * math.pi / 2}[shape]
            out = [Lobe(x, y, r, h)]
            a = ang
            for k in range(1, count):
                if k == count - 1 or shape == "bend":
                    a += turn if k == count - 1 or shape == "bend" else 0
                rr, hh = r * rng.uniform(0.8, 1.15), h * rng.uniform(0.85, 1.15)
                step = (out[-1].r + rr) * rng.uniform(0.45, 0.6)          # overlap: one space, not two
                out.append(Lobe(out[-1].x + step * math.cos(a), out[-1].y + step * math.sin(a), rr, hh))
            return out

        extent = c.get("extent")                  # optional: every lobe within +-extent of the entrance

        def fits(lobes):
            if extent and any(abs(a.x) + a.r > extent or abs(a.y) + a.r > extent for a in lobes):
                return False
            return all(math.hypot(a.x - b.x, a.y - b.y) > a.r + b.r + 25
                       for a in lobes for o in self.rooms for b in o.lobes)

        def grow(parent, kind, drop):
            pl = rng.choice(parent.lobes)
            for _ in range(60):
                ang = rng.uniform(0, 2 * math.pi)
                length = rng.uniform(40, 85)                                   # tunnel, edge to edge
                r0 = 30 if kind == "room" else 64 if kind == "lair" else 45
                x, y = pl.x + (pl.r + r0 + length) * math.cos(ang), pl.y + (pl.r + r0 + length) * math.sin(ang)
                lobes = lobes_for(kind, x, y, ang)
                gap = math.hypot(x - pl.x, y - pl.y) - pl.r - lobes[0].r
                if gap < 30 or not fits(lobes):
                    continue
                room = Room(len(self.rooms), kind, parent.floor - min(drop, 0.45 * gap), lobes)
                self.rooms.append(room)
                self.tunnel(parent, pl, room, lobes[0])
                return room
            return None

        self.rooms.append(Room(0, "room", 0.0, [Lobe(0.0, 0.0, 28.0, 16.0)]))
        self.rooms[0].name = "entrance"
        tries = 0
        while len(self.rooms) < n - 1 and tries < 400:
            tries += 1
            # grow from a room with free exits; room 1 (the hub) is favoured
            free = [r for r in self.rooms if r.degree < (5 if r.i == 1 else 3)]
            a = rng.choices(free, [4 if r.i == 1 else 1 for r in free])[0]
            kind = "hall" if (len(self.rooms) == 1 or rng.random() < 0.55) else "room"
            grow(a, kind, rng.uniform(15, 40))
        if len(self.rooms) > 1:
            self.rooms[1].name = "hub"
        # one loop: the closest pair of lobes in rooms not yet joined
        joined = {(t[0].i, t[2].i) for t in self.tunnels} | {(t[2].i, t[0].i) for t in self.tunnels}
        best = None
        for a in self.rooms:
            for b in self.rooms:
                if a.i < b.i and (a.i, b.i) not in joined:
                    for la in a.lobes:
                        for lb in b.lobes:
                            gap = math.hypot(la.x - lb.x, la.y - lb.y) - la.r - lb.r
                            if 25 < gap < 110 and abs(a.floor - b.floor) < 0.45 * gap and (best is None or gap < best[0]):
                                best = (gap, a, la, b, lb)
        if best:
            self.tunnel(best[1], best[2], best[3], best[4])
        # the lair: one huge cavern off the deepest room, the dead end at the bottom
        deepest = sorted(self.rooms[1:], key=lambda r: r.floor)
        lair = None
        for base in deepest:
            lair = grow(base, "lair", rng.uniform(20, 35))
            if lair:
                break
        if lair:
            lair.name = "lair"
            lair.lava.append(("moat", 0, 0.34, 0.56))
        if lair:
            lair.lava_depth = 8.0
        # every hall gets one feature, dealt from a shuffled deck so neighbours differ
        deck = ["lake", "rim", "river across", "river along", "pillars"]
        rng.shuffle(deck)
        halls = [r for r in self.rooms if r.kind == "hall"]
        for k, r in enumerate(halls):
            r.feature = deck[k % len(deck)]
            big = max(range(len(r.lobes)), key=lambda i: r.lobes[i].r)
            if r.feature == "lake":                       # a lobe that is one deep lava pit, causeways across
                i = len(r.lobes) - 1                      # an end lobe: only one neighbour eats into it
                r.lava.append(("lake", i))
                r.lava_depth = 12.0
            elif r.feature == "rim":                      # a big plateau, lava all round its edge
                r.lava.append(("rim", big, 0.68))
                r.lava_depth = 10.0
            elif r.feature.startswith("river"):
                i = rng.randrange(len(r.lobes))
                lb = r.lobes[i]
                other = r.lobes[i - 1] if i else r.lobes[min(1, len(r.lobes) - 1)]
                along = math.atan2(other.y - lb.y, other.x - lb.x) if other is not lb else rng.uniform(0, math.pi)
                if r.feature == "river along":            # down the gallery, a walkway beside it
                    r.lava.append(("river", i, along, rng.choice((-1, 1)) * 0.35 * lb.r, rng.uniform(6, 9)))
                else:                                     # across it, with a bridge
                    half = rng.uniform(6, 8)
                    r.lava.append(("river", i, along + math.pi / 2, 0.0, half))
                    t = rng.uniform(-0.4, 0.4) * lb.r     # where along the river the bridge crosses
                    cx, cy = lb.x + t * math.cos(along + math.pi / 2), lb.y + t * math.sin(along + math.pi / 2)
                    reach = half + 8
                    r.bridges.append((cx - reach * math.cos(along), cy - reach * math.sin(along),
                                      cx + reach * math.cos(along), cy + reach * math.sin(along), 4.5))
                r.lava_depth = 6.0
            else:                                         # a pillar hall
                for lb in r.lobes:
                    for _ in range(rng.randint(2, 4)):
                        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.3, 0.6) * lb.r
                        r.pillars.append((lb.x + d * math.cos(a), lb.y + d * math.sin(a), rng.uniform(5, 10)))
        # bridges from every entrance into a lobe with a pit, lake or moat
        for r in self.rooms:
            for lava in r.lava:
                if lava[0] not in ("lake", "rim", "moat"):
                    continue
                lb = r.lobes[lava[1]]
                target = 0.0 if lava[0] in ("lake", "moat") else 0.55
                for mx, my in self.mouths(r, lb):
                    ux, uy = lb.x - mx, lb.y - my
                    dist = math.hypot(ux, uy)
                    ex, ey = lb.x - ux / dist * target * lb.r, lb.y - uy / dist * target * lb.r
                    r.bridges.append((mx, my, ex, ey, rng.uniform(4.5, 6)))
                if lava[0] == "lake":
                    r.bridges.append((lb.x, lb.y, lb.x, lb.y, rng.uniform(10, 14)))    # a platform in the middle
                    for other in r.lobes:                                              # causeways to the galleries
                        if other is not lb:
                            r.bridges.append((lb.x, lb.y, other.x, other.y, rng.uniform(5, 7)))
                if lava[0] == "moat" and not self.mouths(r, lb):
                    pass
        # a raised dais in halls without lava: flat top, steep sides, one ramp up (monsters stand on top)
        for r in self.rooms[1:]:
            if r.lava or r.name == "lair" or not r.hall:          # halls only: a dais needs a big room
                continue
            for lb in r.lobes:
                if rng.random() > 0.7:
                    continue
                for _ in range(20):
                    rad = rng.uniform(0.22, 0.32) * lb.r
                    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.1, 0.45) * lb.r
                    x, y = lb.x + d * math.cos(a), lb.y + d * math.sin(a)
                    if any(math.hypot(x - q[0], y - q[1]) < rad + q[2] + 12 for q in r.raises):
                        continue
                    if any(math.hypot(x - mx, y - my) < rad + 12 for mx, my in self.mouths(r, lb)):
                        continue
                    height = rng.uniform(4, 7)
                    ramp = math.atan2(lb.y - y, lb.x - x) if d > 1 else rng.uniform(0, 2 * math.pi)
                    r.raises.append((x, y, rad, height, ramp, height / 0.4, rng.uniform(4, 6)))
                    break
            r.pillars = [p for p in r.pillars
                         if all(math.hypot(p[0] - q[0], p[1] - q[1]) > q[2] + q[5] + p[2] + 4 for q in r.raises)]
        # stalactites everywhere, stalagmites and boulders on dry floor (and spires out of the lava lakes)
        for r in self.rooms:
            for lb in r.lobes:
                size = lb.r / 40.0
                for _ in range(int(rng.uniform(2, 6) * size + 1)):
                    a, d = rng.uniform(0, 2 * math.pi), math.sqrt(rng.uniform(0, 0.8)) * lb.r
                    r.spikes.append(("tite", lb.x + d * math.cos(a), lb.y + d * math.sin(a), 0.0,
                                     rng.uniform(8, 22), rng.uniform(3.5, 7)))
                for _ in range(int(rng.uniform(2, 5) * size + 1)):
                    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.55, 0.9) * lb.r
                    x, y = lb.x + d * math.cos(a), lb.y + d * math.sin(a)
                    r.spikes.append(("mite", x, y, r.floor, rng.uniform(6, 16), rng.uniform(3.5, 6.5)))
                for _ in range(rng.randint(0, 2)):
                    a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.4, 0.85) * lb.r
                    br = rng.uniform(3, 6)
                    r.spikes.append(("boulder", lb.x + d * math.cos(a), lb.y + d * math.sin(a),
                                     r.floor + br * 0.3, br, br))
            for lava in r.lava:
                if lava[0] == "lake":
                    lb = r.lobes[lava[1]]
                    for _ in range(rng.randint(2, 4)):
                        a, d = rng.uniform(0, 2 * math.pi), rng.uniform(0.35, 0.8) * lb.r
                        r.spikes.append(("mite", lb.x + d * math.cos(a), lb.y + d * math.sin(a),
                                         r.floor - r.lava_depth, r.lava_depth + rng.uniform(10, 24),
                                         rng.uniform(5, 9)))
        # nothing may stand on a bridge or a tunnel mouth
        for r in self.rooms:
            clear = [(x0, y0, x1, y1, w + 4) for x0, y0, x1, y1, w in r.bridges]
            for lb in r.lobes:
                clear += [(mx, my, mx, my, 16.0) for mx, my in self.mouths(r, lb)]
            for x, y, rad, h, ang, length, half in r.raises:
                clear.append((x, y, x + length * math.cos(ang), y + length * math.sin(ang), rad + 3))
            r.spikes = [sp for sp in r.spikes if sp[0] == "tite" or
                        all(seg_dist(sp[1], sp[2], *c[:4]) > c[4] + sp[5] for c in clear)]
            r.pillars = [p for p in r.pillars if all(seg_dist(p[0], p[1], *c[:4]) > c[4] + p[2] for c in clear)]
            dry = [sp for sp in r.spikes if sp[0] != "mite" or sp[3] < r.floor - 1
                   or self.channel(r, np.array([sp[1]]), np.array([sp[2]]))[0] < 0.05]
            r.spikes = dry

    def mouths(self, room, lobe):
        """Where tunnels enter this lobe: points on its edge."""
        out = []
        for a, la, b, lb, _ in self.tunnels:
            for mine, other in ((la, lb), (lb, la)):
                if mine is lobe:
                    ux, uy = other.x - mine.x, other.y - mine.y
                    d = math.hypot(ux, uy)
                    out.append((mine.x + ux / d * mine.r * 1.02, mine.y + uy / d * mine.r * 1.02))
        return out

    def tunnel(self, a, la, b, lb):
        wide = a.hall and b.hall and self.rng.random() < 0.5
        radius = (self.rng.uniform(16, 22) if wide else
                  5.0 if self.rng.random() < 0.15 else self.rng.uniform(8, 12))
        self.tunnels.append((a, la, b, lb, radius))
        a.degree += 1
        b.degree += 1

    def room(self, key):
        if isinstance(key, int):
            return self.rooms[key]
        return next(r for r in self.rooms if r.name == key)

    # ------------------------------------------------------------------ the field
    def channel(self, room, x, y):
        """0..1: how deep in this room's lava a point is (bridges are left standing)."""
        out = np.zeros(np.shape(x))
        for lava in room.lava:
            lb = room.lobes[lava[1]]
            dx, dy = x - lb.x, y - lb.y
            rho = np.hypot(dx, dy) / lb.r
            if lava[0] == "moat":
                _, _, lo, hi = lava
                edge = ((hi - lo) / 2 - np.abs(rho - (lo + hi) / 2)) * lb.r        # yards inside the band
            elif lava[0] == "rim":                     # along the whole room's outer wall: the middle
                edge = None                            # of every lobe stays a plateau
                for other in room.lobes:
                    e = (np.hypot(x - other.x, y - other.y) / other.r - lava[2]) * other.r
                    edge = e if edge is None else np.minimum(edge, e)
            elif lava[0] == "lake":
                edge = (0.9 - rho) * lb.r + 4
                for other in room.lobes:                                         # stays in its own lobe
                    if other is not lb:
                        edge = np.minimum(edge, np.hypot(x - other.x, y - other.y) - other.r * 0.5)
            else:
                _, _, ang, off, half = lava
                edge = half - np.abs(-dx * math.sin(ang) + dy * math.cos(ang) - off)
                edge = np.where(rho < 1.3, edge, -9)                              # this lobe's stretch
            out = np.maximum(out, np.clip(edge / 3.0 + 0.5, 0, 1))                  # 3 yd banks
        for x0, y0, x1, y1, w in room.bridges:
            keep = np.clip((w - seg_dist(x, y, x0, y0, x1, y1)) / 2.0 + 0.5, 0, 1)
            out = out * (1 - keep)
        return out

    def raised(self, room, x, y):
        """How far the floor is lifted here by a dais (its top, steep sides, and a ramp up)."""
        out = np.zeros(np.shape(x))
        for cx, cy, rad, h, ang, length, half in room.raises:
            d = np.hypot(x - cx, y - cy)
            top = h * np.clip((rad + 1.5 - d) / 1.5, 0, 1)                    # steep sides
            ux, uy = math.cos(ang), math.sin(ang)
            along = (x - cx) * ux + (y - cy) * uy - rad                       # 0 at the top's edge
            side = np.abs(-(x - cx) * uy + (y - cy) * ux)
            ramp = h * np.clip(1 - along / length, 0, 1) * np.clip((half + 1 - side) / 1.5, 0, 1)
            ramp = np.where(along > -rad, ramp, 0)
            out = np.maximum(out, np.maximum(top, ramp))
        return out

    def shapes(self, x, y, z):
        """Per shape: (distance, floor height) arrays. Room lobes first, then tunnels."""
        out = []
        for r in self.rooms:
            floor = r.floor - r.lava_depth * self.channel(r, x, y) + self.raised(r, x, y)
            for lb in r.lobes:
                cz = r.floor + 0.35 * lb.h
                dz = np.maximum(z - cz, 0)                      # a dome above, straight walls below (pits fit)
                e = (np.sqrt(((x - lb.x) / lb.r) ** 2 + ((y - lb.y) / lb.r) ** 2 + (dz / lb.h) ** 2) - 1) \
                    * min(lb.r, lb.h)
                out.append((e, floor))
        for a, la, b, lb, rad in self.tunnels:
            ax, ay, bx, by = la.x, la.y, lb.x, lb.y
            length = math.hypot(bx - ax, by - ay)
            ux, uy = (bx - ax) / length, (by - ay) / length
            s = np.clip((x - ax) * ux + (y - ay) * uy, 0, length)
            t = np.clip((s - la.r * 0.9) / max(1.0, length - 0.9 * (la.r + lb.r)), 0, 1)   # flat inside each room
            floor = a.floor + (b.floor - a.floor) * t
            px, py, pz = ax + ux * s, ay + uy * s, floor + rad * 0.6
            cap = np.sqrt((x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2) - rad
            out.append((cap, floor))
        return out

    def field(self, x, y, z, noise=None):
        """Density at points, and the floor height of the space each point belongs to."""
        shapes = self.shapes(x, y, z)
        d, floor = shapes[0]
        d, floor = d.copy(), floor.copy()
        for e, f in shapes[1:]:
            floor = np.where(e < d, f, floor)
            h = np.clip(SMOOTH - np.abs(d - e), 0, None) / SMOOTH
            d = np.minimum(d, e) - h * h * SMOOTH / 4
        d = np.maximum(d, floor - z)                                   # flat floors
        if noise is not None:
            d = d + noise * (0.12 + 0.88 * np.clip((z - floor) / 8.0, 0, 1))   # floors nearly flat
        for r in self.rooms:
            for px, py, pr in r.pillars:
                d = np.maximum(d, pr - np.hypot(x - px, y - py))
            for kind, sx, sy, sz, length, rad in r.spikes:
                dist = np.hypot(x - sx, y - sy)
                if kind == "boulder":
                    rock = rad - np.sqrt(dist ** 2 + (z - sz) ** 2)
                elif kind == "mite":                            # a cone standing on its base
                    rock = np.minimum(rad * (sz + length - z) / length, rad * 1.3) - dist
                    rock = np.where(z > sz - 12, rock, -99)
                else:                                           # a cone hanging from the ceiling above it
                    top = self.ceiling(r, sx, sy)
                    tip = top - length
                    rock = np.minimum(rad * (z - tip) / length, rad * 1.3) - dist
                    rock = np.where(z < top + 12, rock, -99)
                d = np.maximum(d, rock)
        return d, floor

    def ceiling(self, room, x, y):
        """Roughly where the dome is above a point (the lobe it's in)."""
        best = None
        for lb in room.lobes:
            rho = math.hypot(x - lb.x, y - lb.y) / lb.r
            if rho < 1:
                z = room.floor + 0.35 * lb.h + lb.h * math.sqrt(1 - rho * rho)
                best = z if best is None else max(best, z)
        return best if best is not None else room.floor + 10

    def bounds(self):
        lobes = [(r, lb) for r in self.rooms for lb in r.lobes]
        lo = [min(lb.x - lb.r for r, lb in lobes) - 20, min(lb.y - lb.r for r, lb in lobes) - 20,
              min(r.floor - r.lava_depth - 12 for r in self.rooms)]
        hi = [max(lb.x + lb.r for r, lb in lobes) + 20, max(lb.y + lb.r for r, lb in lobes) + 20,
              max(r.floor + 1.35 * lb.h for r, lb in lobes) + 15]
        return lo, hi

    def grid(self, voxel):
        lo, hi = self.bounds()
        n = [int(math.ceil((hi[i] - lo[i]) / voxel)) + 1 for i in range(3)]
        xs, ys, zs = (lo[i] + voxel * np.arange(n[i]) for i in range(3))
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        nrng = np.random.default_rng(self.c["seed"])
        noise = np.zeros(X.shape)
        amp = self.c.get("noise", 4.0)
        for octave in range(3):                                        # fBm: 22, 11, 5.5 yd
            step = 22.0 / 2 ** octave
            coarse = nrng.uniform(-1, 1, [int(math.ceil(n[i] * voxel / step)) + 4 for i in range(3)])
            coords = [np.arange(n[i]) * voxel / step + 1 for i in range(3)]
            C = np.meshgrid(*coords, indexing="ij")
            noise += ndimage.map_coordinates(coarse, C, order=3, mode="nearest") * amp / 2 ** octave
        d, floor = self.field(X, Y, Z, noise)
        d[0, :, :] = d[-1, :, :] = d[:, 0, :] = d[:, -1, :] = d[:, :, 0] = d[:, :, -1] = 10.0   # closed box
        self._noise = (noise, lo, voxel)
        return d, floor, lo

    def ground(self, x, y, near_z):
        """Floor height at (x, y): the first air-to-rock step going down from above near_z."""
        zs = np.arange(near_z + 12, near_z - 30, -0.25)
        X, Y = np.full_like(zs, x), np.full_like(zs, y)
        noise, lo, voxel = self._noise
        idx = [(X - lo[0]) / voxel, (Y - lo[1]) / voxel, (zs - lo[2]) / voxel]
        nz = ndimage.map_coordinates(noise, idx, order=1, mode="nearest")
        d, _ = self.field(X, Y, zs, nz)
        for k in range(1, len(zs)):
            if d[k - 1] < 0 <= d[k]:
                return float(zs[k - 1] + (zs[k] - zs[k - 1]) * (-d[k - 1]) / (d[k] - d[k - 1]))
        return None


# ------------------------------------------------------------------ surface nets
EDGES = [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
         ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
         ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]


def surface_nets(d, lo, voxel):
    """Vertices (N, 3) and triangles (M, 3) of the zero surface, faces looking into the air (d < 0)."""
    nx, ny, nz = d.shape
    total = np.zeros((nx - 1, ny - 1, nz - 1, 3))
    count = np.zeros((nx - 1, ny - 1, nz - 1))
    for c0, c1 in EDGES:
        d0 = d[c0[0]:nx - 1 + c0[0], c0[1]:ny - 1 + c0[1], c0[2]:nz - 1 + c0[2]]
        d1 = d[c1[0]:nx - 1 + c1[0], c1[1]:ny - 1 + c1[1], c1[2]:nz - 1 + c1[2]]
        cross = (d0 > 0) != (d1 > 0)
        t = np.where(cross, d0 / np.where(cross, d0 - d1, 1), 0)
        for a in range(3):
            total[..., a] += np.where(cross, c0[a] + t * (c1[a] - c0[a]), 0)
        count += cross
    active = count > 0
    vid = np.full(count.shape, -1, dtype=np.int64)
    vid[active] = np.arange(active.sum())
    cells = np.argwhere(active)
    verts = lo + (cells + total[active] / count[active][:, None]) * voxel

    solid = d > 0
    quads, want = [], []
    for axis in range(3):
        # grid edges along this axis with a sign change, away from the border
        sl = [slice(1, n - 1) for n in d.shape]
        sl[axis] = slice(0, d.shape[axis] - 1)
        a = solid[tuple(sl)]
        sl2 = list(sl)
        sl2[axis] = slice(1, d.shape[axis])
        b = solid[tuple(sl2)]
        idx = np.argwhere(a != b)
        idx[:, [i for i in range(3) if i != axis]] += 1
        rock_low = a[a != b]                                           # rock at the low end: air is +axis
        u, v = [i for i in range(3) if i != axis]
        ring = []
        for du, dv in ((-1, -1), (0, -1), (0, 0), (-1, 0)):
            c = idx.copy()
            c[:, u] += du
            c[:, v] += dv
            ring.append(vid[c[:, 0], c[:, 1], c[:, 2]])
        quads.append(np.stack(ring, 1))
        w = np.zeros((len(idx), 3))
        w[:, axis] = np.where(rock_low, 1.0, -1.0)
        want.append(w)
    quads, want = np.concatenate(quads), np.concatenate(want)
    # turn each quad as a whole toward the air (its diagonals' cross product), then split it along
    # the diagonal whose two triangles agree best: splitting a bent quad the wrong way folds a
    # triangle back into the rock, which the client lights from behind (dark triangles)
    q = verts[quads]
    qn = np.cross(q[:, 2] - q[:, 0], q[:, 3] - q[:, 1])
    back = (qn * want).sum(1) < 0
    quads[back] = quads[back][:, [0, 3, 2, 1]]
    qn[back] = -qn[back]
    q = verts[quads]

    def agree(a, b, c):
        n = np.cross(q[:, b] - q[:, a], q[:, c] - q[:, a])
        return (n * qn).sum(1) / (np.linalg.norm(n, axis=1) * np.linalg.norm(qn, axis=1) + 1e-12)
    split_a = np.minimum(agree(0, 1, 2), agree(0, 2, 3))
    split_b = np.minimum(agree(0, 1, 3), agree(1, 2, 3))
    use_b = split_b > split_a
    t1 = np.where(use_b[:, None], quads[:, [0, 1, 3]], quads[:, [0, 1, 2]])
    t2 = np.where(use_b[:, None], quads[:, [1, 2, 3]], quads[:, [0, 2, 3]])
    tris = np.concatenate([t1, t2])
    keep = (tris[:, 0] != tris[:, 1]) & (tris[:, 1] != tris[:, 2]) & (tris[:, 0] != tris[:, 2])
    return verts, tris[keep]


def vertex_normals(verts, tris):
    """Area-weighted normals on the shared mesh (before any per-material copies)."""
    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, tris[:, k], n)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)

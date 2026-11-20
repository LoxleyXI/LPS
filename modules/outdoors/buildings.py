"""Buildings: another zone's own houses, learned whole and set on level ground beside our roads.

A house in a zone is many pieces: La Vaule's is a base, an entrance, a roof (one of three), walls,
a porch, a lamp and a chimney, each its own model placed round the same spot. learn() takes each
instance of an anchor model in the source and keeps, with it, every piece that stands round most
instances of it (the set) and seldom anywhere else (a rock wall that happens to be near isn't part
of the house). Each instance becomes a template, so its own roof and dressing come with it.

    "buildings": {"from": 85, "sets": ["davoi_h_base", "davoi_co_c"], "count": [8, 12]}

place() finds plots: level ground a template's whole footprint fits on, beside a road and off it,
clear of the other buildings; the front (the side its entrance piece is on) faces the road.

Models are the source zone's own meshes, copied with their textures at build time: nothing shipped.
"""
import math
from collections import Counter

import numpy as np

from zonekit import tiles as T

TILE = T.TILE


def _unrot(v, yaw):
    c, s = math.cos(-yaw), math.sin(-yaw)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def _rot(v, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


def learn(tris, objs, name_of, anchors, share=0.7, own=0.5, radius=None, core=None):
    """{anchor: [template]} for the named anchors. A template: {"parts": [(name16, dyaw, scale,
    offset in the anchor's frame)], "box": (x0, z0, x1, z1) footprint in the anchor's frame,
    "front": unit (x, z) in the anchor's frame}. With `core` (yalms), the footprint is only what
    stands that far above the anchor's own ground (a tower built into a ground tile, not its
    ground), and "full_box" keeps the whole extent."""
    out = {}
    for a in anchors:
        inst = []
        for o in objs:                                     # one per spot: a house can be built of two
            if name_of(o["name"]) == a and o["name"] in tris and not any(   # of its anchor, back to back
                    math.hypot(o["pos"][0] - q["pos"][0], o["pos"][2] - q["pos"][2]) < 1.0 for q in inst):
                inst.append(o)
        if not inst:
            raise ValueError(f"buildings: no {a!r} placed in the source")
        t = tris[inst[0]["name"]].reshape(-1, 3)
        r = radius or (0.5 * float(max(np.ptp(t[:, 0]), np.ptp(t[:, 2]))) + 8.0)
        # (never the land: a building's set is what stands on it)
        near = {id(o): [q for q in objs if q is not o and q["name"] in tris and not T.is_tile(tris[q["name"]])
                        and math.hypot(q["pos"][0] - o["pos"][0], q["pos"][2] - o["pos"][2]) < r] for o in inst}
        # the set: names round most instances ...
        seen = Counter()
        for o in inst:
            seen.update({name_of(q["name"]) for q in near[id(o)]})
        cands = {n for n, k in seen.items() if k >= share * len(inst)}
        # and whatever shares an instance's own spot (its roof, one of several)
        cands |= {name_of(q["name"]) for o in inst for q in near[id(o)]
                  if math.hypot(q["pos"][0] - o["pos"][0], q["pos"][2] - o["pos"][2]) < 1.0}
        # ... that are mostly found round one (not the land's own dressing)
        total = Counter(name_of(q["name"]) for q in objs)
        by = Counter(name_of(q["name"]) for o in inst for q in near[id(o)])
        members = {n for n in cands if by[n] >= own * total[n]} | {a}
        temps = []
        for o in inst:
            yaw = o["rot"][1]
            parts = [(o["name"], 0.0, tuple(o["scale"]), np.zeros(3))]
            for q in near[id(o)]:
                if name_of(q["name"]) in members:
                    parts.append((q["name"], q["rot"][1] - yaw, tuple(q["scale"]),
                                  _unrot(np.array(q["pos"]) - np.array(o["pos"]), yaw)))
            # footprint: every part's placed points, in the anchor's frame
            pts = []
            for name, dyaw, scale, off in parts:
                w = T.placed(tris[name], {"rot": (0.0, dyaw, 0.0), "scale": scale, "pos": tuple(off)}).reshape(-1, 3)
                pts.append(w)
            pts = np.concatenate(pts)
            box = (float(pts[:, 0].min()), float(pts[:, 2].min()), float(pts[:, 0].max()), float(pts[:, 2].max()))
            temp = {"anchor": a, "parts": parts, "box": box, "names": sorted({name_of(p[0]) for p in parts})}
            if core is not None:
                # FFXI y is down: the ground is the largest y, standing parts are well above it
                up = pts[pts[:, 1] < pts[:, 1].max() - core]
                if len(up):
                    temp["full_box"] = box
                    temp["box"] = (float(up[:, 0].min()), float(up[:, 2].min()), float(up[:, 0].max()), float(up[:, 2].max()))
            temp["front"] = _front(tris, parts, name_of)
            temps.append(temp)
        out[a] = temps
    return out


def _front(tris, parts, name_of):
    """Which way the building faces: toward its entrance pieces (names with 'ent', 'do', 'door'),
    else toward the part furthest from the middle."""
    offs = [(name_of(n), off) for n, _, _, off in parts[1:]]
    ent = [off for n, off in offs if any(k in n for k in ("ent", "_do", "door"))]
    pick = ent or [max((off for _, off in offs), key=lambda v: math.hypot(v[0], v[2]), default=np.array([0, 0, 1.0]))]
    v = np.mean(pick, 0)
    d = np.array([v[0], v[2]])
    if np.linalg.norm(d) < 1e-6:
        return (0.0, 1.0)
    # to the nearest quarter: houses sit square to their street
    ang = round(math.atan2(d[0], d[1]) / (math.pi / 2)) * (math.pi / 2)
    return (math.sin(ang), math.cos(ang))


def pick(templates, rng):
    """A template: its group by weight (each set of a group gets its share), then one of the set's."""
    keys = list(templates)
    w = [templates[k][0].get("weight", 1.0) for k in keys]
    return rng.choice(templates[rng.choices(keys, weights=w)[0]])


def footprint(temp, yaw, x, z, pad=0.0):
    """The template's box turned by yaw and set at (x, z): its four corners (world x, z)."""
    x0, z0, x1, z1 = temp["box"]
    pts = []
    for bx, bz in ((x0 - pad, z0 - pad), (x1 + pad, z0 - pad), (x1 + pad, z1 + pad), (x0 - pad, z1 + pad)):
        v = _rot(np.array([bx, 0.0, bz]), yaw)
        pts.append((x + v[0], z + v[2]))
    return pts


def lay(temp, yaw, x, z, h, sink=0.5):
    """The template's pieces placed with its anchor at (x, z) on ground height h:
    [{"name", "pos", "rot", "scale"}] FFXI."""
    out = []
    for name, dyaw, scale, off in temp["parts"]:
        v = _rot(off, yaw)
        out.append({"name": name, "pos": (x + v[0], -(h - sink) + v[1], z + v[2]),
                    "rot": (0.0, yaw + dyaw, 0.0), "scale": scale, "ground": h})
    return out


def place(templates, ground, placed, cfg, rng, road_cells, off_road, count, keep_clear=()):
    """Up to `count` buildings: (template, [pieces], (x, z, r)). Plots are level ground beside a road
    cell, the whole footprint within `level` of one height and off every continuous feature
    (off_road(x, z) True where the ground carries none); fronts face the road."""
    level = cfg.get("level", 0.8)
    gap = cfg.get("gap", 3.0)
    taken = [tuple(k) for k in keep_clear]
    out = []
    pool = [t for ts in templates.values() for t in ts]
    sites = []
    for (i, j) in road_cells:
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (i + di, j + dj)
            if n in placed and n not in road_cells:
                sites.append((n, (-di, -dj)))              # the plot's cell, and the way to the road
    rng.shuffle(sites)
    for (i, j), face in sites:
        if len(out) >= count:
            break
        temp = pick(templates, rng)
        # turn the template so its front looks at the road
        fa = math.atan2(temp["front"][0], temp["front"][1])
        yaw = math.atan2(face[0], face[1]) - fa
        x0, z0, x1, z1 = temp["box"]
        depth = max(abs(x1 - x0), abs(z1 - z0))
        # stand it in its cell, as close to the road side as its footprint allows
        cx, cz = (i + 0.5) * TILE, (j + 0.5) * TILE
        for pull in np.linspace(TILE / 2 - depth / 2 - 2, 0, 5):
            x, z = cx + face[0] * pull, cz + face[1] * pull
            fp = footprint(temp, yaw, x, z)
            xs = np.array([p[0] for p in fp])
            zs = np.array([p[1] for p in fp])
            gxx, gzz = np.meshgrid(np.linspace(xs.min(), xs.max(), 6), np.linspace(zs.min(), zs.max(), 6))
            hs = ground.at(gxx.ravel(), gzz.ravel())
            if np.isnan(hs).any() or np.ptp(hs) > level:
                continue
            if not all(off_road(a, b) for a, b in zip(gxx.ravel(), gzz.ravel())):
                continue
            r = 0.5 * math.hypot(np.ptp(xs), np.ptp(zs))
            if any(math.hypot(x - a, z - b) < r + rr + gap for a, b, rr in taken):
                continue
            h = float(np.min(hs))
            out.append((temp, lay(temp, yaw, x, z, h, cfg.get("sink", 0.5)), (x, z, r)))
            taken.append((x, z, r))
            break
    return out


def site(ground, placed, feats, level, need=(3, 3), most=(6, 5)):
    """The biggest rectangle of cells (at least `need`, at most `most`, in cells) whose ground is
    level to `level` at one height and carries no feature (river, road). Returns (i0, j0, i1, j1, h)
    with i1, j1 exclusive, or None."""
    flat = {}
    for (i, j) in placed:
        if (i, j) in feats:
            continue
        xs, zs = np.meshgrid(np.linspace(i * TILE + 1, (i + 1) * TILE - 1, 7), np.linspace(j * TILE + 1, (j + 1) * TILE - 1, 7))
        h = ground.at(xs.ravel(), zs.ravel())
        if not np.isnan(h).any() and np.ptp(h) <= level:
            flat[(i, j)] = float(np.median(h))
    best = None
    for (i0, j0), h in flat.items():
        for w in range(need[0], most[0] + 1):
            for d in range(need[1], most[1] + 1):
                for a, b in ((w, d), (d, w)):
                    cells = [(i0 + x, j0 + y) for x in range(a) for y in range(b)]
                    if all(c in flat and abs(flat[c] - h) <= level for c in cells):
                        if best is None or a * b > (best[2] - best[0]) * (best[3] - best[1]):
                            best = (i0, j0, i0 + a, j0 + b, h)
    return best


def town(templates, ground, box, spec, rng):
    """A village on the site `box` (cells): streets along its long side, houses shoulder to
    shoulder facing them from both sides. Returns ([(template, pieces, (x, z, r))], streets:
    [((x0, z0), (x1, z1), width)], spots: [(x, z, yaw)] along the streets for townsfolk)."""
    i0, j0, i1, j1, h = box
    x0, z0, x1, z1 = i0 * TILE, j0 * TILE, i1 * TILE, j1 * TILE
    along_x = (x1 - x0) >= (z1 - z0)
    length, breadth = (x1 - x0, z1 - z0) if along_x else (z1 - z0, x1 - x0)
    pool = [t for ts in templates.values() for t in ts]
    street = spec.get("street", 10.0)
    gap = spec.get("gap", [3.0, 6.0])
    level = spec.get("level", 2.0)
    deep = max(max(t["box"][2] - t["box"][0], t["box"][3] - t["box"][1]) for t in pool)
    band = street + 2 * (deep + 2)                  # a street with a row of houses each side
    rows = max(1, int((breadth - 8) // band))
    first = (breadth - rows * band) / 2
    mids = [first + r * band + band / 2 for r in range(rows)]
    if spec.get("snap"):                             # streets down the middle of a row of cells,
        mids = sorted({(math.floor(m / TILE) + 0.5) * TILE for m in mids})   # where a gate can meet them
    edge = spec.get("edge", 6.0)                     # a margin inside the site's rim

    def world(u, v):
        return (x0 + u, z0 + v) if along_x else (x0 + v, z0 + u)
    out, streets, spots = [], [], []
    for mid in mids:                                # each street's centre line across the site
        a, b = world(edge, mid), world(length - edge, mid)
        streets.append((a, b, street))
        for side in (-1, 1):
            u = edge
            while True:
                temp = pick(templates, rng)
                # face the street: the front turns toward -side across the site
                fa = math.atan2(temp["front"][0], temp["front"][1])
                look = (0.0, -side) if along_x else (-side, 0.0)
                yaw = math.atan2(look[0], look[1]) - fa
                fp = footprint(temp, yaw, 0.0, 0.0)
                us = [p[0] if along_x else p[1] for p in fp]
                vs = [p[1] if along_x else p[0] for p in fp]
                wid = max(us) - min(us)
                if u + wid > length - edge:
                    break
                # the anchor so the front edge meets the street's side and the house starts at u
                au = u - min(us)
                front_v = mid + side * street / 2
                av = front_v - (min(vs) if side > 0 else max(vs))
                x, z = world(au, av)
                fp = footprint(temp, yaw, x, z)
                xs, zs = np.array([p[0] for p in fp]), np.array([p[1] for p in fp])
                gx, gz = np.meshgrid(np.linspace(xs.min(), xs.max(), 6), np.linspace(zs.min(), zs.max(), 6))
                hs = ground.at(gx.ravel(), gz.ravel())
                inside = xs.min() >= x0 + 1 and xs.max() <= x1 - 1 and zs.min() >= z0 + 1 and zs.max() <= z1 - 1
                if inside and not np.isnan(hs).any() and np.ptp(hs) <= level:
                    out.append((temp, lay(temp, yaw, x, z, float(np.min(hs)), spec.get("sink", 0.5)),
                                (x, z, 0.5 * math.hypot(np.ptp(xs), np.ptp(zs)))))
                    # townsfolk stand in the street in front of it
                    su, sv = au, mid + side * street * 0.2
                    sx, sz = world(su, sv)
                    spots.append((sx, sz, math.atan2(-look[0], -look[1])))
                u += wid + rng.uniform(*gap)
    return out, streets, spots


def _overlap(a, b, gap=0.0):
    """True when two convex footprints [(x, z)] overlap (or come within `gap`): separating axes."""
    for poly in (a, b):
        for k in range(len(poly)):
            (ax, az), (bx, bz) = poly[k], poly[(k + 1) % len(poly)]
            nx, nz = az - bz, bx - ax
            n = math.hypot(nx, nz) or 1.0
            nx, nz = nx / n, nz / n
            pa = [x * nx + z * nz for x, z in a]
            pb = [x * nx + z * nz for x, z in b]
            if max(pa) + gap < min(pb) or max(pb) + gap < min(pa):
                return False
    return True


HAMLET_ORDER = ["edge", "edge", "fire", "door", "fire", "door", "fire", "door", "fire", "edge"]


def hamlet(templates, ground, box, spec, rng):
    """A few houses round a campfire on the site `box` (cells), their fronts to the fire, with an
    open side where the way in is. Each town differs by its seed and spec: "houses" [lo, hi],
    "arc" (degrees the houses spread over), "clearing" (yalms from the fire to the house fronts),
    "jitter" (degrees). Returns ([(template, pieces, (x, z, r))], [], spots: [(x, z, yaw)], fire:
    (x, z), camp). Spots, in "order" (kinds "edge": by the way in, "door": outside a house, "fire":
    round the fire), each facing the fire; the first is the way in. camp: what was laid, as the
    workbench's camp editor reads and writes it ({"houses": [{"anchor", "variant", "x", "z",
    "yaw"}], "fire": [x, z], "spots": [[x, z, yaw]]}, the site's own frame); spec "fixed", in
    that form, lays exactly that instead of a camp of its own."""
    if spec.get("fixed"):
        return _fixed_camp(templates, ground, box, spec)
    i0, j0, i1, j1, h = box
    x0, z0, x1, z1 = i0 * TILE, j0 * TILE, i1 * TILE, j1 * TILE
    level = spec.get("level", 2.0)
    margin = spec.get("edge", 4.0)
    clearing = spec.get("clearing", 11.0)
    arc = math.radians(spec.get("arc", 240.0))
    jitter = math.radians(spec.get("jitter", 15.0))
    lo, hi = spec.get("houses", [2, 3])

    def inside(x, z, pad=0.0):
        return x0 + margin + pad <= x <= x1 - margin - pad and z0 + margin + pad <= z <= z1 - margin - pad

    roads = spec.get("_roads") or set()
    road_gap = spec.get("road_gap", 4.0)

    def off_road(fp):
        """no part of a footprint (and a margin round it) on a road's cell"""
        xs, zs = [p[0] for p in fp], [p[1] for p in fp]
        for px in np.linspace(min(xs) - road_gap, max(xs) + road_gap, 6):
            for pz in np.linspace(min(zs) - road_gap, max(zs) + road_gap, 6):
                if (int(math.floor(px / TILE)), int(math.floor(pz / TILE))) in roads:
                    return False
        return True

    def level_at(x, z):
        g = ground.at([x], [z])[0]
        return None if np.isnan(g) else float(g)

    def facing(x, z, tx, tz):
        return math.atan2(tx - x, tz - z)

    best = None
    for _attempt in range(40):
        cx = (x0 + x1) / 2 + rng.uniform(-6, 6)
        cz = (z0 + z1) / 2 + rng.uniform(-6, 6)
        opening = rng.uniform(0, 2 * math.pi)        # the way in: the side with no house
        n = rng.randint(lo, hi)
        houses, doors, laid = [], [], []
        for k in range(n):
            a = opening + math.pi + ((k / (n - 1) - 0.5) * arc if n > 1 else 0.0) + rng.uniform(-jitter, jitter)
            d = (math.sin(a), math.cos(a))           # from the fire out to this house
            for _try in range(12):
                temp = pick(templates, rng)
                fa = math.atan2(temp["front"][0], temp["front"][1])
                yaw = math.atan2(-d[0], -d[1]) - fa + rng.uniform(-jitter, jitter) / 2
                fp0 = footprint(temp, yaw, 0.0, 0.0)
                near = min(px * d[0] + pz * d[1] for px, pz in fp0)
                dist = clearing + rng.uniform(0, 4) - near
                x, z = cx + d[0] * dist, cz + d[1] * dist
                fp = footprint(temp, yaw, x, z)
                if not all(inside(px, pz) for px, pz in fp):
                    continue
                if any(_overlap(fp, other, 3.0) for _t, _p, other in houses):
                    continue
                if roads and not off_road(fp):
                    continue
                # a building on its own ground tile: all of that ground must be level too (sunk
                # under the town's, it would show wherever the land falls away)
                full = footprint({**temp, "box": temp["full_box"]}, yaw, x, z) if "full_box" in temp else fp
                xs, zs = np.array([p[0] for p in full]), np.array([p[1] for p in full])
                gx, gz = np.meshgrid(np.linspace(xs.min(), xs.max(), 6), np.linspace(zs.min(), zs.max(), 6))
                hs = ground.at(gx.ravel(), gz.ravel())
                if np.isnan(hs).any() or np.ptp(hs) > level:
                    continue
                houses.append((temp, lay(temp, yaw, x, z, float(np.min(hs)), temp.get("sink", spec.get("sink", 0.5))), fp))
                laid.append((temp, x, z, yaw))
                # outside its door: a step in front of the face toward the fire, a little aside
                side = rng.choice((-1, 1)) * rng.uniform(1.5, 3.0)
                r = clearing - 2.5
                doors.append((cx + d[0] * r + d[1] * side, cz + d[1] * r - d[0] * side))
                break
        if len(houses) >= lo and (best is None or len(houses) > len(best[0])):
            best = (houses, doors, cx, cz, opening, laid)
        if best and len(best[0]) == n:
            break
    if best is None:
        if spec.get("_roads"):
            # no camp keeps off the paths here (a site that is mostly road): lay one anyway, to be
            # moved by hand in the workbench's camp editor
            print("  buildings: no camp off the paths here; laid one across them (place it by hand)")
            return hamlet(templates, ground, box, {**spec, "_roads": set()}, rng)
        return [], [], [], None, None
    houses, doors, cx, cz, opening, laid = best

    o = (math.sin(opening), math.cos(opening))
    edges = []
    for off in (0.0, 4.0, -4.0, 8.0, -8.0):          # the way in, then either side of it
        for r in (clearing + 6, clearing + 3, clearing + 9, clearing):
            x, z = cx + o[0] * r + o[1] * off, cz + o[1] * r - o[0] * off
            if inside(x, z):
                edges.append((x, z))
                break
    ring = []
    start = opening + rng.uniform(0.3, 0.8)
    for k in range(8):                               # round the fire, a step back from it
        a = start + k * 2 * math.pi / 8 + rng.uniform(-0.15, 0.15)
        r = rng.uniform(3.5, 4.5)
        ring.append((cx + math.sin(a) * r, cz + math.cos(a) * r))

    pools = {"edge": edges, "door": doors, "fire": ring}
    spots = []
    def usable(x, z):
        return inside(x, z) and level_at(x, z) is not None and not any(
            _overlap([(x - 0.8, z - 0.8), (x + 0.8, z - 0.8), (x + 0.8, z + 0.8), (x - 0.8, z + 0.8)], fp)
            for _t, _p, fp in houses)

    for kind in spec.get("order", HAMLET_ORDER):
        # every entry in the order gets a spot (the server counts on their places): its own kind's
        # next one, else the fire's
        for pool in (pools.get(kind, []), ring):
            found = None
            while pool:
                x, z = pool.pop(0)
                if usable(x, z):
                    found = (x, z)
                    break
            if found:
                spots.append((found[0], found[1], facing(found[0], found[1], cx, cz)))
                break
    for x, z in ring:                                # what's left of the ring, for anyone else
        if inside(x, z) and level_at(x, z) is not None:
            spots.append((x, z, facing(x, z, cx, cz)))
    out = [(t, p, (sum(q[0] for q in fp) / 4, sum(q[1] for q in fp) / 4,
                   0.5 * math.hypot(np.ptp([q[0] for q in fp]), np.ptp([q[1] for q in fp]))))
           for t, p, fp in houses]
    camp = {"houses": [{"anchor": t["anchor"], "variant": _variant(templates, t), "x": x, "z": z, "yaw": yaw}
                       for t, x, z, yaw in laid],
            "fire": [cx, cz], "spots": [[x, z, yaw] for x, z, yaw in spots]}
    return out, [], spots, (cx, cz), camp


def _variant(templates, temp):
    """Which of its anchor's templates this is (the editor names a house by anchor and variant)."""
    return next(i for i, t in enumerate(templates[temp["anchor"]]) if t is temp)


def _fixed_camp(templates, ground, box, spec):
    """hamlet()'s returns for a camp laid by hand (spec "fixed", from the workbench's camp editor):
    each house on the lowest ground under it (as a laid one is), each spot facing the fire unless
    it says otherwise."""
    fixed = spec["fixed"]
    out = []
    for hse in fixed.get("houses", []):
        ts = templates.get(hse["anchor"]) or []
        if not ts:
            continue
        temp = ts[min(int(hse.get("variant", 0)), len(ts) - 1)]
        x, z, yaw = float(hse["x"]), float(hse["z"]), float(hse["yaw"])
        fp = footprint(temp, yaw, x, z)
        full = footprint({**temp, "box": temp["full_box"]}, yaw, x, z) if "full_box" in temp else fp
        xs, zs = np.array([p[0] for p in full]), np.array([p[1] for p in full])
        gx, gz = np.meshgrid(np.linspace(xs.min(), xs.max(), 6), np.linspace(zs.min(), zs.max(), 6))
        hs = ground.at(gx.ravel(), gz.ravel())
        h = float(np.nanmin(hs)) if np.isfinite(hs).any() else float(box[4])
        pieces = lay(temp, yaw, x, z, h, temp.get("sink", spec.get("sink", 0.5)))
        out.append((temp, pieces, (sum(q[0] for q in fp) / 4, sum(q[1] for q in fp) / 4,
                                   0.5 * math.hypot(np.ptp([q[0] for q in fp]), np.ptp([q[1] for q in fp])))))
    cx, cz = fixed["fire"]
    spots = []
    for s in fixed.get("spots", []):
        x, z = float(s[0]), float(s[1])
        yaw = float(s[2]) if len(s) > 2 and s[2] is not None else math.atan2(cx - x, cz - z)
        spots.append((x, z, yaw))
    camp = {"houses": [dict(hse) for hse in fixed.get("houses", [])], "fire": [cx, cz],
            "spots": [[x, z, yaw] for x, z, yaw in spots], "fixed": True}
    return out, [], spots, (cx, cz), camp


def sites(ground, placed, feats, level, n, need=(3, 3), most=(6, 5), apart=1):
    """Up to n town sites (see site()), each clear of the others by `apart` cells and, where the
    land allows, at a height of its own: a town that climbs its hill terrace by terrace."""
    out, used = [], set(feats)
    for _ in range(n):
        s = site(ground, placed, used, level, need, most)
        if s is None:
            break
        out.append(s)
        i0, j0, i1, j1, h = s
        used |= {(i, j) for i in range(i0 - apart, i1 + apart) for j in range(j0 - apart, j1 + apart)}
        # the next wants another level: every cell at this one is spent
        same = [c for c in placed if c not in used]
        for c in same:
            x, z = (c[0] + 0.5) * TILE, (c[1] + 0.5) * TILE
            g = ground.at([x], [z])[0]
            if not np.isnan(g) and abs(g - h) <= level:
                used.add(c)
    return out


def towers(temps, ground, box, rng, level=2.0, inset=6.0, sink=0.5):
    """A watchtower at each corner of a site that has level ground for it, facing out."""
    i0, j0, i1, j1, _ = box
    pool = [t for ts in temps.values() for t in ts]
    out = []
    for cx, cz, ox, oz in ((i0 * TILE, j0 * TILE, -1, -1), (i1 * TILE, j0 * TILE, 1, -1),
                           (i1 * TILE, j1 * TILE, 1, 1), (i0 * TILE, j1 * TILE, -1, 1)):
        temp = rng.choice(pool)
        fa = math.atan2(temp["front"][0], temp["front"][1])
        look = (ox, 0.0) if rng.random() < 0.5 else (0.0, oz)
        yaw = math.atan2(look[0], look[1]) - fa
        fp = footprint(temp, yaw, 0.0, 0.0)
        half = max(max(abs(p[0]) for p in fp), max(abs(p[1]) for p in fp))
        x, z = cx - ox * (half + inset), cz - oz * (half + inset)
        fp = footprint(temp, yaw, x, z)
        xs, zs = np.array([p[0] for p in fp]), np.array([p[1] for p in fp])
        gx, gz = np.meshgrid(np.linspace(xs.min(), xs.max(), 5), np.linspace(zs.min(), zs.max(), 5))
        hs = ground.at(gx.ravel(), gz.ravel())
        if np.isnan(hs).any() or np.ptp(hs) > level:
            continue
        out.append((temp, lay(temp, yaw, x, z, float(np.min(hs)), sink), (x, z, 0.5 * math.hypot(np.ptp(xs), np.ptp(zs)))))
    return out


def walls(piece, length, ground, box, gates, level=1.5, inset=2.0, sink=0.3, taken=(), mid=(0.0, 0.0, 0.0)):
    """Wall segments (a piece `length` long along its local z) end to end round a site's rim,
    `inset` inside it, leaving the gates open (world (x, z) points, a street's width wide) and
    anywhere the ground won't take one or a building already stands."""
    i0, j0, i1, j1, _ = box
    x0, z0, x1, z1 = i0 * TILE + inset, j0 * TILE + inset, i1 * TILE - inset, j1 * TILE - inset
    out = []
    for (ax, az), (bx, bz) in (((x0, z0), (x1, z0)), ((x1, z0), (x1, z1)), ((x1, z1), (x0, z1)), ((x0, z1), (x0, z0))):
        run = math.hypot(bx - ax, bz - az)
        dx, dz = (bx - ax) / run, (bz - az) / run
        yaw = math.atan2(dx, dz)
        n = int(run // length)
        pad = (run - n * length) / 2
        for k in range(n):
            mx, mz = ax + dx * (pad + (k + 0.5) * length), az + dz * (pad + (k + 0.5) * length)
            if any(math.hypot(mx - gx, mz - gz) < w / 2 + length / 2 for gx, gz, w in gates):
                continue
            if any(math.hypot(mx - x, mz - z) < r + 1 for x, z, r in taken):
                continue
            hs = ground.at([mx - dx * length / 2, mx, mx + dx * length / 2], [mz - dz * length / 2, mz, mz + dz * length / 2])
            if np.isnan(hs).any() or np.ptp(hs) > level:
                continue
            m = _rot(np.array([mid[0], 0.0, mid[2]]), yaw)       # the piece's own middle onto the spot
            out.append({"name": piece, "pos": (mx - m[0], -(float(np.min(hs)) - sink), mz - m[2]),
                        "rot": (0.0, yaw, 0.0), "scale": (1.0, 1.0, 1.0)})
    return out


def scatter(names, tris, ground, placed, avoid, count, rng, level=3.0, sink=0.3, tries=400):
    """Loose pieces (ruined walls) on the open land: any turn, seated at the lowest ground under
    them, on ground no more uneven than `level`, clear of `avoid` [(x, z, r)]."""
    cells = sorted(placed)
    out, taken = [], list(avoid)
    for _ in range(tries):
        if len(out) >= count:
            break
        name = rng.choice(names)
        i, j = rng.choice(cells)
        x, z = (i + rng.random()) * TILE, (j + rng.random()) * TILE
        yaw = rng.uniform(0, 2 * math.pi)
        o = {"rot": (0.0, yaw, 0.0), "scale": (1.0, 1.0, 1.0), "pos": (x, 0.0, z)}
        w = T.placed(tris[name], o).reshape(-1, 3)
        r = 0.5 * math.hypot(np.ptp(w[:, 0]), np.ptp(w[:, 2]))
        if any(math.hypot(x - a, z - b) < r + rr + 4 for a, b, rr in taken):
            continue
        hs = ground.at(w[::7, 0], w[::7, 2])
        if np.isnan(hs).any() or np.ptp(hs) > level:
            continue
        out.append({"name": name, "pos": (x, -(float(np.min(hs)) - sink), z), "rot": (0.0, yaw, 0.0), "scale": (1.0, 1.0, 1.0)})
        taken.append((x, z, r))
    return out


def city_walls_built(box, names, rng, gates_v=(), inset=6.0, every=3, sink=1.5, project=4.0):
    """A curtain wall round a site from our own pieces (architecture.py): each finished on every
    side, so a wall is simply laid along its line (`inset` inside the site's rim cells' middles),
    a corner tower on each joint, a run into a corner built to the length that ends inside it, wall
    towers standing out from the line, gatehouses where the streets leave. names: {"wall",
    "wall_corner", "tower", "gatehouse"} model names (16 bytes). Returns placements, FFXI."""
    i0, j0, i1, j1, h = box
    along_x = (i1 - i0) >= (j1 - j0)
    y = -(h - sink)
    out = []

    def put(name, x, z, inx, inz, scale=(1.0, 1.0, 1.0)):
        # a piece's front (its -z) faces out
        out.append({"name": name, "pos": (x, y, z), "rot": (0.0, math.atan2(inx, inz), 0.0), "scale": scale})
    sides = [((0, 1), [(i, j1 - 1) for i in range(i0, i1)]), ((0, -1), [(i, j0) for i in range(i0, i1)]),
             ((1, 0), [(i1 - 1, j) for j in range(j0, j1)]), ((-1, 0), [(i0, j) for j in range(j0, j1)])]
    gate_cells = set()
    for v in gates_v:
        if along_x:
            j = j0 + int(v // TILE)
            gate_cells |= {(i0, j), (i1 - 1, j)}
        else:
            i = i0 + int(v // TILE)
            gate_cells |= {(i, j0), (i, j1 - 1)}
    corners = {(i0, j0), (i1 - 1, j0), (i0, j1 - 1), (i1 - 1, j1 - 1)}
    for (nx, nz), cells in sides:
        inx, inz = -nx, -nz
        yaw = math.atan2(inx, inz)
        ax, az = math.cos(yaw), -math.sin(yaw)          # along the wall, the piece's local x
        for k, (i, j) in enumerate(cells):
            cx, cz = (i + 0.5) * TILE + inx * inset, (j + 0.5) * TILE + inz * inset
            if (i, j) in corners:
                out_end = 1 if ((ax > 0.5 and i == i1 - 1) or (ax < -0.5 and i == i0) or
                                (az > 0.5 and j == j1 - 1) or (az < -0.5 and j == j0)) else -1
                # from the cell's inner edge to the other run's line, under the corner tower
                a, b = -out_end * TILE / 2, -out_end * inset
                mid = (a + b) / 2
                put(names["wall_corner"], cx + ax * mid, cz + az * mid, inx, inz)
            elif (i, j) in gate_cells and ((nx != 0) == along_x):
                put(names["gatehouse"], cx, cz, inx, inz)
            else:
                put(names["wall"], cx, cz, inx, inz)
                if names.get("tower") and k % every == every // 2:
                    put(names["tower"], cx - inx * project, cz - inz * project, inx, inz)
    for (i, j) in corners:
        ox = 1 if i == i1 - 1 else -1
        oz = 1 if j == j1 - 1 else -1
        # square with the walls, its front (and banner) facing out across the short side's end
        put(names["tower"], (i + 0.5) * TILE - ox * inset, (j + 0.5) * TILE - oz * inset, 0.0, -oz)
    return out

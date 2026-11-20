"""A city laid from a map (maps/<name>.json): the zone is a window onto the map, and what the map
draws inside it is built: its curtain walls (our own wall kit, each run to its own length, towers
at the corners, closed gatehouses at the gates: the country beyond is another zone), its streets
and square paved, houses along the streets facing them, a landmark on its site, and places for
townsfolk to stand.

The map's frame is yalms, x east and z north. The zone's cells start at the window's corner
(x0, z0), snapped to the tile grid, so a map point (x, z) is (x - x0, z - z0) before the zone is
shifted back onto the map's own origin.

    "outdoor": {"kit": "misareaux_coast", "map": {"file": "my_city", "window": "city"},
                "city": {"houses": {"from": 85, "sets": ["davoi_h_base", "davoi_co_c"], "count": 80},
                         "landmarks": [{"at": "cathedral", "from": 85, "set": "dav_chr_base"}],
                         "walls": {"palette": "ronfaure", "prefix": "tav"}, "street_width": 8}}
"""
import json
import math
from pathlib import Path

import numpy as np

from zonekit import tiles as T
from zonekit import writer as fz

from . import props as P

TILE = T.TILE
HERE = Path(__file__).resolve().parent


def load(name):
    """A city map by path, or by name from maps/ beside this module."""
    path = Path(name).expanduser()
    if not (str(name).endswith(".json") and path.exists()):
        path = HERE / "maps" / f"{name}.json"
    return json.loads(path.read_text())


def window(m, name):
    """(x0, z0, W, D): the window's corner snapped to the tile grid, and its size in cells."""
    w = m["windows"][name]
    x0 = math.floor(w["x"][0] / TILE) * TILE
    z0 = math.floor(w["z"][0] / TILE) * TILE
    x1 = math.ceil(w["x"][1] / TILE) * TILE
    z1 = math.ceil(w["z"][1] / TILE) * TILE
    return x0, z0, int((x1 - x0) // TILE), int((z1 - z0) // TILE)


def inside(x, z, poly):
    hit = False
    for (xi, zi), (xj, zj) in zip(poly, poly[-1:] + poly[:-1]):
        if (zi > z) != (zj > z) and x < (xj - xi) * (z - zi) / (zj - zi) + xi:
            hit = not hit
    return hit


def seg_dist(x, z, a, b):
    """Distance from (x, z) to segment a-b, and the nearest point on it."""
    dx, dz = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz + 1e-12)))
    px, pz = a[0] + dx * t, a[1] + dz * t
    return math.hypot(x - px, z - pz), (px, pz)


def near_line(x, z, lines):
    """The nearest point on any of `lines` (polylines): (distance, (px, pz))."""
    best = (1e18, None)
    for line in lines:
        for a, b in zip(line, line[1:]):
            d = seg_dist(x, z, a, b)
            if d[0] < best[0]:
                best = d
    return best


def place(m, win, placed, kit, rng, name_of, base_dat, spec, source):
    """The city: ({"pieces", "made", "keep_clear", "box", "sites", "spots", "streets", "town"},
    {"tris", "sections", "textures", "built"}) in the shape outdoors.place.buildings returns."""
    from . import architecture as A
    from . import buildings as B
    from .place import obj_of
    x0, z0 = win[0], win[1]
    L = lambda pts: [(x - x0, z - z0) for x, z in pts]
    town = L(m["town"])
    streets = [L(s) for s in m["streets"]]
    walls = [L(w) for w in m["walls"]]
    gates = [(g["x"] - x0, g["z"] - z0, g.get("width", 12)) for g in m.get("gates", [])]
    sq = (m["square"]["x"] - x0, m["square"]["z"] - z0, m["square"]["r"])
    cx, cz = np.mean([p[0] for p in town]), np.mean([p[1] for p in town])
    ground = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})

    def h_at(x, z):
        g = ground.at([x], [z])[0]
        return 0.0 if np.isnan(g) else float(g)

    rspec = spec.get("rock")
    if rspec:
        from . import rock as R

        def on_rock(x, z):
            """Whether the map's rock stands at (x, z) (zone cells' frame)."""
            return bool(R.heights(m, np.array([x + x0]), np.array([z + z0]))[0] > 1.0)
    else:
        def on_rock(x, z):
            return False

    # --- our own pieces: the walls, towers and gatehouses, and the paving ---------------------
    wspec = spec.get("walls", {})
    pal = A.palette(wspec.get("palette", "ronfaure"))
    prefix = wspec.get("prefix", "arch")
    sink = wspec.get("sink", 1.5)
    meshes, pieces, extra, made = {}, [], [], []
    solid = []                                   # (x, z, r) our pieces stand on: houses keep clear

    def model(key, mesh):
        n16 = f"{prefix}_{key}"[:16].encode().ljust(16, b" ")
        meshes[n16] = mesh
        return n16

    def put(n16, x, z, inx, inz, h, col="full"):
        pieces.append({"name": n16, "pos": (x, -(h - sink), z), "rot": (0.0, math.atan2(inx, inz), 0.0),
                       "scale": (1.0, 1.0, 1.0), "col": col})

    tower = model("tower", A.tower(pal))
    gate_models = {}
    for k, w in enumerate(walls):
        for a, b in zip(w, w[1:]):
            dx, dz = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dz)
            mx, mz = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            # its front faces out: away from the town's middle
            nx, nz = -dz / length, dx / length
            if (mx - cx) * nx + (mz - cz) * nz < 0:
                nx, nz = -nx, -nz
            n16 = model(f"w{len(meshes):03d}", A.wall(pal, length=length + 2.0))   # into the towers
            hs = [h_at(a[0] + dx * t, a[1] + dz * t) for t in np.linspace(0, 1, 9)]
            put(n16, mx, mz, -nx, -nz, min(hs))
        for x, z in w:                            # a tower at every corner and end
            put(tower, x, z, cx - x, cz - z, h_at(x, z))
            solid.append((x, z, 10))
    for gx, gz, gw in gates:
        # the gap between the wall ends nearest the gate: the gatehouse spans it
        ends = sorted(((math.hypot(x - gx, z - gz), (x, z)) for w in walls for x, z in (w[0], w[-1])))[:2]
        (_, e1), (_, e2) = ends
        dx, dz = e2[0] - e1[0], e2[1] - e1[1]
        span = math.hypot(dx, dz)
        nx, nz = -dz / span, dx / span
        if (gx - cx) * nx + (gz - cz) * nz < 0:
            nx, nz = -nx, -nz
        key = round(span)
        if key not in gate_models:
            gate_models[key] = model(f"g{key:03d}", A.gatehouse(pal, length=span + 2.0, closed=True))
        mx, mz = (e1[0] + e2[0]) / 2, (e1[1] + e2[1]) / 2
        put(gate_models[key], mx, mz, -nx, -nz, h_at(mx, mz))
        solid.append((mx, mz, span / 2 + 12))
        made.append("gatehouse")

    # castles on top of the rock (the movie's): a curtain wall round a square court, corner towers,
    # a tall keep; from the town they're a silhouette on the skyline
    base_h = h_at(win[2] * TILE / 2, win[3] * TILE / 2)   # the land: the map's height 0
    if rspec:
        from . import rock as R_
        cd = {"tower": {"size": 16.0, "height": 34.0}, "keep": {"size": 24.0, "height": 56.0, "at": [0.2, 0.15]},
              "curtain": {"height": 18.0}, **m.get("castle_design", {})}
        big = model("ctower", A.tower(pal, size=cd["tower"]["size"], height=cd["tower"]["height"]))
        keep_ = model("ckeep", A.tower(pal, size=cd["keep"]["size"], height=cd["keep"]["height"]))
        for c in m.get("castles", []):
            lv = R_.castle_level(m, c)
            ccx, ccz = c["x"] - x0, c["z"] - z0
            s2 = c["size"] / 2
            cw = model(f"cw{round(c['size']):03d}", A.wall(pal, length=c["size"] + 2.0, height=cd["curtain"]["height"]))
            corners = [(-s2, -s2), (s2, -s2), (s2, s2), (-s2, s2)]
            for (ax_, az_), (bx_, bz_) in zip(corners, corners[1:] + corners[:1]):
                mx, mz = (ax_ + bx_) / 2, (az_ + bz_) / 2
                put(cw, ccx + mx, ccz + mz, -mx, -mz, base_h + lv, "none")          # its front out from the court
            for ax_, az_ in corners:
                put(big, ccx + ax_, ccz + az_, -ax_, -az_, base_h + lv, "none")
            put(keep_, ccx + s2 * cd["keep"]["at"][0], ccz + s2 * cd["keep"]["at"][1], 0.0, -1.0, base_h + lv, "none")
            made.append("castle")
        rp = m.get("ramp")
        if rp:
            pts = rp["points"]
            k = rp.get("gate_at", len(pts) - 1)
            (ax_, az_, ah), (bx_, bz_, _) = pts[k - 1], pts[k]
            gx_, gz_ = bx_ - x0, bz_ - z0
            up_x, up_z = bx_ - ax_, bz_ - az_
            if k + 1 < len(pts):
                up_x, up_z = pts[k + 1][0] - bx_, pts[k + 1][1] - bz_
            span_ = rp["width"] + rp.get("gate", {}).get("wider", 16.0)
            rg = model("rgate", A.gatehouse(pal, length=span_, closed=True))
            put(rg, gx_, gz_, up_x, up_z, base_h + pts[k][2])
            made.append("castle gate")
            # the side away from the hill: a parapet along the deck's edge (past the town wall the
            # road is outside it), a short piece at a time stepping up with it
            hill = min(m["massifs"], key=lambda q: math.hypot(q["x"] - bx_, q["z"] - bz_))
            w2 = rp["width"] / 2
            curve = R_.ramp_curve(m)
            near = lambda q: int(np.argmin(np.hypot(curve[:, 0] - q[0], curve[:, 2] - q[1])))
            pp = {"from": rp.get("parapet_from", 0), "height": 4.0, "thick": 1.4, "plinth": 0.6, "every": 3,
                  "short": 5, **rp.get("parapet", {})}
            i0, i1 = near(pts[pp["from"]]), near(pts[k]) - pp["short"]   # stop short of the gate
            para = model("parap", A.wall(pal, length=pp["every"] * 2.0 + 0.6, height=pp["height"],
                                         thick=pp["thick"], plinth=pp["plinth"]))
            for i in range(i0 + 1, i1, pp["every"]):    # a piece every few samples (2 yalms apart)
                (ax_, ah, az_), (bx2, bh, bz2) = curve[i - 1], curve[i + 1]
                L = math.hypot(bx2 - ax_, bz2 - az_)
                nx_, nz_ = -(bz2 - az_) / L, (bx2 - ax_) / L
                px, ph, pz = curve[i]
                if (px - hill["x"]) * nx_ + (pz - hill["z"]) * nz_ < 0:
                    nx_, nz_ = -nx_, -nz_                          # out, away from the hill
                put(para, px + nx_ * (w2 - 0.7) - x0, pz + nz_ * (w2 - 0.7) - z0, -nx_, -nz_, base_h + ph - 0.2)
            wtd = {"at": rp.get("tower_at"), "size": 12.0, "above": 8.0, "out": 6.0, "base": 3.0,
                   **rp.get("watch_tower", {})}
            if wtd["at"] is not None:                    # the watch tower on the bend
                tx, tz, th_ = pts[wtd["at"]]
                ox, oz = tx - hill["x"], tz - hill["z"]
                ol = math.hypot(ox, oz)
                ox, oz = ox / ol, oz / ol
                wt = model("watch", A.tower(pal, size=wtd["size"], height=th_ + wtd["above"]))
                put(wt, tx + ox * (w2 + wtd["out"]) - x0, tz + oz * (w2 + wtd["out"]) - z0, -ox, -oz,
                    base_h + wtd["base"])
                made.append("watch tower")

    # paving: the streets and the square as one surface draped on the land a hair above it, with a
    # kerb round its edge dropping below the ground (no gaps where the land dips, no two paving
    # surfaces over each other to flicker). Render only: the land under it is what's solid.
    pv = pal.get("paving")
    if pv:
        width = spec.get("street_width", 8.0)

        pts = [p for s_ in streets for p in s_] + [(sq[0] - sq[2], sq[1] - sq[2]), (sq[0] + sq[2], sq[1] + sq[2])]
        box = (min(p[0] for p in pts) - width, min(p[1] for p in pts) - width,
               max(p[0] for p in pts) + width, max(p[1] for p in pts) + width)
        segs = [(a, b) for s_ in streets for a, b in zip(s_, s_[1:])]

        def inside_by(x, z):
            """How far inside the paving (x, z) arrays are (negative: outside): streets, square."""
            best = np.hypot(x - sq[0], z - sq[1]) - sq[2]
            for a, b in segs:
                dx, dz = b[0] - a[0], b[1] - a[1]
                t = np.clip(((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz), 0, 1)
                best = np.minimum(best, np.hypot(x - a[0] - dx * t, z - a[1] - dz * t) - width / 2)
            return -best
        pm = A.Mesh(pal)
        _pave(pm, inside_by, box, h_at, pv)
        extra.append((pm, False))

    # --- the rock: solid inside the window (it closes the town where the map's crags stand),
    # drawn only beyond it (the rest of the crags, the arch, the hills round the basin) ----------
    rock_tex = ({}, {})
    if rspec:
        rpal = A.palette(rspec.get("palette", "tahrongi"))
        wx0, wz0 = x0, z0
        wx1, wz1 = x0 + win[2] * TILE, z0 + win[3] * TILE
        tsc = rspec.get("texture_scale", 1.0)
        near = R.build(m, rpal, box=(wx0, wz0, wx1, wz1), voxel=rspec.get("voxel", 6.0), scale=tsc)
        extra.append((near, True))
        margin = rspec.get("backdrop", 280.0)
        inside_win = lambda x, z: wx0 + 8 <= x <= wx1 - 8 and wz0 + 8 <= z <= wz1 - 8
        far = R.build(m, rpal, box=(wx0 - margin, wz0 - margin, wx1 + margin, wz1 + margin),
                      voxel=rspec.get("backdrop_voxel", 10.0), keep=lambda x, z: not inside_win(x, z), scale=tsc,
                      floor=rspec.get("floor", -0.6))
        extra.append((far, False))
        rock_tex = palette_textures(rpal, source)
        if m.get("ramp") and pal.get("paving"):         # the castle road: its own paved deck, walked on
            extra.append((_road(pal, R.ramp_curve(m), m["ramp"]["width"] / 2), True))

    # --- landmarks: whole sets from a source zone on their sites -----------------------------
    got = []                                     # (template, pieces, keep circle)
    taken = list(solid)
    rects = []                                   # footprints (4 corners) of what stands: houses fit between
    for lm in spec.get("landmarks", []):
        site = m[lm["at"]]
        tris, objs, _ = source(lm.get("from"))
        temp = B.learn(tris, objs, name_of, [lm["set"]])[lm["set"]][0]
        temp["zone"] = lm.get("from")
        sx, sz = site["x"] - x0, site["z"] - z0
        ax, az = site.get("along", [1, 0])
        fa = math.atan2(temp["front"][0], temp["front"][1])
        # its front on the square, its length along the site
        tx, tz = sq[0] - sx, sq[1] - sz
        yaw = math.atan2(tx, tz) - fa
        fp = B.footprint(temp, yaw, sx, sz)
        hs = ground.at([p[0] for p in fp] + [sx], [p[1] for p in fp] + [sz])
        r = 0.5 * math.hypot(temp["box"][2] - temp["box"][0], temp["box"][3] - temp["box"][1])
        got.append((temp, B.lay(temp, yaw, sx, sz, float(np.nanmin(hs)), 0.5), (sx, sz, r)))
        taken.append((sx, sz, r))
        rects.append(B.footprint(temp, yaw, sx, sz, pad=3.0))
        made.append(lm["set"])

    # --- houses: across the town, each turned to the street nearest it -----------------------
    hs_ = spec.get("houses")
    if hs_:
        tris, objs, _ = source(hs_.get("from"))
        temps = B.learn(tris, objs, name_of, hs_["sets"])
        for a, ts in temps.items():
            for t in ts:
                t["zone"] = hs_.get("from")
        pool = [t for ts in temps.values() for t in ts]
        count = int(hs_.get("count", 80))
        width = spec.get("street_width", 8.0)
        xs = [p[0] for p in town]
        zs = [p[1] for p in town]
        tries, laid = 0, 0
        why = {}
        no = lambda k: why.__setitem__(k, why.get(k, 0) + 1)
        segs = [(a, b, math.hypot(b[0] - a[0], b[1] - a[1])) for st in streets for a, b in zip(st, st[1:])]
        total = sum(n for *_, n in segs)
        while laid < count and tries < count * 60:
            tries += 1
            # a point along a street (by length), one side of it; the house turned to face it and
            # drawn back till its front stands two yalms off the street's edge
            pick = rng.uniform(0, total)
            for a, b, n in segs:
                if pick <= n:
                    break
                pick -= n
            t = pick / n
            side = rng.choice((-1, 1))
            nx_, nz_ = -(b[1] - a[1]) / n * side, (b[0] - a[0]) / n * side
            px, pz = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            temp = rng.choice(pool)
            fa = math.atan2(temp["front"][0], temp["front"][1])
            yaw = math.atan2(-nx_, -nz_) - fa
            x, z = px + nx_ * (width / 2 + 20), pz + nz_ * (width / 2 + 20)
            fp = B.footprint(temp, yaw, x, z)
            back = min((qx - px) * nx_ + (qz - pz) * nz_ for qx, qz in fp) - (width / 2 + 2.0)
            x, z = x - nx_ * back, z - nz_ * back
            if not inside(x, z, town) or math.hypot(x - sq[0], z - sq[1]) < sq[2] + 6:
                no('outside or square'); continue
            if near_line(x, z, walls)[0] < 16:
                no('walls'); continue
            fp = B.footprint(temp, yaw, x, z)
            # it lines the street: its nearest corner a yalm or few back from the street's edge
            edge = min(near_line(p[0], p[1], streets)[0] for p in fp)
            if not width / 2 + 1.0 <= edge <= width / 2 + 5.0:
                no('street distance'); continue
            r = 0.5 * math.hypot(max(p[0] for p in fp) - min(p[0] for p in fp), max(p[1] for p in fp) - min(p[1] for p in fp))
            if any(math.hypot(x - a, z - b) < r + rr for a, b, rr in taken) or \
               any(_overlap(B.footprint(temp, yaw, x, z, pad=1.0), q) for q in rects):
                no('overlap'); continue
            if any(on_rock(qx, qz) for qx, qz in fp + [(x, z)]):
                no('rock'); continue
            gx_, gz_ = zip(*fp)
            hs = ground.at(list(gx_) + [x], list(gz_) + [z])
            if np.isnan(hs).any() or np.ptp(hs) > 2.0:
                no('uneven'); continue
            got.append((temp, B.lay(temp, yaw, x, z, float(np.min(hs)), 0.5), (x, z, r)))
            rects.append(fp)
            laid += 1
        made += [t["anchor"] for t, _, _ in got[len(spec.get("landmarks", [])):]]
        print(f"  city: {laid} houses in {tries} tries; turned away: {why}")

    # houses collide as boxes; landmarks as their real shape
    for t, ps, _ in got:
        for p in ps:
            p["col"] = "base" if t.get("anchor") in (hs_ or {}).get("sets", []) else "full"
    # --- townsfolk: along the streets and round the square ------------------------------------
    spots = []
    for s in streets:
        for a, b in zip(s, s[1:]):
            n = max(1, int(math.hypot(b[0] - a[0], b[1] - a[1]) // 30))
            for k in range(n):
                t = (k + 0.5) / n
                x, z = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                if inside(x, z, town) and not on_rock(x, z):
                    spots.append((x, z, rng.uniform(0, 2 * math.pi), h_at(x, z)))
    for k in range(8):
        ang = 2 * math.pi * k / 8
        x, z = sq[0] + math.cos(ang) * sq[2] * 0.6, sq[1] + math.sin(ang) * sq[2] * 0.6
        spots.append((x, z, math.atan2(sq[0] - x, sq[1] - z), h_at(x, z)))

    # --- models: borrowed from their zones; ours laid as the zone's geometry ------------------
    borrowed = {"tris": {}, "sections": [], "textures": {}}
    by_zone = {}
    for t, ps, _ in got:
        by_zone.setdefault(t.get("zone"), set()).update(name_of(p["name"]) for p in ps)
    from zonekit import borrow
    for zid, names in by_zone.items():
        if zid is None or zid == kit.get("from_zone"):
            continue                             # the base zone's own: already there
        tris, _, dat = source(zid)
        models, textures = borrow.load_kit(dat, sorted(names))
        borrowed["tris"].update({mm["name16"]: tris[mm["name16"]] for mm in models.values()})
        borrowed["sections"] += [mm["section"] for mm in models.values()]
        borrowed["textures"].update(textures)
    tex, full = palette_textures(pal, source, banners=True)
    tex.update(rock_tex[0])
    full.update(rock_tex[1])
    # the paving is laid in the zone's cells' frame; the rock in the map's (its origin at -x0, -z0)
    base = h_at(win[2] * TILE / 2, win[3] * TILE / 2)   # the flat land (y up): the map's height 0
    print(f"  city: rock on land at {base:.2f}")
    at = lambda mm: (-x0, -base, -z0) if mm is not None and getattr(mm, "map_frame", False) else (0.0, 0.0, 0.0)
    built = {"meshes": meshes, "tris": {n: A.triangles(mm) for n, mm in meshes.items()}, "textures": tex,
             "full": full, "extra": [(mm, {"rot": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0), "pos": at(mm)}, solid)
                                      for mm, solid in extra]}
    borrowed["tris"].update(built["tris"])
    borrowed["built"] = built
    all_pieces = pieces + [p for _, ps, _ in got for p in ps]
    return ({"pieces": all_pieces, "made": made, "keep_clear": [c for *_, c in got] + solid,
             "box": None, "sites": [], "spots": spots, "streets": [], "town": town, "on_rock": on_rock, "walls": walls, "square": sq, "rects": rects,
             "clear_lines": (streets, spec.get("street_width", 8.0))}, borrowed)


def palette_textures(pal, source, banners=True):
    """The palette's textures (stone, banner, door, paving) by their full names in their zones:
    ({name16: raw section}, {short: name16})."""
    wanted = [(pal["from"], pal["texture"])]
    for k in ("banner", "door", "paving"):
        if pal.get(k) and (banners or k != "banner"):
            wanted.append((pal[k]["from"], pal[k]["texture"]))
    tex, full = {}, {}
    for zid, short in wanted:
        _, _, dat = source(zid)
        for sec in fz.split_sections(bytes(dat)):
            if sec.kind == fz.T_TEXTURE and sec.raw[0x11:0x21].decode("ascii", "replace").split()[-1:] == [short]:
                tex[sec.raw[0x11:0x21]] = sec.raw
                full[short] = sec.raw[0x11:0x21]
                break
    return tex, full


def _pave(m, inside_by, box, h_at, pv, cell=2.0, lift=0.12, kerb=0.6):
    """A paved surface where inside_by(x, z) >= 0 (a signed distance: straight edges come out
    straight at any angle), traced over cells of `cell` yalms by marching squares; every point at
    the land's own height plus `lift`, the texture tiled by world position; a kerb face down to
    `kerb` under the land along its outline."""
    from .architecture import _ears
    x0, z0, x1, z1 = box
    nx, nz = int(math.ceil((x1 - x0) / cell)), int(math.ceil((z1 - z0) / cell))
    gx, gz = np.meshgrid(x0 + np.arange(nx + 1) * cell, z0 + np.arange(nz + 1) * cell, indexing="ij")
    v = inside_by(gx, gz)
    part = m._part(pv["texture"])
    tile = pv["yalms"]

    def vert(x, z):
        base = len(part["pos"])
        part["pos"].append(np.array([x, h_at(x, z) + lift, z]))
        part["nrm"].append(np.array([0.0, 1.0, 0.0]))
        part["uv"].append((x / tile, -z / tile))
        part["col"].append(0.95)
        return base
    for i in range(nx):
        for j in range(nz):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]      # counter-clockwise, z north
            vals = [v[c] for c in corners]
            if max(vals) < 0:
                continue
            poly, edge_pts = [], []
            for k in range(4):
                a, b = corners[k], corners[(k + 1) % 4]
                va, vb = vals[k], vals[(k + 1) % 4]
                if va >= 0:
                    poly.append((gx[a], gz[a], False))
                if (va >= 0) != (vb >= 0):
                    t = va / (va - vb)
                    poly.append((gx[a] + (gx[b] - gx[a]) * t, gz[a] + (gz[b] - gz[a]) * t, True))
            if len(poly) < 3:
                continue
            # the top: counter-clockwise in (x, z) seen from above is clockwise for a y-up face,
            # so each ear is turned over
            ids = [vert(x, z) for x, z, _ in poly]
            for a, b, c in _ears([(x, z) for x, z, _ in poly]):
                part["tri"].append((ids[a], ids[c], ids[b]))
            # the outline: consecutive crossing points (the inside is on their left, going round)
            for k in range(len(poly)):
                (ax, az, ea), (bx, bz, eb) = poly[k], poly[(k + 1) % len(poly)]
                if not (ea and eb) or math.hypot(bx - ax, bz - az) < 1e-6:
                    continue
                ta, tb = h_at(ax, az) + lift, h_at(bx, bz) + lift
                # outward is to the right of a->b; the face's normal is (b - a) x up: flip to match
                ox, oz = bz - az, -(bx - ax)
                sux, suz = bx - ax, bz - az
                if (-suz) * ox + sux * oz < 0:          # (b-a) x up = (-dz, 0, dx) in (x, z)
                    (ax, az, ta), (bx, bz, tb) = (bx, bz, tb), (ax, az, ta)
                m.quad((ax, ta - lift - kerb, az), (bx, tb - lift - kerb, bz), (bx, tb, bz), (ax, ta, az),
                       (0.0, 0.0, 1.0, 1.0), tex=pv["texture"], tile=(tile, tile), shade=lambda s_, t_, y: 0.7)


def _road(pal, curve, w2, lift=0.15, side=6.0):
    """The road's deck along curve ((x, height, z), map yalms): level across, sloping along, paved,
    with faces down its two edges (into the cut and the bank). A Mesh in map coordinates."""
    from . import architecture as A
    pv = pal["paving"]
    m = A.Mesh(pal)
    part = m._part(pv["texture"])
    tile = pv["yalms"]
    P = np.asarray(curve, float)
    tg = np.gradient(P[:, [0, 2]], axis=0)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)          # left of the way up
    along = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1))])

    def tri(a, b, c, up):
        """A triangle wound to face `up` (a direction)."""
        n = np.cross(b[0] - a[0], c[0] - a[0])
        if np.dot(n, up) < 0:
            b, c = c, b
        n = np.cross(b[0] - a[0], c[0] - a[0])
        n /= np.linalg.norm(n) + 1e-12
        base = len(part["pos"])
        for p_, uv, sh in (a, b, c):
            part["pos"].append(p_)
            part["nrm"].append(n)
            part["uv"].append(uv)
            part["col"].append(sh)
        part["tri"].append((base, base + 1, base + 2))

    for i in range(len(P) - 1):
        q = []
        for j in (i, i + 1):
            x, h, z = P[j]
            for sgn in (1, -1):
                ex, ez = x + nrm[j, 0] * w2 * sgn, z + nrm[j, 1] * w2 * sgn
                q.append((np.array([ex, h + lift, ez]), (ex / tile, -ez / tile), 0.95))
        l0, r0, l1, r1 = q
        tri(l0, r0, r1, (0, 1, 0))
        tri(l0, r1, l1, (0, 1, 0))
        for (a0, a1), sgn in (((l0, l1), 1), ((r0, r1), -1)):     # the edge faces, outward
            out = np.array([nrm[i, 0] * sgn, 0.0, nrm[i, 1] * sgn])
            d0 = (a0[0] - np.array([0, side, 0]), (along[i] / tile, (a0[0][1] - side) / tile), 0.7)
            d1 = (a1[0] - np.array([0, side, 0]), (along[i + 1] / tile, (a1[0][1] - side) / tile), 0.7)
            e0 = (a0[0], (along[i] / tile, a0[0][1] / tile), 0.8)
            e1 = (a1[0], (along[i + 1] / tile, a1[0][1] / tile), 0.8)
            tri(d0, d1, e1, out)
            tri(d0, e1, e0, out)
    m.map_frame = True
    return m


def _overlap(a, b):
    """Two convex quads (4 (x, z) corners each) overlap: no separating axis among their edges."""
    for poly in (a, b):
        for (x1, z1), (x2, z2) in zip(poly, poly[1:] + poly[:1]):
            nx, nz = z1 - z2, x2 - x1
            pa = [nx * x + nz * z for x, z in a]
            pb = [nx * x + nz * z for x, z in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                return False
    return True


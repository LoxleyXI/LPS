"""Props: the source zone's own scatter, placed by rules measured from how it places them.

learn() measures every prop kind from the source's placements:
  band    the height range of the ground it stands on (under water, on sand, up the terraces)
  level   how uneven the ground under its base may be
  rise    how far above its cell's floor it stands
  rates   how many per edge tile and per flat tile
  scale   its size range
Kinds found at only a few sites (a camp's fire and logs, an outpost's crates) are set pieces, not
scatter, and stay out; so do small things that rest on something (buckets and barrels on decks),
things built into the land (fences along cliff tops, tunnel mouths), and anything the source sets
square to the world (walls, fences: built, not grown).

place() adds a few rules of its own (kit file "props"):
  rise_max   nothing stands more than this above its cell's floor: no dressing on the cliff tops
  low, wide pieces (outcrops, mounds) need their whole footprint level to level_wide; tall ones
             (trees) only need level ground under their trunk
  coastal kinds (the source only grows them low down, by its own sea) stand in our sea
  nothing stands on a continuous feature: a stream, a road, a footpath
  big outcrops and trees go at their overall rate anywhere with room, not only where the source
             happened to tuck them
"""
import math
from collections import Counter

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

TILE = T.TILE


class Ground:
    """Heights of laid tiles at any (x, z), FFXI frame: {cell: world triangles}, looked up by cell."""

    def __init__(self, world_by_cell):
        self.cells = world_by_cell

    def at(self, xs, zs):
        xs, zs = np.atleast_1d(xs).astype(float), np.atleast_1d(zs).astype(float)
        out = np.full(len(xs), np.nan)
        keys = np.stack([np.floor(xs / TILE), np.floor(zs / TILE)], 1).astype(int)
        for k in {tuple(x) for x in keys}:
            sel = np.all(keys == k, axis=1)
            w = self.cells.get(k)
            if w is not None:
                out[sel] = T.heights(w, xs[sel], zs[sel])
        return out

    def ring(self, x, z, r, n=8):
        """Heights at the centre and on a ring of radius r."""
        a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        return self.at(np.concatenate([[x], x + r * np.cos(a)]), np.concatenate([[z], z + r * np.sin(a)]))


def reach(size, scale):
    """Half its width (yalms)."""
    return float(np.clip(0.5 * max(size[0], size[2]) * scale, 0.5, 14.0))


def tall(size):
    return size[1] >= 0.6 * max(size[0], size[2])


def base(size, scale):
    """The radius that must stand on ground: a tall piece's trunk, a low piece's whole footprint."""
    return min(reach(size, scale), 2.5) if tall(size) else reach(size, scale)


def learn(kit, name_of, min_placed=8, min_sites=8):
    tris, objs = kit["tris"], kit["objs"]
    cells = {}
    for o in objs:
        if o["name"] in kit["tile_names"]:
            k = (math.floor(o["pos"][0] / TILE), math.floor(o["pos"][2] / TILE))
            w = T.placed(tris[o["name"]], o).reshape(-1, 3, 3)
            cells[k] = w if k not in cells else np.concatenate([cells[k], w])
    ground = Ground(cells)
    kind_of = {k: ("flat" if float(np.ptp(w[..., 1])) < 3 else "edge") for k, w in cells.items()}
    floor = {k: float(-w[..., 1].max()) for k, w in cells.items()}      # a cell's lowest ground (y down)
    counts = Counter(kind_of.values())
    seen = {}
    for o in objs:
        t = tris.get(o["name"])
        if t is None or o["name"] in kit["tile_names"]:
            continue
        pts = t.reshape(-1, 3)
        size = pts.max(0) - pts.min(0)
        if size.max() * max(abs(v) for v in o["scale"]) > 30:
            continue
        x, z = o["pos"][0], o["pos"][2]
        k = (math.floor(x / TILE), math.floor(z / TILE))
        if k not in cells:
            continue
        sc = float(o["scale"][1])
        ring = ground.ring(x, z, base(size, sc))
        if np.isnan(ring[0]):
            continue
        r = seen.setdefault(o["name"], {"label": name_of(o["name"]), "size": size, "offset": [], "h": [],
                                        "level": [], "rise": [], "scale": [], "edge": 0, "flat": 0, "at": [],
                                        "square": []})
        yaw = math.degrees(o["rot"][1]) % 90
        r["square"].append(min(yaw, 90 - yaw) < 1.0)
        r["offset"].append(-o["pos"][1] - ring[0])
        r["h"].append(ring[0])
        r["level"].append(float(np.nanmax(ring) - np.nanmin(ring)))
        r["rise"].append(ring[0] - floor[k])
        r["scale"].append(sc)
        r[kind_of[k]] += 1
        r["at"].append((x, z))
    out = {}
    for n, r in seen.items():
        placed_n = r["edge"] + r["flat"]
        big = reach(r["size"], 1.0) * 2 > 6.0
        if placed_n < (5 if big else min_placed):
            continue
        sites = len({(round(x / 30), round(z / 30)) for x, z in r["at"]})
        if sites < min(min_sites, 0.6 * placed_n):
            continue                                            # a set piece: stays out
        off, height = float(np.median(r["offset"])), float(r["size"][1])
        if off > 0.15 and height < 1.5:
            continue                                            # small things resting on something (decks)
        if off < -0.5 * max(height, 1.0):
            continue                                            # built into the land: fences on cliff tops,
                                                                # tunnel mouths, things hung from ledges
        if np.mean(r["square"]) > 0.5:
            continue                                            # set square to the world: built, not grown
        q = lambda v, p: float(np.percentile(v, p))
        per_edge, per_flat = r["edge"] / max(1, counts["edge"]), r["flat"] / max(1, counts["flat"])
        if big:
            per_edge = per_flat = placed_n / max(1, counts["edge"] + counts["flat"])
        out[n] = {"label": r["label"], "size": r["size"], "placed": placed_n,
                  "per_edge": per_edge, "per_flat": per_flat,
                  "offset": float(np.median(r["offset"])),
                  "band": (q(r["h"], 3) - 0.5, q(r["h"], 97) + 0.5),
                  "level": max(0.6, q(r["level"], 90)),
                  "rise": q(r["rise"], 95) + 1.0,
                  "scale": (q(r["scale"], 5), q(r["scale"], 95)),
                  # rocks, crates, stumps: solid once knee high; cut-out foliage (a leading
                  # underscore: the client's alpha pass) only when tall enough to have a trunk
                  "solid": bool(min(r["size"][0], r["size"][2]) > 1.5 and
                                (r["size"][1] > 4 if (n[:1] in (b"_", "_")) else r["size"][1] > 1.0))}
    return out


def place(placed, kit, rules, cfg, rng, obj_of, tries=20):
    """Props on the laid tiles: [{"name", "pos" FFXI, "yaw", "scale", "solid"}]."""
    pc = cfg.get("props", {})
    density = pc.get("density", 1.0)
    rise_max = pc.get("rise_max", 3.0)
    level_wide = pc.get("level_wide", 1.0)
    sea = float(cfg["coast"]["sea_level"]) if cfg.get("coast") else None
    cells, floor, marked = {}, {}, {}
    feats = kit.get("features", frozenset())
    for k, c in placed.items():
        o = obj_of(c, k)
        w = T.placed(kit["tris"][c["name"]], o).reshape(-1, 3, 3)
        cells[k] = w
        floor[k] = float(-w[..., 1].max())
        # the tile's streams, roads and paths: the triangles textured with a continuous feature
        m = kit["meshes"][c["name"]]
        ft = [p["pos"][t].astype(np.float64) for p in m["parts"] if bytes(p["texture"]) in feats
              for t in [fz.part_triangles(p, bool(m["config"] & 1))] if len(t)]
        if ft:
            marked[k] = T.placed(np.concatenate(ft), o).reshape(-1, 3, 3)
    ground = Ground(cells)
    features = Ground(marked)
    out, taken = [], []
    for (i, j), w in cells.items():
        kind = "per_flat" if float(np.ptp(w[..., 1])) < 3 else "per_edge"
        for n, r in rules.items():
            count = np.random.default_rng(rng.randrange(1 << 30)).poisson(r[kind] * density)
            for _ in range(count):
                for _try in range(tries):
                    s = rng.uniform(*r["scale"])
                    fp, br = reach(r["size"], s), base(r["size"], s)
                    x, z = (i + rng.uniform(0.02, 0.98)) * TILE, (j + rng.uniform(0.02, 0.98)) * TILE
                    ring = ground.ring(x, z, br)
                    if np.isnan(ring).any():
                        continue                              # off the laid ground
                    if marked and not np.isnan(features.ring(x, z, br + 1.0)).all():
                        continue                              # on a stream, road or path
                    h = ring[0]
                    if not (r["band"][0] <= h <= r["band"][1]):
                        continue                              # not at this prop's height
                    if sea is not None and r["band"][1] < sea + 10 and h > sea:
                        continue                              # a coastal kind: in the water only
                    wide = 2 * fp > 6.0 and not tall(r["size"])
                    if np.ptp(ring) > (min(r["level"], level_wide) if wide else r["level"]):
                        continue                              # ground too uneven under its base
                    if h - floor[(i, j)] > min(r["rise"], rise_max):
                        continue                              # up on a cliff top, out of the play area
                    gap = 0.6 * fp if tall(r["size"]) else fp
                    if fp > 3 and any(math.hypot(x - tx, z - tz) < gap + tr for tx, tz, tr in taken):
                        continue                              # big things don't crowd each other
                    # stand on the lowest ground under its base: on a slight slope the uphill side
                    # beds in rather than the downhill side floating
                    out.append({"name": n, "pos": (x, -(float(np.nanmin(ring)) + r["offset"] * s), z),
                                "yaw": rng.uniform(0, 2 * math.pi),
                                "scale": (s, s, s), "solid": r["solid"]})
                    if fp > 3:
                        taken.append((x, z, gap))
                    break
    return out


def scatter(placed, kit, spec, rng, obj_of, avoid=(), source=None):
    """Props from other zones on our level ground (a snowy kit's trees and rocks on Qufim's land).
    spec: [{"from": zone, "names": [...], "per_tile": n, "scale": [lo, hi], "solid": true, "low": bool}]:
    each level land tile gets about per_tile of each set, at random spots where the ground under the
    piece's footprint is level, clear of avoid [(x, z, r)] and of each other. Returns (props
    [{"name", "pos", "yaw", "scale", "solid"}] in the zone's unshifted frame, borrowed {"tris",
    "sections", "textures"})."""
    from zonekit import borrow
    import caves
    ground = Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
    level = [k for k, c in placed.items() if np.ptp(c["corners"]) == 0 and c["corners"].min() > 0]
    out, got = [], {"tris": {}, "sections": [], "textures": {}}
    taken = [tuple(a) for a in avoid]
    for s in spec:
        loaded, tex = borrow.load_kit(caves.zone_dat(s["from"]).read_bytes(), s["names"])
        pieces = {p["name16"]: p for p in loaded.values()}
        names = list(pieces)
        for n, p in pieces.items():
            if n not in got["tris"]:
                got["tris"][n] = p["tris"]
                got["sections"].append(p["section"])
        got["textures"].update(tex)
        lo, hi = s.get("scale", [0.85, 1.15])
        for k in level:
            n_here = int(s.get("per_tile", 1.0) + rng.random())
            for _ in range(n_here):
                name = rng.choice(names)
                size = pieces[name]["hi"] - pieces[name]["lo"]
                sc = rng.uniform(lo, hi)
                r = reach(size, sc)
                for _try in range(6):
                    x = (k[0] + rng.uniform(0.1, 0.9)) * TILE
                    z = (k[1] + rng.uniform(0.1, 0.9)) * TILE
                    if any(math.hypot(x - a, z - b) < r + c for a, b, c in taken):
                        continue
                    h = ground.ring(x, z, base(size, sc))
                    if np.isnan(h).any() or np.ptp(h) > 1.0:
                        continue
                    taken.append((x, z, r))
                    up = -pieces[name]["tris"].reshape(-1, 3)[:, 1]          # by its lowest point, a share buried
                    foot = float(h.min()) - float(up.min()) * sc - s.get("embed", 0.15) * float(up.max() - up.min()) * sc
                    out.append({"name": name, "pos": (x, -foot, z), "yaw": rng.uniform(0, 2 * math.pi),
                                "scale": (sc, sc, sc), "solid": s.get("solid", tall(size))})
                    break
    return out, got

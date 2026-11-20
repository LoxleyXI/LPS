#!/usr/bin/env python3
"""How the original game spreads its monsters over a field zone, measured, for spawns/pattern.py.

For each zone: its walkable ground from the zone's own collision (floor triangles laid on a
2-yalm grid, joined where a monster could walk between them, kept where a spawn can reach), and
its ordinary spawns from the server database (spawn type 0, 1 or 2 with a respawn: no NMs, no
placeholders hidden at the origin). Then:

    density     spawns per 10,000 square yalms of walkable ground (a 100 x 100 yalm field)
    spacing     each spawn's distance to its nearest neighbour (median, quartiles)
    groups      spawns within `link` yalms of each other, chained: how many stand alone, in
                pairs, in threes, bigger
    types       how many types a zone has, the share of the commonest ones, and how much of the
                zone each type covers (its spawns' 60-yalm neighbourhoods, as a share of the land)
    mixing      for each spawn, how many of its 5 nearest are another type
    edges       how far spawns stand from the edge of walkable ground, against the land itself
    openness    spawns per area in the open (30+ yalms from an edge), the middle, and near edges

    python3 modules/spawns/study.py [zone ids] --out modules/spawns/patterns/retail_fields.json --maps DIR
"""
import argparse
import json
import math
import struct
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# the original field zones from the starting areas out to the mid levels
FIELDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 115, 116, 117, 118, 119, 120, 126]
CELL = 2.0          # yalms per grid cell
CLIMB = 1.6         # the most a monster climbs between neighbouring cells (yalms)


# ---------------------------------------------------------------------------------------------
# the zone's walkable ground
# ---------------------------------------------------------------------------------------------

def collision_triangles(zone_id, dat=None):
    """(N, 3, 3) world triangles of the zone's collision, FFXI frame (x east, y down, z north).
    `dat`: a zone DAT to read instead of the game's own (one of ours, from an overlay)."""
    from zonekit import writer as fz
    from xidat.ftable import load_all, resolve
    from xidat.paths import ffxi_dir
    from xidat.zone_files import model_fid
    dat = Path(dat).read_bytes() if dat else resolve(model_fid(zone_id), load_all(ffxi_dir())).read_bytes()
    tables = fz.key_tables()
    for s in fz.split_sections(dat):
        if s.kind != fz.T_ZONEDEF:
            continue
        b = bytearray(s.raw[16:])
        fz.zonedef_decrypt(b, tables)
        z = fz.read_zonedef(b)
        c = z["collision"]
        mesh_at = {m["off"]: m for m in c["meshes"]}
        done, out = set(), []
        for g in c["groups"]:
            for tf, mo in g["pairs"]:
                if (tf, mo) in done or mo not in mesh_at:
                    continue
                done.add((tf, mo))
                M = np.array(struct.unpack_from("<16f", b, tf)).reshape(4, 4)
                pos, _nrm, raw = fz.collision_mesh_arrays(b, mesh_at[mo])
                ia, ib, ic = raw[:, 0] & 0x7FFF, raw[:, 1] & 0x3FFF, raw[:, 2] & 0x3FFF
                if len(pos) == 0 or max(ia.max(), ib.max(), ic.max()) >= len(pos):
                    continue
                w = pos.astype(np.float64) @ M[:3, :3] + M[3, :3]
                out.append(np.stack([w[ia], w[ib], w[ic]], 1))
        return np.concatenate(out)
    raise ValueError(f"zone {zone_id}: no zone definition")


def floor_grid(tris, cell=CELL):
    """Floors on a grid: {(i, j): [heights]} (FFXI y, so lower is higher up), and the grid origin.
    A floor is a triangle facing up or down within 40 degrees of level."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    tris = tris[np.abs(n[:, 1]) > math.cos(math.radians(40))]
    lo = tris[..., [0, 2]].reshape(-1, 2).min(0)
    floors = {}
    for t in tris:
        x, y, z = t[:, 0], t[:, 1], t[:, 2]
        i0, i1 = int((x.min() - lo[0]) // cell), int((x.max() - lo[0]) // cell)
        j0, j1 = int((z.min() - lo[1]) // cell), int((z.max() - lo[1]) // cell)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        px = lo[0] + (ii.ravel() + 0.5) * cell
        pz = lo[1] + (jj.ravel() + 0.5) * cell
        d = (z[1] - z[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (z[0] - z[2])
        if abs(d) < 1e-9:
            continue
        l1 = ((z[1] - z[2]) * (px - x[2]) + (x[2] - x[1]) * (pz - z[2])) / d
        l2 = ((z[2] - z[0]) * (px - x[2]) + (x[0] - x[2]) * (pz - z[2])) / d
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        h = l1 * y[0] + l2 * y[1] + l3 * y[2]
        for i, j, hh in zip(ii.ravel()[inside], jj.ravel()[inside], h[inside]):
            floors.setdefault((int(i), int(j)), []).append(float(hh))
    for k, hs in floors.items():                       # one height per level (a floor met twice)
        hs.sort()
        merged = [hs[0]]
        for h in hs[1:]:
            if h - merged[-1] > 2.5:
                merged.append(h)
        floors[k] = merged
    return floors, lo


def reachable(floors, lo, seeds, cell=CELL, climb=CLIMB):
    """{(i, j, level index)} reached by walking from the seed points (x, y, z) between neighbouring
    floors no more than `climb` apart in height."""
    def node(x, y, z):
        k = (int((x - lo[0]) // cell), int((z - lo[1]) // cell))
        best = None
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                kk = (k[0] + di, k[1] + dj)
                for li, h in enumerate(floors.get(kk, ())):
                    d = abs(h - y) + abs(di) + abs(dj)
                    if best is None or d < best[0]:
                        best = (d, (kk[0], kk[1], li))
        return best and best[0] < 6 and best[1]

    seen = set()
    q = deque()
    for s in seeds:
        n = node(*s)
        if n and n not in seen:
            seen.add(n)
            q.append(n)
    while q:
        i, j, li = q.popleft()
        h = floors[(i, j)][li]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            k = (i + di, j + dj)
            for lj, h2 in enumerate(floors.get(k, ())):
                if abs(h2 - h) <= climb and (k[0], k[1], lj) not in seen:
                    seen.add((k[0], k[1], lj))
                    q.append((k[0], k[1], lj))
    return seen


# ---------------------------------------------------------------------------------------------
# the spawns
# ---------------------------------------------------------------------------------------------

def spawns(zone_id):
    """[(x, y, z, group name, group id)] of the zone's ordinary spawns."""
    import subprocess
    q = ("SELECT sp.pos_x, sp.pos_y, sp.pos_z, g.name, g.groupid FROM mob_spawn_points sp "
         "JOIN mob_groups g ON g.groupid = sp.groupid AND g.zoneid = ((sp.mobid >> 12) & 0xFFF) "
         f"WHERE g.zoneid = {int(zone_id)} AND g.spawntype IN (0, 1, 2) AND g.respawntime > 0 "
         "AND NOT (sp.pos_x = 0 AND sp.pos_y = 0 AND sp.pos_z = 0)")
    # the server's own database login (its settings/network.lua; XI_SERVER: the server's folder)
    from xidat.paths import server_dir, sql_login
    host, port, user, pw, db = sql_login(server_dir())
    out = subprocess.run(["mysql", "-h", host, "-P", port, "-u", user, f"-p{pw}", db, "-N", "-B", "-e", q],
                         capture_output=True, text=True, check=True).stdout
    rows = []
    for line in out.splitlines():
        x, y, z, name, gid = line.split("\t")
        rows.append((float(x), float(y), float(z), name, int(gid)))
    return rows


# ---------------------------------------------------------------------------------------------
# measures
# ---------------------------------------------------------------------------------------------

def _groups(xz, link):
    """Sizes of the chains of spawns within `link` of each other."""
    n = len(xz)
    d = np.hypot(*(xz[:, None, :] - xz[None, :, :]).transpose(2, 0, 1))
    seen, sizes = np.zeros(n, bool), []
    for s in range(n):
        if seen[s]:
            continue
        stack, k = [s], 0
        seen[s] = True
        while stack:
            a = stack.pop()
            k += 1
            for b in np.nonzero((d[a] <= link) & ~seen)[0]:
                seen[b] = True
                stack.append(b)
        sizes.append(k)
    return sizes


def territory(xz, gid, rng=None):
    """How much each type keeps to its own parts: (the median distance from a spawn to the nearest
    of its own type, against the same with the types shuffled: 1 is no preference, lower is
    clustered), and the median type's spread (its spawns' spread against the whole zone's)."""
    rng = rng or np.random.default_rng(1)
    xz, gid = np.asarray(xz, float), np.asarray(gid)
    d = np.hypot(*(xz[:, None, :] - xz[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)

    def same(g):
        out = []
        for k in range(len(g)):
            m = g == g[k]
            m[k] = False
            if m.any():
                out.append(d[k][m].min())
        return float(np.median(out)) if out else float("nan")
    shuffled = float(np.median([same(rng.permutation(gid)) for _ in range(5)]))
    spread = [np.hypot(*xz[gid == t].std(0)) for t in set(gid.tolist()) if (gid == t).sum() >= 4]
    whole = np.hypot(*xz.std(0))
    return same(gid) / shuffled, float(np.median(spread) / whole) if spread else float("nan")


def study(zone_id, link=15.0, draw=None, dat=None):
    rows = spawns(zone_id)
    tris = collision_triangles(zone_id, dat)
    floors, lo = floor_grid(tris)
    reach = reachable(floors, lo, [(x, y, z) for x, y, z, _n, _g in rows])
    cells = {(i, j) for i, j, _l in reach}
    area = len(reach) * CELL * CELL
    # a 2D mask of walkable ground, for distances to its edge
    ij = np.array(sorted(cells))
    i0, j0 = ij.min(0) - 2
    shape = tuple(ij.max(0) - ij.min(0) + 5)
    mask = np.zeros(shape, bool)
    mask[ij[:, 0] - i0, ij[:, 1] - j0] = True
    from scipy import ndimage
    mask = ndimage.binary_closing(mask, iterations=1) | mask
    edge_d = ndimage.distance_transform_edt(mask) * CELL
    xz = np.array([(x, z) for x, _y, z, _n, _g in rows])
    pi = ((xz[:, 0] - lo[0]) // CELL).astype(int) - i0
    pj = ((xz[:, 1] - lo[1]) // CELL).astype(int) - j0
    ok = (pi >= 0) & (pj >= 0) & (pi < shape[0]) & (pj < shape[1])
    spawn_edge = edge_d[pi[ok], pj[ok]]
    land_edge = edge_d[mask]
    names = [n for *_r, n, _g in rows]
    gid = [g for *_r, g in rows]
    # spacing
    d = np.hypot(*(xz[:, None, :] - xz[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1)
    sizes = _groups(xz, link)
    size_hist = Counter(min(s, 5) for s in sizes)
    # types: share and coverage (60-yalm neighbourhoods of its spawns, as a share of the land)
    by_type = Counter(gid)
    total = len(rows)
    walk = np.argwhere(mask)
    wxz = np.stack([lo[0] + (walk[:, 0] + i0 + 0.5) * CELL, lo[1] + (walk[:, 1] + j0 + 0.5) * CELL], 1)
    sub = wxz[:: max(1, len(wxz) // 6000)]
    # the land monsters use: within 25 yalms of a spawn
    near = np.hypot(*(wxz[:: max(1, len(wxz) // 20000)][:, None, :] - xz[None, :, :]).transpose(2, 0, 1)).min(1)
    used = float((near <= 25).mean())
    cover = {}
    for g in by_type:
        pts = xz[[k for k, x in enumerate(gid) if x == g]]
        dd = np.hypot(*(sub[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)).min(1)
        cover[g] = float((dd <= 60).mean())
    # mixing: of each spawn's 5 nearest, how many are another type
    order = np.argsort(d, 1)[:, :5]
    gid_a = np.array(gid)
    mixing = float((gid_a[order] != gid_a[:, None]).mean())
    # openness: spawn density by distance from the edge
    bands = {"near": (0, 10), "middle": (10, 30), "open": (30, 1e9)}
    band_density = {}
    for k, (a, b) in bands.items():
        land_n = ((land_edge >= a) & (land_edge < b)).sum() * CELL * CELL
        sp_n = ((spawn_edge >= a) & (spawn_edge < b)).sum()
        band_density[k] = float(sp_n / land_n * 1e4) if land_n else None
    shares = sorted((c / total for c in by_type.values()), reverse=True)
    same_ratio, spread = territory(xz, gid)
    out = {
        "zone": zone_id,
        "spawns": total,
        "area": round(area),
        "density": round(total / area * 1e4, 2),
        "used": round(used, 2),
        "density_used": round(total / (area * used) * 1e4, 2),
        "spacing": {"median": round(float(np.median(nn)), 1), "p25": round(float(np.percentile(nn, 25)), 1),
                    "p75": round(float(np.percentile(nn, 75)), 1)},
        "groups": {str(k): v for k, v in sorted(size_hist.items())},
        "alone": round(size_hist.get(1, 0) / len(sizes), 2),
        "types": len(by_type),
        "top_shares": [round(s, 2) for s in shares[:5]],
        "coverage": {"median": round(float(np.median(list(cover.values()))), 2),
                     "min": round(min(cover.values()), 2), "max": round(max(cover.values()), 2)},
        "mixing": round(mixing, 2),
        "same_type": round(same_ratio, 2),
        "type_spread": round(spread, 2),
        "edge": {"spawns_median": round(float(np.median(spawn_edge)), 1),
                 "land_median": round(float(np.median(land_edge)), 1)},
        "band_density": {k: (round(v, 2) if v is not None else None) for k, v in band_density.items()},
        "type_names": {str(g): next(n for n, x in zip(names, gid) if x == g) for g in by_type},
    }
    if draw:
        _draw(draw, mask, lo, i0, j0, xz, gid, out)
    return out


def _draw(path, mask, lo, i0, j0, xz, gid, info):
    """The walkable ground in grey with each type's spawns in its own colour."""
    from PIL import Image, ImageDraw
    import colorsys
    s = 2                                               # pixels per cell
    img = Image.new("RGB", (mask.shape[0] * s, mask.shape[1] * s), (250, 248, 240))
    land = np.where(mask.T[::-1], 205, 250).astype(np.uint8)
    img.paste(Image.fromarray(np.dstack([land, land, land - 10]).repeat(s, 0).repeat(s, 1)))
    d = ImageDraw.Draw(img)
    types = sorted(set(gid), key=lambda g: -gid.count(g))
    col = {g: tuple(int(255 * c) for c in colorsys.hsv_to_rgb(k / len(types), 0.85, 0.8)) for k, g in enumerate(types)}
    H = mask.shape[1] * s
    for (x, z), g in zip(xz, gid):
        px = ((x - lo[0]) / CELL - i0) * s
        py = H - ((z - lo[1]) / CELL - j0) * s
        d.ellipse([px - 4, py - 4, px + 4, py + 4], fill=col[g], outline=(30, 30, 30))
    y = 6
    for g in types:
        d.rectangle([6, y, 16, y + 10], fill=col[g])
        d.text((20, y - 1), f"{info['type_names'][str(g)]} ({gid.count(g)})", fill=(20, 20, 20))
        y += 14
    img.save(path)


def summarise(zones):
    """The pattern: medians over the zones studied."""
    def med(f):
        return round(float(np.median([f(z) for z in zones])), 2)
    groups = Counter()
    for z in zones:
        for k, v in z["groups"].items():
            groups[k] += v
    n = sum(groups.values())
    return {
        "zones": [z["zone"] for z in zones],
        "density": med(lambda z: z["density"]),
        "density_range": [round(min(z["density"] for z in zones), 2), round(max(z["density"] for z in zones), 2)],
        "density_used": med(lambda z: z["density_used"]),
        "used": med(lambda z: z["used"]),
        "spacing_median": med(lambda z: z["spacing"]["median"]),
        "spacing_p25": med(lambda z: z["spacing"]["p25"]),
        "groups": {k: round(v / n, 3) for k, v in sorted(groups.items())},
        "types": med(lambda z: z["types"]),
        "top_share": med(lambda z: z["top_shares"][0]),
        "coverage": med(lambda z: z["coverage"]["median"]),
        "mixing": med(lambda z: z["mixing"]),
        "same_type": med(lambda z: z["same_type"]),
        "type_spread": med(lambda z: z["type_spread"]),
        "edge_spawns": med(lambda z: z["edge"]["spawns_median"]),
        "edge_land": med(lambda z: z["edge"]["land_median"]),
        "band_density": {k: med(lambda z, k=k: z["band_density"][k] or 0) for k in ("near", "middle", "open")},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zones", nargs="*", type=int, help=f"zone ids (default: {FIELDS})")
    ap.add_argument("--out", help="write the pattern (and each zone's figures) here as JSON")
    ap.add_argument("--maps", help="folder for a picture of each zone's spawns")
    ap.add_argument("--dat", help="with one zone id: read this zone DAT (one of ours) instead of the game's")
    args = ap.parse_args()
    found = []
    for z in args.zones or FIELDS:
        draw = Path(args.maps) / f"zone{z}.png" if args.maps else None
        if draw:
            draw.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = study(z, draw=draw, dat=args.dat)
        except Exception as e:                          # a zone without collision or spawns
            print(f"zone {z}: skipped ({e})")
            continue
        found.append(r)
        print(f"zone {z}: {r['spawns']} spawns on {r['area']:,} sq yalms: {r['density']} per 10k "
              f"({r['density_used']} on the {r['used']:.0%} they use); "
              f"nearest {r['spacing']['median']}; alone {r['alone']}; {r['types']} types, "
              f"top {r['top_shares'][:3]}; same type {r['same_type']}; spread {r['type_spread']}; "
              f"bands {r['band_density']}")
    if args.dat:
        return
    pattern = summarise(found)
    print(json.dumps(pattern, indent=1))
    if args.out:
        # figures only: the zones' monster names (for the pictures) stay out of the saved file
        figures = [{k: v for k, v in z.items() if k != "type_names"} for z in found]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"pattern": pattern, "zones": figures}, indent=1) + "\n")


if __name__ == "__main__":
    main()

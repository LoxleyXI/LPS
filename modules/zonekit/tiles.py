"""Outdoor tile kits: a zone's 40x40-yalm land tiles, read as shapes.

FFXI's outdoor zones are laid from tiles on a 40-yalm grid (centres at 20 mod 40), turned in quarter
turns and sometimes mirrored, with floors in fixed height steps (8 yalms in Valkurm, 10 in Ronfaure).
Their names say their shape (fl flat, st straight edge, coi / coo inside / outside corner, sla ramp),
but not in every zone, so the shape is read from the geometry: the top surface's height sampled along
each side of the tile. Two tiles fit side by side when their shared sides have the same profile.

Frames: FFXI's, x east, y down, z north; heights here are up (-y). A tile's local origin is its centre.
"""
import math
import re
from collections import Counter

import numpy as np

from . import writer as fz

TILE = 40.0
SIDES = ("n", "e", "s", "w")          # +z, +x, -z, -x


def _rot_zyx(r):
    rx, ry, rz = r
    cx, sx, cy, sy, cz, sz = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def read(dat, tables=None):
    """({mesh name16: opaque triangles (k,3,3) local}, zonedef objects, {name16: mesh})."""
    tables = tables or fz.key_tables()
    meshes, zd = {}, None
    for s in fz.split_sections(bytes(dat)):
        if s.kind == fz.T_MESH:
            b = bytearray(s.raw[0x10:])
            fz.mesh_decrypt(b, tables)
            m = fz.read_mesh(b)
            meshes.setdefault(bytes(m["name"]), m)
        elif s.kind == fz.T_ZONEDEF and zd is None:
            b = bytearray(s.raw[0x10:])
            fz.zonedef_decrypt(b, tables)
            zd = fz.read_zonedef(b)
    tris = {}
    for name, m in meshes.items():
        t = [p["pos"][fz.part_triangles(p, bool(m["config"] & 1))].astype(np.float64)
             for p in m["parts"] if not p["flags"] & 0x8000]
        t = [a for a in t if len(a)]
        if t:
            tris[name] = np.concatenate(t)
    return tris, zd["objects"], meshes


def is_tile(tris):
    pts = tris.reshape(-1, 3)
    size = pts.max(0) - pts.min(0)
    return abs(size[0] - TILE) < 1 and abs(size[2] - TILE) < 1


def heights(tris, xs, zs):
    """The top surface's height (up) at points (x, z): the highest triangle over each; nan off it."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    out = np.full(len(xs), np.nan)
    v0 = (c - a)[:, [0, 2]]
    v1 = (b - a)[:, [0, 2]]
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    ok = np.abs(den) > 1e-9
    for i, (x, z) in enumerate(zip(xs, zs)):
        v2 = np.stack([x - a[:, 0], z - a[:, 2]], 1)
        d02 = (v0 * v2).sum(1); d12 = (v1 * v2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (d11 * d02 - d01 * d12) / den
            v = (d00 * d12 - d01 * d02) / den
            inside = ok & (u >= -1e-4) & (v >= -1e-4) & (u + v <= 1 + 1e-4)
        if inside.any():
            y = a[inside, 1] + u[inside] * (c[inside, 1] - a[inside, 1]) + v[inside] * (b[inside, 1] - a[inside, 1])
            out[i] = -y.min()                              # y down: the top is the smallest y
    return out


def side_points(n=9, inset=0.5):
    """Sample points along each side, in order walking clockwise (seen from above: n, e, s, w)."""
    h = TILE / 2 - inset
    t = np.linspace(-h, h, n)
    return {"n": (t, np.full(n, h)), "e": (np.full(n, h), -t), "s": (-t, np.full(n, -h)), "w": (np.full(n, -h), t)}


def profile(tris, n=9):
    """{side: heights along it (clockwise)} plus "centre", in the tile's own frame."""
    out = {}
    for s, (xs, zs) in side_points(n).items():
        out[s] = heights(tris, xs, zs)
    out["centre"] = heights(tris, np.array([0.0]), np.array([0.0]))[0]
    return out


def placed(tris, o):
    """A placed object's triangles in world space (rotation, scale, position)."""
    R = _rot_zyx(o["rot"]) * np.array(o["scale"])
    return tris.reshape(-1, 3) @ R.T + np.array(o["pos"])


def world_profile(tris, o, n=9):
    """Profile of a placed tile in the world: sides named by world direction."""
    w = placed(tris, o).reshape(-1, 3, 3)
    cx, cz = o["pos"][0], o["pos"][2]
    local = w - np.array([cx, 0, cz])
    return profile(local, n)


CASES = {(0, 0, 0, 0): "flat", (1, 0, 0, 0): "one corner up", (1, 1, 0, 0): "straight",
         (1, 0, 1, 0): "diagonal", (1, 1, 1, 0): "three up"}
SHAPE = re.compile(r"_(coi|coo|co|st|fl|fla|ed|jo|joa|joc|sl|do|bt|tun)\d*(?=_|$)")


def canonical(levels):
    """The corner levels (NW, NE, SE, SW) in a fixed form under quarter turns and mirroring."""
    lv = np.asarray(levels)
    return max(tuple(int(x) for x in np.roll(m, r)) for m in (lv, lv[::-1]) for r in range(4))


def family(name):
    """The tile's family from its name ("ke_moa_coi2_m" -> "ke_moa"); names without a shape
    token keep their stem ("yama_2_m" -> "yama", "jime_b_m" -> "jime")."""
    hits = list(SHAPE.finditer(name))
    if hits:
        return name[:hits[-1].start()] or name                # the last shape word ends the family
    stem = re.sub(r"(_\d+\w*|_m|\d+)$", "", re.sub(r"_m$", "", name))
    return re.sub(r"_[a-z]$", "", stem)                  # Ronfaure's jime_a..d: the letter is the shape


def zone_step(objs, tile_names):
    """The zone's height step: the most common gap between the levels tiles are placed at."""
    count = Counter(round(-o["pos"][1]) for o in objs if o["name"] in tile_names)
    levels = []                                          # near levels (0 and -1) merged into the busier
    for y, _ in sorted(count.items(), key=lambda kv: -kv[1]):
        if all(abs(y - l) > 2 for l in levels):
            levels.append(y)
    levels.sort()
    gaps = [b - a for a, b in zip(levels, levels[1:])]
    return float(Counter(gaps).most_common(1)[0][0]) if gaps else 8.0


def scan(dat, name_of=lambda b: b.split(b"\0")[0].decode("ascii", "replace").rstrip()):
    """The zone's land tiles as shapes: {"step", "seams_matching", "tiles": {...}, "families": {...}}."""
    tris, objs, _ = read(dat)
    used = Counter(o["name"] for o in objs)
    tile_names = {n for n, t in tris.items() if used[n] and is_tile(t)}
    if not tile_names:
        return None
    step = zone_step(objs, tile_names)
    out = {}
    for n in sorted(tile_names, key=lambda n: -used[n]):
        p = profile(tris[n], 17)
        corners = np.array([p["n"][0], p["e"][0], p["s"][0], p["w"][0]])       # NW NE SE SW
        allh = np.concatenate([p[k] for k in SIDES])
        name = name_of(n)
        entry = {"placed": used[n], "family": family(name)}
        if np.isnan(corners).any():
            entry["case"] = "has holes"
            out[name] = entry
            continue
        rel = corners - corners.min()
        span = float(np.nanmax(allh) - np.nanmin(allh))
        top = rel.max()
        binary = top < 1.0 or np.all((rel < 0.25 * top) | (rel > 0.75 * top))
        if binary and top >= 1.0:
            case = CASES.get(canonical((rel > top / 2).astype(int)), "?")
        elif top < 1.0:
            case = "flat" if span < 2 else "flat edges, raised inside"
        else:
            case = "levels " + "/".join(str(x) for x in canonical(np.round(rel / step).astype(int)))
        entry.update({"case": case, "step": round(float(top), 1), "corners": [round(float(c), 1) for c in rel]})
        out[name] = entry
    fams = {}
    for name, e in out.items():
        f = fams.setdefault(e["family"], {"placed": 0, "step": [], "cases": {}})
        f["placed"] += e["placed"]
        if e.get("step"):
            f["step"].append(e["step"])
        f["cases"].setdefault(e["case"], []).append(name)
    for f in fams.values():
        f["step"] = round(float(np.median(f["step"])), 1) if f["step"] else 0.0
        f["missing"] = [c for c in ("one corner up", "straight", "three up", "diagonal")
                        if f["step"] >= 1 and c not in f["cases"]]
    fams = dict(sorted(fams.items(), key=lambda kv: -kv[1]["placed"]))
    return {"step": step, "seams_matching": seam_match(tris, objs, tile_names), "tiles": out, "families": fams}


def seam_match(tris, objs, tile_names, tol=1.0):
    """Share of the zone's shared tile sides whose profiles agree within tol yalms: how well side
    profiles describe how this kit fits together."""
    grid = {}
    for o in objs:
        if o["name"] in tile_names:
            grid.setdefault((math.floor(o["pos"][0] / TILE), math.floor(o["pos"][2] / TILE)), o)
    prof = {k: world_profile(tris[o["name"]], o) for k, o in grid.items()}
    good = total = 0
    for (i, j), p in prof.items():
        for side, (di, dj), other in (("n", (0, 1), "s"), ("e", (1, 0), "w")):
            q = prof.get((i + di, j + dj))
            if q is None:
                continue
            a, b = p[side], q[other][::-1]
            m = ~np.isnan(a) & ~np.isnan(b)
            if m.sum() < 3:
                continue
            total += 1
            good += np.abs(a[m] - b[m]).max() <= tol
    return round(good / total, 3) if total else None

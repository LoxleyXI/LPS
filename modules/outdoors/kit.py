"""A source zone as an outdoor kit: its land tiles as poses, plus what its kit file says about it.

The kit file (kits/<name>.json) names the source zone and the roles its pieces play; everything the
code needs to know about one zone lives there:

    {"name": "Valkurm Dunes", "from": 103,
     "levels": [0, 8, 16],        ground heights basins may sit at
     "step": 8,                   the height a walkable slope climbs in one tile
     "wall_rise": 17,             a wall corner stands at least this far above the lowest floor beside it
     "avoid": ["ke_roa"],         families never used as filler (roads only make sense as a network)
     "coast": {...},              optional: a sea along the south edge (see layout.py, effects.py)
     "networks": {"road": {"texture": "ron_ro"}, "stream": {"texture": "ron_riv", "length": [3, 6]}},
                                  optional: features laid as networks (see layout.py); the tiles that
                                  carry them and the sides they leave by are measured
     "props": {"rise_max": 3}}    see props.py

The tiles themselves are read from the player's own copy of the source zone at build time.
"""
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

HERE = Path(__file__).resolve().parent
TILE = T.TILE


def config(name):
    """A kit file by name (kits/<name>.json) or path."""
    p = Path(name) if str(name).endswith(".json") else HERE / "kits" / f"{name}.json"
    return json.loads(p.read_text())


HOLE = -1000.0                      # a corner with no ground: open sea (the sea effect draws the water)


def compound(tris, objs, meshes, name_of):
    """Tiles the source builds from more than one mesh placed together (Qufim: a cliff's body, ga_cl_coi1_m,
    with its foot, ga_cl_coi1_r, or its beach, ga_clb_st1_s): meshes of one stem (the name less its _m, _r,
    _s, b_m) at the same grid centre in the same pose, taken as one tile. Adds each to tris and meshes under a
    name of its own (the stem, a +, its parts' last letters) and returns ({name: [part names]}, [placings])."""
    import re
    groups = {}
    for o in objs:
        n = o["name"]
        x, z = o["pos"][0], o["pos"][2]
        if n not in meshes or abs((x % TILE) - TILE / 2) > 0.5 or abs((z % TILE) - TILE / 2) > 0.5:
            continue
        stem = re.sub(r"b?_[a-z]$", "", name_of(n))
        key = (round(x), round(o["pos"][1], 1), round(z), tuple(np.round(o["rot"], 3)), tuple(o["scale"]), stem)
        groups.setdefault(key, set()).add(n)
    parts, placings, by_sig = {}, [], {}
    for key, names in groups.items():
        if len(names) < 2:
            continue
        sig = tuple(sorted(names))
        if sig not in by_sig:
            stem = key[-1]
            label = (stem + "+" + "".join(name_of(n)[-1] for n in sig))[:16]
            k = 0
            while label.ljust(16).encode() in meshes:
                k += 1
                label = label[:15] + str(k)
            n16 = label.ljust(16).encode()
            by_sig[sig] = n16
            parts[n16] = list(sig)
            t = [tris[n] for n in sig if n in tris]
            if not t:
                continue
            tris[n16] = np.concatenate(t)
            meshes[n16] = {"name": n16, "config": meshes[sig[0]]["config"],
                           "parts": [p for n in sig for p in meshes[n]["parts"]]}
        x, y, z, rot, scale, _ = key
        placings.append({"name": by_sig[sig], "pos": (float(x), float(y), float(z)), "rot": rot, "scale": scale})
    for n in {n for sig in parts.values() for n in sig}:     # a part with nothing solid still needs its bounds
        if n not in tris:
            m = meshes[n]
            tris[n] = np.concatenate([p["pos"][fz.part_triangles(p, bool(m["config"] & 1))].astype(np.float64)
                                      for p in m["parts"]])
    return parts, placings


def load(dat, name_of, samples=33, levels=(), holes=False, compound_tiles=False, open_below=None):
    """Candidates: every land tile the zone uses, in every turn and mirror, at every height the zone
    places it at and at each of `levels`, with its corner heights and side profiles (sampled densely: a steep bit between
    sparse samples can hide a slit) in that pose. Also the zone's placed objects, for props.
    compound_tiles: also tiles of several meshes placed together (see compound()). open_below: a corner
    at or under this height counts as open sea (a beach that runs out into the water, beside a sea cliff)."""
    tris, objs, meshes = T.read(dat)
    parts = {}
    if compound_tiles:
        parts, placings = compound(tris, objs, meshes, name_of)
        in_parts = {n for sig in parts.values() for n in sig}
        objs = [o for o in objs if o["name"] not in in_parts] + placings
    used = Counter(o["name"] for o in objs)
    tile_names = [n for n, t in tris.items() if used[n] and T.is_tile(t)]
    if holes:
        # a sea cliff has no ground on its sea side, so it measures short of a full tile: it's still
        # a tile if the zone always places it on the grid (centres at 20 mod 40) and it fits one
        on_grid = Counter(o["name"] for o in objs
                          if abs((o["pos"][0] % TILE) - TILE / 2) < 0.5 and abs((o["pos"][2] % TILE) - TILE / 2) < 0.5)
        for n, t in tris.items():
            if n in tile_names or not used[n] or on_grid[n] != used[n]:
                continue
            pts = t.reshape(-1, 3)
            size = pts.max(0) - pts.min(0)
            if 15 <= size[0] <= TILE + 1 and 15 <= size[2] <= TILE + 1:
                tile_names.append(n)
    bases = {}
    for o in objs:
        if o["name"] in tile_names:
            bases.setdefault(o["name"], set()).add(round(-o["pos"][1]))
    for n in tile_names:                                       # a tile doesn't care how high it sits:
        bases[n] |= {round(float(x)) for x in levels}          # every piece at every level of the kit
    cands = []
    for n in tile_names:
        seen = set()
        for b in sorted(bases[n]):
            for k in range(4):
                for sx in (1.0, -1.0):
                    o = {"rot": (0.0, k * math.pi / 2, 0.0), "scale": (sx, 1.0, 1.0), "pos": (0.0, -float(b), 0.0)}
                    looks = side_looks(meshes[n], o)
                    p = T.world_profile(tris[n], o, samples)
                    corners = np.array([p["n"][0], p["e"][0], p["s"][0], p["w"][0]])
                    if np.isnan(corners).any():
                        if not holes or np.isnan(corners).all():
                            continue
                        corners = np.where(np.isnan(corners), HOLE, corners)   # a sea cliff's open side
                    if open_below is not None:                 # ground at the sea's own floor is open sea
                        corners = np.where(corners <= open_below, HOLE, corners)   # (Qufim's beach runs out at 0)
                    key = (tuple(np.round(corners).astype(int)),
                           tuple(np.round(np.nan_to_num(np.concatenate([p[s] for s in T.SIDES]), nan=-99), 0)))
                    if key in seen:
                        continue                              # a symmetric tile: same pose twice
                    seen.add(key)
                    cands.append({"name": n, "label": name_of(n), "rot": o["rot"], "scale": o["scale"],
                                  "base": float(b), "corners": corners, "sides": p, "looks": looks,
                                  "weight": used[n]})
    feats = features(tris, objs, meshes, {n for n in tile_names if T.is_tile(tris[n])})   # full land tiles
    for c in cands:                                            # what must carry across a seam
        c["looks"] = {k: [v & feats for v in seq] for k, seq in c["looks"].items()}
        # which sides each feature leaves by: the tile's place in a road or stream network
        c["links"] = {f: frozenset(side for side in "nesw" if any(f in v for v in c["looks"][side])) for f in feats}
    return {"tris": tris, "meshes": meshes, "objs": objs, "cands": cands, "tile_names": set(tile_names),
            "terrain": terrain(dat), "features": feats, "parts": parts}


def features(tris, objs, meshes, tile_names, rare=0.05, continues=0.95):
    """The source's continuous features: textures that are rare along its seams but carry across
    nearly every one (a footpath, a road, a river). Measured from its own neighbouring tiles: the
    common ground textures blend loosely even in the real zone (Valkurm's sandy grass carries over
    5% of the time), a path never stops dead. Only these have to match across our seams."""
    grid = {}
    for o in objs:
        if o["name"] in tile_names:
            grid[(math.floor(o["pos"][0] / TILE), math.floor(o["pos"][2] / TILE))] = o
    looks = {k: side_looks(meshes[o["name"]], {"rot": o["rot"], "scale": o["scale"]}) for k, o in grid.items()}
    seen, both, total = Counter(), Counter(), 0
    for (i, j), L in looks.items():
        for side, (di, dj), other in (("e", (1, 0), "w"), ("n", (0, 1), "s")):
            M = looks.get((i + di, j + dj))
            if not M:
                continue
            for x, y in zip(L[side], M[other][::-1]):
                total += 1
                for t in x | y:
                    seen[t] += 1
                    both[t] += (t in x and t in y)
    return frozenset(t for t, n in seen.items() if n / max(1, total) < rare and both[t] / n >= continues)


_looks_cache = {}


def side_looks(mesh, o, samples=17, inset=0.8):
    """What the surface looks like along each side: at each sample, the textures on top there
    (the ground's, plus any painted layer over it: a footpath, a blend of grass into sand). Two
    tiles only fit if these agree, or a path stops dead at a seam."""
    key = (bytes(mesh["name"]), o["rot"], o["scale"])
    if key in _looks_cache:
        return _looks_cache[key]
    R = T._rot_zyx(o["rot"]) * np.array(o["scale"])
    parts = []
    for p in mesh["parts"]:
        t = fz.part_triangles(p, bool(mesh["config"] & 1))
        if len(t):
            parts.append((bytes(p["texture"]), (p["pos"][t].astype(np.float64).reshape(-1, 3) @ R.T).reshape(-1, 3, 3)))
    out = {}
    for side, (xs, zs) in T.side_points(samples, inset).items():
        hs = [T.heights(w, xs, zs) for _, w in parts]
        top = np.nanmax(np.array(hs), axis=0) if hs else np.full(len(xs), np.nan)
        out[side] = [frozenset(parts[k][0] for k in range(len(parts)) if not np.isnan(hs[k][i]) and hs[k][i] >= top[i] - 0.3)
                     for i in range(len(xs))]
    _looks_cache[key] = out
    return out


def terrain(dat):
    """The source's footstep sounds: its commonest collision terrain underfoot and on steep faces."""
    tables = fz.key_tables()
    for s in fz.split_sections(bytes(dat)):
        if s.kind != fz.T_ZONEDEF:
            continue
        b = bytearray(s.raw[16:])
        fz.zonedef_decrypt(b, tables)
        z = fz.read_zonedef(b)
        floor, wall = Counter(), Counter()
        for m in z.get("collision", {}).get("meshes", []):
            pos, nrm, q = fz.collision_mesh_arrays(b, m)
            q = q.astype(np.uint32)
            t = (q[:, 0] >> 15) | ((q[:, 1] >> 15) << 1) | ((q[:, 2] >> 15) << 2) | ((q[:, 3] >> 15) << 3)
            ni = q[:, 3] & 0x0FFF
            ok = ni < len(nrm)
            up = np.zeros(len(q), bool)
            up[ok] = np.abs(nrm[ni[ok], 1]) > 0.6
            floor.update(t[up].tolist())
            wall.update(t[~up].tolist())
        return {"floor": floor.most_common(1)[0][0] if floor else 5,
                "wall": wall.most_common(1)[0][0] if wall else 5}
    return {"floor": 5, "wall": 5}


def feature(kit, short):
    """A kit feature (16-byte texture name) by its short name ("ron_ro")."""
    hits = [f for f in kit["features"] if f.decode("ascii", "replace").split()[-1] == short]
    if len(hits) != 1:
        raise ValueError(f"network texture {short!r}: features are "
                         f"{sorted(f.decode('ascii', 'replace').split()[-1] for f in kit['features'])}")
    return hits[0]

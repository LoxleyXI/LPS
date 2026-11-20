"""Monsters spread the way the original field zones spread them (measured by spawns/study.py, kept
in patterns/retail_fields.json), on any land with a walkable surface.

What the original zones do, and what this copies:
    - a density: spawns per 10,000 square yalms of walkable ground (about 7)
    - quiet ground: about 40% of a zone has no spawn near it (patches, not the edges alone)
    - fewer near the edge: within 10 yalms of it, about half the density of the open
    - mostly alone: about three in four stand alone, the rest in twos and threes, a few in packs
    - every type everywhere, but loosely: a type ranges over most of the zone (its spread about
      0.84 of the whole zone's), and a spawn's nearest of its own type is about two thirds as far
      as it would be if types were dealt at random (soft territories, not camps)

    placements = place(land, types, seed, pattern=load())
    print(measure(placements, land))        # the same figures, to set against the pattern's

`land`: {"surface" (outdoors.props.Ground, local coords), "placed", "shift", "start": (x, z) local
(a point on the main ground, such as the way in), "avoid": [(x, z, r)] local (camps)}. `types`: the
zone file's types; their "count" is each one's share of the whole (the total comes from the
density), unless a type sets "fixed": true to keep its count as it is.
"""
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STEP = 3.0                                   # yalms between the samples of the walkable ground
CLIMB = 2.4                                  # the most a monster climbs between samples


def load(name="retail_fields"):
    """The pattern: the study's figures (patterns/<name>.json, written by study.py) and the placer's
    settings that meet them (patterns/<name>.placer.json, set by hand against measure())."""
    p = dict(json.loads((HERE / "patterns" / f"{name}.json").read_text())["pattern"])
    tuning = HERE / "patterns" / f"{name}.placer.json"
    if tuning.exists():
        p.update({k: v for k, v in json.loads(tuning.read_text()).items() if not k.startswith("_")})
    return p


# ---------------------------------------------------------------------------------------------
# the walkable ground
# ---------------------------------------------------------------------------------------------

def ground(land):
    """Walkable samples of a land: {"xz" (n, 2) local, "h" (n,), "edge" (n,) yalms to the edge,
    "grid", "mask", "origin"}. Walkable: a surface there, not steep, and reached from the land's
    start by small climbs (no cliff tops, no wall tops)."""
    from scipy import ndimage
    import zonekit.tiles as T
    from outdoors.kit import HOLE
    cells = np.array(list(land["placed"]))
    x0, z0 = cells.min(0) * T.TILE
    x1, z1 = (cells.max(0) + 1) * T.TILE
    xs = np.arange(x0 + STEP / 2, x1, STEP)
    zs = np.arange(z0 + STEP / 2, z1, STEP)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    H = land["surface"].at(X.ravel(), Z.ravel()).reshape(X.shape)
    ok = np.isfinite(H) & (H > HOLE / 2)
    Hf = np.where(ok, H, np.nan)
    gx, gz = np.gradient(np.nan_to_num(Hf, nan=-1e4), STEP)
    ok &= np.hypot(gx, gz) < 0.6
    # reached from the start by small climbs (the camp cut out after: it often stands at the start)
    sx, sz = land.get("start", ((x0 + x1) / 2, (z0 + z1) / 2))
    si = int(np.clip((sx - xs[0]) / STEP, 0, len(xs) - 1))
    sj = int(np.clip((sz - zs[0]) / STEP, 0, len(zs) - 1))
    if not ok[si, sj]:                                  # the nearest walkable sample to the start
        pts = np.argwhere(ok)
        if not len(pts):
            raise ValueError("no walkable ground")
        si, sj = pts[np.argmin((pts[:, 0] - si) ** 2 + (pts[:, 1] - sj) ** 2)]
    seen = np.zeros_like(ok)
    seen[si, sj] = True
    stack = [(si, sj)]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < ok.shape[0] and 0 <= b < ok.shape[1] and ok[a, b] and not seen[a, b] \
                    and abs(H[a, b] - H[i, j]) <= CLIMB:
                seen[a, b] = True
                stack.append((a, b))
    for x, z, r in land.get("avoid", ()):
        seen &= np.hypot(X - x, Z - z) > r
    edge = ndimage.distance_transform_edt(np.pad(seen, 1))[1:-1, 1:-1] * STEP
    idx = np.nonzero(seen)
    return {"xz": np.stack([X[idx], Z[idx]], 1), "h": H[idx], "edge": edge[idx], "mask": seen,
            "origin": (xs[0], zs[0]), "ij": np.stack(idx, 1)}


def land_of(summary, clear=45.0):
    """The land a placement needs, from a generate() summary (or a region's): its surface and
    tiles, its way in (the first basin) as the start, and its outpost (the camp's patch) kept
    clear by `clear` yalms. Local coords: the summary's shift is added to the placements."""
    import zonekit.tiles as T
    b0 = summary["basins"][0]
    avoid = []
    patch = summary.get("patch")
    if patch:
        ci = sum(c[0] for c in patch) / len(patch) + 0.5
        cj = sum(c[1] for c in patch) / len(patch) + 0.5
        avoid.append((ci * T.TILE, cj * T.TILE, clear))
    return {"surface": summary["surface"], "placed": summary["placed"], "shift": summary["shift"],
            "start": (b0[0] * T.TILE, b0[1] * T.TILE), "avoid": avoid}


def _field(shape, cells, rng):
    """Smooth noise in 0..1 on the sample grid, varying over about `cells` samples."""
    from scipy import ndimage
    n = (max(2, int(shape[0] / cells) + 3), max(2, int(shape[1] / cells) + 3))
    small = rng.random(n)
    big = ndimage.zoom(small, (shape[0] / (n[0] - 2), shape[1] / (n[1] - 2)), order=3, mode="nearest")
    big = big[:shape[0], :shape[1]]
    if big.shape != shape:
        big = np.pad(big, ((0, shape[0] - big.shape[0]), (0, shape[1] - big.shape[1])), mode="edge")
    lo, hi = big.min(), big.max()
    return (big - lo) / (hi - lo + 1e-9)


# ---------------------------------------------------------------------------------------------
# placing
# ---------------------------------------------------------------------------------------------

def place(land, types, seed=1, pattern=None, density=None):
    """[{"type", "pos": (x, north, height), "facing"}] in the zone's coordinates (the land's shift
    added), as outdoors/mobs.py gives them."""
    p = pattern or load()
    rng = np.random.default_rng(seed)
    g = ground(land)
    xz, h, edge = g["xz"], g["h"], g["edge"]
    shape = g["mask"].shape
    area = len(xz) * STEP * STEP
    density = float(density or p["density"])
    # where monsters are: quiet patches (about `used` of the land busy) and fewer by the edge
    busy = _field(shape, p.get("busy_scale", 140) / STEP, rng)[tuple(g["ij"].T)]
    cut = np.quantile(busy, 1 - p["used"])
    weight = np.where(busy >= cut, 1.0, p.get("quiet", 0.12))
    band = p["band_density"]
    open_ = max(band["open"], 1e-6)
    # (each placed group's members stand round its anchor, some of them nearer the edge than it,
    # so the anchors keep further off it than the figures alone would say: `edge_lean`)
    lean_e = float(p.get("edge_lean", 1.0))
    weight *= np.where(edge < 10, (band["near"] / open_) ** lean_e,
                       np.where(edge < 30, (band["middle"] / open_) ** lean_e, 1.0))
    weight /= weight.sum()
    # each type's own parts: a smooth field per type, leaned on by `territory`
    keys = list(types)
    fixed = {k: int(t.get("count", 0)) for k, t in types.items() if t.get("fixed")}
    share = np.array([float(types[k].get("count", 1)) for k in keys])
    share /= share.sum()
    lean = float(p.get("territory", 2.0))
    prefs = np.stack([_field(shape, p.get("territory_scale", 110) / STEP, rng)[tuple(g["ij"].T)] for _ in keys])
    pull = share[:, None] * np.exp(lean * (prefs - 0.5) * 2)
    total = int(round(density * area / 1e4))
    want = {k: (fixed[k] if k in fixed else None) for k in keys}
    free_total = total - sum(fixed.values())
    left = {k: (fixed[k] if k in fixed else int(round(free_total * share[i] / share[[j for j, kk in enumerate(keys) if kk not in fixed]].sum())))
            for i, k in enumerate(keys)}
    sizes = sorted(p["groups"].items())
    size_n = np.array([int(k) for k, _ in sizes])
    size_p = np.array([v for _, v in sizes])
    size_p /= size_p.sum()
    apart = float(p.get("apart", 11.0))                 # an anchor's room from every spawn so far
    near = float(p.get("near", 6.0))                    # a pack's spacing
    out, pts = [], np.zeros((0, 2))
    tries = 0
    while sum(left.values()) > 0 and tries < 20000:
        tries += 1
        a = rng.choice(len(xz), p=weight)
        if len(pts) and np.hypot(*(pts - xz[a]).T).min() < apart * (0.5 if tries > 8000 else 1.0):
            continue
        # the group's type: leaning to the types whose parts these are, among those still wanted
        w = pull[:, a] * np.array([1.0 if left[k] > 0 else 0.0 for k in keys])
        if w.sum() <= 0:
            break
        k = keys[rng.choice(len(keys), p=w / w.sum())]
        n = int(rng.choice(size_n, p=size_p))
        members = [a]
        for _ in range(n - 1):
            for _try in range(20):
                ang, dist = rng.uniform(0, 2 * math.pi), rng.uniform(near * 0.7, near * 1.6)
                q = xz[members[-1]] + dist * np.array([math.cos(ang), math.sin(ang)])
                b = int(np.argmin(np.hypot(*(xz - q).T)))
                if np.hypot(*(xz[b] - q)) > STEP * 1.5:
                    continue
                if min(np.hypot(*(xz[m] - xz[b])) for m in members) < near * 0.6:
                    continue
                members.append(b)
                break
        for m in members:
            kk = k
            if m != a and rng.random() > p.get("pack_same", 0.75):
                w2 = pull[:, m] * np.array([1.0 if left[t] > 0 else 0.0 for t in keys])
                if w2.sum() > 0:
                    kk = keys[rng.choice(len(keys), p=w2 / w2.sum())]
            if left[kk] <= 0:
                continue
            left[kk] -= 1
            x, z = xz[m] + rng.uniform(-STEP / 2, STEP / 2, 2)
            out.append({"type": kk, "pos": (float(x), float(z), float(h[m])),
                        "facing": float(rng.uniform(-math.pi, math.pi))})
            pts = np.vstack([pts, xz[m][None]])
    sx, sz = land["shift"][0], land["shift"][2]
    for o in out:
        x, z, hh = o["pos"]
        o["pos"] = (x + sx, z + sz, hh)
    return out


# ---------------------------------------------------------------------------------------------
# measuring (the study's own figures, for a land of ours)
# ---------------------------------------------------------------------------------------------

def measure(placements, land, link=15.0):
    """The study's figures for these placements on this land."""
    import sys
    sys.path.insert(0, str(HERE.parent))
    from spawns.study import territory, _groups
    g = ground(land)
    sx, sz = land["shift"][0], land["shift"][2]
    xz = np.array([(o["pos"][0] - sx, o["pos"][1] - sz) for o in placements]).reshape(-1, 2)
    if len(xz) < 2:
        raise ValueError(f"only {len(xz)} spawns placed: no ground to measure")
    gid = [o["type"] for o in placements]
    area = len(g["xz"]) * STEP * STEP
    d = np.hypot(*(xz[:, None, :] - xz[None, :, :]).transpose(2, 0, 1))
    np.fill_diagonal(d, np.inf)
    sizes = _groups(xz, link)
    near_land = np.hypot(*(g["xz"][::3][:, None, :] - xz[None, :, :]).transpose(2, 0, 1)).min(1)
    # spawn edge distances from the nearest walkable sample
    k = np.argmin(np.hypot(*(g["xz"][:, None, :] - xz[None, :, :]).transpose(2, 0, 1)), 0)
    sp_edge, land_edge = g["edge"][k], g["edge"]
    bands = {}
    for name, (a, b) in {"near": (0, 10), "middle": (10, 30), "open": (30, 1e9)}.items():
        la = ((land_edge >= a) & (land_edge < b)).sum() * STEP * STEP
        bands[name] = round(float(((sp_edge >= a) & (sp_edge < b)).sum() / la * 1e4), 2) if la else None
    same, spread = territory(xz, gid)
    return {"spawns": len(xz), "area": round(area), "density": round(len(xz) / area * 1e4, 2),
            "used": round(float((near_land <= 25).mean()), 2),
            "alone": round(sum(1 for s in sizes if s == 1) / len(sizes), 2),
            "spacing_median": round(float(np.median(d.min(1))), 1),
            "same_type": round(same, 2), "type_spread": round(spread, 2), "band_density": bands}

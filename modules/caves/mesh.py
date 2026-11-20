"""A generated cave as engine-neutral geometry: per-texture meshes plus a collision mesh.

Frame: x east, y north, z up, in yalms, entrance room centred on the origin with its floor at z = 0.
The zone writer (zonekit/writer.py) converts to FFXI's own axes.

Vertex colours follow FFXI's terrain shader: the colour MODULATES the live light, 128 is neutral
("lit as the engine lights it"), lower darkens. We bake mild darkening up the walls and in the
dome, and a cold glow near the ice sheets.
"""

import numpy as np
from scipy import ndimage

from .network import Network, surface_nets, vertex_normals

FLOOR, WALL_X, WALL_Y, CEILING, BED, SHORE, FOOT_X, FOOT_Y, PILLAR, PILLAR_FOOT = range(10)
SURFACES = ["floor", "wall", "wall", "ceiling", "bed", "shore", "foot", "foot", "wall", "foot"]
# walls and wall feet: two flat uv projections each; pillars: wrapped around their axis


def build(cave, textures, uv_scale=8.0, max_triangles=400000, water=None, uv=None, foot=None, overlay=None):
    """cave: the generator settings (seed, rooms, voxel, noise, depth, extent). textures: surface ->
    texture name. water: {"texture", "alpha", "features": ["lake", "river", "moat", "rim"]} fills
    those features' pits with a translucent surface at wading depth; the rest stay frozen sheets.
    foot: {"height": 7.7, "repeat": 16, "u_max": 0.86, "v": [0.5, 1.0]}: the lower part of every wall uses textures["foot"]
    as a strip from the floor (v[1], snow) up to that height (v[0], rock), tiling along the wall within [0, u_max):
    how Beaucedine lays its cliff-foot texture (gar_ykk1, measured from its meshes).
    overlay: {"texture", "alpha", "scale", "seed"}: a translucent pass over the floors whose opacity follows
    smooth noise, so a repeating floor texture stops reading as a grid.
    Returns {"meshes": [...], "collision": {...}, "rooms": [...], "voxel": v}."""
    net = Network(cave)
    wet_kinds = set((water or {}).get("features", []))
    for r in net.rooms:
        r.lava_depth *= cave.get("depth", 0.35)
        r.frozen = not any(lv[0] in wet_kinds for lv in r.lava)
    voxel = cave.get("voxel", 3.0)
    while True:
        d, floor, lo = net.grid(voxel)
        verts, tris = surface_nets(d, np.array(lo), voxel)
        if len(tris) <= max_triangles:
            break
        voxel += 0.5
    lo = np.array(lo)
    nx, ny = d.shape[:2]
    gx, gy = np.meshgrid(lo[0] + voxel * np.arange(nx), lo[1] + voxel * np.arange(ny), indexing="ij")

    # where the (frozen) features are, and a soft glow around them
    sheet = np.zeros(gx.shape, dtype=bool)          # every pit (frozen or under water): the glow
    ice = np.zeros(gx.shape, dtype=bool)            # frozen pits only: ice-sheet bed and shore band
    for r in net.rooms:
        if r.lava:
            pit = r.inside(gx, gy) & (net.channel(r, gx, gy) > 0.5)
            sheet |= pit
            if r.frozen:
                ice |= pit
    glow2d = ndimage.gaussian_filter(sheet.astype(float), 14.0 / voxel)
    glow2d = np.clip(glow2d / max(glow2d.max(), 1e-6), 0, 1) ** 0.6
    # under water the bed and banks keep the floor texture: the water surface gives the effect, and a
    # texture change there only draws a hard seam at the waterline
    near2d = ndimage.binary_dilation(ice, iterations=max(1, int(round(4 / voxel))))

    def at2d(a, p):
        i = np.clip(np.round((p[:, 0] - lo[0]) / voxel).astype(int), 0, nx - 1)
        j = np.clip(np.round((p[:, 1] - lo[1]) / voxel).astype(int), 0, ny - 1)
        return a[i, j]

    def smooth2d(a, p):
        return ndimage.map_coordinates(a, [(p[:, 0] - lo[0]) / voxel, (p[:, 1] - lo[1]) / voxel], order=1, mode="nearest")

    floor_soft = ndimage.gaussian_filter(floor, 2.0)

    def floor_at(p):
        return ndimage.map_coordinates(floor_soft, [(p[:, a] - lo[a]) / voxel for a in range(3)], order=1, mode="nearest")

    p = verts[tris]
    n = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    cen = p.mean(1)
    normals = vertex_normals(verts, tris)
    ns = normals[tris].mean(1)                             # the surface's slope, not each triangle's: a
    ns /= np.linalg.norm(ns, axis=1, keepdims=True) + 1e-12   # stepped slope alternates steep and flat
    # the ground under every spot: the lowest open air in its column; wall points (no air in their
    # column) take the nearest column's. Heights up a wall are measured from this.
    air = d < 0
    has_air = air.any(2)
    lowest = np.where(has_air, lo[2] + voxel * np.argmax(air, axis=2), np.nan)
    _, (ii, jj) = ndimage.distance_transform_edt(~has_air, return_indices=True)
    ground2d = ndimage.gaussian_filter(lowest[ii, jj], 1.0)

    def ground_at(q):
        return ndimage.map_coordinates(ground2d, [(q[:, 0] - lo[0]) / voxel, (q[:, 1] - lo[1]) / voxel],
                                       order=1, mode="nearest")

    mat = np.where(np.abs(n[:, 0]) > np.abs(n[:, 1]), WALL_X, WALL_Y)
    mat = np.where((n[:, 2] < -0.4) & (cen[:, 2] - floor_at(cen) > 12.0), CEILING, mat)   # overhangs low
                                                                        # on a wall stay wall (were dark diamonds)
    # floor: level ground, or a gentle slope at ground level. A steep face up the side of a wall is
    # wall even where it leans back (were floor-textured teeth cut into the wall foot)
    up = (ns[:, 2] > 0.85) | ((ns[:, 2] > 0.5) & (cen[:, 2] - ground_at(cen) < 1.5))
    mat = np.where(up, FLOOR, mat)
    low = cen[:, 2] < floor_at(cen) - 0.5                 # floor faces down in a frozen pit: the ice sheet
    mat = np.where(up & at2d(ice, cen) & low, BED, mat)
    mat = np.where(up & at2d(near2d, cen) & (mat == FLOOR), SHORE, mat)

    # pillars, stalagmites and stalactites: wrapped around their own axis, so no seams where a flat
    # projection changes axis and no streaks where it runs along the surface
    pillar_of = np.full(len(tris), -1)
    pillars = []                                        # (x, y, widest radius, radius at height z)
    for r in net.rooms:
        for px, py, pr in r.pillars:
            pillars.append((px, py, pr, lambda z, pr=pr: np.full(len(z), pr)))
        for kind, sx, sy, sz, length, rad in r.spikes:
            if kind == "mite":
                pillars.append((sx, sy, rad * 1.3, lambda z, sz=sz, L=length, rad=rad:
                                np.clip(rad * (sz + L - z) / L, 0, rad * 1.3)))
            elif kind == "tite":
                tip = net.ceiling(r, sx, sy) - length
                pillars.append((sx, sy, rad * 1.3, lambda z, tip=tip, L=length, rad=rad:
                                np.clip(rad * (z - tip) / L, 0, rad * 1.3)))
    best = np.full(len(tris), voxel * 0.9)                # on the shaft's surface, give or take a voxel
    walls = np.isin(mat, (WALL_X, WALL_Y))
    for k, (px, py, pr, radius) in enumerate(pillars):
        dist = np.hypot(cen[:, 0] - px, cen[:, 1] - py)
        cand = walls & (dist < pr + voxel)
        if not cand.any():
            continue
        off = np.full(len(tris), np.inf)
        off[cand] = np.abs(dist[cand] - radius(cen[cand, 2]))
        take = off < best
        pillar_of[take], best[take] = k, off[take]
    mat = np.where(pillar_of >= 0, PILLAR, mat)

    foot_h = (foot or {}).get("height", 7.7)
    if foot and "foot" in textures:                        # the bottom of the walls: the cliff-foot strip
        low_wall = cen[:, 2] - ground_at(cen) < foot_h
        mat = np.where((mat == WALL_X) & low_wall, FOOT_X, mat)
        mat = np.where((mat == WALL_Y) & low_wall, FOOT_Y, mat)
        mat = np.where((mat == PILLAR) & low_wall, PILLAR_FOOT, mat)

    def colours_at(q, m):
        shade = {FLOOR: 0.95, WALL_X: 0.9, WALL_Y: 0.9, CEILING: 0.6, BED: 1.0, SHORE: 0.95, FOOT_X: 0.92,
                 FOOT_Y: 0.92, PILLAR: 0.9, PILLAR_FOOT: 0.92}[m]
        if m in (WALL_X, WALL_Y, PILLAR):
            s = shade * (1 - 0.35 * np.clip((q[:, 2] - floor_at(q)) / 40.0, 0, 1))
        else:
            s = np.full(len(q), shade)
        g = smooth2d(glow2d, q) * np.clip(1 - (q[:, 2] - floor_at(q)) / 50.0, 0.1, 1)
        b = np.clip(128 * s + 70 * g, 0, 255)            # cold glow: more blue, a little green
        gg = np.clip(128 * s + 40 * g, 0, 255)
        r = np.clip(128 * s + 10 * g, 0, 255)
        return np.stack([b, gg, r, np.full(len(q), 128)], 1).astype(np.uint8)   # BGRA

    def unwrap(t):
        """A triangle's turn values made continuous across the 1 -> 0 wrap."""
        return np.where(t - t.min(1, keepdims=True) > 0.5, t - 1.0, t)

    meshes = []
    for m, surface in enumerate(SURFACES):
        sel_mask = mat == m
        if not sel_mask.any():
            continue
        scale = (uv or {}).get(surface, uv_scale)       # yalms per texture repeat, per surface
        tri_v = tris[sel_mask]
        if m in (FOOT_X, FOOT_Y, PILLAR_FOOT, PILLAR):
            # own vertices per triangle, with "along" (the wall's run in repeats) worked out per triangle
            P = verts[tri_v].astype(np.float64)
            N = normals[tri_v].astype(np.float64)
            if m in (PILLAR, PILLAR_FOOT):
                ks = pillar_of[sel_mask]
                circ = np.array([2 * np.pi * pillars[k][2] for k in ks])
                reps = foot.get("repeat", 16.0) if m == PILLAR_FOOT else scale
                count = np.maximum(1, np.round(circ / reps))
                if m == PILLAR_FOOT:
                    count = np.maximum(2, 2 * np.round(circ / reps / 2))   # even: the mirror closes
                centre = np.array([pillars[k][:2] for k in ks])[:, None, :]
                d2 = P[:, :, :2] - centre
                turns = (np.arctan2(d2[..., 1], d2[..., 0]) / (2 * np.pi)) % 1.0
                S = unwrap(turns) * count[:, None]
            else:
                S = (P[:, :, 1] if m == FOOT_X else P[:, :, 0]) / foot.get("repeat", 16.0)
            if m == PILLAR:                                # the wall texture tiles: no cuts needed
                pos = P.reshape(-1, 3)
                tex_uv = np.stack([S.reshape(-1), pos[:, 2] / scale], 1)
                nrm = N.reshape(-1, 3)
            else:
                # gar_ykk1 is an atlas: snow at both ends of v with rock between, and a separate strip in
                # the last eighth of u. The foot uses the rock-to-snow half (v 0.5 -> 1 at the ground) and
                # u inside [0, u_max), mirrored every other repeat: triangles are cut where a repeat ends,
                # so neighbours always agree and none reaches the strip.
                P, N, S = cut_at_integers(P, N, S)
                band = np.floor(S.min(1, keepdims=True) + 1e-9)
                frac = np.clip(S - band, 0, 1)
                umax = foot.get("u_max", 0.86)
                u = umax * np.where(band % 2 == 0, frac, 1 - frac)
                pos, nrm = P.reshape(-1, 3), N.reshape(-1, 3)
                up_h = np.abs(pos[:, 2] - ground_at(pos))          # below the ground (banks): mirror the rock
                v_top, v_ground = foot.get("v", (0.5, 1.0))           # the strip's part of the atlas
                tex_uv = np.stack([u.reshape(-1),
                                   v_ground + (v_top - v_ground) * np.clip(up_h / foot_h, 0, 1)], 1)
            nrm = nrm / (np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12)
            local = np.arange(len(pos))
        else:
            used, local = np.unique(tri_v, return_inverse=True)
            pos, nrm = verts[used], normals[used]
            if m == WALL_X:
                tex_uv = pos[:, [1, 2]] / scale
            elif m == WALL_Y:
                tex_uv = pos[:, [0, 2]] / scale
            else:
                tex_uv = pos[:, [0, 1]] / scale
        meshes.append({"surface": surface, "texture": textures[surface], "positions": pos.astype(np.float32),
                       "normals": nrm.astype(np.float32), "uvs": tex_uv.astype(np.float32),
                       "colours": colours_at(pos, m), "triangles": local.reshape(-1, 3).astype(np.int32)})

    if water:
        meshes += water_surfaces(net, water, uv_scale)
    if overlay:
        meshes += floor_overlay([m for m in meshes if m["surface"] == "floor"], overlay)
    wall = np.abs(n[:, 2]) < 0.5
    return {"meshes": meshes,
            "collision": {"positions": verts.astype(np.float32), "triangles": tris.astype(np.int32), "wall": wall},
            "rooms": [{"name": r.name, "kind": r.kind, "floor": r.floor, "feature": r.feature,
                       "lobes": [(lb.x, lb.y, lb.r) for lb in r.lobes]} for r in net.rooms],
            "net": net, "voxel": voxel}


def cut_at_integers(P, N, S):
    """Triangles (T,3,3) with a per-corner value S (T,3), cut along S = whole numbers so no triangle
    spans one: each crossing triangle becomes a triangle and a quad (two triangles), winding kept."""
    outP, outN, outS = [], [], []
    for _ in range(64):
        lo_s = S.min(1)
        c = np.floor(lo_s + 1e-9) + 1
        cross = S.max(1) > c + 1e-6
        if not cross.any():
            break
        outP.append(P[~cross]); outN.append(N[~cross]); outS.append(S[~cross])
        P, N, S, c = P[cross], N[cross], S[cross], c[cross]
        below = S < c[:, None]
        # the lone corner: the one on its own side of the cut
        lone = np.where(below.sum(1) == 1, np.argmax(below, 1), np.argmax(~below, 1))
        idx = np.stack([lone, (lone + 1) % 3, (lone + 2) % 3], 1)
        r = np.arange(len(P))[:, None]
        P, N, S = P[r, idx], N[r, idx], S[r, idx]

        def at(i, j):
            t = ((c - S[:, i]) / (S[:, j] - S[:, i]))[:, None]
            return P[:, i] + t * (P[:, j] - P[:, i]), N[:, i] + t * (N[:, j] - N[:, i]), c

        p01, n01, s01 = at(0, 1)
        p02, n02, s02 = at(0, 2)
        P = np.concatenate([np.stack([P[:, 0], p01, p02], 1), np.stack([p01, P[:, 1], P[:, 2]], 1),
                            np.stack([p01, P[:, 2], p02], 1)])
        N = np.concatenate([np.stack([N[:, 0], n01, n02], 1), np.stack([n01, N[:, 1], N[:, 2]], 1),
                            np.stack([n01, N[:, 2], n02], 1)])
        S = np.concatenate([np.stack([S[:, 0], s01, s02], 1), np.stack([s01, S[:, 1], S[:, 2]], 1),
                            np.stack([s01, S[:, 2], s02], 1)])
    outP.append(P); outN.append(N); outS.append(S)
    return np.concatenate(outP), np.concatenate(outN), np.concatenate(outS)


def water_surfaces(net, water, uv_scale, cell=1.5):
    """Flat translucent water over every unfrozen pit, the way Dragon's Aery draws its own: alpha
    blended (part flag 0x8000), vertex alpha ~110 in open water fading to 0 at the shore."""
    out = []
    alpha = water.get("alpha", 110)
    for r in net.rooms:
        if not r.lava or r.frozen:
            continue
        xs = [lb.x for lb in r.lobes]
        ys = [lb.y for lb in r.lobes]
        pad = max(lb.r for lb in r.lobes) + 4
        gx, gy = np.meshgrid(np.arange(min(xs) - pad, max(xs) + pad, cell),
                             np.arange(min(ys) - pad, max(ys) + pad, cell), indexing="ij")
        wet = r.inside(gx, gy) & (net.channel(r, gx, gy) > 0.4)
        if not wet.any():
            continue
        wet = ndimage.binary_dilation(wet, iterations=1)          # run under the banks: no gaps
        fade = np.clip(ndimage.distance_transform_edt(wet) / 3.0, 0, 1)
        nx, ny = gx.shape
        vid = -np.ones((nx + 1, ny + 1), dtype=np.int64)
        corners = np.zeros((nx + 1, ny + 1), dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                corners[di:nx + di, dj:ny + dj] |= wet
        vid[corners] = np.arange(corners.sum())
        ci, cj = np.nonzero(corners)
        vx = gx[0, 0] - cell / 2 + ci * cell
        vy = gy[0, 0] - cell / 2 + cj * cell
        z = np.full(len(vx), r.lava_level)
        # alpha from the nearest cell's shore distance
        a = fade[np.clip(ci, 0, nx - 1), np.clip(cj, 0, ny - 1)] * alpha
        qi, qj = np.nonzero(wet)
        v00, v10 = vid[qi, qj], vid[qi + 1, qj]
        v01, v11 = vid[qi, qj + 1], vid[qi + 1, qj + 1]
        tris = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])  # face up
        pos = np.stack([vx, vy, z], 1).astype(np.float32)
        colours = np.stack([np.full(len(vx), 128), np.full(len(vx), 128), np.full(len(vx), 128),
                            a], 1).astype(np.uint8)
        out.append({"surface": "water", "texture": water["texture"], "flags": 0x8000,
                    "positions": pos, "normals": np.tile([0, 0, 1], (len(vx), 1)).astype(np.float32),
                    "uvs": (pos[:, :2] / (uv_scale * 1.5)).astype(np.float32), "colours": colours,
                    "triangles": tris.astype(np.int32)})
    return out


def floor_overlay(floors, spec):
    """A second, alpha-blended pass over the floors (lifted a hair to avoid z-fighting), its opacity
    following large smooth noise: drifts of plain snow and patches of the ground below."""
    rng = np.random.default_rng(spec.get("seed", 7))
    cell = spec.get("patch", 18.0)                          # yalms per noise cell
    grid = rng.uniform(0, 1, (64, 64))
    out = []
    for f in floors:
        pos = f["positions"].copy()
        pos[:, 2] += 0.06
        gx = (pos[:, 0] / cell) % 60 + 1
        gy = (pos[:, 1] / cell) % 60 + 1
        nz = ndimage.map_coordinates(grid, [gx, gy], order=3, mode="wrap")
        a = np.clip((nz - 0.35) / 0.4, 0, 1) * spec.get("alpha", 100)
        col = f["colours"].copy()
        col[:, 3] = a.astype(np.uint8)
        out.append({"surface": "overlay", "texture": spec["texture"], "flags": 0x8000, "positions": pos,
                    "normals": f["normals"], "uvs": (pos[:, :2] / spec.get("scale", 20.0)).astype(np.float32),
                    "colours": col, "triangles": f["triangles"]})
    return out

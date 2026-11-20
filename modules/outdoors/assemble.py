"""Turning what a zone places into what the writer takes: objects, collision (a house as a box, a
wall as a block, fences round the map and along its cliff tops), our own pieces laid as the zone's
geometry, and the sea."""
import math

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

from .place import obj_of

TILE = T.TILE


def lay_geometry(mesh, o, full, laid):
    """A built piece (architecture.Mesh, y up, its own frame) placed by o (FFXI) into `laid`:
    {texture16: [pos, nrm, uv, bgra, tris]} in the cave frame, wound to face the air."""
    R = T._rot_zyx(o["rot"]) * np.array(o["scale"])
    for tex, (pos, nrm, uv, col, tri) in mesh.arrays().items():
        pf = pos * np.array([1, -1, 1], np.float32)          # its own frame, FFXI (y down)
        nf = nrm * np.array([1, -1, 1], np.float32)
        wp = pf @ R.T + np.array(o["pos"])
        wn = nf @ R.T
        wn /= np.linalg.norm(wn, axis=1, keepdims=True) + 1e-12
        c = np.clip(col * 128, 0, 255).astype(np.uint8)
        bgra = np.stack([c, c, c, np.full_like(c, 128)], 1)
        acc = laid.setdefault(full[tex], [[], [], [], [], [], 0])
        base = acc[5]
        acc[0].append(fz.from_ffxi(wp))
        acc[1].append(fz.from_ffxi(wn))
        acc[2].append(uv)
        acc[3].append(bgra)
        # retail winding in FFXI's frame is normal = -(b-a)x(c-a); a proper rotation into the cave
        # frame keeps it: turned over to face the air
        acc[4].append(tri[:, [0, 2, 1]] + base)
        acc[5] = base + len(pos)


def laid_meshes(laid):
    return [{"texture": tex, "flags": 0,
             "positions": np.concatenate(a[0]).astype(np.float32), "normals": np.concatenate(a[1]).astype(np.float32),
             "uvs": np.concatenate(a[2]).astype(np.float32), "colours": np.concatenate(a[3]),
             "triangles": np.concatenate(a[4]).astype(np.int32)} for tex, a in laid.items()]


def box(tris, o, ground, up=6.0):
    """A piece's collision as its own box (its bounds in its own frame, placed), from a yalm under
    the ground to `up` over it: you walk into a house's walls, never onto its roof. Twelve
    triangles in the cave frame, wound outward, in place of the piece's hundreds."""
    t = tris.reshape(-1, 3)
    lo, hi = t.min(0), t.max(0)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = T.placed(c, o)                                   # FFXI: y is down
    top = max(w[:, 1].min(), -(ground + up))
    bot = min(w[:, 1].max(), -(ground - 1.0))
    if top >= bot:
        return np.zeros((0, 3, 3))
    w[:, 1] = np.where((np.arange(8) // 2) % 2 == 0, top, bot) if lo[1] < hi[1] else w[:, 1]
    v = fz.from_ffxi(w)
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    out = np.array([tri for a, b, cc, d in faces for tri in ([v[a], v[b], v[cc]], [v[a], v[cc], v[d]])])
    mid = v.mean(0)
    n = np.cross(out[:, 1] - out[:, 0], out[:, 2] - out[:, 0])
    flip = ((out.mean(1) - mid) * n).sum(1) < 0
    out[flip] = out[flip][:, [0, 2, 1]]
    return out


def writer_obj(kit, name16, o, lod):
    t = kit["tris"][name16].reshape(-1, 3)
    corners = np.array([[x, y, z] for x in (t[:, 0].min(), t[:, 0].max()) for y in (t[:, 1].min(), t[:, 1].max())
                        for z in (t[:, 2].min(), t[:, 2].max())])
    R = T._rot_zyx(o["rot"]) * np.array(o["scale"])
    w = corners @ R.T + np.array(o["pos"])
    return {"name": name16, "pos": np.array(o["pos"], dtype=np.float64), "rot": tuple(o["rot"]),
            "scale": tuple(o["scale"]), "lo": w.min(0), "hi": w.max(0), "lod": lod}


def ours(w_ffxi, scale):
    """Placed FFXI triangles -> the cave frame, wound our way (a mirrored placement is already
    turned over, so it keeps its order)."""
    cave = fz.from_ffxi(w_ffxi.reshape(-1, 3)).reshape(-1, 3, 3)
    if np.prod(np.sign(scale)) > 0:
        cave = cave[:, [0, 2, 1]]
    return cave


def fence(W, D, shift, height=120.0):
    """Invisible walls round the map (collision only), both faces."""
    x0, z0 = shift[0], shift[2]
    x1, z1 = x0 + W * TILE, z0 + D * TILE
    ring = [(x0, z0), (x1, z0), (x1, z1), (x0, z1), (x0, z0)]
    quads = []
    for (ax, az), (bx, bz) in zip(ring, ring[1:]):
        a, b = np.array([ax, 40.0, az]), np.array([bx, 40.0, bz])       # FFXI y down: 40 below ground
        a2, b2 = a - [0, height, 0], b - [0, height, 0]
        for tri in ((a, b, b2), (a, b2, a2)):
            quads.append(fz.from_ffxi(np.array(tri)))
            quads.append(fz.from_ffxi(np.array(tri))[[0, 2, 1]])
    return np.array(quads)


def sea_tiles(ground, kit, spec, shift, W, D, rng, name_of):
    """The sea laid from the source's own sea tiles (Misareaux's are 80-yalm flats at -36 that meet
    the foot of its sea cliffs): under every tile with open sea at a corner, and out past the map."""
    from .kit import HOLE
    names = [n for n in kit["tris"] if name_of(n) in spec["names"]]
    if not names:
        return []
    size = float(spec.get("size", 80))
    per = int(round(size / TILE))
    margin = int(round(spec.get("margin", 240) / size))
    wet = set()
    for i in range(W):
        for j in range(D):
            if (ground[i:i + 2, j:j + 2] <= HOLE / 2).any():
                wet.add((i // per, j // per))
    nx, nz = (W + per - 1) // per, (D + per - 1) // per
    out = []
    for a in range(-margin, nx + margin):
        for b in range(-margin, nz + margin):
            inside = 0 <= a < nx and 0 <= b < nz
            if inside and (a, b) not in wet:
                continue
            name = rng.choice(names)
            yaw = rng.choice([0, 1, 2, 3]) * math.pi / 2
            pts = kit["tris"][name].reshape(-1, 3)
            mid = (pts.min(0) + pts.max(0)) / 2                # these pieces aren't centred on their origin
            mid[1] = 0.0
            off = T._rot_zyx((0.0, yaw, 0.0)) @ mid
            out.append({"name": name, "rot": (0.0, yaw, 0.0), "scale": (1.0, 1.0, 1.0),
                        "pos": ((a + 0.5) * size + shift[0] - off[0],
                                -(float(spec.get("height", -36)) - spec.get("sink", 0.3)),   # just under the
                                (b + 0.5) * size + shift[2] - off[2])})                       # cliffs' own foot
    return out


def brink_fence(ground, placed, kit, shift, drop=2.0, step=1.0, height=40.0, gaps=(), gap=10.0):
    """An invisible wall at the brink of an island's sea cliffs: each shore tile's ground sampled
    every `step` yalms, and the wall traced (marching squares) where it falls `drop` below the land
    beside it. A wall through the middle of the tile let players slide down the face first and stick."""
    from .kit import HOLE
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    n = int(round(TILE / step))
    quads = []
    for (i, j), c in placed.items():
        g = [ground[i, j + 1], ground[i + 1, j + 1], ground[i + 1, j], ground[i, j]]
        land = [x for x in g if x > HOLE / 2]
        if len(land) == 4 or not land:
            continue                                          # not a shore tile
        top = max(land)
        o = obj_of(c, (i, j))
        w = T.placed(kit["tris"][c["name"]], o).reshape(-1, 3, 3)
        x0, z0 = i * TILE, j * TILE
        xs = x0 + np.arange(n + 1) * step
        zs = z0 + np.arange(n + 1) * step
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        h = T.heights(w, X.ravel(), Z.ravel()).reshape(X.shape)
        up = np.nan_to_num(h, nan=-1e9) >= top - drop            # standing ground
        for a in range(n):
            for b in range(n):
                sq = [up[a, b], up[a + 1, b], up[a + 1, b + 1], up[a, b + 1]]   # (x,z) (x+1,z) (x+1,z+1) (x,z+1)
                if all(sq) or not any(sq):
                    continue
                mids = {0: (a + 0.5, b), 1: (a + 1, b + 0.5), 2: (a + 0.5, b + 1), 3: (a, b + 0.5)}
                cut = [k for k in range(4) if sq[k] != sq[(k + 1) % 4]]
                for e, f in zip(cut[0::2], cut[1::2]):
                    (ax, az), (bx, bz) = mids[e], mids[f]
                    mx, mz = x0 + (ax + bx) / 2 * step, z0 + (az + bz) / 2 * step
                    if any(math.hypot(mx - gx, mz - gz) < gap for gx, gz in gaps):
                        continue                                  # the way onto a bridge
                    pa = np.array([x0 + ax * step + shift[0], -(top - height / 2), z0 + az * step + shift[2]])
                    pb = np.array([x0 + bx * step + shift[0], -(top - height / 2), z0 + bz * step + shift[2]])
                    pa2, pb2 = pa - [0, height, 0], pb - [0, height, 0]
                    for tri in ((pa, pb, pb2), (pa, pb2, pa2)):
                        quads.append(fz.from_ffxi(np.array(tri)))
                        quads.append(fz.from_ffxi(np.array(tri))[[0, 2, 1]])
    return np.array(quads).reshape(-1, 3, 3) if quads else np.zeros((0, 3, 3))


def shore_fence(ground, shift, sea, height=80.0):
    """An invisible wall along an island's shore: the line between its land corners and its open-sea
    corners, through the middle of each cliff tile (a marching-squares contour), so nobody walks off
    the cliff top into a sea with no ground under it."""
    from .kit import HOLE
    land = np.nan_to_num(ground, nan=HOLE) > HOLE / 2
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    segs = []
    for i in range(W):
        for j in range(D):
            c = [land[i, j + 1], land[i + 1, j + 1], land[i + 1, j], land[i, j]]          # NW NE SE SW
            if all(c) or not any(c):
                continue
            mids = {"n": (i + 0.5, j + 1), "e": (i + 1, j + 0.5), "s": (i + 0.5, j), "w": (i, j + 0.5)}
            cut = [side for side, (a, b) in {"n": (0, 1), "e": (1, 2), "s": (2, 3), "w": (3, 0)}.items() if c[a] != c[b]]
            for a, b in zip(cut[0::2], cut[1::2]):
                segs.append((mids[a], mids[b]))
    quads = []
    for (ax, az), (bx, bz) in segs:
        a = np.array([ax * TILE + shift[0], -sea + 10.0, az * TILE + shift[2]])
        b = np.array([bx * TILE + shift[0], -sea + 10.0, bz * TILE + shift[2]])
        a2, b2 = a - [0, height, 0], b - [0, height, 0]
        for tri in ((a, b, b2), (a, b2, a2)):
            quads.append(fz.from_ffxi(np.array(tri)))
            quads.append(fz.from_ffxi(np.array(tri))[[0, 2, 1]])
    return np.array(quads).reshape(-1, 3, 3) if quads else np.zeros((0, 3, 3))


def sea(W, shift, cfg, D=None):
    """A still sea over the coast rows and out past the map, translucent over the seabed, at the
    source's sea level (where its beach slope meets the water)."""
    coast = cfg["coast"]
    level = float(coast["sea_level"])
    margin, side = coast.get("sea_margin", 280), coast.get("sea_side_margin", 560)
    rows = coast.get("sea_rows", 3)
    x0, z0 = shift[0] - side, shift[2] - margin
    x1, z1 = shift[0] + W * TILE + side, shift[2] + (rows + 0.8) * TILE
    if D is not None:                                   # an island: sea all round, under the land too
        z1 = shift[2] + D * TILE + margin
    xs, zs = np.arange(x0, x1 + 1, 40.0), np.arange(z0, z1 + 1, 40.0)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    pos = fz.from_ffxi(np.stack([gx.ravel(), np.full(gx.size, -level), gz.ravel()], 1))
    idx = np.arange(len(xs) * len(zs)).reshape(len(xs), len(zs))
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    tris = np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)])
    p = pos[tris]
    down = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])[:, 2] <= 0
    tris[down] = tris[down][:, [0, 2, 1]]
    shade = coast.get("sea_shade", 118)                      # 128 is the engine's own light
    col = np.tile(np.array([shade, shade, shade, coast.get("sea_alpha", 150)], np.uint8), (len(pos), 1))
    return {"surface": "water", "texture": coast.get("sea_texture", "effect  umi1"), "flags": 0x8000,
            "positions": pos.astype(np.float32), "normals": np.tile([0, 0, 1], (len(pos), 1)).astype(np.float32),
            "uvs": (pos[:, :2] / coast.get("sea_repeat", 24.0)).astype(np.float32), "colours": col,
            "triangles": tris.astype(np.int32)}


def water_fence(cells, kit, shift, gaps=(), step=1.0, low=-40.0, high=120.0):
    """An invisible wall where the ground of shore cells you walk down to (a beach running out into
    the water, with nothing under it beyond) ends: each cell's ground sampled every `step` yalms and
    the wall traced (marching squares) round where it stops, cut where a gap (x, z, r) crosses it (a
    jetty going out). Walking into the shallows is fine; walking off the end of the world isn't."""
    n = int(round(TILE / step))
    quads = []
    for (i, j), c in cells.items():
        w = T.placed(kit["tris"][c["name"]], obj_of(c, (i, j))).reshape(-1, 3, 3)
        xs = i * TILE + np.arange(n + 1) * step
        zs = j * TILE + np.arange(n + 1) * step
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        has = ~np.isnan(T.heights(w, X.ravel(), Z.ravel()).reshape(X.shape))
        for a in range(n):
            for b in range(n):
                sq = [has[a, b], has[a + 1, b], has[a + 1, b + 1], has[a, b + 1]]
                if all(sq) or not any(sq):
                    continue
                mids = {0: (a + 0.5, b), 1: (a + 1, b + 0.5), 2: (a + 0.5, b + 1), 3: (a, b + 0.5)}
                cut = [k for k in range(4) if sq[k] != sq[(k + 1) % 4]]
                for e, f in zip(cut[0::2], cut[1::2]):
                    (ax, az), (bx, bz) = mids[e], mids[f]
                    pa = (i * TILE + ax * step + shift[0], j * TILE + az * step + shift[2])
                    pb = (i * TILE + bx * step + shift[0], j * TILE + bz * step + shift[2])
                    mx, mz = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
                    if any(math.hypot(mx - gx, mz - gz) < r for gx, gz, r in gaps):
                        continue
                    A, B = np.array([pa[0], -low, pa[1]]), np.array([pb[0], -low, pb[1]])     # FFXI: y down
                    A2, B2 = np.array([pa[0], -high, pa[1]]), np.array([pb[0], -high, pb[1]])
                    for tri in ((A, B, B2), (A, B2, A2)):
                        quads.append(fz.from_ffxi(np.array(tri)))
                        quads.append(fz.from_ffxi(np.array(tri))[[0, 2, 1]])
    return np.array(quads).reshape(-1, 3, 3) if quads else np.zeros((0, 3, 3))


class Laid:
    """Raw geometry for lay_geometry (y up, its own frame): {texture: (pos, nrm, uv, col, tri)}."""

    def __init__(self, parts):
        self.parts = parts

    def arrays(self):
        return self.parts


def seabed(land, box, spec):
    """A floor under the sea, for a source whose sea has none (Qufim's water is effects over
    nothing): shallow by the land, sloping down away from it, in the source's own rock.
    land: [(x, z)] centres of land cells (world); box: (x0, z0, x1, z1) world.
    spec: {"texture", "yalms", "shallow", "deep", "slope" (yalms to full depth), "cell", "shade"}.
    Returns (Laid, collision triangles FFXI (k, 3, 3))."""
    from .rock import noise
    cell = spec.get("cell", 8.0)
    x0, z0, x1, z1 = box
    xs = np.arange(x0, x1 + cell, cell)
    zs = np.arange(z0, z1 + cell, cell)
    X, Z = np.meshgrid(xs, zs, indexing="ij")
    L = np.array(land, float)
    d = np.full(X.shape, 1e9)
    for lx, lz in L:                                       # distance to the nearest land cell's edge
        d = np.minimum(d, np.maximum(np.abs(X - lx), np.abs(Z - lz)) - TILE / 2)
    t = np.clip(d / spec.get("slope", 60.0), 0.0, 1.0)
    H = spec.get("shallow", -2.0) + (spec.get("deep", -9.0) - spec.get("shallow", -2.0)) * t
    H = H + noise(X * 0.08, 1.7, Z * 0.08) * 1.2
    nx, nz = X.shape
    pos = np.stack([X.ravel(), H.ravel(), Z.ravel()], 1).astype(np.float32)
    idx = np.arange(nx * nz).reshape(nx, nz)
    a, b, c, dd = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    tri = np.concatenate([np.stack([a, c, b], 1), np.stack([a, dd, c], 1)])
    P = pos[tri]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    flip = n[:, 1] < 0                                     # faces up
    tri[flip] = tri[flip][:, [0, 2, 1]]
    nrm = np.tile([0.0, 1.0, 0.0], (len(pos), 1)).astype(np.float32)
    y = spec.get("yalms", 12.0)
    uv = (pos[:, [0, 2]] / y).astype(np.float32)
    col = np.full(len(pos), spec.get("shade", 0.7), np.float32)
    laid = Laid({spec.get("texture", "quf_wll1"): (pos, nrm, uv, col, tri.astype(np.int64))})
    solid = (pos * np.array([1, -1, 1], np.float32))[tri]
    return laid, solid

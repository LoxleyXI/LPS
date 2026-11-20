"""Rock from a map: the massifs, spires, buttresses and the arch of maps/<name>.json as one solid,
meshed like the caves (a density field, surface nets) and textured from a canyon palette
(kits/architecture/<name>.json: its strata on the steep faces with the bands level, its ground on
the tops, cut on the atlas's own tile lines).

The shapes are a three.js terrain prototype's: the same Perlin noise
(its permutation table), terraced mesas, spires, buttresses under the arch's legs, uplands round
the basin. The prototype's z runs south; the map's runs north, so noise is read at -z.

    mesh = rock.build(map, palette("tahrongi"), box=(x0, z0, x1, z1), voxel=6.0)

Frames: the mesh is y up in map coordinates (yalms): lay it at the zone's map origin.
"""
import math

import numpy as np

from .architecture import Mesh, _clip

_P = np.array([
    151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99, 37, 240, 21, 10,
    23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32, 57, 177, 33, 88, 237, 149, 56, 87,
    174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27, 166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211,
    133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102, 143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208,
    89, 18, 169, 200, 196, 135, 130, 116, 188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5,
    202, 38, 147, 118, 126, 255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119,
    248, 152, 2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224, 232,
    178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81, 51, 145, 235, 249,
    14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50, 45, 127, 4, 150, 254, 138, 236, 205,
    93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61, 156, 180], dtype=np.int64)
_P = np.concatenate([_P, _P])


def noise(x, y, z):
    """Ken Perlin's improved noise, as three.js's ImprovedNoise (arrays in, array out)."""
    x, y, z = (np.asarray(a, dtype=np.float64) for a in np.broadcast_arrays(x, y, z))
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    X, Y, Z = (fx.astype(np.int64) & 255), (fy.astype(np.int64) & 255), (fz.astype(np.int64) & 255)
    x, y, z = x - fx, y - fy, z - fz
    fade = lambda t: t * t * t * (t * (t * 6 - 15) + 10)
    u, v, w = fade(x), fade(y), fade(z)

    def grad(h, x, y, z):
        h = h & 15
        a = np.where(h < 8, x, y)
        b = np.where(h < 4, y, np.where((h == 12) | (h == 14), x, z))
        return np.where(h & 1, -a, a) + np.where(h & 2, -b, b)
    A = _P[X] + Y
    AA, AB = _P[A] + Z, _P[A + 1] + Z
    B = _P[X + 1] + Y
    BA, BB = _P[B] + Z, _P[B + 1] + Z
    lerp = lambda t, a, b: a + t * (b - a)
    return lerp(w, lerp(v, lerp(u, grad(_P[AA], x, y, z), grad(_P[BA], x - 1, y, z)),
                        lerp(u, grad(_P[AB], x, y - 1, z), grad(_P[BB], x - 1, y - 1, z))),
                lerp(v, lerp(u, grad(_P[AA + 1], x, y, z - 1), grad(_P[BA + 1], x - 1, y, z - 1)),
                        lerp(u, grad(_P[AB + 1], x, y - 1, z - 1), grad(_P[BB + 1], x - 1, y - 1, z - 1))))


def fbm(x, z, octaves, s=0.0):
    a, f, total, n = 1.0, 1.0, 0.0, 0.0
    for i in range(octaves):
        total = total + a * noise(x * f, z * f, s + i * 7.1)
        n += a
        a *= 0.5
        f *= 2.03
    return total / n


def smooth(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def terrace(v, k, sharp=0.75):
    """Flat shelves with steep risers: layered sandstone."""
    s = v * k
    f = np.floor(s)
    return (f + smooth(sharp, 1.0, s - f)) / k


def _massif(x, z, m):
    """A terraced mesa (prototype coords: z south)."""
    dx, dz = x - m["x"], z - (-m["z"])
    d = np.hypot(dx, dz)
    a = np.arctan2(dz, dx)
    s = m["s"]
    R = m["r"] * (1 + 0.18 * noise(np.cos(a) * 1.5, np.sin(a) * 1.5, s * 13.7)) + noise(x * 0.06, z * 0.06, s * 5) * 6
    if m.get("ribs"):                                  # fluted walls: columns with creases between
        n = m.get("rib_n", 24)
        wob = noise(np.cos(a) * 3, np.sin(a) * 3, s * 19.3) * 0.6
        R = R + m["ribs"] * (np.abs(np.sin(a * n / 2 + wob)) * 2 - 1)
    body = 1 - smooth(R - m["edge"], R, d)
    h = terrace(body, m.get("steps", 5), 0.6) * (m["h"] + fbm(x * 0.03, z * 0.03, 3, s * 3) * m["rough"])
    foot = (1 - smooth(R - 5, R + m["foot"], d)) * m["footH"] + noise(x * 0.05, z * 0.05, s * 7) * 4
    h = np.maximum(h, foot)
    return np.where(d > m["r"] + m["foot"] + 25, -1e9, h)


def _catmull(points, n=161):
    """A Catmull-Rom curve through the points, resampled evenly by length: (n, 3) and t (n,)."""
    P = np.array(points, dtype=np.float64)
    P = np.concatenate([P[:1] * 2 - P[1:2], P, P[-1:] * 2 - P[-2:-1]])
    dense = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, 40, endpoint=False):
            t2, t3 = t * t, t * t * t
            dense.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    dense.append(P[-2])
    dense = np.array(dense)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)]) / seg.sum()
    ts = np.linspace(0, 1, n)
    return np.stack([np.interp(ts, cum, dense[:, k]) for k in range(3)], 1), ts


def heights(m, x, z):
    """The rock's height field at map points (x east, z north): uplands round the basin, the
    massifs, the buttresses under the arch's legs and the spires. Below 0.5 there's no rock (the
    zone's land is there)."""
    pz = -z                                            # the prototype's z
    b = m["basin"]
    ex, ez = (x - b["x"]) / b["rx"], (pz - (-b["z"])) / b["rz"]
    basin_t = smooth(0.8, 1.15, np.sqrt(ex * ex + ez * ez) + fbm(x * 0.006, pz * 0.006, 2, 41) * 0.25)
    n = fbm(x * 0.0032, pz * 0.0032, 4, 11)
    up = terrace(np.clip(0.45 + n * 1.3, 0, 1), 6) * 150 + fbm(x * 0.018, pz * 0.018, 3, 21) * 9
    h = up * basin_t
    for mm in m["massifs"]:
        h = np.maximum(h, _massif(x, pz, mm))
    # buttresses: the ground swells under each end of the arch so it flares out of its pillars
    curve, ts = _catmull([(p[0], p[1], -p[2]) for p in m["arch"]])     # (x, height, prototype z)
    legs, mid = m["arch_radius"]["legs"], m["arch_radius"]["middle"]
    for (cx, cy, cz), t in zip(curve, ts):
        e = 1 - smooth(0.1, 0.34, min(t, 1 - t))
        if e <= 0.01:
            continue
        rv, rh = _radius(m, t)
        top = (cy - rv * 0.55) * e
        spread = m["arch_radius"].get("buttress", [1.5, 1.5])[0 if t < 0.5 else 1]
        f = 1 - smooth(rh * 0.45, rh * spread, np.hypot(x - cx, pz - cz))
        h = np.maximum(h, np.where(f > 0, top * terrace(f, 4, 0.55), -1e9))
    for sp in m["spires"]:
        h = h + sp["h"] * (1 - smooth(sp["r"] * 0.35, sp["r"], np.hypot(x - sp["x"], pz - (-sp["z"]))))
    h = np.maximum(h, _banks(m, x, z))
    for c in m.get("castles", []):                     # a level top for the castle to stand on
        level = castle_level(m, c)
        f = 1 - smooth(c["size"] * 0.75, c["size"] * 1.1, np.hypot(x - c["x"], z - c["z"]))
        h = np.where((f > 0) & (h > level - 40), h * (1 - f) + level * f, h)
    if m.get("ramp"):                                   # a ledge cut up a rock face, banked below
        hr, d, qx, qz = ramp_at(m, x, z)
        w2 = m["ramp"]["width"] / 2
        # the side away from the massif it climbs is cut down to the road's bank: a ledge open to
        # the town, not a trench
        mm = min(m["massifs"], key=lambda q: math.hypot(q["x"] - np.mean([p[0] for p in m["ramp"]["points"]]),
                                                         q["z"] - np.mean([p[1] for p in m["ramp"]["points"]])))
        outer = np.hypot(x - mm["x"], z - mm["z"]) > np.hypot(qx - mm["x"], qz - mm["z"])
        rc = {"back": 7.0, "below": 3.5, "bank": 1.6, "landing": 9.0, **m["ramp"].get("cut", {})}
        cut = w2 + rc["back"]                           # the rock kept back from the road's edges
        bank = hr - rc["below"] - (d - cut) * rc["bank"]
        h = np.where(d < cut, hr - rc["below"],
                     np.where(outer & (d < cut + 40), np.where(bank > 0, bank, -10.0), np.maximum(h, bank)))
        k = m["ramp"].get("gate_at")
        if k is not None:                               # a landing at the gate, wide as the gatehouse
            gx, gz, gh = m["ramp"]["points"][k]
            r = m["ramp"]["width"] / 2 + rc["landing"]
            dg = np.hypot(x - gx, z - gz)
            h = np.maximum(h, np.where(dg < r, gh - rc["below"], gh - rc["below"] - (dg - r) * rc["bank"]))
            h = np.where(d < cut, np.minimum(h, hr - rc["below"]), h)   # the road below the gate stays cut
    # below a couple of yalms there's no rock: the scree skirt ends cleanly instead of breaking
    # into specks over the land
    return np.where(h < 2.5, -10.0, h)


def ramp_curve(m, spacing=2.0):
    """The road's line: a smooth curve through its points, (n, 3) as (x, height, z), evenly spaced."""
    pts = [(p[0], p[2], p[1]) for p in m["ramp"]["points"]]
    P = np.array(pts, dtype=np.float64)
    length = float(np.linalg.norm(np.diff(P[:, [0, 2]], axis=0), axis=1).sum())
    curve, _ = _catmull(pts, n=max(8, int(length / spacing)))
    return curve


def ramp_at(m, x, z):
    """The ramp's height at (x, z), the distance to its line and the nearest point on it."""
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    best_d = np.full(np.broadcast(x, z).shape, 1e9)
    best_h = np.zeros_like(best_d)
    qx, qz = np.zeros_like(best_d), np.zeros_like(best_d)
    c = ramp_curve(m, 4.0)
    for a, b in zip([(p[0], p[2], p[1]) for p in c[:-1]], [(p[0], p[2], p[1]) for p in c[1:]]):
        dx, dz = b[0] - a[0], b[1] - a[1]
        t = np.clip(((x - a[0]) * dx + (z - a[1]) * dz) / (dx * dx + dz * dz + 1e-12), 0, 1)
        d = np.hypot(x - a[0] - dx * t, z - a[1] - dz * t)
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_h = np.where(closer, a[2] + (b[2] - a[2]) * t, best_h)
        qx = np.where(closer, a[0] + dx * t, qx)
        qz = np.where(closer, a[1] + dz * t, qz)
    return best_h, best_d, qx, qz


def castle_level(m, c):
    """The height the castle's top is levelled to: its crag's own height there, less the spires."""
    mm = min(m["massifs"], key=lambda q: math.hypot(q["x"] - c["x"], q["z"] - c["z"]))
    return float(mm["h"] * c.get("level", 0.97))


def _banks(m, x, z):
    """A rocky footing along the town walls' outer faces (map["wall_bank"]: height h, width w),
    none in the gateways."""
    wb = m.get("wall_bank")
    x, z = np.asarray(x, dtype=np.float64), np.asarray(z, dtype=np.float64)
    out = np.full(np.broadcast(x, z).shape, -1e9)
    if not wb:
        return out
    cx, cz = m["square"]["x"], m["square"]["z"]
    hb, w, off = wb.get("h", 5.0), wb.get("w", 12.0), wb.get("offset", 5.0)
    for run in m["walls"]:
        for a, b in zip(run, run[1:]):
            dx, dz = b[0] - a[0], b[1] - a[1]
            L = math.hypot(dx, dz)
            nx, nz = -dz / L, dx / L
            if ((a[0] + b[0]) / 2 - cx) * nx + ((a[1] + b[1]) / 2 - cz) * nz < 0:
                nx, nz = -nx, -nz
            ax, az = a[0] + nx * off, a[1] + nz * off       # its line, out from the wall
            t = np.clip(((x - ax) * dx + (z - az) * dz) / (L * L), 0, 1)
            d = np.hypot(x - ax - dx * t, z - az - dz * t)
            side = (x - a[0]) * nx + (z - a[1]) * nz           # none on the town side of the wall
            bank = hb * (1 - smooth(w * 0.3, w, d)) * smooth(-2.0, 1.0, side)
            bank = bank + noise(x * 0.12, z * 0.12, 57.0) * 1.5 * (bank > 0.5)
            out = np.maximum(out, np.where(bank > 0.5, bank, -1e9))
    for g in m.get("gates", []):                        # the gateways stay open
        gw = g.get("width", 12)
        out = np.where(np.hypot(x - g["x"], z - g["z"]) < gw + 10, -1e9, out)
    return out


def _radius(m, t):
    """The arch's section (vertical, horizontal half-sizes) at t along it: each end's own
    thickness (arch_radius "west" and "east", else "legs") holding, then thinning to the middle."""
    ar = m["arch_radius"]
    end = ar.get("west", ar["legs"]) if t < 0.5 else ar.get("east", ar["legs"])
    s = math.sin(math.pi * t) ** 2
    return end[0] + (ar["middle"][0] - end[0]) * s, end[1] + (ar["middle"][1] - end[1]) * s


def arch_field(m, x, y, z):
    """How far inside the arch's rock (x, y, z) are, in yalms (negative: outside): an elliptical
    section swept along its curve, thicker at the legs."""
    curve, ts = _catmull([(p[0], p[1], p[2]) for p in m["arch"]])     # map coords (x, height, z north)
    legs, mid = m["arch_radius"]["legs"], m["arch_radius"]["middle"]
    best = np.full(x.shape, -1e9)
    tang = np.gradient(curve, axis=0)
    for (cx, cy, cz), t, tg in zip(curve[::2], ts[::2], tang[::2]):
        rv, rh = _radius(m, t)
        dx, dy, dz = x - cx, y - cy, z - cz
        th = np.array([tg[0], tg[2]]) / (np.hypot(tg[0], tg[2]) + 1e-9)
        along = dx * th[0] + dz * th[1]
        across = -dx * th[1] + dz * th[0]
        q = np.sqrt((dy / rv) ** 2 + (across / rh) ** 2 + (along / (rh * 0.6)) ** 2)
        best = np.maximum(best, (1 - q) * min(rv, rh))
    return best + fbm(x * 0.05, z * 0.05, 2, 91) * 3


def build(m, pal, box, voxel=6.0, top=290.0, keep=None, scale=1.0, floor=None):
    """The rock over box (x0, z0, x1, z1: map yalms) as a Mesh (y up, map coordinates), faceted,
    textured from pal. keep(x, z) -> bool leaves rock only where it says (a zone's backdrop ring)."""
    from caves.network import surface_nets
    x0, z0, x1, z1 = box
    xs = np.arange(x0, x1 + voxel, voxel)
    ys = np.arange(-6.0, top + voxel, voxel)
    zs = np.arange(z0, z1 + voxel, voxel)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    H = heights(m, gx, gz)
    if floor is not None:                              # land where there's no rock (beyond the zone)
        H = np.maximum(H, floor)
    d = H[:, None, :] - ys[None, :, None]
    GX, GY, GZ = np.meshgrid(xs, ys, zs, indexing="ij")
    d = np.maximum(d, arch_field(m, GX, GY, GZ))
    verts, tris = surface_nets(d, np.array([x0, -6.0, z0]), voxel)
    P = verts[tris]
    if keep is not None:
        c = P.mean(1)
        P = P[np.array([keep(a, b) for a, b in zip(c[:, 0], c[:, 2])], dtype=bool)]
    mesh = _texture(P, pal, scale, {**_LIGHT, **m.get("light", {})})
    mesh.map_frame = True                              # laid at the zone's map origin
    return mesh


_LIGHT = {"sun": [0.45, 0.7, -0.55], "sun_floor": 0.62, "sun_gain": 0.5, "tint": [0.86, 0.3, 0.12], "patch": 0.015}


def _texture(P, pal, scale=1.0, light=_LIGHT):
    """Faceted rock: each triangle flat, its steep faces in the strata (bands level, running round
    the face), its tops in the ground; cut on the atlas's tile lines."""
    mesh = Mesh(pal)
    part = mesh._part(pal["texture"])
    sun_dir = np.array(light["sun"]) / np.linalg.norm(light["sun"])   # y up, z north
    t0, t1, t2 = light["tint"]
    pf = light["patch"]
    yu, yv = pal["yalms"] if isinstance(pal["yalms"], list) else (pal["yalms"], pal["yalms"])
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    for tri, nn in zip(P, n):
        steep = nn[1] < 0.6                            # walls and overhangs (the arch's underside): rock
        if steep:
            u0, v0, u1, v1 = pal["regions"]["strata"]
            # along the face: the horizontal axis it faces least; down it: height
            # projected from the axis it faces most (no stretching on the slopes between): from
            # the side the strata run level round it, from below they run along the underside
            ax = int(np.argmax(np.abs(nn)))
            if ax == 1:
                s, t = tri[:, 0], tri[:, 2]
            else:
                s, t = (tri[:, 2] if ax == 0 else tri[:, 0]), -tri[:, 1]
        else:
            u0, v0, u1, v1 = pal["regions"]["ground"]
            s, t = tri[:, 0], -tri[:, 2]
        tw, th = (u1 - u0) * yu * scale, (v1 - v0) * yv * scale
        # baked light: darker at the foot, and each facet lit by a low sun from the south east so
        # the columns and shelves read (the game draws these colours as they are)
        sun = light["sun_floor"] + light["sun_gain"] * max(0.0, float(nn @ sun_dir))
        c = tri.mean(0)                                # broad patches, lighter and darker, and bands
        tint = t0 + t1 * float(noise(c[0] * pf, c[2] * pf, 3.3)) + t2 * float(noise(c[0] * 0.004, c[1] * 0.05, c[2] * 0.004))
        shade = [min(1.3, (0.55 + 0.45 * min(1.0, max(0.0, y / 40.0))) * sun * tint) for y in tri[:, 1]]
        poly = [(s[k], t[k], tri[k, 0], tri[k, 1], tri[k, 2], shade[k]) for k in range(3)]
        for i in range(int(math.floor(min(s) / tw)), int(math.ceil(max(s) / tw))):
            for j in range(int(math.floor(min(t) / th)), int(math.ceil(max(t) / th))):
                cell = poly
                for axis, val, keep_hi in ((0, i * tw, True), (0, (i + 1) * tw, False), (1, j * th, True), (1, (j + 1) * th, False)):
                    cell = _clip(cell, axis, val, keep_hi)
                    if len(cell) < 3:
                        break
                if len(cell) < 3:
                    continue
                base = len(part["pos"])
                for cs, ct, px, py, pz, sh in cell:
                    part["pos"].append(np.array([px, py, pz]))
                    part["nrm"].append(nn)
                    part["uv"].append((u0 + (cs - i * tw) / tw * (u1 - u0), v0 + (ct - j * th) / th * (v1 - v0)))
                    part["col"].append(sh)
                for k in range(1, len(cell) - 1):
                    part["tri"].append((base, base + k, base + k + 1))
    return mesh

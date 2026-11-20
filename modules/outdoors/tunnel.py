"""Our own rock among the tiles: a block of rock standing in cells a drawn grid reserves (no tiles
there), with a passage carved through it that climbs from one face to another: the way from a landing
up through the cliff and out above the land behind. Meshed like the caves and outdoors/rock.py (a
density field, surface nets), its steep faces in one texture and its tops in another (the source
zone's own cliff rock and ground), lit darker inside the passage. The passage's own carved floor is
what's walked on (it climbs evenly: the field is a plane there).

    "rocks": [{"box": [x0, z0, x1, z1],              world, the reserved cells' outline
               "margin": [w, e, s, n],              how far the rock reaches past each side (into the tiles round it)
               "top": 40, "bottom": -14, "voxel": 2,
               "rough": 3,                           how far the faces wander (yalms)
               "wall": ["quf_wll1", 10], "ground": ["quf_f1", 10],   texture and yalms it spans
               "tunnel": {"path": [[x, z], ...], "floor": [3.6, 20.2], "width": 8, "height": 7}}]

Frames: world (the zone's own, after the layout's shift), y up.
"""
import math

import numpy as np

from .architecture import Mesh, triangles
from .rock import _texture, noise


def _path(points, spacing=1.0):
    """The path resampled every `spacing` yalms: (points (n, 2), arc length at each)."""
    pts = [np.array(points[0], float)]
    for a, b in zip(points, points[1:]):
        a, b = np.array(a, float), np.array(b, float)
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / spacing)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    pts = np.array(pts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    return pts, s


def _nearest(px, pz, pts, s):
    """For points (px, pz): distance across the path (signed), arc length along it."""
    best = np.full(px.shape, 1e9)
    across = np.zeros(px.shape)
    along = np.zeros(px.shape)
    for k in range(len(pts) - 1):
        a, b = pts[k], pts[k + 1]
        d = b - a
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        t = np.clip(((px - a[0]) * d[0] + (pz - a[1]) * d[1]) / (L * L), 0.0, 1.0)
        qx, qz = a[0] + d[0] * t, a[1] + d[1] * t
        dist = np.hypot(px - qx, pz - qz)
        better = dist < best
        side = np.sign(d[0] * (pz - a[1]) - d[1] * (px - a[0]))
        best = np.where(better, dist, best)
        across = np.where(better, dist * side, across)
        along = np.where(better, s[k] + t * L, along)
    return across, along


def _inside_tunnel(x, y, z, tun, pts, s):
    """How far inside the passage (x, y, z) are (negative: in the rock)."""
    across, along = _nearest(x, z, pts, s)
    f0, f1 = tun["floor"]
    floor = f0 + (f1 - f0) * np.clip(along / s[-1], 0.0, 1.0)
    w2, h = tun["width"] / 2.0, tun["height"]
    side = h * 0.6                                       # walls rise straight, then a rounded roof
    roof = floor + side + (h - side) * np.sqrt(np.clip(1.0 - (across / w2) ** 2, 0.0, 1.0))
    # the passage runs on past both ends of its path, so it opens through the faces
    beyond = np.maximum(-along, along - s[-1])
    reach = np.where(beyond > 0, tun.get("extend", 10.0) - beyond, 1e9)
    return np.minimum.reduce([w2 - np.abs(across), y - floor, roof - y, reach])


def build(spec):
    """(mesh, collision triangles (k, 3, 3) FFXI frame, y down) for one rock spec."""
    x0, z0, x1, z1 = spec["box"]
    mw, me, ms, mn = spec.get("margin", [6, 6, 0, 2])
    top, bottom, voxel, rough = spec.get("top", 40.0), spec.get("bottom", -14.0), spec.get("voxel", 2.0), spec.get("rough", 3.0)
    xs = np.arange(x0 - mw - 2 * voxel, x1 + me + 2 * voxel + voxel, voxel)
    ys = np.arange(bottom - 2 * voxel, top + 6 + voxel, voxel)
    zs = np.arange(z0 - ms - rough - 2 * voxel, z1 + mn + rough + 2 * voxel + voxel, voxel)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    wob = lambda f, k: rough * noise(X * f, Y * f, Z * f + k)
    d = np.minimum.reduce([
        (X - (x0 - mw)), ((x1 + me) - X),
        (Z - (z0 - ms) + wob(0.07, 11.0) + 0.5 * wob(0.2, 3.0)),          # the faces wander
        ((z1 + mn) - Z + wob(0.07, 23.0) + 0.5 * wob(0.2, 5.0)),
        (top + 1.2 * noise(X * 0.05, 7.0, Z * 0.05) - Y),
        (Y - bottom)])
    tun = spec.get("tunnel")
    if tun:
        pts, s = _path(tun["path"])
        d = np.minimum(d, -_inside_tunnel(X, Y, Z, tun, pts, s))
    from caves.network import surface_nets
    verts, tris = surface_nets(d, np.array([xs[0], ys[0], zs[0]]), voxel)
    P = verts[tris]
    n = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    steep = n[:, 1] < 0.6

    def pal(tex, yalms):
        return {"texture": tex, "yalms": yalms, "regions": {"strata": [0, 0, 1, 1], "ground": [0, 0, 1, 1]}}
    light = {"sun": [0.45, 0.7, -0.55], "sun_floor": 0.62, "sun_gain": 0.5, "tint": [0.95, 0.12, 0.05], "patch": 0.015}
    mesh = _texture(P[steep], pal(*spec.get("wall", ["quf_wll1", 10.0])), 1.0, light)
    mesh.add(_texture(P[~steep], pal(*spec.get("ground", ["quf_f1", 10.0])), 1.0, light))
    solid = [triangles(mesh)]
    if tun:
        # darker inside the passage, darkest half way through
        for part in mesh.parts.values():
            if not part["pos"]:
                continue
            p = np.array(part["pos"])
            inside = _inside_tunnel(p[:, 0], p[:, 1], p[:, 2], tun, pts, s) > -1.5
            _, along = _nearest(p[:, 0], p[:, 2], pts, s)
            depth = np.clip(np.minimum(along, s[-1] - along) / tun.get("dusk", 14.0), 0.0, 1.0)
            dim = np.where(inside, 1.0 - tun.get("dark", 0.6) * depth, 1.0)
            part["col"] = list(np.array(part["col"]) * dim)
    return mesh, np.concatenate(solid)

"""Our own architecture, in the game's own stone: clean standalone pieces (walls, towers, gatehouses)
built for the 40-yalm grid, finished on every side, textured from another zone's atlas at that
zone's own scale. Nothing of the game's is shipped: the shapes are ours, the texture is copied from
the player's own files at build time, like any borrowed model's.

A palette (kits/architecture/<name>.json) names the atlas and its regions (u0, v0, u1, v1) and how
many yalms the whole texture spans on the source's own models; a face is cut along the region's
tile lines, so its courses line up across faces and pieces.

Frames: pieces are built y up (x along a wall, z across it: -z its front, the outside), and written
in FFXI's frame (y down). Retail winding: a face's normal is -(b-a)x(c-a) in FFXI's frame.
"""
import json
import math
from pathlib import Path

import numpy as np

KITS = Path(__file__).resolve().parent / "kits" / "architecture"


def palette(name):
    """A palette (kits/architecture/<name>.json): the atlas, its regions and its scale."""
    d = json.loads((KITS / f"{name}.json").read_text())
    d["regions"] = {k: tuple(v) for k, v in d["regions"].items()}
    if d.get("banner"):
        d["banner"] = {k: tuple(v) if isinstance(v, list) else v for k, v in d["banner"].items()}
    return d



def _clip(poly, axis, value, keep_greater):
    """Sutherland-Hodgman against one line in 2D (poly: [(s, t, ...)])."""
    out = []
    n = len(poly)
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        ina = (a[axis] >= value) if keep_greater else (a[axis] <= value)
        inb = (b[axis] >= value) if keep_greater else (b[axis] <= value)
        if ina:
            out.append(a)
        if ina != inb:
            f = (value - a[axis]) / (b[axis] - a[axis])
            out.append(tuple(a[i] + f * (b[i] - a[i]) for i in range(len(a))))
    return out


def _ears(poly):
    """Triangles (index triples) of a simple counter-clockwise polygon, concave or not (the face
    over an arch is), by ear clipping. Repeated and collinear points are skipped."""
    pts = []
    for k, p in enumerate(poly):
        if not pts or math.dist(poly[pts[-1]], p) > 1e-6:
            pts.append(k)
    if len(pts) > 1 and math.dist(poly[pts[0]], poly[pts[-1]]) < 1e-6:
        pts.pop()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    out, guard = [], 0
    while len(pts) > 3 and guard < 10000:
        guard += 1
        n = len(pts)
        for k in range(n):
            i, j, l = pts[k - 1], pts[k], pts[(k + 1) % n]
            A, B, C = poly[i], poly[j], poly[l]
            c = cross(A, B, C)
            if abs(c) < 1e-9:                         # collinear: drop the middle point
                pts.pop(k)
                break
            if c < 0:
                continue                              # a reflex corner, not an ear
            if any(cross(A, B, poly[q]) >= -1e-9 and cross(B, C, poly[q]) >= -1e-9 and cross(C, A, poly[q]) >= -1e-9
                   for q in pts if q not in (i, j, l)):
                continue                              # another point inside
            out.append((i, j, l))
            pts.pop(k)
            break
        else:
            break                                     # nothing clips: give up on the rest
    if len(pts) == 3 and abs(cross(*[poly[q] for q in pts])) > 1e-9:
        out.append(tuple(pts))
    return out


class Mesh:
    """Triangles by texture, y up, with UVs and vertex colours."""

    def __init__(self, palette):
        self.pal = palette
        self.parts = {}          # texture short name -> {"pos", "nrm", "uv", "col", "tri"}

    def _part(self, tex):
        return self.parts.setdefault(tex, {"pos": [], "nrm": [], "uv": [], "col": [], "tri": []})

    def face(self, poly, origin, su, sv, region, shade=None, tex=None, anchor=(0.0, 0.0), inset=0.0, tile=None):
        """A flat polygon (2D points in the face's own (s, t) frame, counter-clockwise seen from its
        front) placed at origin + s*su + t*sv, its normal su x sv. Cut on the region's tile lines
        (a tile is the region's size in yalms), each piece mapped into the region. `anchor` shifts
        the tiling (so courses meet across faces); `shade(s, t, y)` gives a vertex brightness."""
        pal = self.pal
        u0, v0, u1, v1 = pal["regions"][region] if isinstance(region, str) else region
        tw, th = tile or ((u1 - u0) * pal["yalms"], (v1 - v0) * pal["yalms"])
        origin, su, sv = np.asarray(origin, float), np.asarray(su, float), np.asarray(sv, float)
        n = np.cross(su, sv)
        n = n / (np.linalg.norm(n) + 1e-12)
        ss = [p[0] for p in poly]
        ts = [p[1] for p in poly]
        part = self._part(tex or pal["texture"])
        for i in range(math.floor((min(ss) + anchor[0]) / tw), math.ceil((max(ss) + anchor[0]) / tw)):
            for j in range(math.floor((min(ts) + anchor[1]) / th), math.ceil((max(ts) + anchor[1]) / th)):
                s0, t0 = i * tw - anchor[0], j * th - anchor[1]
                cell = list(poly)
                for axis, val, keep in ((0, s0, True), (0, s0 + tw, False), (1, t0, True), (1, t0 + th, False)):
                    cell = _clip(cell, axis, val, keep)
                    if len(cell) < 3:
                        break
                if len(cell) < 3:
                    continue
                base = len(part["pos"])
                for s, t in cell:
                    p = origin + s * su + t * sv + n * inset
                    part["pos"].append(p)
                    part["nrm"].append(n)
                    # t runs up the face; the texture's v runs down it
                    part["uv"].append((u0 + (s - s0) / tw * (u1 - u0), v1 - (t - t0) / th * (v1 - v0)))
                    b = shade(s, t, p[1]) if shade else 1.0
                    part["col"].append(b)
                for a, b, c in _ears([(x[0], x[1]) for x in cell]):
                    part["tri"].append((base + a, base + b, base + c))

    def quad(self, a, b, c, d, region, **kw):
        """A flat quad a-b-c-d (counter-clockwise from its front: a bottom left, b bottom right)."""
        a, b, c, d = (np.asarray(x, float) for x in (a, b, c, d))
        su = b - a
        w = np.linalg.norm(su)
        su = su / w
        up = d - a
        sv = up - su * (up @ su)
        h = np.linalg.norm(sv)
        sv = sv / h
        pts = [(float((x - a) @ su), float((x - a) @ sv)) for x in (a, b, c, d)]
        self.face(pts, a, su, sv, region, **kw)

    def box(self, x0, x1, y0, y1, z0, z1, side, top=None, bottom=False, ends=True, shade=None, tex=None):
        """An axis-aligned box: `side` on its four sides (only front and back without `ends`),
        `top` on top. Tiles anchored to the piece's own axes, so courses line up across boxes."""
        kw = {"shade": shade, "tex": tex}
        self.quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), side, anchor=(-x1, y0), **kw)  # front, -z
        self.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), side, anchor=(x0, y0), **kw)   # back, +z
        if ends:
            self.quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), side, anchor=(z0, y0), **kw)
            self.quad((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), side, anchor=(-z1, y0), **kw)
        if top:
            self.quad((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), top, anchor=(x0, -z1), **kw)
        if bottom:                                   # the underside of an overhang
            self.quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1), side, anchor=(x0, z0), **kw)

    def batter(self, x0, x1, z0, z1, y1, out, region, shade=None, sides=(True, True, True, True)):
        """A sloping foot from the ground up to y1: `out` wider at the ground than the rectangle
        (x0..x1, z0..z1) it meets at the top. sides: front (-z), back (+z), left (-x), right (+x)."""
        kw = {"shade": shade}
        X0, X1, Z0, Z1 = x0 - out, x1 + out, z0 - out, z1 + out
        if sides[0]:
            self.quad((X1, 0, Z0), (X0, 0, Z0), (x0, y1, z0), (x1, y1, z0), region, **kw)
        if sides[1]:
            self.quad((X0, 0, Z1), (X1, 0, Z1), (x1, y1, z1), (x0, y1, z1), region, **kw)
        if sides[2]:
            self.quad((X0, 0, Z0), (X0, 0, Z1), (x0, y1, z1), (x0, y1, z0), region, **kw)
        if sides[3]:
            self.quad((X1, 0, Z1), (X1, 0, Z0), (x1, y1, z0), (x1, y1, z1), region, **kw)

    def course(self, x0, x1, z0, z1, y, height=0.6, proud=0.3, region="plinth", shade=None, ends=True):
        """A string course: a thin band standing proud of a face, round a block."""
        self.box(x0 - proud, x1 + proud, y, y + height, z0 - proud, z1 + proud, region, top=region, bottom=True,
                 ends=ends, shade=shade)

    def add(self, other, offset=(0.0, 0.0, 0.0)):
        """Another piece's faces into this one, moved by offset."""
        off = np.asarray(offset, float)
        for tex, q in other.parts.items():
            part = self._part(tex)
            base = len(part["pos"])
            part["pos"] += [np.asarray(p) + off for p in q["pos"]]
            part["nrm"] += list(q["nrm"])
            part["uv"] += list(q["uv"])
            part["col"] += list(q["col"])
            part["tri"] += [(a + base, b + base, c + base) for a, b, c in q["tri"]]

    def arrays(self):
        """{texture: (pos (n,3) y up, nrm, uv, col 0..2, tri)}"""
        out = {}
        for tex, p in self.parts.items():
            if p["tri"]:
                out[tex] = tuple(np.array(p[k], np.float32) for k in ("pos", "nrm", "uv", "col")) + \
                           (np.array(p["tri"], np.int64),)
        return out


def banner(m, cx, top, cz, ax, az, width=3.0, length=12.0, off=0.35):
    """The Safehold's banner hung flat against a face whose outward normal is (ax, az) (horizontal),
    its top at `top`, centred on (cx, cz) on the face: the cloth in two mirrored halves, both
    sides, and a rod over it."""
    b = m.pal.get("banner")
    if not b:
        return
    ux, uz = -az, ax                                   # along the face, left to right from outside
    ox, oz = cx + ax * off, cz + az * off
    w2 = width / 2
    u0, v0, u1, v1 = b["cloth"]
    halves = (((u0, v0, u1, v1), -w2, 0.0), ((u1, v0, u0, v1), 0.0, w2))   # edge to middle, middle to edge
    for (region, s0, s1), front in ((h, True) for h in halves):
        a = (ox + ux * s0, top - length, oz + uz * s0)
        bb = (ox + ux * s1, top - length, oz + uz * s1)
        c = (ox + ux * s1, top, oz + uz * s1)
        d = (ox + ux * s0, top, oz + uz * s0)
        m.quad(a, bb, c, d, region, tex=b["texture"], tile=(w2, length), shade=lambda s, t, y: 0.9)
        # its back, a hair behind, seen from behind: the same half, mirrored
        back = [(p[0] - ax * 0.03, p[1], p[2] - az * 0.03) for p in (bb, a, d, c)]
        m.quad(*back, (region[2], region[1], region[0], region[3]), tex=b["texture"], tile=(w2, length),
               shade=lambda s, t, y: 0.6)
    # the rod: a thin box across the top
    r = 0.18
    ends = [(ox - ux * (w2 + 0.4), oz - uz * (w2 + 0.4)), (ox + ux * (w2 + 0.4), oz + uz * (w2 + 0.4))]
    rx0, rx1 = sorted(e[0] for e in ends)
    rz0, rz1 = sorted(e[1] for e in ends)
    m.box(rx0 - r, rx1 + r, top, top + 2 * r, rz0 - r, rz1 + r, b["rod"], top=b["rod"], bottom=True,
          shade=lambda s, t, y: 0.9, tex=b["texture"])


def _shade(foot=0.55, rise=6.0):
    """Darker toward the foot, as FFXI's baked vertex colours are."""
    return lambda s, t, y: foot + (1.0 - foot) * min(1.0, max(0.0, y / rise))


def _crenels(m, x0, x1, y0, z0, z1, height=3.0, merlon=2.5, gap=1.5, shade=None):
    """A crenellated parapet between x0 and x1 on top of y0: merlons with their caps."""
    length = x1 - x0
    n = max(1, int(round((length + gap) / (merlon + gap))))
    step = (length + gap) / n
    w = step - gap
    for k in range(n):
        a = x0 + k * step
        m.box(a, a + w, y0, y0 + height, z0, z1, "ashlar", top="coping", shade=shade)


def wall(pal, length=40.0, height=16.0, thick=6.0, plinth=3.0, ends=True):
    """A curtain wall: battered plinth, ashlar body, walkway on top between a crenellated parapet
    outside (-z) and a low one inside; finished on both faces and at its ends."""
    m = Mesh(pal)
    sh = _shade()
    x0, x1 = -length / 2, length / 2
    h2, t2 = height - 1.0, thick / 2
    m.batter(x0, x1, -t2, t2, plinth, 1.2, "plinth", shade=sh, sides=(True, True, False, False))
    if ends:
        for x, sgn in ((x0, -1), (x1, 1)):             # the foot's ends: trapezoids
            pts = [(-t2 - 1.2, 0.0), (t2 + 1.2, 0.0), (t2, plinth), (-t2, plinth)]
            if sgn < 0:
                m.face(pts, (x, 0, 0), (0, 0, 1), (0, 1, 0), "plinth", shade=sh)
            else:
                m.face([(-a, b) for a, b in pts], (x, 0, 0), (0, 0, -1), (0, 1, 0), "plinth", shade=sh)
    m.box(x0, x1, plinth, h2, -t2, t2, "ashlar", ends=ends, shade=sh)
    m.box(x0, x1, h2, height, -t2 - 0.4, t2 + 0.4, "coping", top="smooth", bottom=True, ends=ends, shade=sh)
    _crenels(m, x0, x1, height, -t2 - 0.4, -t2 + 0.8, shade=sh)
    m.box(x0, x1, height, height + 1.2, t2 - 0.8, t2 + 0.4, "ashlar", top="coping", ends=ends, shade=sh)
    return m


def tower(pal, size=14.0, height=22.0, plinth=3.0, windows=True, banners=True):
    """A square tower: battered plinth, ashlar faces with slit windows, a band of arched
    loopholes, a crenellated top round a flat roof."""
    m = Mesh(pal)
    sh = _shade()
    s2 = size / 2
    m.batter(-s2, s2, -s2, s2, plinth + 1.0, 1.4, "plinth", shade=sh)
    stage = plinth + 1.0 + (height - 4.0 - plinth - 1.0) * 0.62     # the upper stage steps in
    m.box(-s2, s2, plinth + 1.0, stage, -s2, s2, "ashlar", shade=sh)
    u2 = s2 - 0.5
    m.box(-u2, u2, stage, height - 3.0, -u2, u2, "ashlar", shade=sh)
    m.course(-s2, s2, -s2, s2, stage - 0.6, shade=sh)                  # at the step
    m.course(-s2, s2, -s2, s2, plinth + 1.0 + (stage - plinth - 1.0) * 0.45, shade=sh)
    s2 = u2
    m.box(-s2 - 0.3, s2 + 0.3, height - 3.0, height - 1.0, -s2 - 0.3, s2 + 0.3, "arcade", bottom=True, shade=sh)
    m.box(-s2 - 0.5, s2 + 0.5, height - 1.0, height, -s2 - 0.5, s2 + 0.5, "coping", top="smooth", bottom=True, shade=sh)
    if banners:                                        # on its front, under the loopholes, clear
        banner(m, 0.0, height - 3.4, -(size / 2), 0, -1,   # of the wider lower stage
               length=min(12.0, height - 3.4 - plinth - 2.0))
    for (ax, az) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        _crenels_side(m, s2 + 0.5, height, ax, az, sh)
        if windows:
            for y, r in ((plinth + 4.0, s2 + 0.5), (stage + 1.5, s2)):
                if banners and (ax, az) == (0, -1):
                    continue                           # the banner hangs there
                # a slit on each face, just proud of it
                cx, cz = ax * (r + 0.02), az * (r + 0.02)
                ux, uz = -az, ax                       # along the face, left to right from outside
                a = (cx - ux * 0.8, y, cz - uz * 0.8)
                b = (cx + ux * 0.8, y, cz + uz * 0.8)
                c = (cx + ux * 0.8, y + 2.2, cz + uz * 0.8)
                d = (cx - ux * 0.8, y + 2.2, cz - uz * 0.8)
                m.quad(a, b, c, d, "window", shade=sh)
    return m


def _crenels_side(m, r, y0, ax, az, shade, merlon=2.5, gap=1.5, depth=1.2):
    """Merlons along one side of a square top of half-size r (outer face at r); the sides facing
    x stop short of the corners, which the sides facing z take."""
    lo, hi = (-r + depth, r - depth) if ax else (-r, r)
    length = hi - lo
    n = max(1, int(round((length + gap) / (merlon + gap))))
    step = (length + gap) / n
    w = step - gap
    for k in range(n):
        a = lo + k * step
        if ax:
            x0, x1 = (r - depth, r) if ax > 0 else (-r, -r + depth)
            m.box(x0, x1, y0, y0 + 3.0, a, a + w, "ashlar", top="coping", shade=shade)
        else:
            z0, z1 = (r - depth, r) if az > 0 else (-r, -r + depth)
            m.box(a, a + w, y0, y0 + 3.0, z0, z1, "ashlar", top="coping", shade=shade)


def gatehouse(pal, length=40.0, height=16.0, thick=6.0, plinth=3.0, opening=8.0, spring=8.0, banners=True,
              closed=False):
    """A wall with a round-headed gateway through it (paved floor, a vault), two towers flanking it
    on the outside (-z). closed: its leaves shut across the arch (the city beyond is another zone)."""
    m = Mesh(pal)
    sh = _shade()
    x0, x1, t2, o2 = -length / 2, length / 2, thick / 2, opening / 2
    h2 = height - 1.0
    # the wall either side of the gateway; the ends of these boxes are the gateway's reveals
    for a, b in ((x0, -o2), (o2, x1)):
        m.batter(a, b, -t2, t2, plinth, 1.2, "plinth", shade=sh, sides=(True, True, False, False))
        for x, sgn in ((a, -1), (b, 1)):
            pts = [(-t2 - 1.2, 0.0), (t2 + 1.2, 0.0), (t2, plinth), (-t2, plinth)]
            if sgn < 0:
                m.face(pts, (x, 0, 0), (0, 0, 1), (0, 1, 0), "plinth", shade=sh)
            else:
                m.face([(-p, q) for p, q in pts], (x, 0, 0), (0, 0, -1), (0, 1, 0), "plinth", shade=sh)
        m.box(a, b, plinth, h2, -t2, t2, "ashlar", shade=sh)
    # over the gateway: the face from the arch up to the coping, front and back
    k = 12
    arch = [(o2 * math.cos(math.pi * i / k), spring + o2 * math.sin(math.pi * i / k)) for i in range(k + 1)]
    over = arch + [(-o2, h2), (o2, h2)]                    # clockwise in (x, y)
    m.face([(-x, y) for x, y in over], (0, 0, -t2), (-1, 0, 0), (0, 1, 0), "ashlar", shade=sh)   # front: s = -x
    m.face([(x, y) for x, y in reversed(over)], (0, 0, t2), (1, 0, 0), (0, 1, 0), "ashlar", shade=sh)
    # the vault, seen from the passage
    for i in range(k):
        (xa, ya), (xb, yb) = arch[i], arch[i + 1]
        m.quad((xb, yb, -t2), (xa, ya, -t2), (xa, ya, t2), (xb, yb, t2), "narrow", shade=sh)
    m.quad((-o2, 0.02, t2 + 0.6), (o2, 0.02, t2 + 0.6), (o2, 0.02, -t2 - 0.6), (-o2, 0.02, -t2 - 0.6), "smooth", shade=sh)
    if closed and pal.get("door"):
        # two leaves meeting in the middle, each the arch's own shape cut down the centre, shaded
        # apart so they read as two, planks both sides, two timber bands across the front of each
        dr = pal["door"]
        opening_shape = [(o2, 0.0)] + arch + [(-o2, 0.0)]            # counter-clockwise in (x, y)
        size = (opening / 2, spring + o2)
        seam = 0.0                                                    # the leaves meet: no gap between them
        for side, keep_hi in ((1, True), (-1, False)):
            leaf = _clip(opening_shape, 0, seam * side, keep_hi)
            shade = 0.72 if side > 0 else 0.66                        # the two leaves catch the light apart
            for z, front in ((-0.15, True), (0.15, False)):
                # seen from the front (-z) s runs toward -x, which turns the outline over
                pts = [(-x, y) for x, y in reversed(leaf)] if front else [(x, y) for x, y in leaf]
                su = (-1, 0, 0) if front else (1, 0, 0)
                m.face(pts, (0, 0, z), su, (0, 1, 0), tuple(dr["region"]), tex=dr["texture"], tile=size,
                       anchor=(o2, 0.0), shade=lambda s_, t_, y, sh_=shade: sh_)
            xa, xb = sorted((seam * side, (o2 - 0.3) * side))
            for yb in (1.6, spring - 1.2):
                m.box(xa, xb, yb, yb + 0.5, -0.45, -0.15, tuple(dr["region"]), top=tuple(dr["region"]), bottom=True,
                      shade=lambda s_, t_, y: 0.45, tex=dr["texture"])
    # coping, walkway and parapets over the whole length
    m.box(x0, x1, h2, height, -t2 - 0.4, t2 + 0.4, "coping", top="smooth", bottom=True, shade=sh)
    _crenels(m, x0, x1, height, -t2 - 0.4, -t2 + 0.8, shade=sh)
    m.box(x0, x1, height, height + 1.2, t2 - 0.8, t2 + 0.4, "ashlar", top="coping", shade=sh)
    # a band of loopholes over the gateway, between the towers (as San d'Oria's own gate has)
    m.box(-o2 - 1.0, o2 + 1.0, h2 - 3.0, h2 - 1.0, -t2 - 0.3, -t2, "arcade", bottom=True, ends=True, shade=sh)
    # two towers flanking the gateway, standing out in front of the wall
    for cx in (-(o2 + 6.0), o2 + 6.0):
        m.add(tower(pal, size=10.0, height=height + 5.0, windows=False, banners=banners), (cx, 0.0, -t2 - 3.0))
    return m


def _timber(m, x0, x1, y0, y1, z0, z1, side, top, tile, shade):
    """A box of wood: grain up its sides (`tile` yalms of the region across and up), `top` on top."""
    kw = {"shade": shade, "tile": tile}
    m.quad((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0), side, **kw)
    m.quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1), side, **kw)
    m.quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0), side, **kw)
    m.quad((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1), side, **kw)
    m.quad((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0), top, shade=shade, tile=(x1 - x0, z1 - z0))


def jetty(pal, length=66.0, width=4.4, deck=3.0, foot=-6.0, bay=5.0, cribs=(15.0, 33.0, 51.0), water=1.8):
    """A timber jetty running out along +z from its land end at z = 0: boards along its length (staggered
    joints), two stringers on capped post pairs every `bay`, stone-filled cribs where it crosses deep water,
    a stone abutment on the shore and tall mooring posts at the far end. y up, deck top at `deck`; the posts
    go down to `foot` (under the water: the sea has no bed here)."""
    m = Mesh(pal)
    rock = pal["rock"]
    wet = lambda s, t, y: 0.45 + 0.55 * min(1.0, max(0.0, (y - water + 0.3) / 1.5))   # darker at the waterline
    dry = lambda s, t, y: 0.85
    w2, board, gap, thick = width / 2, 0.8, 0.07, 0.22
    n = int(round(width / board))
    board = (width - gap * (n - 1)) / n
    u0, v0, u1, v1 = pal["regions"]["board"]
    du = u1 - u0
    for k in range(n):                                  # the boards, each its own strip of grain
        x0 = -w2 + k * (board + gap)
        region = (u0 + (k % 8) * du, v0, u0 + (k % 8 + 1) * du, v1) if u0 + (k % 8 + 1) * du <= 1.0 else (u0, v0, u1, v1)
        z = -((k * 2.3) % 6.0)                          # staggered joints
        while z < length:
            za, zb = max(0.0, z), min(length, z + 6.0)
            if zb - za > 0.3:
                _timber(m, x0, x0 + board, deck - thick, deck, za + 0.02, zb - 0.02, region, region, (board, 6.0), dry)
            z += 6.0
    sy = deck - thick
    for x in (-w2 + 0.6, w2 - 0.6):                     # stringers under the boards
        _timber(m, x - 0.25, x + 0.25, sy - 0.5, sy, 0.0, length, "post", "post", (0.5, 4.0), dry)
    z = bay
    while z < length - 0.5:
        _timber(m, -w2 - 0.2, w2 + 0.2, sy - 1.0, sy - 0.5, z - 0.25, z + 0.25, "post", "post", (0.5, 4.0), dry)
        for x in (-w2 + 0.1, w2 - 0.1):
            _timber(m, x - 0.25, x + 0.25, foot, sy - 1.0, z - 0.25, z + 0.25, "post", "end", (0.5, 4.0), wet)
        z += bay
    for zc in cribs:                                    # timber boxes filled with stone, under the deck
        if zc >= length - 3:
            continue
        x0, x1, z0, z1, top = -w2 - 0.6, w2 + 0.6, zc - 2.4, zc + 2.4, water + 0.5
        m.box(x0, x1, foot, top, z0, z1, rock["region"], top=rock["region"], shade=wet, tex=rock["texture"])
        for y in (top - 0.35, water - 0.4):             # log courses round it
            _timber(m, x0 - 0.15, x1 + 0.15, y, y + 0.35, z0 - 0.15, z0 + 0.15, "post", "post", (0.5, 4.0), wet)
            _timber(m, x0 - 0.15, x1 + 0.15, y, y + 0.35, z1 - 0.15, z1 + 0.15, "post", "post", (0.5, 4.0), wet)
            _timber(m, x0 - 0.15, x0 + 0.15, y, y + 0.35, z0, z1, "post", "post", (0.5, 4.0), wet)
            _timber(m, x1 - 0.15, x1 + 0.15, y, y + 0.35, z0, z1, "post", "post", (0.5, 4.0), wet)
    m.box(-w2 - 0.8, w2 + 0.8, foot, sy - 0.02, -2.0, 1.5, rock["region"], top=rock["region"], shade=wet,
          tex=rock["texture"])                                # the shore end on its stone abutment
    for x in (-w2 + 0.1, w2 - 0.1):                     # mooring posts at the far end, standing proud
        _timber(m, x - 0.3, x + 0.3, foot, deck + 1.3, length - 0.6, length, "post", "end", (0.6, 4.0), wet)
    return m


def jetty_collision(length=66.0, width=4.4, deck=3.0):
    """What you walk on: the deck as one slab (the boards' gaps would catch feet), y up."""
    m = Mesh({"regions": {"x": (0, 0, 1, 1)}, "yalms": 1000.0, "texture": "none"})
    m.box(-width / 2, width / 2, deck - 0.6, deck, 0.0, length, "x", top="x", bottom=True)
    return m


PIECES = {"wall": wall, "tower": tower, "gatehouse": gatehouse, "jetty": jetty}


def build(name, palette="ronfaure", **kw):
    return PIECES[name](globals()["palette"](palette) if isinstance(palette, str) else palette, **kw)


def triangles(mesh):
    """All its triangles (k, 3, 3), FFXI frame, local."""
    out = []
    for pos, nrm, uv, col, tri in mesh.arrays().values():
        out.append((pos * np.array([1, -1, 1], np.float32))[tri])
    return np.concatenate(out)

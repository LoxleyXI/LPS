"""The layout: heights on the tile corners, before any tile is chosen.

Corners are floors (a ground height) or walls (nan: any cliff top, settled later by the tiles).
Basins sit at the kit's ground levels and are joined by passes, a tree over the basins, each an L
two corners wide, climbing one `step` per corner on a slope in the middle of its longer leg. The
optional coast runs along the south edge as short bays between rocky headlands (a long straight
beach looks wrong and outruns the source's sea effects); the land behind the bays runs on, so the
beaches connect. The first basin (the shore, or the southmost basin) is the way in.

_tidy() then makes the corners something a tile kit can lay: no diagonal cells, no single cliff
corner between two floor levels, no step over `step` between neighbouring floors, a wall round the
edge of the map.
"""
import math
import random

import numpy as np


class Plan:
    def __init__(self, W, D, cfg):
        self.W, self.D, self.cfg = W, D, cfg
        self.coast = cfg.get("coast")
        self.levels = [float(x) for x in cfg["levels"]]
        self.step = float(cfg["step"])                          # what a pass climbs in one tile
        self.max_rise = float(cfg.get("max_rise", cfg["step"]))  # the most between neighbouring floors
        if self.coast:
            self.sea_rows = self.coast.get("sea_rows", 3)       # corner rows 0.. sea, then the beach row
            self.beach_row = self.sea_rows
            self.land0 = self.beach_row + 1
        else:
            self.sea_rows = self.beach_row = 0
            self.land0 = 1


def layout(W, D, cfg, rng, basins=5):
    """(ground (W+1, D+1) with nan at walls, basins [(i, j, level)], bays [(i0, i1)])."""
    P = Plan(W, D, cfg)
    ground = np.full((W + 1, D + 1), np.nan)
    bays = []
    lo_i, hi_i = 2, W - 2
    if P.coast:
        sea, beach = float(P.coast["sea"]), float(P.coast["beach"])
        widths = P.coast.get("bay_widths", [4, 5, 6])
        ground[:, 0] = sea
        ground[lo_i:hi_i + 1, P.land0:P.land0 + 3] = P.levels[0]      # the shore: the way in
        a = lo_i
        while a < hi_i:
            b = min(a + rng.choice(widths), hi_i)
            if hi_i - b < min(widths):
                b = hi_i                                          # no stub of a bay at the end
            ground[a:b + 1, 1:P.sea_rows] = sea
            ground[a:b + 1, P.beach_row] = beach
            bays.append((a, b))
            a = b + 3                                             # two corners of headland between
        centres = [(W / 2, P.land0 + 1, P.levels[0])]
    else:
        centres = []
    tries = 0
    ii, jj = np.meshgrid(np.arange(W + 1), np.arange(D + 1), indexing="ij")
    while len(centres) < basins + (1 if P.coast else 0) and tries < 500:
        tries += 1
        r = rng.uniform(1.5, 2.8)
        lo_j = P.land0 + (4 if P.coast else 1) + r
        if not centres and not P.coast:
            ci, cj, lv = W / 2 + rng.uniform(-2, 2), lo_j, P.levels[0]     # the way in: low, south
        else:
            ci, cj, lv = rng.uniform(2 + r, W - 2 - r), rng.uniform(lo_j, D - 2 - r), rng.choice(P.levels)
        if cj > D - 2 - r or any(math.hypot(ci - a, cj - b) < r + 3.5 for a, b, _ in centres):
            continue
        centres.append((ci, cj, lv))
        blob = np.hypot((ii - ci) / r, (jj - cj) / (r * rng.uniform(0.7, 1.3))) <= 1.0
        ground[blob & np.isnan(ground) & (jj >= P.land0)] = lv
    order, rest, edges = [0], list(range(1, len(centres))), []
    while rest:
        a, b = min(((a, b) for a in order for b in rest),
                   key=lambda ab: math.hypot(centres[ab[0]][0] - centres[ab[1]][0], centres[ab[0]][1] - centres[ab[1]][1]))
        edges.append((a, b))
        order.append(b)
        rest.remove(b)
    routes = [_pass(ground, centres[a], centres[b], rng, P) for a, b in edges]
    _tidy(ground, P)
    return ground, centres, bays, _networks(ground, cfg, rng, P, routes)


def _networks(ground, cfg, rng, P, routes):
    """The kit's roads (along the passes, graded) and streams, as {name: {cell: sides}}."""
    nets = {}
    specs = cfg.get("networks") or {}
    kind = {name: spec.get("kind", "road" if name == "road" else "stream") for name, spec in specs.items()}
    for name in specs:
        if kind[name] == "road":
            nets[name] = _links(routes)                       # along the passes, basin to basin
            for _ in range(10):                               # graded: flat, or climbing straight ahead
                if not _grade(ground, nets[name], P.step):
                    break
                _tidy(ground, P)
    for name, spec in specs.items():
        if kind[name] == "stream":
            taken = set().union(*[set(v) for v in nets.values()]) if nets else set()
            nets[name] = _stream(ground, rng, spec, taken)
    return nets


def _noise(shape, cells, rng):
    """Smooth value noise in 0..1 over `shape`, varying over about `cells` corners."""
    n0 = max(2, int(math.ceil(shape[0] / cells)) + 2)
    n1 = max(2, int(math.ceil(shape[1] / cells)) + 2)
    small = np.array([[rng.random() for _ in range(n1)] for _ in range(n0)])
    x = np.arange(shape[0]) / cells
    y = np.arange(shape[1]) / cells
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)       # smooth between the lattice points
    a = small[np.ix_(x0, y0)]
    b = small[np.ix_(x0 + 1, y0)]
    c = small[np.ix_(x0, y0 + 1)]
    d = small[np.ix_(x0 + 1, y0 + 1)]
    fx, fy = fx[:, None], fy[None, :]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _shift_any(mask, r=1):
    """Corners with any True within `r` corners (a square), the mask grown."""
    out = mask.copy()
    W, D = mask.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di or dj:
                out[max(di, 0):W + min(di, 0), max(dj, 0):D + min(dj, 0)] |= \
                    mask[max(-di, 0):W + min(-di, 0), max(-dj, 0):D + min(-dj, 0)]
    return out


def _open(mask, r=1):
    """Morphological opening: what's left when every strip narrower than 2r+1 corners goes."""
    return _shift_any(~_shift_any(~mask, r), r)


def fields(W, D, cfg, rng, basins=4, spec=None):
    """Open country (see _fields()), drawn again a little smaller (from the same draw) while the
    finished land, banks and roads included, takes more than its "tiles"."""
    spec = dict(spec or {})
    tiles = int(spec.get("tiles", 110))
    base = rng.random()
    aim, best = tiles, None
    for _ in range(8):
        out = _fields(W, D, cfg, random.Random(base), basins, {**spec, "tiles": aim})
        g = out[0]
        used = sum(1 for i in range(W) for j in range(D) if (~np.isnan(g[i:i + 2, j:j + 2])).any())
        if used <= tiles and (best is None or used > best[0]):
            best = (used, out)
        if tiles * 0.92 <= used <= tiles:
            break
        aim = max(20, aim + (tiles * 0.96 - used))            # (some draws lose more to rocks and banks)
    return best[1] if best else out


def _fields(W, D, cfg, rng, basins=4, spec=None):
    """Open country, the way the original field zones lie: one wide stretch of land with a ragged
    edge (bays, capes, lobes), rock outcrops standing in it, rising in places to a field a step
    higher with banks between. The land grows from the way in, best ground first (the ground the
    noise favours, kept off the map's edge), until it takes `tiles` tiles; so it is all one piece
    and needs no corridors to join it. Its basins are then spread over it, each with a level, and
    the land takes the level of the basin nearest; where levels meet the kit lays a bank.

    spec (the zone file's "fields" block, all optional):
        tiles    the most tiles the land may take (each cell with land at a corner, banks and
                 walls included): its size (110)
        ragged   how much the noise shapes the land: 0 a round blob, 2 wild (1.0)
        rocks    outcrops standing in the fields (4), each `rock` corners across ([2, 3])
        flat     chance a basin lies at its neighbour's level rather than a step up or down (0.6)
    Returns as layout(): (ground, basins [(i, j, level)] (the first is the way in), [], nets)."""
    spec = spec or {}
    P = Plan(W, D, cfg)
    tiles = int(spec.get("tiles", 110))
    ragged = float(spec.get("ragged", 1.0))
    shape = (W + 1, D + 1)
    ii, jj = np.meshgrid(np.arange(W + 1), np.arange(D + 1), indexing="ij")
    inner = (ii >= 2) & (ii <= W - 2) & (jj >= 2) & (jj <= D - 2)
    rough = 0.6 * _noise(shape, 5, rng) + 0.4 * _noise(shape, 2.5, rng)
    rough = (rough - rough.min()) / (np.ptp(rough) + 1e-9)
    cx, cz = W / 2 + rng.uniform(-1.5, 1.5), D / 2 + rng.uniform(-1, 1)
    centre = np.hypot((ii - cx) / (W / 2), (jj - cz) / (D / 2))     # 0 at the middle, 1 at the edge
    start = (int(round(cx)), 3)
    near_start = np.exp(-((ii - start[0]) ** 2 + (jj - start[1]) ** 2) / 8.0)
    score = ragged * rough - 0.9 * centre + 0.8 * near_start
    # the land: the ground scoring over a line, the piece of it holding the way in, cleaned of
    # one-corner strips; the line set (by halving) so the land takes `tiles` tiles, less a margin
    # for what the banks, rocks and roads add

    def flood(m):
        out = np.zeros(shape, bool)
        if not m[start]:
            return out
        out[start] = True
        stack = [start]
        while stack:
            a, b = stack.pop()
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (a + da, b + db)
                if 0 <= n[0] <= W and 0 <= n[1] <= D and m[n] and not out[n]:
                    out[n] = True
                    stack.append(n)
        return out

    def grown(t):
        m = flood(_open(flood((score > t) & inner), 1))
        c = np.zeros((W, D), bool)
        for a, b in np.argwhere(m):
            c[max(a - 1, 0):a + 1, max(b - 1, 0):b + 1] = True
        return m, int(c.sum())

    lo_t, hi_t = float(score.min()), float(score.max())
    land = grown(hi_t - 1e-6)[0]
    for _ in range(28):
        mid = (lo_t + hi_t) / 2
        m, n = grown(mid)
        if n <= tiles * 0.9:
            land, hi_t = m, mid
        else:
            lo_t = mid
    # the basins: the way in, then the land's farthest corners from those already chosen
    pts = np.argwhere(land)
    centres = [(float(start[0]), float(start[1]), P.levels[0])]
    for _ in range(basins - 1):
        d = np.min([np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) for c in centres], 0)
        k = int(np.argmax(d * (0.7 + 0.6 * np.array([rng.random() for _ in range(len(pts))]))))
        ci, cj = pts[k]
        near = min(centres, key=lambda c: math.hypot(ci - c[0], cj - c[1]))
        lv = near[2]
        if rng.random() >= float(spec.get("flat", 0.6)):
            at = P.levels.index(near[2])
            lv = P.levels[max(0, min(len(P.levels) - 1, at + rng.choice([-1, 1])))]
        centres.append((float(ci), float(cj), lv))
    dist = np.stack([np.hypot(ii - c[0], jj - c[1]) + 1.5 * (rough - 0.5) for c in centres])
    levels = np.array([c[2] for c in centres])[dist.argmin(0)]
    # rock outcrops in the open: well inside the land, off the way in
    deep = land.copy()
    for _ in range(2):
        deep &= ~_shift_any(~deep, 1)
    spots = [tuple(x) for x in np.argwhere(deep)]
    lo_r, hi_r = spec.get("rock", [2, 3])
    for _ in range(int(spec.get("rocks", 4))):
        spots = [q for q in spots if math.hypot(q[0] - start[0], q[1] - start[1]) > 4]
        if not spots:
            break
        ci, cj = spots[rng.randrange(len(spots))]
        r = rng.uniform(lo_r, hi_r) / 2
        land &= ~(np.hypot(ii - ci, (jj - cj) * rng.uniform(0.7, 1.3)) <= r)
        spots = [q for q in spots if math.hypot(q[0] - ci, q[1] - cj) > r + 3]
    ground = np.full(shape, np.nan)
    ground[land] = levels[land]
    _tidy(ground, P)
    _reach_only(ground, centres[0], P)
    _tidy(ground, P)
    # the roads: basin to basin over open ground (a tree from the way in), as layout()'s passes
    routes = []
    for a, b in _tree(centres):
        route = _walk(ground, centres[a], centres[b])
        if route:
            routes.append(route)
    return ground, centres, [], _networks(ground, cfg, rng, P, routes)


def _tree(centres):
    order, rest, edges = [0], list(range(1, len(centres))), []
    while rest:
        a, b = min(((a, b) for a in order for b in rest),
                   key=lambda ab: math.hypot(centres[ab[0]][0] - centres[ab[1]][0], centres[ab[0]][1] - centres[ab[1]][1]))
        edges.append((a, b))
        order.append(b)
        rest.remove(b)
    return edges


def _walk(ground, A, B):
    """The shortest run of cells from basin A to basin B through cells open at all four corners."""
    from collections import deque
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    ok = lambda c: 0 <= c[0] < W and 0 <= c[1] < D and not np.isnan(ground[c[0]:c[0] + 2, c[1]:c[1] + 2]).any()
    a = (int(round(A[0])), int(round(A[1])))
    b = (int(round(B[0])), int(round(B[1])))
    a = min((c for c in [(a[0] + x, a[1] + y) for x in (-1, 0, 1) for y in (-1, 0, 1)] if ok(c)),
            key=lambda c: abs(c[0] - a[0]) + abs(c[1] - a[1]), default=None)
    b = min((c for c in [(b[0] + x, b[1] + y) for x in (-1, 0, 1) for y in (-1, 0, 1)] if ok(c)),
            key=lambda c: abs(c[0] - b[0]) + abs(c[1] - b[1]), default=None)
    if a is None or b is None:
        return None
    prev, q = {a: None}, deque([a])
    while q:
        c = q.popleft()
        if c == b:
            break
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n not in prev and ok(n):
                prev[n] = c
                q.append(n)
    if b not in prev:
        return None
    route, c = [], b
    while c is not None:
        route.append(c)
        c = prev[c]
    return route[::-1]


def _reach_only(ground, start, P):
    """Ground that can't be walked to from `start` (a field the passes missed) goes back to wall."""
    W, D = ground.shape
    a, b = int(round(start[0])), int(round(start[1]))
    if np.isnan(ground[a, b]):
        return
    seen = np.zeros(ground.shape, bool)
    seen[a, b] = True
    stack = [(a, b)]
    while stack:
        i, j = stack.pop()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x, y = i + di, j + dj
            if 0 <= x < W and 0 <= y < D and not seen[x, y] and not np.isnan(ground[x, y]) \
                    and abs(ground[x, y] - ground[i, j]) <= P.max_rise:
                seen[x, y] = True
                stack.append((x, y))
    ground[~seen] = np.nan


def _grade(ground, links, step):
    """Road cells the kit can lay: flat, or a straight slope climbing in the road's own direction.
    Anything else is dug down level with its lowest corner. Returns whether anything changed."""
    changed = False
    for (i, j), sides in links.items():
        idx = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)]           # NW NE SE SW
        c = [ground[x] for x in idx]
        if np.isnan(c).any():
            lv = np.nanmin(c) if not np.isnan(c).all() else None
        else:
            nw, ne, se, sw = c
            if max(c) == min(c):
                continue
            up_ns = nw == ne and sw == se and abs(nw - sw) == step and sides == frozenset("ns")
            up_ew = nw == sw and ne == se and abs(nw - ne) == step and sides == frozenset("ew")
            if up_ns or up_ew:
                continue
            lv = min(c)
        if lv is None:
            continue
        for x in idx:
            ground[x] = lv
        changed = True
    return changed


def _span(a, b):
    return range(a, b + 1) if b >= a else range(a, b - 1, -1)


def _pass(ground, A, B, rng, P):
    (ai, aj, la), (bi, bj, lb) = A, B
    ai, aj, bi, bj = int(round(ai)), int(round(aj)), int(round(bi)), int(round(bj))
    if rng.random() < 0.5:
        leg1, leg2 = [(ai, j) for j in _span(aj, bj)], [(i, bj) for i in _span(ai, bi)]
    else:
        leg1, leg2 = [(i, aj) for i in _span(ai, bi)], [(bi, j) for j in _span(aj, bj)]
    pts = leg1 + leg2
    steps = int(round((lb - la) / P.step))
    # a slope at the turn needs pieces kits don't have: it goes in the middle of the longer leg
    mid = len(leg1) // 2 if len(leg1) >= len(leg2) else len(leg1) + len(leg2) // 2
    for k, (i, j) in enumerate(pts):
        s = min(max(k - (mid - abs(steps) // 2), 0), abs(steps))
        lv = la + math.copysign(P.step * s, steps) if steps else la
        for di in (0, 1):
            for dj in (0, 1):
                x, y = i + di, j + dj
                if 0 <= x < ground.shape[0] and P.land0 <= y < ground.shape[1] and np.isnan(ground[x, y]):
                    ground[x, y] = lv
    # the cells down the middle of the pass (each point's corners are all open): a road's route
    route = []
    for p in pts:
        if not route or route[-1] != p:
            route.append(p)
    return route


SIDE = {(0, 1): "n", (1, 0): "e", (0, -1): "s", (-1, 0): "w"}


def _links(routes):
    """{cell: sides the network leaves by} for a set of routes (lists of neighbouring cells)."""
    out = {}
    for route in routes:
        for a, b in zip(route, route[1:]):
            d = (b[0] - a[0], b[1] - a[1])
            if d not in SIDE:
                continue
            out.setdefault(a, set()).add(SIDE[d])
            out.setdefault(b, set()).add(SIDE[(-d[0], -d[1])])
    return {k: frozenset(v) for k, v in out.items()}


def _stream(ground, rng, spec, taken):
    """A short stream across level ground: a walk of `length` cells through cells whose corners
    are all open and level, springing up at one end and sinking away at the other."""
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    lo, hi = spec.get("length", [3, 6])

    def flat(i, j):
        """Level itself, with open ground all round (its banks can slope, but not be cliffs)."""
        if not (1 <= i < W - 1 and 1 <= j < D - 1) or (i, j) in taken:
            return False
        c = ground[i:i + 2, j:j + 2]
        return not np.isnan(ground[i - 1:i + 3, j - 1:j + 3]).any() and c.max() == c.min()

    cells = [(i, j) for i in range(W) for j in range(D) if flat(i, j)]
    for _ in range(200):
        if not cells:
            break
        route = [rng.choice(cells)]
        want = rng.randint(lo, hi)
        heading = rng.choice(list(SIDE))
        while len(route) < want:
            if rng.random() < 0.35:
                heading = rng.choice([d for d in SIDE if d != (-heading[0], -heading[1])])
            nxt = (route[-1][0] + heading[0], route[-1][1] + heading[1])
            if not flat(*nxt) or nxt in route or any(abs(nxt[0] - r[0]) + abs(nxt[1] - r[1]) == 1 for r in route[:-1]):
                break
            route.append(nxt)
        if len(route) >= lo:
            return _links([route])
    return {}


def _tidy(ground, P):
    W, D = ground.shape
    land = P.land0
    ground[0, land + 3 if P.coast else 0:] = np.nan
    ground[-1, land + 3 if P.coast else 0:] = np.nan
    ground[:, -1] = np.nan
    if not P.coast:
        ground[:, 0] = np.nan
    low = float(P.coast["beach"]) if P.coast else -1e9          # coast rows don't count as land floors
    for _ in range(20):
        changed = False
        for i in range(W - 1):
            for j in range(land, D - 1):
                c = [ground[i, j + 1], ground[i + 1, j + 1], ground[i + 1, j], ground[i, j]]   # NW NE SE SW
                open_ = [not np.isnan(x) for x in c]
                if open_ in ([True, False, True, False], [False, True, False, True]):
                    lv = np.nanmin(c)                             # a diagonal cell: open it up
                    if not open_[1]:
                        ground[i + 1, j + 1] = lv; ground[i, j] = lv
                    else:
                        ground[i, j + 1] = lv; ground[i + 1, j] = lv
                    changed = True
        for i in range(1, W - 1):
            for j in range(land + 1, D - 1):
                if not np.isnan(ground[i, j]):
                    continue
                around = {ground[x, y] for x in (i - 1, i, i + 1) for y in (j - 1, j, j + 1)
                          if not np.isnan(ground[x, y]) and ground[x, y] > low}
                if len(around) > 1:
                    # a cliff corner between floors at two levels would need two heights at once:
                    # the wall gives way and the floors meet in a slope bank
                    ground[i, j] = min(around)
                    changed = True
        for i in range(W - 1):
            for j in range(land, D - 1):
                idx = [(i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)]
                c = [ground[x] for x in idx]
                walls = [k for k in range(4) if np.isnan(c[k])]
                lv = {x for x in c if not np.isnan(x)}
                if len(walls) == 1 and len(lv) > 1:
                    # a slope bank ending in a single cliff corner: kits have no such piece, so the
                    # bank carries on, the corner taking the level that makes a straight slope
                    k = walls[0]
                    side = c[(k + 1) % 4] if c[(k + 1) % 4] != c[(k + 2) % 4] else c[(k + 3) % 4]
                    ground[idx[k]] = side
                    changed = True
        for i in range(W):
            for j in range(land, D):
                g = ground[i, j]
                if np.isnan(g):
                    continue
                for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
                    x, y = i + di, j + dj
                    if 0 <= x < W and land <= y < D and not np.isnan(ground[x, y]) and abs(ground[x, y] - g) > P.max_rise:
                        ground[x, y] = g + math.copysign(P.max_rise, ground[x, y] - g)
                        changed = True
        if not changed:
            break


def wall_floor(ground):
    """For each wall corner, the lowest floor beside it (straight neighbours, else diagonal): the
    tiles' cliff top there must stand at least the kit's wall_rise above it."""
    W, D = ground.shape
    out = np.full(ground.shape, np.nan)
    for i in range(W):
        for j in range(D):
            if not np.isnan(ground[i, j]):
                continue
            near = [ground[x, y] for x, y in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1))
                    if 0 <= x < W and 0 <= y < D and not np.isnan(ground[x, y])]
            if not near:
                near = [ground[x, y] for x in (i - 1, i + 1) for y in (j - 1, j + 1)
                        if 0 <= x < W and 0 <= y < D and not np.isnan(ground[x, y])]
            if near:
                out[i, j] = min(near)
    return out


def terraces(W, D, cfg, rng, regions=6, min_size=3, max_size=7):
    """Mesas, the way Misareaux is built: the base level fills the play area and runs all along the
    outer wall (a terrace edge that met the wall would need a cliff whose foot steps partway along
    it, which such kits don't have); raised blocks stand on it, one tile clear of the wall and of
    the way in, a tile of ground between them (a mesa is one ring of one family: some kits can't
    mix cliff and slope along an edge, or turn a cliff through an inside corner).

    Built in tiers for height: first the kit's plateaus (big blocks, always walkable hills), then
    smaller mesas on the low ground or standing on a hill a tier up. Each is a hill (its ring in the
    kit's slope families) or a rock (cliffs all round, scenery); nothing needed stands on a rock.
    Returns (ground, seeds [(i, j, level)] (the first is the way in), slopes {cell: True})."""
    P = Plan(W, D, cfg)
    sl = cfg.get("slopes") or {}
    ground = np.full((W + 1, D + 1), np.nan)
    ground[1:W, 1:D] = P.levels[0]
    ei, ej = W // 2, 2                                       # the way in
    isle = (cfg.get("coast") or {}).get("island")
    crossings_pre = []
    if isle:
        # an island: open sea all round, sea cliffs for a shore, no walls (a sea cliff can't meet a
        # wall: nothing joins a wall corner to open sea)
        from .kit import HOLE
        ring = int(isle.get("sea", 2)) if isinstance(isle, dict) else 2
        ground[:, :] = HOLE
        ground[ring:W - ring + 1, ring:D - ring + 1] = P.levels[0]
        ei, ej = W // 2, ring + 3
        _inlets(ground, ring, (ei, ej), isle if isinstance(isle, dict) else {}, rng)
        if isinstance(isle, dict) and isle.get("channels"):
            _channels(ground, ring, (ei, ej), isle, rng, crossings_pre, P.levels[0])
    seeds = [(ei, ej, P.levels[0])]
    blocks = []                                              # (i0, j0, i1, j1, level, walkable, form)
    edge = (ring if isle else 0) + 2                         # blocks stay a tile clear of wall or shore
    river = cfg.get("river")
    crossings = list(crossings_pre)
    if river:                                                # the river first: the mesas keep clear of it
        path = _carve_river(ground, blocks, (ei, ej), P, river, rng) or []
        straight = [(a, b, c) for a, b, c in zip(path, path[1:], path[2:])
                    if (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1])]
        rng.shuffle(straight)
        for a, b, c in straight[:river.get("bridges", 2)]:     # a footbridge across a straight reach
            run = (b[0] - a[0], b[1] - a[1])
            across = (run[1], run[0])
            crossings.append({"kind": "river", "at": (float(b[0]), float(b[1])), "dir": across,
                              "level": P.levels[0],
                              "join": [(b[0] - across[0], b[1] - across[1]), (b[0] + across[0], b[1] + across[1])]})
    g0 = np.nan_to_num(ground, nan=0)
    wet = (g0 < P.levels[0]) & (g0 > -500)                   # the river (not the open sea round an island)
    sea = g0 <= -500
    bridged = (cfg.get("bridges") or {}).get("chasm")
    twins = min((b.get("twins", 0.6) for b in (bridged if isinstance(bridged, list) else [bridged] if bridged else [])), default=0.6)
    plan = [("plateau", 6, max(7, W // 3))] * sl.get("plateaus", 2) + \
           [("mesa", min_size, max_size)] * regions
    for kind, lo, hi in plan:
        for _ in range(200):
            w, h = rng.randint(lo, hi), rng.randint(lo, min(hi, D - 6))
            if W - edge - w < edge or D - edge - h < edge:
                continue
            i0, j0 = rng.randint(edge, W - edge - w), rng.randint(edge, D - edge - h)
            i1, j1 = i0 + w, j0 + h
            if i0 - 2 <= ei <= i1 + 2 and j0 - 2 <= ej <= j1 + 2:
                continue
            if wet[max(i0 - 2, 0):i1 + 3, max(j0 - 2, 0):j1 + 3].any():
                continue                                     # a tile of ground between mesa and river
            if sea[max(i0 - 2, 0):i1 + 3, max(j0 - 2, 0):j1 + 3].any():
                continue                                     # and between mesa and sea cliff
            apart = lambda b: i0 - b[2] >= 3 or b[0] - i1 >= 3 or j0 - b[3] >= 3 or b[1] - j1 >= 3
            inside = lambda b: i0 - b[0] >= 2 and b[2] - i1 >= 2 and j0 - b[1] >= 2 and b[3] - j1 >= 2
            hosts = [b for b in blocks if inside(b)]
            if not all(apart(b) or inside(b) for b in blocks):
                continue
            if hosts and not all(b[5] for b in hosts):
                continue                                     # nothing stands on a rock
            base = max([P.levels[0]] + [b[4] for b in hosts])
            r = rng.random()
            walk = kind == "plateau" or not blocks or bool(hosts)
            if kind == "plateau":
                form = "ramp"                                # the big ones: a rock line with a way up
            elif r < sl.get("hills", 0.2):
                form = "hill"
            elif walk or r < sl.get("hills", 0.2) + sl.get("ramps", 0.5):
                form = "ramp"
            else:
                form = "rock"
            # a way up climbs one step; a rock can stand taller (as tall as the kit's cliffs go)
            lv = base + (rng.choice([P.step, P.max_rise]) if form == "rock" else P.step)
            if lv > max(P.levels):
                continue
            # a mesa is a rock (cliffs all round), a ramp (cliffs, one side a walkable slope, the way
            # Misareaux cuts a way up through its rock lines: the switch at that side's two corners),
            # or a hill (slopes all round). Plateaus and anything standing on one are walkable; the
            # outer edge of the map never has a way up
            ground[i0:i1 + 1, j0:j1 + 1] = lv
            blocks.append((i0, j0, i1, j1, lv, form != "rock", form))
            seeds.append(((i0 + i1) / 2, (j0 + j1) / 2, lv))
            if bridged and not hosts and rng.random() < twins:
                _twin(ground, blocks, seeds, crossings, (i0, j0, i1, j1, lv), P, edge, (ei, ej), wet, sea, rng)
            break
    if bridged:
        _chasms(blocks, crossings, ground)
    slopes = {}
    for i0, j0, i1, j1, lv, walk, form in blocks:
        ring = [(i, j) for i in range(i0 - 1, i1 + 1) for j in range(j0 - 1, j1 + 1)
                if not np.isnan(ground[i:i + 2, j:j + 2]).any() and np.ptp(ground[i:i + 2, j:j + 2]) > 0
                and ground[i:i + 2, j:j + 2].max() == lv]
        if form == "hill":
            slopes.update({c: True for c in ring})
        elif form == "ramp":
            side = rng.choice(["s", "n", "w", "e"])
            line = {"s": lambda c: c[1] == j0 - 1, "n": lambda c: c[1] == j1,
                    "w": lambda c: c[0] == i0 - 1, "e": lambda c: c[0] == i1}[side]
            cells = sorted(c for c in ring if line(c))
            for c in cells:
                slopes[c] = True
            for c in (cells[0], cells[-1]):                  # the corners: where cliff turns to ramp
                slopes[c] = "free"
    return ground, seeds, slopes, crossings


def _twin(ground, blocks, seeds, crossings, src, P, edge, entry, wet, sea, rng):
    """A second mesa at the same height one flat tile across from `src` (a chasm: cliff, a tile of
    low ground, cliff), often a rock reachable only by the bridge laid across it."""
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    i0, j0, i1, j1, lv = src
    base = float(np.nanmin(ground[i0 - 1:i1 + 2, j0 - 1:j1 + 2]))
    for _ in range(60):
        side = rng.choice("nsew")
        w, h = rng.randint(3, 6), rng.randint(3, 6)
        if side in "ew":
            a0 = i1 + 3 if side == "e" else i0 - 3 - w
            b0 = rng.randint(max(j0 - h + 3, edge), min(j1 - 3, D - edge - h)) if min(j1 - 3, D - edge - h) >= max(j0 - h + 3, edge) else None
            if b0 is None:
                continue
            box = (a0, b0, a0 + w, b0 + h)
        else:
            b0 = j1 + 3 if side == "n" else j0 - 3 - h
            a0 = rng.randint(max(i0 - w + 3, edge), min(i1 - 3, W - edge - w)) if min(i1 - 3, W - edge - w) >= max(i0 - w + 3, edge) else None
            if a0 is None:
                continue
            box = (a0, b0, a0 + w, b0 + h)
        x0, y0, x1, y1 = box
        if x0 < edge or y0 < edge or x1 > W - edge or y1 > D - edge:
            continue
        if x0 - 2 <= entry[0] <= x1 + 2 and y0 - 2 <= entry[1] <= y1 + 2:
            continue
        if wet[max(x0 - 2, 0):x1 + 3, max(y0 - 2, 0):y1 + 3].any() or sea[max(x0 - 2, 0):x1 + 3, max(y0 - 2, 0):y1 + 3].any():
            continue
        if any(not (x0 - b[2] >= 3 or b[0] - x1 >= 3 or y0 - b[3] >= 3 or b[1] - y1 >= 3) for b in blocks):
            continue
        if (np.nan_to_num(ground[x0 - 1:x1 + 2, y0 - 1:y1 + 2], nan=-1e9) != base).any():
            continue                                         # low ground all round, as the chasm floor
        ground[x0:x1 + 1, y0:y1 + 1] = lv
        form = "rock" if rng.random() < 0.6 else "ramp"
        blocks.append((x0, y0, x1, y1, lv, form != "rock", form))
        seeds.append(((x0 + x1) / 2, (y0 + y1) / 2, lv))
        # the bridge: across the chasm, in the middle of where the two faces overlap
        if side in "ew":
            m = (max(j0, y0) + min(j1, y1)) / 2
            a, b = (i1, x0) if side == "e" else (x1, i0)
            crossings.append({"kind": "chasm", "at": ((a + b) / 2, m), "dir": (1, 0), "level": lv,
                              "join": [(a, int(round(m))), (b, int(round(m)))]})
        else:
            m = (max(i0, x0) + min(i1, x1)) / 2
            a, b = (j1, y0) if side == "n" else (y1, j0)
            crossings.append({"kind": "chasm", "at": (m, (a + b) / 2), "dir": (0, 1), "level": lv,
                              "join": [(int(round(m)), a), (int(round(m)), b)]})
        return


def _narrow(g):
    """Is there a strip one corner wide between higher ground on both sides (a trench no tile kit
    has pieces for: two cliffs facing each other across a single tile)?"""
    c = g[1:-1, 1:-1]
    left, right = g[:-2, 1:-1], g[2:, 1:-1]
    down, up = g[1:-1, :-2], g[1:-1, 2:]
    with np.errstate(invalid="ignore"):
        return bool(((left > c) & (right > c)).any() or ((down > c) & (up > c)).any())


def _carve_river(ground, blocks, entry, P, spec, rng):
    """A river the way Misareaux cuts one: a channel two corners wide, sunk `depth` below the low
    ground, winding across it in straight reaches with turns, clear of the mesas, the walls and
    the way in. Its banks and water are the kit's water pieces (see water_links)."""
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    low = P.levels[0]
    water = low - float(spec.get("depth", P.step))
    lo, hi = spec.get("length", [8, 16])

    def free(i, j):
        if not (3 <= i <= W - 3 and 3 <= j <= D - 3):
            return False
        if abs(i - entry[0]) + abs(j - entry[1]) < 5:
            return False
        near = ground[i - 2:i + 3, j - 2:j + 3]
        return not np.isnan(near).any() and np.all((near == low) | (near == water))

    for _ in range(60):
        cells = [(i, j) for i in range(3, W - 3) for j in range(3, D - 3) if free(i, j)]
        if not cells:
            return
        pos = rng.choice(cells)
        heading = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
        path, reach = [pos], rng.randint(2, 4)
        while len(path) < rng.randint(lo, hi):
            if reach == 0:
                heading = rng.choice([h for h in ((1, 0), (-1, 0), (0, 1), (0, -1)) if h != (-heading[0], -heading[1])])
                reach = rng.randint(2, 4)
            nxt = (path[-1][0] + heading[0], path[-1][1] + heading[1])
            if not free(*nxt) or nxt in path:
                break
            path.append(nxt)
            reach -= 1
        if len(path) >= lo:
            for i, j in path:
                ground[i, j] = water                           # a line of corners: the banks face
            return path                                        # each other across a narrow channel


SIDES_OF = {"n": (0, 1), "e": (1, 2), "s": (2, 3), "w": (3, 0)}      # corner indices (NW NE SE SW)


def water_links(ground, level):
    """{cell: sides with water} for water at `level`: a side carries water when either of its end
    corners is at the water (how Misareaux's river pieces are made: open water on every side, a
    straight bank on its low side and both ends, a bank corner by its corners)."""
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    out = {}
    for i in range(W):
        for j in range(D):
            c = [ground[i, j + 1], ground[i + 1, j + 1], ground[i + 1, j], ground[i, j]]
            wet = [not np.isnan(x) and abs(x - level) < 0.5 for x in c]
            if any(wet):
                out[(i, j)] = frozenset(side for side, (a, b) in SIDES_OF.items() if wet[a] or wet[b])
    return out


def _inlets(ground, ring, entry, spec, rng):
    """Break an island's shore into bays and headlands: rectangles of sea cut in from the edge,
    `width` corners wide and `depth` deep, never leaving a strip of land or sea one corner across
    (no sea-cliff piece lays that) and never near the way in."""
    from .kit import HOLE
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    lo_w, hi_w = spec.get("width", [2, 5])
    lo_d, hi_d = spec.get("depth", [1, 3])
    for _ in range(spec.get("inlets", 6) * 20):
        if spec.get("_made", 0) >= spec.get("inlets", 6):
            break
        side = rng.choice("nsew")
        w, d = rng.randint(lo_w, hi_w), rng.randint(lo_d, hi_d)
        if side in "ns":
            a = rng.randint(ring + 2, W - ring - 2 - w)
            i0, i1 = a, a + w
            j0, j1 = (ring, ring + d - 1) if side == "s" else (D - ring - d + 1, D - ring)
        else:
            a = rng.randint(ring + 2, D - ring - 2 - w)
            j0, j1 = a, a + w
            i0, i1 = (ring, ring + d - 1) if side == "w" else (W - ring - d + 1, W - ring)
        if abs((i0 + i1) / 2 - entry[0]) < w / 2 + 4 and abs((j0 + j1) / 2 - entry[1]) < d + 5:
            continue
        trial = ground.copy()
        trial[i0:i1 + 1, j0:j1 + 1] = HOLE
        land = trial > HOLE / 2
        thin = False
        for axis in (0, 1):
            a_, b_ = np.roll(land, 1, axis), np.roll(land, -1, axis)
            if (land & ~a_ & ~b_).any() or (~land & a_ & b_).any():
                thin = True                                   # land or sea one corner across
        if thin:
            continue
        ground[:, :] = trial
        spec["_made"] = spec.get("_made", 0) + 1
    spec.pop("_made", None)


def _chasms(blocks, crossings, ground):
    """Any two mesas at the same height with one flat tile between them (cliff, low ground, cliff)
    and faces overlapping by 3 or more get a bridge across, if they haven't one already."""
    have = {tuple(sorted(map(tuple, c["join"]))) for c in crossings if c["kind"] == "chasm"}
    for n, A in enumerate(blocks):
        for B in blocks[n + 1:]:
            if A[4] != B[4]:
                continue
            for axis in (0, 1):
                lo_a, hi_a = (A[0], A[2]) if axis == 0 else (A[1], A[3])
                lo_b, hi_b = (B[0], B[2]) if axis == 0 else (B[1], B[3])
                ov_lo, ov_hi = ((max(A[1], B[1]), min(A[3], B[3])) if axis == 0 else (max(A[0], B[0]), min(A[2], B[2])))
                if ov_hi - ov_lo < 3:
                    continue
                if lo_b - hi_a == 3:
                    a, b = hi_a, lo_b
                elif lo_a - hi_b == 3:
                    a, b = hi_b, lo_a
                else:
                    continue
                m = (ov_lo + ov_hi) / 2
                if axis == 0:
                    at, d, join = ((a + b) / 2, m), (1, 0), [(a, int(round(m))), (b, int(round(m)))]
                else:
                    at, d, join = (m, (a + b) / 2), (0, 1), [(int(round(m)), a), (int(round(m)), b)]
                key = tuple(sorted(join))
                if key in have:
                    continue
                low = ground[join[0][0]:join[1][0] + 1, join[0][1]:join[1][1] + 1] if axis == 0 else \
                    ground[join[0][0]:join[0][0] + 1, join[0][1]:join[1][1] + 1]
                have.add(key)
                crossings.append({"kind": "chasm", "at": at, "dir": d, "level": A[4], "join": join})


def _channels(ground, ring, entry, spec, rng, crossings, land):
    """Split an island into landmasses with sea channels cut coast to coast (the way Misareaux's
    islands lie apart), `width` corners of open sea across, each spanned by the rope bridge."""
    from .kit import HOLE
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    lo_w, hi_w = spec.get("channel_width", [2, 3])
    for n in range(spec.get("channels", 1)):
        for _ in range(40):
            w = rng.randint(lo_w, hi_w)
            if rng.random() < 0.5:                            # a channel running north-south
                a = rng.randint(ring + 6, W - ring - 6 - w)
                if abs(a + w / 2 - entry[0]) < w / 2 + 4:
                    continue
                trial = ground.copy()
                trial[a:a + w, :] = HOLE
                line = [(a - 1, j) for j in range(ring + 3, D - ring - 3)]
                d, across = (1, 0), (a - 1, a + w)
            else:                                             # east-west
                b = rng.randint(ring + 7, D - ring - 6 - w)
                trial = ground.copy()
                trial[:, b:b + w] = HOLE
                line = [(i, b - 1) for i in range(ring + 3, W - ring - 3)]
                d, across = (0, 1), (b - 1, b + w)
            dry = trial > HOLE / 2
            thin = False
            for axis in (0, 1):
                a_, b_ = np.roll(dry, 1, axis), np.roll(dry, -1, axis)
                if (dry & ~a_ & ~b_).any():
                    thin = True                               # land one corner across: no piece lays it
                sea1 = ~dry & a_ & b_
                if w > 1 and sea1.any():
                    thin = True                               # (a one-corner channel is the point when w is 1:
                                                              # two sea cliffs face each other across it)
            if thin:
                continue
            ground[:, :] = trial
            # the bridge: somewhere along the channel where both shores are straight
            ok = [p for p in line if all(trial[p[0] + d[0] * k, p[1] + d[1] * k] > HOLE / 2 for k in (0, -1))
                  and all(trial[p[0] + d[0] * (w + 1) + d[0] * k, p[1] + d[1] * (w + 1) + d[1] * k] > HOLE / 2 for k in (0, 1))]
            if ok:
                p = ok[len(ok) // 2 + rng.randint(-len(ok) // 4, len(ok) // 4)] if len(ok) > 4 else ok[len(ok) // 2]
                if d == (1, 0):
                    at, join = ((across[0] + across[1]) / 2, p[1] + 0.5), [(across[0], p[1]), (across[1], p[1])]
                else:
                    at, join = (p[0] + 0.5, (across[0] + across[1]) / 2), [(p[0], across[0]), (p[0], across[1])]
                crossings.append({"kind": "chasm", "at": at, "dir": d, "level": land, "join": join, "sea": True})
            break

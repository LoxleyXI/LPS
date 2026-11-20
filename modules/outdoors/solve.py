"""Choosing the tiles: for every cell, a tile pose whose corners are the cell's corner heights and
whose sides agree with its neighbours' sides.

Floor corners must match exactly; a wall corner takes any cliff top at least the kit's wall_rise
above the lowest floor beside it, settled by the neighbours through the side profiles (which include
the corners). Constraint propagation (every cell keeps only the poses that fit some pose of each
neighbour; the way "wave function collapse" works), then a search that fixes the most constrained
cell first, propagating after each choice and undoing on a dead end. A layout the kit can't lay is
found by the propagation alone, at the cell that empties.
"""
import math

import numpy as np

from .layout import wall_floor


def compatibility(cands, side_tol, look_misses=1):
    """(E, N): E[a, b] when b can sit east of a, N[a, b] when b can sit north of a: the shared
    side's heights agree within side_tol, and so does what's on the surface there (the same
    textures on top at all but look_misses samples: a footpath carries on across the seam)."""
    def stack(side):
        return np.array([c["sides"][side] for c in cands])

    def steep(P):
        """How fast each side's height changes at each sample (yalms per sample)."""
        g = np.abs(np.gradient(np.nan_to_num(P, nan=0.0), axis=1))
        return np.where(np.isnan(P), 0.0, g)

    def agree(A, B):
        # on a sheer face a tiny sideways offset reads as a big height difference though nothing
        # gapes, so the allowance grows with the steepness there; on gentle ground (where a
        # mismatch is a visible slit) it stays side_tol
        Br = B[:, ::-1]
        d = np.abs(A[:, None, :] - Br[None, :, :])
        allow = side_tol + 0.5 * np.maximum(steep(A)[:, None, :], steep(Br)[None, :, :])
        m = ~np.isnan(d)
        return (m.sum(2) < 3) | np.all(~m | (d <= allow), axis=2)

    def looks(side, other):
        # each distinct look along a side gets a number; sides compare as integer rows
        table = {}
        enc = lambda seq: [table.setdefault(x, len(table)) for x in seq]
        A = np.array([enc(c["looks"][side]) for c in cands])
        B = np.array([enc(c["looks"][other][::-1]) for c in cands])
        return (A[:, None, :] != B[None, :, :]).sum(2) <= look_misses
    return (agree(stack("e"), stack("w")) & looks("e", "w"),
            agree(stack("n"), stack("s")) & looks("n", "s"))


def solve(ground, kit, cfg, rng, nets=None, slopes=None, tol=1.2, side_tol=0.5, budget=20000, repairs=400,
          require=None, skip=()):
    """({(i, j): candidate}, missing [(i, j, wanted corners, 0)]).

    A cell with a wall corner takes a piece from the kit's wall families ("walls": {"families"}):
    a slope there would lead up onto the cliff tops and off the edge of the world.

    nets: {network name: {cell: sides it leaves by}} from the layout. A network cell takes a tile
    whose network feature leaves by exactly those sides; every other cell takes one without it (no
    stray bits of road). Network tiles are used even from families the kit otherwise avoids.

    require: {(i, j): label prefix} cells that must take a piece of that family (a drawn grid's
    beach), whatever the kit otherwise avoids. skip: cells left empty (our own rock stands there)."""
    from .kit import feature
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    nets = nets or {}
    avoid = tuple(cfg.get("avoid", []))
    netf = tuple(feature(kit, spec["texture"]) for spec in (cfg.get("networks") or {}).values())
    key = (avoid, side_tol, netf, tuple(sorted(set((require or {}).values()))))
    if kit.get("_compat_key") != key:
        kit["_filler"] = [c for c in kit["cands"]
                          if not (avoid and c["label"].startswith(avoid)) or any(c["links"].get(f) for f in netf)
                          or c["label"].startswith(tuple((require or {}).values()) or ("\0",))]
        kit["_E"], kit["_N"] = compatibility(kit["_filler"], side_tol)
        kit["_compat_key"] = key
    cands, E, N = kit["_filler"], kit["_E"], kit["_N"]
    corners = np.array([c["corners"] for c in cands])
    # cells kept for something built on them (an outpost): only tiles whose surface is level
    flat_cells = set(map(tuple, cfg.get("flat_cells", [])))
    if flat_cells:
        bump = np.array([float(np.ptp(np.asarray(kit["tris"][c["name"]]).reshape(-1, 3)[:, 1])) for c in cands])
        flat_ok = bump <= cfg.get("flat_tol", 2.0)
    walls = tuple((cfg.get("walls") or {}).get("families", []))
    wall_ok = np.array([not walls or c["label"].startswith(walls) for c in cands])   # pieces you can't climb
    links = {name: (feature(kit, spec["texture"]), np.array([c["links"].get(feature(kit, spec["texture"]), frozenset()) for c in cands], object))
             for name, spec in (cfg.get("networks") or {}).items()}
    weights = np.array([math.sqrt(c["weight"]) for c in cands])
    slopes = slopes or {}
    slope_fams = tuple((cfg.get("slopes") or {}).get("families", []))
    is_slope = np.array([bool(slope_fams) and c["label"].startswith(slope_fams) for c in cands])
    # how readily the search picks a slope over a cliff (a kit whose look is cliffs sets it low)
    weights = weights * np.where(is_slope, (cfg.get("slopes") or {}).get("weight", 1.0), 1.0)
    need = {name: np.array([h for h in have], object) for name, (f, have) in links.items()}

    def four(a, i, j):
        return np.array([a[i, j + 1], a[i + 1, j + 1], a[i + 1, j], a[i, j]])   # NW NE SE SW

    def domain(ground, lowest, i, j):
        g = four(ground, i, j)
        if np.isnan(g).all():
            return None
        floor = ~np.isnan(g)
        wmin = np.nan_to_num(four(lowest, i, j), nan=-1e9)
        ok = np.all(np.where(floor, np.abs(corners - np.nan_to_num(g)) <= tol, corners >= wmin - tol), axis=1)
        for name, have in need.items():
            want = nets.get(name, {}).get((i, j), frozenset())
            ok &= np.array([h == want for h in have])
        if (i, j) in flat_cells:
            ok &= flat_ok
        if require and (i, j) in require:
            ok &= np.array([c["label"].startswith(require[(i, j)]) for c in cands])
        elif require:
            ok &= ~np.array([c["label"].startswith(tuple(require.values())) for c in cands])   # only where asked
        if not floor.all():
            ok &= wall_ok                                       # a wall is a cliff, never a slope up
        elif slope_fams and slopes and np.ptp(g) > 0 and slopes.get((i, j)) != "free":
            ok &= is_slope if (i, j) in slopes else ~is_slope   # marked ways up are slopes; the rest, cliffs
                                                                # ("free": a join, either, e.g. the
                                                                # transition corner where a ramp meets a cliff)
        return ok

    ground = ground.copy()
    lowest = wall_floor(ground) + float(cfg["wall_rise"])     # a wall corner's least height
    dom = {}
    from .kit import HOLE
    for j in range(D):
        for i in range(W):
            if (four(ground, i, j) <= HOLE / 2).all() or (i, j) in skip:
                continue                                        # open sea, or our own rock: no tile
            d = domain(ground, lowest, i, j)
            if d is not None:
                dom[(i, j)] = d

    # repair: a cell no piece fits gets one corner moved to a neighbour's height (or opened to wall),
    # whichever fixes it and breaks least round about; only the kit's own pieces decide
    def around(x, y):
        return [(a, b) for a in (x - 1, x) for b in (y - 1, y) if 0 <= a < W and 0 <= b < D]

    for _ in range(repairs):
        bad = [c for c, d in dom.items() if not d.any()]
        if not bad:
            break
        i, j = bad[0]
        best = None
        for x, y in ((i, j + 1), (i + 1, j + 1), (i + 1, j), (i, j)):
            if x in (0, W) or y in (0, D):
                continue                                        # the outer ring stays wall
            old = ground[x, y]
            values = {v for v in four(ground, i, j) if not np.isnan(v)} | {np.nan}
            for v in values:
                if (np.isnan(v) and np.isnan(old)) or v == old:
                    continue
                ground[x, y] = v
                low2 = wall_floor(ground) + float(cfg["wall_rise"])
                cells = {c for c in around(x, y)} | {c for a, b in around(x, y) for c in around(a, b)}
                score = 0
                for c in cells:
                    d = domain(ground, low2, *c)
                    score += d is not None and not d.any()
                if best is None or score < best[0]:
                    best = (score, (x, y), v)
                ground[x, y] = old
        if best is None:
            break
        _, (x, y), v = best
        ground[x, y] = v
        lowest = wall_floor(ground) + float(cfg["wall_rise"])
        for c in {c for a, b in around(x, y) for c in around(a, b)} | set(around(x, y)):
            d = domain(ground, lowest, *c)
            if d is None:
                dom.pop(c, None)
            else:
                dom[c] = d
    solve.repaired = ground

    def want(i, j):
        return [None if np.isnan(x) else float(x) for x in four(ground, i, j)]

    def propagate(dom, queue):
        while queue:
            i, j = queue.pop()
            d = dom[(i, j)]
            for (di, dj), M, forward in (((1, 0), E, True), ((-1, 0), E, False), ((0, 1), N, True), ((0, -1), N, False)):
                n = (i + di, j + dj)
                if n not in dom:
                    continue
                nd = dom[n] & (M[d].any(0) if forward else M[:, d].any(1))
                if nd.sum() < dom[n].sum():
                    dom[n] = nd
                    if not nd.any():
                        return n
                    queue.append(n)
        return None

    empty = [c for c, d in dom.items() if not d.any()]
    if empty:
        return {}, [(i, j, want(i, j), 0) for i, j in empty]
    bad = propagate(dom, list(dom))
    if bad:
        return {}, [(bad[0], bad[1], want(*bad), 0)]
    stack, steps = [], 0
    while True:
        open_ = [(int(d.sum()), c) for c, d in dom.items() if d.sum() > 1]
        if not open_:
            break
        steps += 1
        if steps > budget:
            return {}, [(-1, -1, ["search budget"], 0)]
        _, cell = min(open_, key=lambda t: (t[0], rng.random()))
        order = sorted(np.nonzero(dom[cell])[0], key=lambda k: -weights[k] * rng.random())
        stack.append((cell, {c: d.copy() for c, d in dom.items()}, order))
        while stack:
            cell, saved, order = stack[-1]
            if not order:
                stack.pop()
                if not stack:
                    return {}, [(cell[0], cell[1], want(*cell), 0)]
                continue
            pick = order.pop(0)
            dom = {c: d.copy() for c, d in saved.items()}
            dom[cell] = np.zeros(len(cands), bool)
            dom[cell][pick] = True
            if propagate(dom, [cell]) is None:
                break
        else:
            return {}, [(-1, -1, ["no solution"], 0)]
    return {c: cands[int(np.nonzero(d)[0][0])] for c, d in dom.items()}, []


def reachable(ground, placed, cfg, start, below=None, joins=()):
    """The share of open ground a player can walk to from `start` (a corner): along level ground,
    and up or down a step only through a piece from the kit's slope families."""
    import collections
    fams = tuple((cfg.get("slopes") or {}).get("families", []))
    W, D = ground.shape
    seen, q = {start}, collections.deque([start])
    extra = {}
    for a, b in joins:                                        # bridges: a way across a gap
        extra.setdefault(tuple(a), []).append(tuple(b))
        extra.setdefault(tuple(b), []).append(tuple(a))
    while q:
        x, y = q.popleft()
        for n in extra.get((x, y), []):
            if n not in seen and 0 <= n[0] < W and 0 <= n[1] < D:
                seen.add(n)
                q.append(n)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = x + dx, y + dy
            if not (0 <= a < W and 0 <= b < D) or (a, b) in seen or np.isnan(ground[a, b]):
                continue
            if ground[a, b] != ground[x, y]:
                cells = [(min(x, a), min(y, b)), (min(x, a) - (dx == 0), min(y, b) - (dy == 0))]
                if not any(c in placed and fams and placed[c]["label"].startswith(fams) for c in cells):
                    continue
            seen.add((a, b))
            q.append((a, b))
    counted = ~np.isnan(ground) if below is None else (~np.isnan(ground) & (np.nan_to_num(ground, nan=-1e9) >= below))
    return sum(1 for x, y in seen if counted[x, y]) / max(1, int(counted.sum()))       # the river bed
                                                                                         # isn't ground

"""Monsters for a generated outdoor zone: spawn points on its level ground, for the caves module's
server SQL, client models and name list.

A zone file's "mobs" block is the caves one ("sql", "overlay", "models_from", "pools_from",
"types"), with each type's "count" and a "gap" (yalms kept between spawns). Spawns stand on cells
whose four corners share a height (level ground, not slopes, cliffs or sea), a few to a cell,
anywhere but its edges.

A type's "camps" ({"count", "radius"}) gathers it into that many camps, each a circle of that
radius on its own patch of ground, apart from other types' camps (Abyssea style: you know where
the tigers are). Types without it spread over the whole zone (beastmen and saplings, as the
original game has them).
"""
import math
import random

import numpy as np

from zonekit import tiles as T

from .kit import HOLE

TILE = T.TILE


def level_cells(summary, clear=1, slope=0.0):
    """[(i, j, height)] for cells of level ground, kept `clear` cells off any town (an outpost
    stays safe). With `slope` and the summary's "surface" (the laid tiles' real heights), a cell
    counts when its surface rises no more than `slope` yalms across it: gentle ramps as well as
    flat ground."""
    ground = summary["ground"]
    towns = (summary.get("town") or {}).get("sites", [])
    out = []
    for (i, j) in summary["placed"]:
        if any(i0 - clear <= i < i1 + clear and j0 - clear <= j < j1 + clear for i0, j0, i1, j1, _h in towns):
            continue
        g4 = [ground[i, j], ground[i + 1, j], ground[i, j + 1], ground[i + 1, j + 1]]
        if not all(math.isfinite(g) for g in g4) or min(g4) <= HOLE / 2:
            continue
        if max(g4) - min(g4) <= 1e-3:
            out.append((i, j, float(g4[0])))
            continue
        surface = summary.get("surface")
        if slope and surface is not None:
            u = [(a + 0.5) / 5 for a in range(5)]
            xs = [(i + a) * TILE for a in u for _ in u]
            zs = [(j + b) * TILE for _ in u for b in u]
            hs = surface.at(xs, zs)
            if np.isfinite(hs).all() and np.ptp(hs) <= slope:
                out.append((i, j, None))          # a gentle slope: each spawn takes its own height
    return out


def place(summary, types, seed=1, gap=12.0, slope=0.0):
    """[{"type", "pos": (x, north, height), "facing"}] for every type's "count", spread over the
    zone's level ground, at least `gap` yalms apart."""
    rng = random.Random(seed)
    if "shift" in summary:
        sx, sz = summary["shift"][0], summary["shift"][2]
    else:
        b0 = summary["basins"][0]
        # the layout puts the first basin at the zone's origin (zone.py: shift)
        sx, sz = round(-b0[0]) * TILE, round(-b0[1]) * TILE
    cells = level_cells(summary, slope=slope)
    if not cells:
        raise ValueError("no level ground to stand monsters on")
    def centre(c):
        return (c[0] + 0.5) * TILE + sx, (c[1] + 0.5) * TILE + sz

    # camps first: each a level cell well away from every camp already made
    camps = []                                          # (x, z, radius, type)
    for key, t in types.items():
        spec = t.get("camps")
        if not spec:
            continue
        radius = spec.get("radius", 50.0)
        for _ in range(spec.get("count", 2)):
            best = None
            for _try in range(400):
                x, z = centre(rng.choice(cells))
                clear = min(((x - a) ** 2 + (z - b) ** 2) ** 0.5 - r for a, b, r, _k in camps) if camps else 1e9
                if best is None or clear > best[0]:
                    best = (clear, x, z)
                if clear > 2.5 * radius:
                    break
            camps.append((best[1], best[2], radius, key))

    def where(key):
        """level cells a type may stand on: its camps' cells, or anywhere outside every camp"""
        own = [(x, z, r) for x, z, r, k in camps if k == key]
        if own:
            return [c for c in cells if any((centre(c)[0] - x) ** 2 + (centre(c)[1] - z) ** 2 <= r * r for x, z, r in own)]
        return [c for c in cells if all((centre(c)[0] - x) ** 2 + (centre(c)[1] - z) ** 2 > r * r for x, z, r, _k in camps)]

    # each spawn at the best of a few candidate points: the one furthest from every spawn so far
    # (spread over all its ground, never piled into one corner when the ground runs short)
    taken = []
    out = []
    for key, t in types.items():
        pool = where(key) or cells
        for _ in range(t.get("count", 0)):
            best = None
            for _try in range(30):
                i, j, h = rng.choice(pool)
                x = (i + rng.uniform(0.15, 0.85)) * TILE + sx
                z = (j + rng.uniform(0.15, 0.85)) * TILE + sz
                room = min(((x - a) ** 2 + (z - b) ** 2 for a, b in taken), default=1e12)
                if best is None or room > best[0]:
                    best = (room, x, z, h)
                if room >= (4 * gap) ** 2:
                    break
            _room, x, z, h = best
            if h is None:
                # on a slope: the surface's height at this very point (local, before the shift)
                h = float(summary["surface"].at([x - sx], [z - sz])[0])
            taken.append((x, z))
            out.append({"type": key, "pos": (x, z, h), "facing": rng.uniform(-math.pi, math.pi)})
    return out

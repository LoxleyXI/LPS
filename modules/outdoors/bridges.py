"""Bridges: the source zone's own, learned whole and laid across our gaps.

A bridge in a zone is one or more pieces placed at the same point (Misareaux's rope bridge is a
deck and four mirrored rails and supports). learn() finds each named bridge in the source, keeps
its pieces' placement relative to the first, and measures its long axis, its length, and how high
its origin sits above the ground at its ends. place() lays it across a gap in our zone: the span
measured on the laid tiles brink to brink along its line, the bridge stretched to fit.

    "bridges": {"chasm": {"parts": ["ri_hasi3_h", "ri_hasi4_h"]},
                "river": {"parts": ["fa_hasi00"]}}
"""
import math

import numpy as np

from zonekit import tiles as T

from .buildings import _unrot

TILE = T.TILE


def learn(kit, name_of, spec, ground, source=None):
    """{kind: [bridge]} for the kit file's bridges. ground: heights of the source's own laid tiles.
    A kind is one bridge ({"parts": [...]}) or a list to choose from; a bridge with "from": <zone>
    is learned in that zone (source(zone) gives its (kit-like {"objs", "tris"}, ground))."""
    out = {}
    for kind, opts in (spec or {}).items():
        for b in (opts if isinstance(opts, list) else [opts]):
            k, g = (kit, ground) if "from" not in b else source(b["from"])
            got = learn_one(k, name_of, b, g, kind)
            if got:
                got["from"] = b.get("from")
                out.setdefault(kind, []).append(got)
    return out


def learn_one(kit, name_of, b, ground, kind=""):
    parts = [o for o in kit["objs"] if name_of(o["name"]) in b["parts"] and o["name"] in kit["tris"]]
    if not parts:
        return None
    anchor = next(o for o in parts if name_of(o["name"]) == b["parts"][0])
    ax, az, ay = anchor["pos"][0], anchor["pos"][2], anchor["pos"][1]
    members = [o for o in parts if math.hypot(o["pos"][0] - ax, o["pos"][2] - az) < 1.0]
    yaw = anchor["rot"][1]
    # its shape is all its pieces together (a deck can be two mirrored halves), in the anchor's frame
    local = []
    for o in members:
        w = T.placed(kit["tris"][o["name"]], o).reshape(-1, 3) - np.array(anchor["pos"])
        local.append(np.stack([_unrot(v, yaw) for v in w[:: max(1, len(w) // 400)]]))
    t = np.concatenate(local)
    size = t.max(0) - t.min(0)
    axis = np.array([0.0, 0.0, 1.0]) if size[2] >= size[0] else np.array([1.0, 0.0, 0.0])
    mid_local = (t.max(0) + t.min(0)) / 2
    mid_local[1] = 0.0                                     # centred along the ground, not in height
    c, sn = math.cos(yaw), math.sin(yaw)
    centre = np.array(anchor["pos"]) + np.array([c * mid_local[0] + sn * mid_local[2], 0.0, -sn * mid_local[0] + c * mid_local[2]])
    d = np.array([c * axis[0] + sn * axis[2], 0.0, -sn * axis[0] + c * axis[2]])
    length = float(max(size[0], size[2]))
    ends = [ground.at([centre[0] + s * d[0] * (length / 2 + 4)], [centre[2] + s * d[2] * (length / 2 + 4)])[0]
            for s in (-1, 1)]
    end = float(np.nanmean(ends)) if not np.isnan(ends).all() else -ay
    under = ground.at([centre[0]], [centre[2]])[0]
    if not np.isnan(under) and under > end - 4:
        # nothing to cross where the source stands it: laid over a gap you'd walk off its sides
        print(f"  bridge {kind!r}: {name_of(anchor['name'])} crosses no gap in its source, not used")
        return None
    # the gap it was built for: brink to brink under it (the deck runs well onto the land either
    # side; what has to match is the gap, not the deck)
    gap = span(ground, centre[0], centre[2], d[0], d[2], end)
    return {"parts": [(o["name"], o["rot"][1] - yaw, tuple(o["scale"]),
                       _unrot(np.array(o["pos"]) - np.array(anchor["pos"]), yaw)) for o in members],
            "names": sorted({name_of(o["name"]) for o in members}),
            "local_axis": axis, "length": length, "mid": mid_local,
            "gap": float(sum(gap)) if gap else length, "gap_mid": float((gap[1] - gap[0]) / 2) if gap else 0.0,
            "offset": float(-ay - end)}                     # origin above the ground at its ends




def span(ground, cx, cz, dx, dz, top, reach=150.0, step=1.0):
    """Brink to brink along a line through (cx, cz): how far out, each way, the ground first stands
    within a yalm of `top`. Returns (back, forward) distances or None."""
    out = []
    for sgn in (-1, 1):
        hit = None
        for r in np.arange(0, reach, step):
            h = ground.at([cx + sgn * dx * r], [cz + sgn * dz * r])[0]
            if not np.isnan(h) and h >= top - 2.5:           # the brink: where the land starts to fall
                hit = r
                break
        if hit is None:
            return None
        out.append(hit)
    return tuple(out)


def place(bridge, cx, cz, dx, dz, top, back, fwd, lo=0.7, hi=1.2):
    """The bridge's pieces laid across the gap from `back` behind (cx, cz) to `fwd` ahead along
    (dx, dz), at the height its ends stand at (`top`): [{"name", "pos", "rot", "scale"}] FFXI,
    or [] if the gap is out of the stretch it can take (stretched much, its posts and anchors land
    off the ground: gaps are made to fit the bridge, not the other way round)."""
    s = (back + fwd) / bridge["gap"]                          # our gap against the one it was made for
    if not lo <= s <= hi:
        return []
    shift = (fwd - back) / 2 - bridge["gap_mid"] * s           # the gaps' middles line up
    mx, mz = cx + dx * shift, cz + dz * shift
    # yaw that turns the bridge's local long axis onto (dx, dz)
    la = bridge["local_axis"]
    yaw = math.atan2(dx, dz) - math.atan2(la[0], la[2])
    c, sn = math.cos(yaw), math.sin(yaw)
    out = []
    stretch = np.where(la > 0.5, s, 1.0)                        # along the long axis only
    for name, dyaw, scale, off in bridge["parts"]:
        mid = bridge["mid"] * stretch
        o = off * stretch - mid
        wx, wz = c * o[0] + sn * o[2], -sn * o[0] + c * o[2]
        sc = tuple(float(a * b) for a, b in zip(scale, stretch))
        out.append({"name": name, "pos": (mx + wx, -(top + bridge["offset"]) + o[1], mz + wz),
                    "rot": (0.0, yaw + dyaw, 0.0), "scale": sc})
    return out


def seat(pieces, kit, ground, cx, cz, dx, dz, gap, sink=0.3):
    """The laid bridge let down onto the land: each piece's lowest point over the land at either end
    measured, and the whole bridge lowered till the highest of those sits `sink` under
    the ground, the way the source's own rest (0.3 to 1.6 under). Stretching and the land's own
    slope leave them floating otherwise."""
    lifts = []
    for p in pieces:
        v = T.placed(kit["tris"][p["name"]], p).reshape(-1, 3)
        along = (v[:, 0] - cx) * dx + (v[:, 2] - cz) * dz
        g = ground.at(v[:, 0], v[:, 2])
        on = ((along <= -gap[0]) | (along >= gap[1])) & ~np.isnan(g)
        for end in (along <= -gap[0], along >= gap[1]):
            sel = on & end
            if sel.any():
                lifts.append(float(np.min(-v[sel, 1] - g[sel])))
    if not lifts:
        return pieces
    drop = max(0.0, max(lifts) + sink)                   # only ever let down: a bridge whose piers
                                                          # stand in the riverbed is already down
    return [dict(p, pos=(p["pos"][0], p["pos"][1] + drop, p["pos"][2])) for p in pieces]

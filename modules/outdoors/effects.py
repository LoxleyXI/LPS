"""The source zone's effects in a new layout.

Effect generators (0x05 sections, unencrypted) carry their world position as the xyz after the
first resource they name. Ambient ones (weather, birds, the sky) sit at the origin and apply
anywhere: they stay. Placed ones belong to the source zone's own places (its campfires, cave
mouths, outpost lights), so a folder of them is dropped, unless it is the coast's sea or surf, which
is moved onto ours (kit file "coast"):

    "coast": {"beach_tile": "be_ssl",       the beach-slope tiles the sea is measured from
              "sea_groups": "umi",          folders of sea (surface, foam, breakers): one per bay
              "surf": "sakn"}               a folder of surf lines, dealt across the bays

Only the source's south-facing coasts are used (sea to -z, like ours). Each generator keeps its own
distance from its own nearest beach tile, which straightens the source's curving bays onto ours;
a sea group used for a second bay is a copy of its folder under a new name (its controller starts
its generators by their own names, so a copy runs on its own).
"""
import math
import re
import struct
from collections import Counter

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

EFFECT = 0x05


def _pos_offset(body, resources):
    for off in range(0, len(body) - 20):
        if body[off:off + 4] in resources:
            return off + 8
    return None


def plan(dat, name_of, cfg):
    """(drop_dirs for the writer, coast groups {folder: [generator]})."""
    tris, objs, _ = T.read(dat)
    secs = fz.split_sections(bytes(dat))
    paths = dict(fz.walk_tree(secs))
    resources = {s.name for s in secs if s.kind not in (EFFECT, fz.T_DIR, fz.T_END)}
    coast = cfg.get("coast") or {}
    keep = set()
    if coast:
        keep = {s.name for i, s in enumerate(secs) if s.kind == fz.T_DIR and paths.get(i, ())[1:] == (b"effe",)
                and ((coast.get("surf") and s.name == coast["surf"].encode())
                     or (coast.get("sea_groups") and s.name.startswith(coast["sea_groups"].encode())))}
    beach = []
    if coast:
        for o in objs:
            if name_of(o["name"]).startswith(coast["beach_tile"]) and o["name"] in tris:
                p = T.world_profile(tris[o["name"]], o, 5)
                beach.append((o["pos"][0], o["pos"][2], np.nanmean(p["n"]) - np.nanmean(p["s"])))
    beach = np.array(beach) if beach else np.zeros((0, 3))
    placed_dirs, groups = set(), {}
    for i, s in enumerate(secs):
        pth = paths.get(i, ())
        if s.kind != EFFECT or len(pth) < 3 or pth[1] not in (b"effe", b"mode"):
            continue
        off = _pos_offset(s.raw[16:], resources)
        if off is None:
            continue
        x, y, z = struct.unpack_from("<3f", s.raw, 16 + off)
        if abs(x) <= 1 and abs(z) <= 1:
            continue                                         # ambient: stays wherever it is
        folder = (pth[1], pth[2])
        placed_dirs.add(folder)
        if pth[1] != b"effe" or pth[2] not in keep or not len(beach):
            continue
        d = np.hypot(beach[:, 0] - x, beach[:, 1] - z)
        k = int(np.argmin(d))
        if beach[k, 2] <= 1.0 or d[k] > 80:
            continue                                         # not by a south-facing beach
        groups.setdefault(pth[2], []).append({"name": s.name, "pos": (x, y, z), "off": off,
                                              "anchor": (float(beach[k, 0]), float(beach[k, 1]))})
    anim = coast.get("animate", "")
    drop = [f for f in placed_dirs if not (f[0] == b"effe" and (f[1] in groups or (anim and f[1].startswith(anim.encode()))))]
    return drop, groups


def move(out, groups, bays, zb, cfg):
    """Rewrite the kept generators into our bays. bays: FFXI (x0, x1) per bay; zb: our beach-slope
    row's FFXI z. Returns (bytes, generators moved)."""
    coast = cfg["coast"]
    seas_prefix, surf_name = coast.get("sea_groups", "~").encode(), coast.get("surf", "").encode()
    secs = [bytearray(x.raw) for x in fz.split_sections(out)]
    kinds = [x[4] & 0x7F for x in secs]                        # the section type: low 7 bits after the name
    names = [bytes(x[:4]) for x in secs]

    def span(dirname, parent=b"effe"):
        depth, start, stack = 0, None, []
        for i, k in enumerate(kinds):
            if k == fz.T_DIR:
                if names[i] == dirname and stack and stack[-1] == parent and start is None:
                    start, depth = i, len(stack)
                stack.append(names[i])
            elif k == fz.T_END:
                stack.pop()
                if start is not None and len(stack) == depth:
                    return start, i
        return None

    def put(sections, gens, cx, gxm):
        idx = {bytes(sec[:4]): n for n, sec in enumerate(sections) if sec[4] & 0x7F == EFFECT}
        for e in gens:
            n = idx.get(e["name"])
            if n is not None:
                x, y, z = e["pos"]
                struct.pack_into("<3f", sections[n], 16 + e["off"], cx + (x - gxm), y, zb + (z - e["anchor"][1]))

    moved, extra = 0, []
    seas = [g for g in sorted(groups) if g.startswith(seas_prefix) and span(g)]
    if seas:
        xs = {g: [e["pos"][0] for e in groups[g]] for g in seas}
        width = {g: max(xs[g]) - min(xs[g]) + 30.0 for g in seas}
        mid = {g: (max(xs[g]) + min(xs[g])) / 2 for g in seas}
        used = Counter()
        spare = iter([seas_prefix[:2] + b"j%d" % k for k in range(1, 10)] + [seas_prefix[:2] + b"k%d" % k for k in range(1, 10)])
        for x0, x1 in bays:
            g = min(seas, key=lambda g: (abs(width[g] - (x1 - x0)), used[g]))
            sp = span(g)
            if used[g] == 0:
                put(secs[sp[0]:sp[1] + 1], groups[g], (x0 + x1) / 2, mid[g])
            else:
                block = [bytearray(x) for x in secs[sp[0]:sp[1] + 1]]
                block[0][:4] = next(spare)                      # the copy's folder name
                put(block, groups[g], (x0 + x1) / 2, mid[g])
                extra.append((sp[1], block))
            used[g] += 1
            moved += len(groups[g])
    surf = groups.get(surf_name, [])
    sp = span(surf_name) if surf else None
    if surf and sp:
        block = secs[sp[0]:sp[1] + 1]
        idx = {bytes(sec[:4]): k for k, sec in enumerate(block) if sec[4] & 0x7F == EFFECT}
        per = (len(surf) + len(bays) - 1) // len(bays)
        for k, e in enumerate(surf):
            if e["name"] in idx:
                x0, x1 = bays[k % len(bays)]
                tx = x0 + (k // len(bays) + 0.5) / per * (x1 - x0)
                x, y, z = e["pos"]
                struct.pack_into("<3f", block[idx[e["name"]]], 16 + e["off"], tx, y, zb + (z - e["anchor"][1]))
                moved += 1
    for after, block in sorted(extra, key=lambda t: -t[0]):
        secs[after + 1:after + 1] = block
    return b"".join(bytes(x) for x in secs), moved


# --- islands: a shore facing every way -----------------------------------------------------------

def facing_of(tris, o):
    """Which way a source sea-cliff tile faces the sea: the side with the most open (hole) samples."""
    p = T.world_profile(tris, o, 9)
    holes = {side: int(np.isnan(p[side]).sum()) for side in "nesw"}
    return max(holes, key=holes.get)


def plan_by_facing(dat, name_of, cfg, reach=120.0):
    """The source's sea generators, each with the way its nearest sea cliff faces and its offset
    from that cliff tile's centre. {folder: [generator]} for the kit's sea groups."""
    coast = cfg.get("coast") or {}
    prefix = coast.get("animate", "")
    if not prefix:
        return {}
    tris, objs, _ = T.read(dat)
    cliffs = [(o["pos"][0], o["pos"][2], facing_of(tris[o["name"]], o)) for o in objs
              if name_of(o["name"]).startswith(coast.get("cliff_tile", "se_md")) and o["name"] in tris]
    xy = np.array([(x, z) for x, z, _ in cliffs])
    secs = fz.split_sections(bytes(dat))
    paths = dict(fz.walk_tree(secs))
    resources = {s.name for s in secs if s.kind not in (EFFECT, fz.T_DIR, fz.T_END)}
    groups = {}
    for i, s in enumerate(secs):
        pth = paths.get(i, ())
        if s.kind != EFFECT or len(pth) < 3 or pth[1] != b"effe" or not pth[2].startswith(prefix.encode()):
            continue
        off = _pos_offset(s.raw[16:], resources)
        if off is None:
            continue
        x, y, z = struct.unpack_from("<3f", s.raw, 16 + off)
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - z)
        k = int(np.argmin(d))
        # what it draws: the source's own flat sea sheets (umi0, umi3) are cut to its coast and
        # show as boxes on ours, so the kit names them to leave out; surf and splashes travel
        draws = {m.group(0).decode() for m in re.finditer(rb"[a-z0-9_]{4}", s.raw[16:]) if m.group(0) in resources}
        skip = bool(draws & set(coast.get("animate_skip", [])))
        groups.setdefault(pth[2], []).append({"name": s.name, "pos": (x, y, z), "off": off,
                                              "face": cliffs[k][2] if d[k] < reach and not skip else None,
                                              "rel": (x - cliffs[k][0], z - cliffs[k][1])})
    return groups


def move_by_facing(out, groups, shores, copies_max=6):
    """Lay the source's sea generators along our shores: each at the same offset from one of our
    sea-cliff tiles facing the same way as its own did (no turning needed: its surf lies along the
    shore as it did). shores: {facing: [(x, z) FFXI centres of our sea-cliff tiles]}. Folders are
    copied (under new names) until every stretch of shore has its share; a generator whose facing
    we don't have is parked far out of sight. Returns (bytes, placed)."""
    secs = [bytearray(x.raw) for x in fz.split_sections(out)]
    kinds = [x[4] & 0x7F for x in secs]
    names = [bytes(x[:4]) for x in secs]

    def span(dirname):
        depth, start, stack = 0, None, []
        for i, k in enumerate(kinds):
            if k == fz.T_DIR:
                if names[i] == dirname and stack and stack[-1] == b"effe" and start is None:
                    start, depth = i, len(stack)
                stack.append(names[i])
            elif k == fz.T_END:
                stack.pop()
                if start is not None and len(stack) == depth:
                    return start, i
        return None

    by_face = {}
    for g in sorted(groups):
        for e in groups[g]:
            if e["face"]:
                by_face.setdefault(e["face"], []).append((g, e["name"]))
    need = [len(shores.get(f, [])) / max(1, len(v)) for f, v in by_face.items() if shores.get(f)]
    copies = max(1, min(copies_max, int(np.ceil(max(need)))) if need else 1)
    spare = iter([bytes([c1, c2]) + b"%02d" % n for c1, c2 in (b"uv", b"uw", b"ux", b"uy") for n in range(100)])
    placed, extra = 0, []
    for g in sorted(groups):
        sp = span(g)
        if sp is None:
            continue
        for k in range(copies):
            block = secs[sp[0]:sp[1] + 1] if k == 0 else [bytearray(x) for x in secs[sp[0]:sp[1] + 1]]
            if k:
                block[0][:4] = next(spare)
            idx = {bytes(sec[:4]): n for n, sec in enumerate(block) if sec[4] & 0x7F == EFFECT}
            for e in groups[g]:
                n = idx.get(e["name"])
                if n is None:
                    continue
                x, y, z = e["pos"]
                targets = shores.get(e["face"]) if e["face"] else None
                if targets:
                    order = by_face[e["face"]].index((g, e["name"]))
                    tx, tz = targets[(k * len(by_face[e["face"]]) + order) % len(targets)]
                    struct.pack_into("<3f", block[n], 16 + e["off"], tx + e["rel"][0], y, tz + e["rel"][1])
                    placed += 1
                else:
                    struct.pack_into("<3f", block[n], 16 + e["off"], 0.0, -5000.0, 0.0)   # out of sight
            if k:
                extra.append((sp[1], block))
    for after, block in sorted(extra, key=lambda t: -t[0]):
        secs[after + 1:after + 1] = block
    return b"".join(bytes(x) for x in secs), placed


# --- a sea of sheets: Qufim's way --------------------------------------------------------------

def lay_sheets(out, dat, name_of, spec, points, beach):
    """Qufim draws its sea as animated sheets (each generator a patch some 400 yalms across) dotted
    round the island, with foam and waves of their own at its one beach. spec (kit coast "sheets"):
    {"folder": "umi1", "beach_tile": "qfim_03", "beach_reach": 40}. The folder's generators within
    beach_reach of the source's beach tile go to ours (beach: its world pose {"pos", "rot", "scale"}),
    keeping their place on the tile; the rest, the sheets, go one to each of `points` (FFXI (x, z)),
    any left over out of sight. Returns (bytes, placed)."""
    tris, objs, _ = T.read(dat)
    src = next(o for o in objs if name_of(o["name"]) == spec["beach_tile"])
    R_src = T._rot_zyx(src["rot"]) * np.array(src["scale"])
    secs = [bytearray(x.raw) for x in fz.split_sections(out)]
    paths = dict(fz.walk_tree(fz.split_sections(out)))
    resources = {bytes(x[:4]) for x in secs if x[4] & 0x7F not in (EFFECT, fz.T_DIR, fz.T_END)}
    folder = spec["folder"].encode()
    reach = spec.get("beach_reach", 40.0)
    placed, sheets = 0, []
    for i, s in enumerate(secs):
        if s[4] & 0x7F != EFFECT or paths.get(i, ())[1:3] != (b"effe", folder):
            continue
        off = _pos_offset(bytes(s[16:]), resources)
        if off is None:
            continue
        p = np.array(struct.unpack_from("<3f", s, 16 + off))
        if math.hypot(p[0] - src["pos"][0], p[2] - src["pos"][2]) <= reach:
            if beach is None:                                                   # no beach of ours: out of sight
                struct.pack_into("<3f", s, 16 + off, 0.0, -5000.0, 0.0)
                continue
            local = np.linalg.solve(R_src, p - np.array(src["pos"]))           # its place on the tile
            R = T._rot_zyx(beach["rot"]) * np.array(beach["scale"])
            q = R @ local + np.array(beach["pos"])
            struct.pack_into("<3f", s, 16 + off, *q)
            placed += 1
        else:
            sheets.append((s, off, p[1]))
    for k, (s, off, y) in enumerate(sheets):
        if k < len(points):
            struct.pack_into("<3f", s, 16 + off, points[k][0], y, points[k][1])
            placed += 1
        else:
            struct.pack_into("<3f", s, 16 + off, 0.0, -5000.0, 0.0)                 # out of sight
    return b"".join(bytes(x) for x in secs), placed


# --- effects from another zone -------------------------------------------------------------------

def _span(kinds, names, parents, name, parent):
    """(start, end) section indices of the directory `name` directly under `parent`."""
    depth, start, stack = 0, None, []
    for i, k in enumerate(kinds):
        if k == fz.T_DIR:
            if names[i] == name and stack and stack[-1] == parent and start is None:
                start, depth = i, len(stack)
            stack.append(names[i])
        elif k == fz.T_END:
            stack.pop()
            if start is not None and len(stack) == depth:
                return start, i
    return None


def graft(out, src, grafts):
    """Copy effect folders from another zone's DAT into ours, their generators moved to our spots:
    grafts [{"folder": "hi01", "at": [[x, y, z] FFXI, ...], "models": ["hi12"]}] (a folder's
    generators take the spots in order, the rest go out of sight; "models": the source's model-folder
    resources the folder draws, copied into ours). A torch's flame from Vunkerl Inlet [S], say.
    Returns (bytes, generators placed)."""
    S = fz.split_sections(bytes(src))
    s_kinds, s_names = [x.kind for x in S], [x.name for x in S]
    s_res = {x.name for x in S if x.kind not in (EFFECT, fz.T_DIR, fz.T_END)}
    secs = [bytearray(x.raw) for x in fz.split_sections(out)]
    kinds = [x[4] & 0x7F for x in secs]
    names = [bytes(x[:4]) for x in secs]
    placed = 0
    effe_blocks, mode_blocks = [], []
    for g in grafts:
        sp = _span(s_kinds, s_names, None, g["folder"].encode(), b"effe")
        if sp is None:
            raise ValueError(f"graft: no effect folder {g['folder']!r} in the source")
        block = [bytearray(x.raw) for x in S[sp[0]:sp[1] + 1]]
        if _span(kinds, names, None, g["folder"].encode(), b"effe"):
            raise ValueError(f"graft: our zone already has an effect folder {g['folder']!r}")
        spots = list(g["at"])
        only = {n.encode() for n in g.get("names", [])}       # just these generators (one fire of many)
        k = 0
        for b in block[1:-1]:
            if b[4] & 0x7F != EFFECT:
                continue
            if only and bytes(b[:4]).rstrip() not in only:
                off = _pos_offset(bytes(b[16:]), s_res)
                if off is not None:
                    struct.pack_into("<3f", b, 16 + off, 0.0, -5000.0, 0.0)        # out of sight
                continue
            off = _pos_offset(bytes(b[16:]), s_res)
            if off is None:
                continue
            xyz = spots[k] if k < len(spots) else (0.0, -5000.0, 0.0)          # spare: out of sight
            struct.pack_into("<3f", b, 16 + off, *list(xyz)[:3])
            placed += k < len(spots)
            k += 1
        effe_blocks += block
        for m in g.get("models", []):
            if m.encode() in names:
                continue                                                     # ours already has it
            hit = next(x for x in S if x.name == m.encode() and x.kind not in (EFFECT, fz.T_DIR, fz.T_END))
            mode_blocks.append(bytearray(hit.raw))
            names.append(m.encode())
    # into our effe and mode folders, before each one's end
    for folder, blocks in ((b"mode", mode_blocks), (b"effe", effe_blocks)):
        if not blocks:
            continue
        kinds = [x[4] & 0x7F for x in secs]
        names = [bytes(x[:4]) for x in secs]
        sp = _span(kinds, names, None, folder, names[0])            # directly under the zone's root
        if sp is None:
            raise ValueError(f"graft: our zone has no {folder!r} folder")
        secs[sp[1]:sp[1]] = blocks
    return b"".join(bytes(x) for x in secs), placed


def _hue_texture(raw, degrees, tint=None):
    """A texture section (0x20) with its colours turned `degrees` round the hue wheel, or with
    `tint` all set to that one hue (DXT or paletted), brightness, saturation and alpha kept."""
    import colorsys
    from zonekit import textures as X
    b = bytearray(raw)
    h0 = 16
    if b[h0] == 0xA1:
        w, h = struct.unpack_from("<II", b, h0 + 0x15)
        kind = bytes(b[h0 + 0x39:h0 + 0x3D]).decode()
        size = struct.unpack_from("<I", b, h0 + 0x3D)[0]
        img = X.decode(bytes(b[h0 + 0x45:h0 + 0x45 + size]), w, h, kind)
        rgb = img[..., :3].astype(np.float64) / 255.0
        out = _hue_rgb(rgb, degrees, tint)
        img = np.concatenate([(out * 255).round().astype(np.uint8), img[..., 3:]], -1)
        enc = X.encode(img, kind)
        b[h0 + 0x45:h0 + 0x45 + len(enc)] = enc
    elif struct.unpack_from("<H", b, h0 + 0x1F)[0] == 8:                  # paletted: turn the palette
        pal = np.frombuffer(bytes(b[h0 + 0x3D:h0 + 0x3D + 1024]), np.uint8).reshape(256, 4).copy()
        rgb = pal[:, [2, 1, 0]].astype(np.float64) / 255.0                  # BGRA
        out = (_hue_rgb(rgb[None], degrees, tint)[0] * 255).round().astype(np.uint8)
        pal[:, 2], pal[:, 1], pal[:, 0] = out[:, 0], out[:, 1], out[:, 2]
        b[h0 + 0x3D:h0 + 0x3D + 1024] = pal.tobytes()
    return b


def _hue_rgb(rgb, degrees, tint=None):
    """(..., 3) RGB in 0..1 turned round the hue wheel (or all set to the hue `tint`)."""
    r, g, bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(-1), rgb.min(-1)
    d = mx - mn
    h = np.zeros_like(mx)
    m = d > 1e-9
    rr = m & (mx == r)
    gg = m & (mx == g) & ~rr
    bb = m & ~rr & ~gg
    h[rr] = ((g - bl)[rr] / d[rr]) % 6
    h[gg] = (bl - r)[gg] / d[gg] + 2
    h[bb] = (r - g)[bb] / d[bb] + 4
    h = ((h * 60 + degrees) if tint is None else np.full_like(h, float(tint))) % 360 / 60
    c = d
    x = c * (1 - np.abs(h % 2 - 1))
    z = np.zeros_like(c)
    k = np.floor(h).astype(int) % 6
    out = np.select([k[..., None] == i for i in range(6)],
                    [np.stack(t, -1) for t in ((c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x))])
    return out + mn[..., None]


def graft_set(out, src, spec):
    """A set piece of effects from another zone, once per spot: spec {"folder": "zcry", "at": [[x, y, z,
    scale], ...] FFXI, "hue": degrees (its generators' colours, and textures unless "tint": one hue for
    the textures), "resources": ["cry_"] (things in the source's effe root it draws,
    copied once, recoloured too)}. Each copy of the folder (named folder[:2] + a number) moves as a
    whole, its generators keeping their offsets from the first (scaled), the generators themselves
    scaled. Returns (bytes, generators placed)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))           # LPS: its particle patcher
    from modules.particles import patch_particle_overrides
    S = fz.split_sections(bytes(src))
    s_kinds, s_names = [x.kind for x in S], [x.name for x in S]
    s_res = {x.name for x in S if x.kind not in (EFFECT, fz.T_DIR, fz.T_END)}
    sp = _span(s_kinds, s_names, None, spec["folder"].encode(), b"effe")
    if sp is None:
        raise ValueError(f"graft_set: no effect folder {spec['folder']!r} in the source")
    hue, tint = spec.get("hue", 0.0), spec.get("tint")
    base = [bytearray(x.raw) for x in S[sp[0]:sp[1] + 1]]
    gens = []
    for n, b in enumerate(base):
        if b[4] & 0x7F == EFFECT:
            off = _pos_offset(bytes(b[16:]), s_res)
            if off is not None:
                gens.append((n, off, np.array(struct.unpack_from("<3f", b, 16 + off))))
            if hue:                                                          # its colours
                one = patch_particle_overrides(bytes(b), [{"id": bytes(b[:4]).decode("ascii", "replace").strip(),
                                                             "overrides": {"hue": hue}}])
                base[n] = bytearray(one)
        elif b[4] & 0x7F == fz.T_TEXTURE and (hue or tint is not None):
            base[n] = _hue_texture(b, hue, tint)
    anchor = gens[0][2]
    blocks, placed = [], 0
    for k, spot in enumerate(spec["at"]):
        x, y, z = spot[:3]
        sc = spot[3] if len(spot) > 3 else 1.0
        block = [bytearray(b) for b in base]
        block[0][:4] = (spec["folder"][:2] + "%02d" % k).encode()
        for n, off, p in gens:
            q = np.array([x, y, z]) + (p - anchor) * sc
            struct.pack_into("<3f", block[n], 16 + off, *q)
            if sc != 1.0:
                one = patch_particle_overrides(bytes(block[n]), [{"id": bytes(block[n][:4]).decode("ascii", "replace").strip(),
                                                                    "overrides": {"scale": sc}}])
                block[n] = bytearray(one)
            placed += 1
        blocks += block
    secs = [bytearray(x.raw) for x in fz.split_sections(out)]
    names = [bytes(x[:4]) for x in secs]
    extra = []
    for r in spec.get("resources", []):
        if r.encode() in names:
            continue
        hit = bytearray(next(x for x in S if x.name == r.encode() and x.kind not in (EFFECT, fz.T_DIR, fz.T_END)).raw)
        extra.append(_hue_texture(hit, hue, tint) if (hue or tint is not None) and hit[4] & 0x7F == fz.T_TEXTURE else hit)
    kinds = [x[4] & 0x7F for x in secs]
    effe = _span(kinds, names, None, b"effe", names[0])
    secs[effe[1]:effe[1]] = extra + blocks
    return b"".join(bytes(x) for x in secs), placed


def adjust_texture(raw, value=1.0, saturation=1.0, lift=0.0):
    """A texture section (DXT or paletted) washed: its colour pulled toward grey (saturation), its
    brightness scaled (value) and lifted toward white (lift 0..1). Weathered bone from a cave's."""
    from zonekit import textures as X
    b = bytearray(raw)
    h0 = 16

    def wash(rgb):
        grey = rgb.mean(-1, keepdims=True)
        out = grey + (rgb - grey) * saturation
        out = out * value
        return np.clip(out + (1.0 - out) * lift, 0.0, 1.0)
    if b[h0] == 0xA1:
        w, h = struct.unpack_from("<II", b, h0 + 0x15)
        kind = bytes(b[h0 + 0x39:h0 + 0x3D]).decode()
        size = struct.unpack_from("<I", b, h0 + 0x3D)[0]
        img = X.decode(bytes(b[h0 + 0x45:h0 + 0x45 + size]), w, h, kind)
        rgb = wash(img[..., :3].astype(np.float64) / 255.0)
        img = np.concatenate([(rgb * 255).round().astype(np.uint8), img[..., 3:]], -1)
        enc = X.encode(img, kind)
        b[h0 + 0x45:h0 + 0x45 + len(enc)] = enc
    else:
        pal = np.frombuffer(bytes(b[h0 + 0x3D:h0 + 0x3D + 1024]), np.uint8).reshape(256, 4).copy()
        rgb = wash(pal[:, [2, 1, 0]].astype(np.float64) / 255.0)
        out = (rgb * 255).round().astype(np.uint8)
        pal[:, 2], pal[:, 1], pal[:, 0] = out[:, 0], out[:, 1], out[:, 2]
        b[h0 + 0x3D:h0 + 0x3D + 1024] = pal.tobytes()
    return bytes(b)

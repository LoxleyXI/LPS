"""Putting it together: the source zone's own DAT (its textures, sky, lighting, effects) with our
tiles and props as placed objects, collision from their triangles (trees only at the trunk), the
source's footstep sounds, invisible fences round the map, and, with a coast, a still sea at the
source's sea level under its moved sea effects. And installing it into a client overlay.
"""
import math
import random
from collections import Counter

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

from . import effects, kit as K, layout as L, props as P, solve as S
from .assemble import (box, brink_fence, fence, laid_meshes, lay_geometry, ours, sea, sea_tiles, water_fence,
                       writer_obj)
from .place import bridges, buildings, obj_of

TILE = T.TILE


def _grid(spec):
    """Corner heights drawn as rows of characters, north row first, one character per corner:
    "." open sea, "#" wall (any cliff top), any other key of spec["heights"] that floor height.
    spec["start"] (i, j) is the way in (a corner, i east, j north from the south row)."""
    from .kit import HOLE
    rows = [r.replace(" ", "") for r in spec["corners"]]
    W, D = len(rows[0]) - 1, len(rows) - 1
    ground = np.full((W + 1, D + 1), np.nan)
    for r, row in enumerate(rows):
        if len(row) != W + 1:
            raise ValueError(f"grid row {r} has {len(row)} corners, not {W + 1}")
        for i, ch in enumerate(row):
            if ch == ".":
                ground[i, D - r] = HOLE
            elif ch != "#":
                ground[i, D - r] = float(spec["heights"][ch])
    i, j = spec["start"]
    return ground, (float(i), float(j), float(ground[i, j]))


def outpost_patch(ground, nets, basins, size):
    """Cells [(i, j)] for an outpost: a rectangle of `size` cells (either way round) on one basin
    floor (every corner at the same height), carrying no road or stream, off the zone's edge,
    nearest the first basin. None if there is none."""
    W, D = ground.shape[0] - 1, ground.shape[1] - 1
    busy = {c for cells in nets.values() for c in cells}
    b0 = basins[0]
    best = None
    for a, b in {tuple(size), tuple(size[::-1])}:
        for i0 in range(1, W - a):
            for j0 in range(1, D - b):
                block = ground[i0:i0 + a + 1, j0:j0 + b + 1]
                if np.isnan(block).any() or np.ptp(block) > 0:
                    continue
                cells = [(i, j) for i in range(i0, i0 + a) for j in range(j0, j0 + b)]
                if any(c in busy for c in cells):
                    continue
                d = (i0 + a / 2 - b0[0]) ** 2 + (j0 + b / 2 - b0[1]) ** 2
                if best is None or d < best[0]:
                    best = (d, cells)
    return best and best[1]


def lay(base_dat, settings, name_of, kit=None):
    """The land of an "outdoor" block, before anything stands on it: the layout from its seed and
    the kit's tiles laid on it (generate() goes on from here; outdoors/preview.py stops here).
    Returns a dict of what generate() uses: cfg, mp, win, city_map, rng, W, D, coast, kit, slopes,
    crossings, ground, basins, bays, nets, placed, attempt, patch (the outpost's cells, or None)."""
    win = city_map = patch = None
    cfg = K.config(settings["kit"])
    mp = settings.get("map")
    if mp:
        # a window onto a map (maps/<name>.json): flat land over it, the map's city built on it
        from . import city
        city_map = city.load(mp["file"])
        win = city.window(city_map, mp["window"])
        cfg = {k: v for k, v in cfg.items() if k not in ("coast", "river")}
    if "prop_density" in settings:
        cfg.setdefault("props", {})["density"] = settings["prop_density"]
    rng = random.Random(settings.get("seed", 1))
    grid = settings.get("grid")
    if grid:                                            # corners drawn by hand (see _grid)
        g_ground, g_start = _grid(grid)
        W, D = g_ground.shape[0] - 1, g_ground.shape[1] - 1
    else:
        W, D = (win[2], win[3]) if mp else settings.get("size", [16, 14])
    levels = list(cfg["levels"])
    if cfg.get("river"):                                # the river's water level: its pieces go there too
        levels.append(cfg["levels"][0] - cfg["river"].get("depth", cfg["step"]))
    coast = cfg.get("coast") or {}
    if coast.get("island"):
        levels.append(float(coast.get("cliff_base", cfg["levels"][0] - 32)))   # sea cliffs stand on it
    kit = kit or K.load(base_dat, name_of, levels=levels, holes=bool(coast.get("island")),
                        compound_tiles=cfg.get("compound", False), open_below=coast.get("open_below"))
    kit["from_zone"] = cfg["from"]                      # the base zone: its own models need no copying
    for attempt in range(300):                          # layouts from the seed until one the kit can lay
        slopes, crossings = {}, []
        if grid:
            ground, basins, bays, nets = g_ground.copy(), [g_start], [], {}
        elif mp:
            ground = np.full((W + 1, D + 1), float(levels[0]))
            basins = [(-win[0] / TILE, -win[1] / TILE, levels[0])]   # the map's origin: the zone's
            bays, nets = [], {}
        elif cfg.get("layout") == "terraces":
            ground, basins, slopes, crossings = L.terraces(W, D, cfg, rng, settings.get("basins", 6))
            if not slopes:
                continue
            bays, nets = [], {}
            if cfg.get("river"):
                water = cfg["levels"][0] - cfg["river"].get("depth", cfg["step"])
                nets["river"] = L.water_links(ground, water)
        elif settings.get("layout") == "fields" and not cfg.get("coast"):
            ground, basins, bays, nets = L.fields(W, D, cfg, rng, settings.get("basins", 4), settings.get("fields"))
        else:
            ground, basins, bays, nets = L.layout(W, D, cfg, rng, settings.get("basins", 5))
        solve_cfg = cfg
        if settings.get("outpost") and not mp:
            patch = outpost_patch(ground, nets, basins, settings["outpost"].get("size", [3, 2]))
            if patch is None:
                continue                                # no basin floor big enough: another layout
            solve_cfg = {**cfg, "flat_cells": patch, "flat_tol": settings["outpost"].get("flat_tol", 2.0)}
        require = {tuple(int(v) for v in k.split(",")): t for k, t in (grid or {}).get("tiles", {}).items()}
        skip = {(i, j) for i0, j0, i1, j1 in (grid or {}).get("reserve", [])
                for i in range(i0, i1 + 1) for j in range(j0, j1 + 1)}
        placed, missing = S.solve(ground, kit, solve_cfg, rng, nets, slopes, require=require, skip=skip)
        if not missing and cfg.get("layout") == "terraces" and not mp:
            start = (int(round(basins[0][0])), int(round(basins[0][1])))
            if S.reachable(S.solve.repaired, placed, cfg, start, below=cfg["levels"][0],
                           joins=[c["join"] for c in crossings]) < cfg.get("reach", 0.97):
                missing = [(-1, -1, ["a terrace can't be walked to"], 0)]
        if not missing:
            ground = S.solve.repaired
            break
        if grid and missing[0][0] >= 0:
            # a drawn grid the kit can't lay as drawn: one land corner of the cell that failed becomes
            # a cliff top or takes a neighbour's height (the sea and the corners marked fixed stay)
            i, j = missing[0][:2]
            fixed = grid.get("fixed", [])
            opts = [(x, y) for x, y in ((i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1))
                    if 0 < x < W and 0 < y < D and g_ground[x, y] > -999 and [x, y] not in fixed]
            if opts:
                x, y = rng.choice(opts)
                near = [g_ground[a, b] for a, b in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1))
                        if not np.isnan(g_ground[a, b]) and g_ground[a, b] > -999 and g_ground[a, b] != g_ground[x, y]]
                if np.isnan(g_ground[x, y]):
                    if near:
                        g_ground[x, y] = rng.choice(near)
                else:
                    g_ground[x, y] = rng.choice(near + [np.nan])
    else:
        raise ValueError(f"no layout the kit can lay in 300 tries (last: {missing[:5]})")
    return {"cfg": cfg, "mp": mp, "win": win, "city_map": city_map, "rng": rng, "W": W, "D": D,
            "coast": coast, "kit": kit, "slopes": slopes, "crossings": crossings, "ground": ground,
            "basins": basins, "bays": bays, "nets": nets, "placed": placed, "attempt": attempt, "patch": patch,
            "skip": skip}


def generate(base_dat, settings, name_of, report=None, kit=None):
    """The zone DAT (bytes) for an "outdoor" block ({"kit": name, "seed", "size", "basins", "draw",
    "prop_density"}), plus a summary."""
    land = lay(base_dat, settings, name_of, kit)
    cfg, mp, win, city_map, rng, W, D, coast, kit = (land[k] for k in ("cfg", "mp", "win", "city_map", "rng", "W", "D", "coast", "kit"))
    slopes, crossings, ground, basins, bays, nets, placed, attempt, skip = (
        land[k] for k in ("slopes", "crossings", "ground", "basins", "bays", "nets", "placed", "attempt", "skip"))
    if mp:
        from . import city
    rules = P.learn(kit, name_of)
    spans = bridges(placed, kit, cfg, crossings, name_of)
    props = P.place(placed, kit, rules, cfg, rng, obj_of)
    props = [p for p in props if not any(abs(p["pos"][0] - x) < r and abs(p["pos"][2] - z) < r for x, z, r in spans["keep_clear"])]
    if mp:
        from .place import sources
        houses, borrowed = city.place(city_map, win, placed, kit, rng, name_of, base_dat, settings.get("city", {}),
                                      sources(kit, base_dat))
        # wild scatter outside the town; inside it only greenery (the kit's trees, shrubs and flowers,
        # not its boulders), off the streets and the square, clear of the walls
        lines, width = houses["clear_lines"]
        green = settings.get("city", {}).get("greenery", {})
        not_green = [n.encode() for n in green.get("not", ["fa_iwa"])]
        sq = houses["square"]

        def keep(p):
            x, z = p["pos"][0], p["pos"][2]
            if houses["on_rock"](x, z) or city.near_line(x, z, lines)[0] <= width / 2 + 2:
                return False
            if not city.inside(x, z, houses["town"]):
                return True
            return (green.get("in_town", True) and not any(p["name"].startswith(n) for n in not_green)
                    and city.near_line(x, z, lines)[0] > width / 2 + green.get("street_gap", 6.0)   # doorways face the streets
                    and city.near_line(x, z, houses["walls"])[0] > green.get("wall_gap", 10.0)
                    and math.hypot(x - sq[0], z - sq[1]) > sq[2] + 3)
        props = [p for p in props if keep(p)]
        if green.get("density"):                            # a denser pass that only fills the town
            cfg2 = dict(cfg, props={**cfg.get("props", {}), "density": green["density"]})
            more = P.place(placed, kit, rules, cfg2, random.Random(settings.get("seed", 1) + 7), obj_of)
            more = [p for p in more if city.inside(p["pos"][0], p["pos"][2], houses["town"]) and keep(p)
                    and all(math.hypot(p["pos"][0] - q["pos"][0], p["pos"][2] - q["pos"][2]) > 5 for q in props)]
            props += more
    else:
        houses, borrowed = buildings(settings.get("buildings"), placed, kit, nets, rng, name_of, spans["keep_clear"], base_dat)
    shift_xz = (round(-basins[0][0]) * TILE, round(-basins[0][1]) * TILE)    # the way in on the origin (as below)

    def clear(p):
        x, z = p["pos"][0], p["pos"][2]
        if mp and houses.get("rects") and city.inside(x, z, houses["town"]):
            # in the town: off the houses' own footprints (greenery grows between them)
            gap = settings.get("city", {}).get("greenery", {}).get("house_gap", 5.0)
            return not any(city.inside(x, z, r) or city.near_line(x, z, [list(r) + [r[0]]])[0] < gap
                           for r in houses["rects"])
        return not any(math.hypot(x - cx_, z - cz_) < r + 2 for cx_, cz_, r in houses["keep_clear"])
    props = [p for p in props if clear(p)]
    if settings.get("scatter_clear"):                    # the source's own scatter kept off them too
        props = [p for p in props if not any(math.hypot(p["pos"][0] - (x - shift_xz[0]), p["pos"][2] - (z - shift_xz[1])) < r
                                             for x, z, r in settings["scatter_clear"])]
    for i0, j0, i1, j1, _ in houses["sites"]:           # the village's streets and yards stay clear
        props = [p for p in props if not (i0 * TILE - 4 <= p["pos"][0] <= i1 * TILE + 4 and j0 * TILE - 4 <= p["pos"][2] <= j1 * TILE + 4)]

    # the way in (the first basin) sits on the origin; the grid keeps the source's alignment
    shift = np.round(np.array([-basins[0][0] * TILE, 0.0, -basins[0][1] * TILE]) / TILE) * TILE
    off = settings.get("_offset")                       # a region's island (regions.py), in yalms
    if off:
        shift = shift + np.round(np.array([off[0], 0.0, off[1]]) / TILE) * TILE

    objects, col = [], []
    used = set()
    for k, c in placed.items():
        o = obj_of(c, k)
        o["pos"] = tuple(np.array(o["pos"]) + shift)
        for part in kit.get("parts", {}).get(c["name"], [c["name"]]):   # a compound tile: each of its meshes
            objects.append(writer_obj(kit, part, o, lod=(1000.0, 5000.0, 5000.0)))
            used.add(part)
        col.append(ours(T.placed(kit["tris"][c["name"]], o).reshape(-1, 3, 3), o["scale"]))
    for part in (spans["borrowed"],):                   # models from other zones, all in one table
        for k in ("tris", "textures"):
            borrowed[k] = {**borrowed[k], **part[k]}
        borrowed["sections"] = borrowed["sections"] + part["sections"]
    if settings.get("max_texture"):                       # a zone file has a size the client won't
        from zonekit import textures as X                         # load past (24 MiB): textures capped
        borrowed["textures"] = {k: X.cap(v, settings["max_texture"]) for k, v in borrowed["textures"].items()}
    if settings.get("scatter"):                           # other zones' props on our land (props.scatter)
        avoid = [(x - shift[0], z - shift[2], r) for x, z, r in settings.get("scatter_clear", [])]
        more, got = P.scatter(placed, kit, settings["scatter"], random.Random(settings.get("seed", 1) + 11),
                              obj_of, avoid + [(p["pos"][0], p["pos"][2], 2.0) for p in props])
        borrowed["tris"].update(got["tris"])
        if settings.get("max_texture"):
            from zonekit import textures as X
            got["textures"] = {k: X.cap(v, settings["max_texture"]) for k, v in got["textures"].items()}
        borrowed["textures"].update(got["textures"])
        borrowed["sections"] = borrowed["sections"] + got["sections"]
        scattered = more
    else:
        scattered = []
    for spot in settings.get("placed", []):              # single borrowed models, placed by hand
        from zonekit import borrow
        import caves
        loaded, tex = borrow.load_kit(caves.zone_dat(spot["from"]).read_bytes(), [spot["name"]])
        piece = next(iter(loaded.values()))
        n16 = piece["name16"]
        if spot.get("tint") is not None:                 # its textures turned one hue (a green crystal)
            tex = {k: (bytes(effects._hue_texture(v, 0.0, spot["tint"])) if k.split()[-1].decode() in spot.get("tint_only", [k.split()[-1].decode()]) else v)
                   for k, v in tex.items()}
        for short, (zid, other) in spot.get("swap", {}).items():
            # a texture of its own drawn with another zone's instead (Zi'Tah's forest floor round a
            # crystal's foot, as Qufim's snow): that texture under this one's name
            key = next(k for k in tex if k.split()[-1].decode() == short)
            src = next(s_ for s_ in fz.split_sections(caves.zone_dat(zid).read_bytes())
                       if s_.kind == fz.T_TEXTURE and s_.raw[0x11:0x21].split()[-1].decode() == other)
            raw = bytearray(src.raw)
            raw[0x11:0x21] = key
            tex[key] = bytes(raw)
        if settings.get("max_texture"):
            from zonekit import textures as X
            tex = {k: X.cap(v, settings["max_texture"]) for k, v in tex.items()}
        if n16 not in borrowed["tris"]:
            borrowed["tris"][n16] = piece["tris"]
            borrowed["sections"] = borrowed["sections"] + [piece["section"]]
            borrowed["textures"].update(tex)
        x, h, z = spot["pos"]
        sc = spot.get("scale", 1.0)
        if h is None:                                    # on the ground there, by its own lowest point,
            gr = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
            up = -piece["tris"].reshape(-1, 3)[:, 1]     # with a share of its height buried
            h = (float(gr.ring(x - shift[0], z - shift[2], 2.0).min()) - float(up.min()) * sc
                 - spot.get("embed", 0.15) * float(up.max() - up.min()) * sc)
        scattered.append({"name": n16, "pos": (x - shift[0], -h, z - shift[2]), "yaw": spot.get("yaw", 0.0),
                          "scale": (-sc if spot.get("mirror") else sc, sc, sc), "solid": spot.get("solid", False)})
    gr2 = None                                           # the laid ground, made when a set first needs it
    for st in settings.get("sets", []):                  # a group of another zone's objects, as they stand
        from zonekit import borrow
        import caves
        sdat = caves.zone_dat(st["from"]).read_bytes()
        _, sobjs, _ = T.read(sdat)
        sx, sy, sz = st["source"]
        group = [o for o in sobjs if math.hypot(o["pos"][0] - sx, o["pos"][2] - sz) <= st.get("radius", 6.0)
                 and not any(o["name"].startswith(k.encode()) for k in st.get("skip", ["hit_"]))
                 and (not st.get("only") or any(o["name"].startswith(k.encode()) for k in st["only"]))]
        loaded, tex = borrow.load_kit(sdat, sorted({o["name"].split(b"\0")[0].decode("ascii", "replace").rstrip() for o in group}))
        if settings.get("max_texture"):
            from zonekit import textures as X
            tex = {k: X.cap(v, settings["max_texture"]) for k, v in tex.items()}
        by16 = {p["name16"]: p for p in loaded.values()}
        for n16, p in by16.items():
            if n16 not in borrowed["tris"]:
                borrowed["tris"][n16] = p["tris"]
                borrowed["sections"] = borrowed["sections"] + [p["section"]]
        borrowed["textures"].update(tex)
        x, h, z = st["at"]
        if h is None:
            gr = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
            h = float(gr.ring(x - shift[0], z - shift[2], 2.0).min())
        yaw = st.get("yaw", 0.0)
        cy, sy_ = math.cos(yaw), math.sin(yaw)
        for o in group:
            n16 = o["name"] if o["name"] in by16 else next((k for k in by16 if k.rstrip() == o["name"].rstrip()), None)
            if n16 is None:
                continue
            dx, dy, dz = o["pos"][0] - sx, o["pos"][1] - sy, o["pos"][2] - sz
            rx, rz = dx * cy + dz * sy_, -dx * sy_ + dz * cy           # turned with the set (FFXI yaw)
            py = -h + dy
            if st.get("settle"):                         # each piece on our ground by its own lowest point
                if gr2 is None:
                    gr2 = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
                g = gr2.ring(x + rx - shift[0], z + rz - shift[2], 1.0)
                if not np.isnan(g).all():
                    up = -by16[n16]["tris"].reshape(-1, 3)[:, 1] * o["scale"][1]
                    py = -(float(np.nanmin(g)) - float(up.min()) - st.get("embed", 0.15) * float(up.max() - up.min()))
            scattered.append({"name": n16, "pos": (x + rx - shift[0], py, z + rz - shift[2]),
                              "yaw": o["rot"][1] + yaw, "scale": tuple(o["scale"]),
                              "solid": any(o["name"].startswith(k.encode()) for k in st.get("solid", []))})
    adjust = settings.get("texture_adjust") or {}
    if adjust:                                           # borrowed textures washed (cave-lit bone, weathered)
        borrowed["textures"] = {k: (effects.adjust_texture(v, **adjust[k.split()[-1].decode()])
                                    if k.split()[-1].decode() in adjust else v)
                                for k, v in borrowed["textures"].items()}
    both = {"tris": {**kit["tris"], **borrowed["tris"]}}
    for b in spans["pieces"]:                           # bridges: walked on, so all of each is solid
        o = {"rot": b["rot"], "scale": b["scale"], "pos": tuple(np.array(b["pos"]) + shift)}
        objects.append(writer_obj(both, b["name"], o, lod=(1000.0, 5000.0, 5000.0)))
        used.add(b["name"])
        col.append(ours(T.placed(both["tris"][b["name"]], o).reshape(-1, 3, 3), o["scale"]))
    built = borrowed.pop("built", None)
    laid = {}                                           # built pieces as world geometry, by texture
    for b in houses["pieces"]:                          # buildings (see _buildings for what's solid)
        o = {"rot": b["rot"], "scale": b["scale"], "pos": tuple(np.array(b["pos"]) + shift)}
        if built and b["name"] in built["meshes"]:
            lay_geometry(built["meshes"][b["name"]], o, built["full"], laid)
        else:
            objects.append(writer_obj(both, b["name"], o, lod=(1000.0, 5000.0, 5000.0)))
            used.add(b["name"])
        if b.get("col") == "none":                      # out of reach (the castles on the crags)
            continue
        if b.get("col") == "base" and "ground" in b:
            solid = box(both["tris"][b["name"]], o, b["ground"])   # already in the cave frame
            if len(solid):
                col.append(solid)
        else:
            col.append(ours(T.placed(both["tris"][b["name"]], o).reshape(-1, 3, 3), o["scale"]))
    for mesh, o, solid in (built or {}).get("extra", []):  # the city's own geometry: paving, rock
        o = dict(o, pos=tuple(np.array(o["pos"]) + shift))
        lay_geometry(mesh, o, built["full"], laid)
        if solid:                                           # the rock you walk into
            for pos, nrm, uv, colr, tri in mesh.arrays().values():
                w = (pos * np.array([1.0, -1.0, 1.0]))[tri] + np.array(o["pos"])   # FFXI: y down
                col.append(ours(w, (1.0, 1.0, 1.0)))
    for p in scattered:                                     # borrowed: in the props block, not the base's meshes
        o = {"rot": (0.0, p["yaw"], 0.0), "scale": p["scale"], "pos": tuple(np.array(p["pos"]) + shift)}
        objects.append(writer_obj(both, p["name"], o, lod=(10.0, 150.0, 400.0)))
        if p["solid"]:
            w = T.placed(both["tris"][p["name"]], o).reshape(-1, 3, 3)
            low = (-w[..., 1]).min(1) < -o["pos"][1] + 3.0
            t = both["tris"][p["name"]].reshape(-1, 3)
            if P.tall(t.max(0) - t.min(0)):                  # a tree: its trunk, not its lowest boughs
                d = np.hypot(w[..., 0] - o["pos"][0], w[..., 2] - o["pos"][2]).max(1)
                low &= d < 1.2 * o["scale"][0]
            col.append(ours(w[low], o["scale"]))
    for p in props:
        o = {"rot": (0.0, p["yaw"], 0.0), "scale": p["scale"], "pos": tuple(np.array(p["pos"]) + shift)}
        objects.append(writer_obj(kit, p["name"], o, lod=(10.0, 150.0, 400.0)))
        used.add(p["name"])
        if p["solid"]:
            w = T.placed(kit["tris"][p["name"]], o).reshape(-1, 3, 3)
            low = (-w[..., 1]).min(1) < -o["pos"][1] + 3.0      # only what you walk into: trunks and
            col.append(ours(w[low], o["scale"]))               # bases, not canopies overhead
    piece_tex = {}
    for piece in settings.get("pieces", []):            # our own pieces, placed by hand (architecture.py)
        from . import architecture as A
        from .scene import textures_by_short
        kw = piece.get("args", {})
        pal = A.palette(piece.get("palette", "ronfaure"))
        mesh = A.build(piece["piece"], pal, **kw)
        px, h, pz = piece["pos"]
        o = {"rot": (0.0, piece.get("yaw", 0.0), 0.0), "scale": (1.0, 1.0, 1.0), "pos": (px, -h, pz)}
        full = textures_by_short(base_dat)
        for zid, short in [(pal["from"], pal["texture"])] + [(pal[k].get("from", pal["from"]), pal[k]["texture"])
                                                             for k in ("door", "banner") if pal.get(k)]:
            if short in full:
                continue                                 # the palette's textures from their own zones
            import caves
            sec = next(s_ for s_ in fz.split_sections(caves.zone_dat(zid).read_bytes())
                       if s_.kind == fz.T_TEXTURE and s_.raw[0x11:0x21].split()[-1:] == [short.encode()])
            raw = sec.raw
            if settings.get("max_texture"):
                from zonekit import textures as X
                raw = X.cap(raw, settings["max_texture"])
            piece_tex[sec.raw[0x11:0x21]] = raw
            full[short] = sec.raw[0x11:0x21]
        lay_geometry(mesh, o, full, laid)
        solid = getattr(A, piece["piece"] + "_collision", None)
        if solid:
            keep = {k: kw[k] for k in ("length", "width", "deck") if k in kw}
            col.append(ours(T.placed(A.triangles(solid(**keep)), o).reshape(-1, 3, 3), o["scale"]))
        elif piece.get("solid"):                         # walls and towers: all of it
            col.append(ours(T.placed(A.triangles(mesh), o).reshape(-1, 3, 3), o["scale"]))
    if coast.get("seabed"):                              # a floor under the sea (assemble.seabed)
        from .assemble import seabed
        from .scene import textures_by_short
        ashore = [((i + 0.5) * TILE + shift[0], (j + 0.5) * TILE + shift[2]) for (i, j) in placed
                if np.nanmax(ground[i:i + 2, j:j + 2]) > 0] + \
               [((i + 0.5) * TILE + shift[0], (j + 0.5) * TILE + shift[2]) for (i, j) in skip]
        m = coast["seabed"].get("margin", 240.0)
        bed, solid = seabed(ashore, (shift[0] - m, shift[2] - m, shift[0] + W * TILE + m, shift[2] + D * TILE + m),
                            coast["seabed"])
        o = {"rot": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0), "pos": (0.0, 0.0, 0.0)}
        lay_geometry(bed, o, textures_by_short(base_dat), laid)    # seen, not walked: the fences keep
                                                                    # players off the sea (its collision
                                                                    # filled the client's grid cells)
    crystal_meshes, crystal_tex = [], {}
    if settings.get("crystals"):                         # giant glowing crystals (crystals.py)
        from . import crystals
        gr = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
        crystal_meshes, solid, crystal_tex = crystals.build(
            settings["crystals"], lambda x, z: float(gr.ring(x - shift[0], z - shift[2], 3.0).min()))
        col.append(solid)
    for spec in settings.get("rocks", []):              # our own rock in reserved cells (tunnel.py)
        from . import tunnel
        from .scene import textures_by_short
        mesh, solid = tunnel.build(spec)
        o = {"rot": (0.0, 0.0, 0.0), "scale": (1.0, 1.0, 1.0), "pos": (0.0, 0.0, 0.0)}
        lay_geometry(mesh, o, textures_by_short(base_dat), laid)
        col.append(ours(solid, o["scale"]))
    tris = np.concatenate(col)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    fences = fence(W, D, shift)
    if coast.get("island"):
        # shores you walk down to the water (kit "walk_down": a beach) get no fence at the cliff top,
        # but one along their open sides, at the water's edge, with gaps where a piece leaves it
        walk = tuple(coast.get("walk_down", ())) or ("\0",)
        cliffs = {k: c for k, c in placed.items() if not c["label"].startswith(walk)}
        beaches = {k: c for k, c in placed.items() if c["label"].startswith(walk)}
        fences = np.concatenate([fences, brink_fence(ground, cliffs, kit, shift, gaps=spans["ends"]),
                                 water_fence(beaches, kit, shift, settings.get("fence_gaps", []))])
    if settings.get("fences"):                          # by hand: a polyline, or {"line", "low", "high"}
        from .scene import _fence
        fences = np.concatenate([fences] + [_fence(f["line"], f.get("low", -40.0), f.get("high", 120.0))
                                            if isinstance(f, dict) else _fence(f) for f in settings["fences"]])
    tris = np.concatenate([tris, fences])
    up = np.concatenate([n[:, 2], np.zeros(len(fences))])
    terr = kit["terrain"]
    sea_spec = coast.get("sea_tiles")
    if sea_spec:
        for o in sea_tiles(ground, kit, sea_spec, shift, W, D, rng, name_of):
            objects.append(writer_obj(kit, o["name"], o, lod=(1000.0, 5000.0, 5000.0)))
            used.add(o["name"])
    surface = cfg.get("coast") and not coast.get("sheets") and (not sea_spec or coast.get("surface"))
    geometry = {"meshes": ([sea(W, shift, cfg, D if coast.get("island") else None)] if surface else []) +
                          laid_meshes(laid) + crystal_meshes,
                "extra_textures": {**({k: (X.cap(v, settings["max_texture"]) if settings.get("max_texture") else v)
                                       for k, v in built["textures"].items()} if built else {}), **crystal_tex, **piece_tex},
                "collision": {"positions": tris.reshape(-1, 3).astype(np.float32),
                              "triangles": np.arange(len(tris) * 3).reshape(-1, 3).astype(np.int32),
                              "wall": np.abs(up) < 0.5,
                              "terrain": np.where(up > 0.6, terr["floor"], terr["wall"]).astype(np.uint32)},
                "props": {"sections": borrowed["sections"], "textures": borrowed["textures"], "objects": objects}}
    def summarise(moved, dropped):
        return {"cells": len(placed), "props": len(props), "attempts": attempt + 1,
                "bridges": spans["made"], "buildings": houses["made"],
                "town": (houses["box"] or houses.get("town")) and {"sites": houses["sites"],
                                           "spots": [(x + shift[0], -h, z + shift[2], yaw) for x, z, yaw, h in houses["spots"]],
                                           "fire": houses.get("fire") and (houses["fire"][0] + shift[0], -houses["fire"][2],
                                                                           houses["fire"][1] + shift[2]),
                                           "camp": houses.get("camp")},
                "roads": sorted({c for cells in nets.values() for c in cells}
                                | {c for c, v in placed.items() if "miti" in (v.get("label") or "")}),
                "effects_moved": moved, "effects_dropped": dropped,
                "basins": [(round(a, 1), round(b, 1), lv) for a, b, lv in basins], "bays": bays,
                "patch": land["patch"],
                "networks": {k: len(v) for k, v in nets.items()},
                "tiles": Counter(c["label"] for c in placed.values()).most_common(),
                "ground": ground, "placed": placed, "shift": tuple(float(v) for v in shift),
                # the laid tiles' real surface, for standing monsters on (mobs.py)
                "surface": P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3)
                                    for k, c in placed.items()})}

    if settings.get("_parts"):
        # a region of another zone (regions.py): its pieces, for the main zone to take in
        own = {n for n in used if n in kit["tris"]}
        return {"geometry": geometry, "used": own, "summary": summarise(0, [])}
    regions = settings.get("regions")
    if regions:
        from . import regions as R
        import caves
        region_summaries = R.merge(geometry, regions, name_of, generate, caves.zone_dat, K.config,
                                   settings.get("max_texture"))
    stats = report if report is not None else {}
    drop, groups = effects.plan(base_dat, name_of, cfg)
    sheets = coast.get("sheets")
    if sheets:                                           # the source's sea sheets stay, to be laid on ours
        drop = [f for f in drop if f != (b"effe", sheets["folder"].encode())]
    writer_base = base_dat
    if settings.get("max_texture"):
        writer_base = X.cap(base_dat, settings["max_texture"])     # the base zone's own too
    out = fz.build_zone(writer_base, geometry, draw=settings.get("draw", 400), keep_meshes=used,
                        report=stats, drop_dirs=tuple(drop))
    moved = 0
    if sheets:
        from .kit import HOLE
        step, margin = sheets.get("spacing", 220.0), sheets.get("margin", 200.0)

        def at_sea(x, z):
            k = (math.floor((x - shift[0]) / TILE), math.floor((z - shift[2]) / TILE))
            if k not in placed:
                return True
            i, j = k
            return (ground[i:i + 2, j:j + 2] <= HOLE / 2).any()
        pts = [(x, z) for x in np.arange(shift[0] - margin, shift[0] + W * TILE + margin + 1, step)
               for z in np.arange(shift[2] - margin, shift[2] + D * TILE + margin + 1, step) if at_sea(x, z)]
        beach = next(({**obj_of(c, k), "pos": tuple(np.array(obj_of(c, k)["pos"]) + shift)}
                      for k, c in placed.items() if c["label"] == sheets["beach_tile"]), None)
        out, moved = effects.lay_sheets(out, base_dat, name_of, sheets, pts, beach)
    ground_at = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
    for g in settings.get("grafts", []):                # effects from other zones (effects.graft, graft_set)
        import caves
        at = []
        for spot in g["at"]:
            x, h, z = spot[:3]
            if h is None:                                # on the ground there
                h = float(ground_at.at(x - shift[0], z - shift[2])[0])
            lift = spot[4] if len(spot) > 4 else 0.0     # and this far above it (a crystal's tip)
            at.append([x, -(h + lift), z] + list(spot[3:4]))
        src = caves.zone_dat(g["from"]).read_bytes()
        if g.get("whole"):
            out, n = effects.graft_set(out, src, {**g, "at": at})
        else:
            out, n = effects.graft(out, src, [{**g, "at": at}])
        moved += n
    if coast.get("island") and coast.get("animate"):
        # the source's own sea effects, each by a shore of ours facing the way its own did
        from .kit import HOLE
        shores = {}
        side_of = {"n": (0, 1), "e": (1, 2), "s": (2, 3), "w": (3, 0)}
        for (i, j), c in placed.items():
            g4 = [ground[i, j + 1], ground[i + 1, j + 1], ground[i + 1, j], ground[i, j]]
            hole = [x <= HOLE / 2 for x in g4]
            if not any(hole) or all(hole):
                continue
            open_sides = [f for f, (a, b) in side_of.items() if hole[a] and hole[b]]
            for f in open_sides[:1]:
                shores.setdefault(f, []).append(((i + 0.5) * TILE + shift[0], (j + 0.5) * TILE + shift[2]))
        fx = effects.plan_by_facing(base_dat, name_of, cfg)
        if fx:
            out, moved = effects.move_by_facing(out, fx, shores)
    if coast.get("island") and groups:
        # the island's south shore, in bay-sized reaches
        south = sorted(k for k, c in placed.items() if c["label"].startswith(coast["beach_tile"]))
        row = min((k[1] for k in south), default=None)
        xs = sorted(k[0] for k in south if k[1] == row)
        if xs:
            n = max(1, round((xs[-1] - xs[0] + 1) / coast.get("reach", 6)))
            edges = np.linspace(xs[0], xs[-1] + 1, n + 1)
            bays = [(int(round(a)), int(round(b))) for a, b in zip(edges, edges[1:])]
    if cfg.get("coast") and bays and groups:
        beach = [k for k, c in placed.items() if c["label"].startswith(cfg["coast"]["beach_tile"])]
        zb = float(np.mean([(k[1] + 0.5) * TILE + shift[2] for k in beach]))
        reaches = [(a * TILE + shift[0], b * TILE + shift[0]) for a, b in bays]
        out, moved = effects.move(out, groups, reaches, zb, cfg)
    summary = summarise(moved, sorted(set(stats.get("dropped_dirs") or [])))
    if regions:
        summary["regions"] = region_summaries
    folk = (settings.get("townsfolk") or {}).get("spots")
    if folk:                                             # people standing where the zone file says
        spots = []
        for x, h, z, yaw in folk:
            if h is None:
                h = float(ground_at.at(x - shift[0], z - shift[2])[0])
            spots.append((x, -h, z, yaw))
        summary["town"] = {"sites": [], "spots": spots}
    return out, summary

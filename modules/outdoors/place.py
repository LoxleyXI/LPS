"""What an outdoor zone places beyond its land: bridges across its gaps, and buildings (houses along
roads, or towns on its level terraces with streets, walls, towers and gates), from this zone's own
models, another zone's, or our own (architecture.py)."""
import math

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

TILE = T.TILE

from . import props as P


def obj_of(c, k):
    i, j = k
    return {"rot": c["rot"], "scale": c["scale"], "pos": ((i + 0.5) * TILE, -c["base"], (j + 0.5) * TILE)}


def source_ground(tris, objs, tile_names=None):
    """Heights of a zone's own laid tiles."""
    src = {}
    for o in objs:
        if o["name"] in tris and (o["name"] in tile_names if tile_names is not None else T.is_tile(tris[o["name"]])):
            k = (int(np.floor(o["pos"][0] / TILE)), int(np.floor(o["pos"][2] / TILE)))
            w = T.placed(tris[o["name"]], o).reshape(-1, 3, 3)
            src[k] = w if k not in src else np.concatenate([src[k], w])
    return P.Ground(src)


def sources(kit, base_dat):
    """source(zone) -> (tris, objs, dat) of a zone, cached; None or the base zone's own id: the base."""
    import caves
    zones = {}

    def source(zid):
        if zid is None or zid == kit.get("from_zone"):
            return kit["tris"], kit["objs"], base_dat
        if zid not in zones:
            dat = caves.zone_dat(zid).read_bytes()
            tris, objs, _ = T.read(dat)
            zones[zid] = (tris, objs, dat)
        return zones[zid]
    return source


def bridges(placed, kit, cfg, crossings, name_of):
    """The layout's crossings laid with bridges (the source's own, or another zone's), each the one
    built for the gap nearest ours, fitted to it on the laid tiles. Returns {"pieces", "made",
    "keep_clear", "ends", "borrowed": {"tris", "sections", "textures"}}."""
    import caves
    from zonekit import borrow as CP
    from . import bridges as B
    out = {"pieces": [], "made": [], "keep_clear": [], "ends": [],
           "borrowed": {"tris": {}, "sections": [], "textures": {}}}
    spec = cfg.get("bridges")
    if not spec or not crossings:
        return out
    others = {}

    def source(zid):
        if zid not in others:
            dat = caves.zone_dat(zid).read_bytes()
            tris, objs, _ = T.read(dat)
            others[zid] = ({"tris": tris, "objs": objs}, source_ground(tris, objs), dat)
        return others[zid][:2]
    learned = B.learn(kit, name_of, spec, source_ground(kit["tris"], kit["objs"], kit["tile_names"]), source)
    ours = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
    used = {}
    for c in crossings:
        opts = learned.get(c["kind"])
        if not opts:
            continue
        cx, cz = c["at"][0] * TILE, c["at"][1] * TILE
        dx, dz = c["dir"]
        gap = B.span(ours, cx, cz, dx, dz, c["level"])
        if gap is None:
            continue
        bridge = min(opts, key=lambda b: abs(math.log(sum(gap) / b["gap"])))
        pieces = B.place(bridge, cx, cz, dx, dz, c["level"], *gap)
        if pieces:
            pieces = B.seat(pieces, {"tris": {**kit["tris"], **(others[bridge["from"]][0]["tris"] if bridge["from"] else {})}},
                            ours, cx, cz, dx, dz, gap)
            out["pieces"] += pieces
            out["made"].append((c["kind"], "+".join(bridge["names"]), round(sum(gap), 1)))
            out["keep_clear"].append((cx, cz, (sum(gap) / 2 + 4)))
            out["ends"] += [(cx - dx * gap[0], cz - dz * gap[0]), (cx + dx * gap[1], cz + dz * gap[1])]
            if bridge["from"]:
                used.setdefault(bridge["from"], set()).update(bridge["names"])
    for zid, names in used.items():
        tris, dat = others[zid][0]["tris"], others[zid][2]
        models, textures = CP.load_kit(dat, sorted(names))
        out["borrowed"]["tris"].update({m["name16"]: tris[m["name16"]] for m in models.values()})
        out["borrowed"]["sections"] += [m["section"] for m in models.values()]
        out["borrowed"]["textures"].update(textures)
    return out


def base_raw(dat, name16):
    for sec in fz.split_sections(bytes(dat)):
        if sec.kind == fz.T_TEXTURE and sec.raw[0x11:0x21] == name16:
            return sec.raw


def buildings(spec, placed, kit, nets, rng, name_of, keep_clear, base_dat):
    """Buildings (buildings.py): houses from this zone or others, beside our roads or as a town of
    streets on the level terraces, with watchtowers at its corners and walls round it. Returns
    ({"pieces", "made", "keep_clear", "box", "sites", "spots", "streets"}, {"tris", "sections",
    "textures"}: the models borrowed from other zones).

        "buildings": {"groups": [{"from": 85, "sets": ["davoi_h_base"], "weight": 3},
                                 {"sets": ["koya_waku", "sairo"]}],     no "from": the base zone's own
                      "town": {"sites": 3, "level": 2.0, "street": 10},
                      "towers": {"sets": ["miharidai"]}, "walls": {"piece": "tab_hei2"}}
    """
    import caves
    from zonekit import borrow as CP
    from . import buildings as B
    none = ({"pieces": [], "made": [], "keep_clear": [], "box": None, "sites": [], "spots": [], "streets": []},
            {"tris": {}, "sections": [], "textures": {}})
    if not spec:
        return none
    zones = {}

    def source(zid):
        """(tris, objs, dat) of a zone; None: the base zone itself."""
        if zid is None:
            return kit["tris"], kit["objs"], base_dat
        if zid not in zones:
            dat = caves.zone_dat(zid).read_bytes()
            tris, objs, _ = T.read(dat)
            zones[zid] = (tris, objs, dat)
        return zones[zid]
    ours = set(kit["tris"]) | set(kit["meshes"])

    def learn(group):
        zid = group.get("from")
        zid = None if zid == kit.get("from_zone") else zid
        tris, objs, _ = source(zid)
        temps = B.learn(tris, objs, name_of, group["sets"], core=group.get("core"))
        drop = set(group.get("exclude", []))            # neighbours learned with it but not of it (a palm's fronds)
        if drop:
            for ts in temps.values():
                for t in ts:
                    t["parts"] = [p for p in t["parts"] if name_of(p[0]) not in drop]
                    t["names"] = [n for n in t["names"] if n not in drop]
        for a in list(temps):
            if zid is not None:
                # a borrowed model whose name the base zone already has would be drawn as the base's own
                clash = {name_of(n) for t in temps[a] for n, *_ in t["parts"] if n in ours}
                if clash:
                    print(f"  buildings: {a!r} left out, the base zone has models named {sorted(clash)}")
                    del temps[a]
                    continue
            for t in temps[a]:
                t["zone"], t["weight"] = zid, float(group.get("weight", 1.0)) / max(1, len(group["sets"]))
                if "sink" in group:
                    t["sink"] = group["sink"]       # set lower than a house: a tower's own ground hides
                if "front" in group:
                    # the way in, stated (x, z in the anchor's frame): a ramp the guess can't see
                    fx, fz = group["front"]
                    k = math.hypot(fx, fz) or 1.0
                    t["front"] = (fx / k, fz / k)
        return temps
    groups = spec.get("groups") or [{"from": spec["from"], "sets": spec["sets"]}]
    temps = {}
    for g in groups:
        temps.update(learn(g))
    if not temps:
        return none
    feats = {c for cells in nets.values() for c in cells}
    # road-surface tiles laid as plain terrain (Gustaberg's and Sarutabaruta's "miti" pieces) are
    # paths too: a camp's buildings keep off them (its site may still take them: the flat ground)
    paths = feats | {c for c, v in placed.items() if "miti" in (v.get("label") or "")}
    ground = P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3) for k, c in placed.items()})
    extra = {"box": None, "sites": [], "spots": [], "streets": []}
    built = {}                                          # our own wall kit's models, when a town has one
    town = spec.get("town")
    if town:
        town = town if isinstance(town, dict) else {}
        cfg = {**spec, **town}
        near_bridge = {(int(math.floor(x / TILE)), int(math.floor(z / TILE))) for x, z, _ in keep_clear}
        boxes = B.sites(ground, placed, feats | near_bridge, cfg.get("level", 2.0), int(cfg.get("sites", 1)),
                        tuple(cfg.get("need", [3, 3])), tuple(cfg.get("most", [6, 5])))
        if not boxes:
            print("  buildings: no level stretch big enough for a town")
            return none
        got, extra = [], {"box": boxes[0], "sites": boxes, "spots": [], "streets": []}
        tower_t = learn(spec["towers"]) if spec.get("towers") else {}
        for ts in tower_t.values():
            for t in ts:
                t["solid"] = "full"                     # climbed: all of it stands
        wall = spec.get("walls")
        wall_parts = []
        city = spec.get("city_walls")
        if city:
            # a curtain wall of our own pieces (architecture.py), round the terraces big enough for one
            built = built_pieces(city, base_dat, source, name_of)
        for box in boxes:
            i0, j0, i1, j1, hh = box
            walled = city and min(i1 - i0, j1 - j0) >= 4
            inner = (i0 + 1, j0 + 1, i1 - 1, j1 - 1, hh) if walled else box
            # walled: houses keep clear of the gate passages at the streets' ends
            if cfg.get("layout") == "hamlet":
                # a few houses round a campfire (B.hamlet); spots at their own ground height
                g, streets, spots, fire, camp = B.hamlet(temps, ground, inner, {**cfg, "_roads": paths}, rng)
                got += g
                if camp is not None and not extra.get("camp"):
                    # for the workbench's camp editor: what was laid, and every template it can lay
                    home = kit.get("from_zone", kit.get("from"))
                    camp["site"] = list(box)
                    camp["templates"] = [{"anchor": a, "variant": i, "zone": t.get("zone") or home,
                                          "box": list(t["box"]), "front": list(t["front"]),
                                          "sink": t.get("sink", cfg.get("sink", 0.5)),
                                          "parts": [[name_of(n), float(dy), [float(v) for v in sc], [float(v) for v in off]]
                                                    for n, dy, sc, off in t["parts"]]}
                                         for a, ts in temps.items() for i, t in enumerate(ts)]
                    extra["camp"] = camp
                if fire and not extra.get("fire"):
                    fh = ground.at([fire[0]], [fire[1]])[0]
                    extra["fire"] = (fire[0], fire[1], box[4] if np.isnan(fh) else float(fh))
                for x, z, yaw in spots:
                    gh = ground.at([x], [z])[0]
                    extra["spots"].append((x, z, yaw, box[4] if np.isnan(gh) else float(gh)))
                continue
            g, streets, spots = B.town(temps, ground, inner, {**cfg, "snap": bool(walled),
                                                             **({"edge": 14.0} if walled else {})}, rng)
            got += g
            extra["streets"] += streets
            extra["spots"] += [(x, z, yaw, box[4]) for x, z, yaw in spots]
            if walled:
                # streets' centre lines across the long side, in the whole site's frame
                along_x = (i1 - i0) >= (j1 - j0)
                vs = [((a[1] if along_x else a[0]) - (j0 if along_x else i0) * TILE) for a, b, w in streets]
                got.append(({"zone": None, "anchor": "built_wall"},
                            B.city_walls_built(box, built["names"], rng, vs, city.get("inset", 6.0),
                                               city.get("every", 3), city.get("sink", 1.5)), None))
                continue
            if tower_t:
                got += B.towers(tower_t, ground, box, rng, cfg.get("level", 2.0))
            if wall:
                zid = wall.get("from")
                tris, _, _ = source(None if zid == kit.get("from_zone") else zid)
                n16 = next(n for n in tris if name_of(n) == wall["piece"])
                t = tris[n16].reshape(-1, 3)
                gates = [(x, z, w) for a, b, w in streets for x, z in (a, b)]
                wall_parts += B.walls(n16, float(np.ptp(t[:, 2])), ground, box, gates,
                                      taken=[c for *_, c in got if c], mid=(t.max(0) + t.min(0)) / 2)
        got.append((None, wall_parts, None))
        ruins = spec.get("ruins")
        if ruins:
            # ruined walls on the open land round the town, clear of it and of the bridges
            rtris, _, _ = source(ruins["from"])
            names = [n for n in rtris if name_of(n) in ruins["pieces"]]
            avoid = [((a + c) / 2 * TILE, (b + d) / 2 * TILE, 0.5 * math.hypot(c - a, d - b) * TILE + 10)
                     for a, b, c, d, _ in boxes] + [tuple(k) for k in keep_clear] + [c for *_, c in got if c]
            lo, hi = ruins.get("count", [10, 16])
            laid = B.scatter(names, rtris, ground, placed, avoid, rng.randint(lo, hi), rng, ruins.get("level", 3.0))
            got.append(({"zone": ruins["from"], "anchor": "ruin"}, laid, None))
    else:
        roads = set()
        for name, cells in nets.items():
            if (spec.get("along") or name) == name:
                roads |= set(cells)
        if not roads:
            print("  buildings: no road to set them along")
            return none

        def off_road(x, z):
            return (int(math.floor(x / TILE)), int(math.floor(z / TILE))) not in feats
        lo, hi = spec.get("count", [6, 10])
        got = B.place(temps, ground, placed, spec, rng, sorted(roads), off_road, rng.randint(lo, hi),
                      keep_clear=[(x, z, r) for x, z, r in keep_clear])
    # what is solid: a house only where you can walk into it (its walls near the ground; roofs and
    # eaves overhead cost collision for nothing); walls, towers and ruins you climb, all of them
    for t, ps, _ in got:
        full = t is None or t.get("anchor") == "ruin" or t.get("solid") == "full"
        for p in ps:
            if t is not None and t.get("anchor") == "built_wall":
                p["col"] = "full"                             # a few hundred triangles: its real shape
            else:
                p["col"] = "full" if full else "base"
    pieces = [p for _, ps, _ in got for p in ps]
    # borrowed models: each from its own zone
    borrowed = {"tris": {}, "sections": [], "textures": {}}
    by_zone = {}
    for t, ps, _ in got:
        if t is not None and t.get("zone") is not None:
            by_zone.setdefault(t["zone"], set()).update(name_of(p["name"]) for p in ps)
    base = set(fz.texture_sections(fz.split_sections(bytes(base_dat))))
    # models only ever seen from one side where they came from: baked double sided if the zone says
    both_sides = set(spec.get("double_sided", []))
    for zid, names in by_zone.items():
        tris, _, dat = source(zid)
        models, textures = CP.load_kit(dat, sorted(names))
        borrowed["tris"].update({m["name16"]: tris[m["name16"]] for m in models.values()})
        borrowed["sections"] += [fz.double_sided(m["section"]) if n in both_sides else m["section"]
                                 for n, m in models.items()]
        borrowed["textures"].update(textures)
        same = sorted(t.decode("ascii", "replace").split()[-1] for t in textures if t in base and textures[t] != base_raw(base_dat, t))
        if same:
            print(f"  buildings: textures from zone {zid} named like the base zone's, drawn with the base's: {same}")
    if built:
        # laid as the zone's own geometry (as the caves and the sea are), cut into its tiles by the
        # writer, not placed as models
        borrowed["tris"].update(built["tris"])
        borrowed["built"] = built
    return ({"pieces": pieces, "made": [t["anchor"] for t, ps, _ in got if t is not None and ps],
             "keep_clear": [c for t, _, c in got if c is not None], **extra}, borrowed)


def built_pieces(city, base_dat, source, name_of):
    """Our own wall kit (architecture.py) for a city wall: {"names", "tris", "meshes", "textures",
    "full"}. city: {"palette": "ronfaure" (kits/architecture/<name>.json), "prefix": model names'
    prefix, "inset", "banners"}."""
    from . import architecture as A
    pal = A.palette(city.get("palette", "ronfaure"))
    inset = city.get("inset", 6.0)
    flags = {"banners": city.get("banners", True)}
    meshes = {"wall": A.build("wall", pal), "wall_corner": A.wall(pal, length=TILE / 2 - inset),
              "tower": A.build("tower", pal, **flags), "gatehouse": A.build("gatehouse", pal, **flags)}
    # the palette's textures (the stone, the banner), by their full names in their zones
    tex, full = {}, {}
    wanted = [(pal["from"], pal["texture"])] + ([(pal["banner"]["from"], pal["banner"]["texture"])]
                                                if pal.get("banner") and flags["banners"] else [])
    for zid, short in wanted:
        _, _, dat = source(zid)
        for sec in fz.split_sections(bytes(dat)):
            if sec.kind == fz.T_TEXTURE and sec.raw[0x11:0x21].decode("ascii", "replace").split()[-1:] == [short]:
                tex[sec.raw[0x11:0x21]] = sec.raw
                full[short] = sec.raw[0x11:0x21]
                break
    prefix = city.get("prefix", "arch")
    out = {"names": {}, "tris": {}, "textures": tex, "meshes": {}, "full": full}
    for k, m in meshes.items():
        n16 = f"{prefix}_{k}"[:16].encode().ljust(16, b" ")
        out["names"][k] = n16
        out["tris"][n16] = A.triangles(m)
        out["meshes"][n16] = m
    return out

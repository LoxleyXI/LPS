#!/usr/bin/env python3
"""Map previews of a zone's lands: new layouts and their monsters, without building the zone.

Each land of the zone file is laid (layout and tiles, as a build lays it, but no props, houses or
DAT), its monsters placed, and the result drawn as a parchment map with a dot per spawn, one
colour per type. Sheet 0 is each land as the zone file has it now (with its monsters placed the
current way); sheets 1 on are new layouts from new seeds, with monsters spread in the retail
pattern (spawns/pattern.py). A contact sheet per land puts them side by side.

    python3 modules/outdoors/preview.py examples/zones/sand_coast.json OUT
    python3 modules/outdoors/preview.py my_zone.json OUT --land "North Field" --variants 9 \\
        --fields '{"fill": 0.7, "rocks": 6}' --basins 5 --seeds 4200

To build one: put its seed (and "layout": "fields", the "fields" block and "basins" printed under
it) into that land's block in the zone file, and give the zone's "mobs" a "pattern" block (the
preview's "density"). The camps were laid by hand for the old layouts: lay them again in the camp
editor after a land changes.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

COLOURS = [(200, 40, 40), (40, 110, 210), (240, 160, 20), (30, 150, 60), (150, 60, 190),
           (20, 170, 170), (230, 90, 160), (120, 80, 30), (90, 90, 90), (170, 200, 30),
           (250, 220, 60), (0, 70, 110)]


def lands_of(entry):
    """[(name, outdoor settings, monster types)] for the main land and each region."""
    spec = entry["outdoor"]
    main = {k: v for k, v in spec.items() if k != "regions"}
    out = [(spec.get("land", entry["name"]), main, (entry.get("mobs") or {}).get("types", {}))]
    for r in spec.get("regions", []):
        out.append((r["name"], dict(r), (r.get("mobs") or {}).get("types", {})))
    return out


def surface_of(land):
    import zonekit.tiles as T
    from outdoors import props as P
    from outdoors.place import obj_of
    kit = land["kit"]
    return P.Ground({k: T.placed(kit["tris"][c["name"]], obj_of(c, k)).reshape(-1, 3, 3)
                     for k, c in land["placed"].items()})


BUDGET = 120          # tiles a land can have: three lands of about 100 tiles each
                      # (props with them) put its ZoneDef at 79% of the client's limit


def preview_land(name, settings, types, out_dir, variants, seeds, layout, fields, basins, density, walls, size=None):
    import caves
    from caves.kits import _name
    from outdoors import kit as K, zone as Z, maps, mobs as omobs
    from spawns import pattern as PT
    import zonekit.tiles as T
    cfg = K.config(settings["kit"])
    base = caves.zone_dat(cfg["from"]).read_bytes()
    pat = PT.load()
    stem = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    out_dir = Path(out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = list(types)
    colour = {k: COLOURS[i % len(COLOURS)] for i, k in enumerate(keys)}
    kit = None
    sheets = []
    current_tiles = None
    for v in range(variants + 1):
        s = {k: val for k, val in settings.items() if k not in ("buildings", "mobs", "offset", "name")}
        if v:
            s["seed"] = seeds + v
            if layout == "fields":
                s["layout"] = "fields"
                s["fields"] = {"tiles": BUDGET - 10, **fields}
            if basins:
                s["basins"] = basins
            if size:
                s["size"] = list(size)
        t0 = time.time()
        try:
            land = Z.lay(base, s, _name, kit=kit)
        except ValueError as e:
            print(f"  {name} v{v}: no layout ({e})")
            continue
        kit = land["kit"]
        surf = surface_of(land)
        placed = land["placed"]
        # what a build hands the placer (its summary, as generate() makes it)
        world = PT.land_of({"surface": surf, "placed": placed, "shift": (0.0, 0.0, 0.0), "patch": land["patch"],
                            "basins": [(round(a, 1), round(b, 1), lv) for a, b, lv in land["basins"]]})
        labels = [{"text": "camp", "at": [x, z]} for x, z, _r in world["avoid"]]
        if v == 0:
            # as it is now: the zone file's own monster placement
            current_tiles = len(placed)
            summary = {"ground": land["ground"], "placed": placed, "surface": surf, "shift": (0.0, 0.0, 0.0),
                       "basins": land["basins"],
                       "town": {"sites": [(min(c[0] for c in land["patch"]), min(c[1] for c in land["patch"]),
                                           max(c[0] for c in land["patch"]) + 1, max(c[1] for c in land["patch"]) + 1, 0)]}
                       if land["patch"] else None}
            mobs = omobs.place(summary, dict(types), settings.get("seed", 1), 14.0, 4.5)
            title = f"{name}: as it is"
        else:
            mobs = PT.place(world, types, seed=s["seed"], pattern=pat, density=density)
            title = f"{name} {v}"
        m = PT.measure(mobs, world)
        counts = {k: sum(1 for o in mobs if o["type"] == k) for k in keys}
        legend = [(f"{types[k]['name'].replace('_', ' ')} ({counts[k]})", colour[k]) for k in keys]
        over = "  OVER BUDGET" if len(placed) > BUDGET else ""
        notes = [f"seed {s.get('seed')}{', fields' if v and layout == 'fields' else ''}: {len(placed)} tiles"
                 + (f" (now {current_tiles}, budget {BUDGET}){over}" if v else "") + f", {m['area'] / 1000:.0f}k sq yalms walkable",
                 f"{m['spawns']} spawns, {m['density']} per 10k (retail {pat['density']}); "
                 f"{m['used']:.0%} of the land busy (retail {pat['used']:.0%})",
                 f"alone {m['alone']:.0%} (retail {pat['groups']['1']:.0%}); nearest {m['spacing_median']} yalms "
                 f"(retail {pat['spacing_median']}); same type {m['same_type']} (retail {pat['same_type']})",
                 f"by the edge {m['band_density']['near']} per 10k, open {m['band_density']['open']} "
                 f"(retail {pat['band_density']['near']}, {pat['band_density']['open']})"]
        dots = [(o["pos"][0], o["pos"][1], colour[o["type"]]) for o in mobs]
        path = out_dir / f"{v:02d}.png"
        maps.draw_land(title, {"surface": surf, "placed": placed, "shift": (0.0, 0.0, 0.0)}, labels, path,
                       seed=v + 1, walls=walls, dots=dots, legend=legend, notes=notes)
        sheets.append((v, path, s))
        print(f"  {name} {v}: seed {s.get('seed')}, {len(placed)} tiles, {m['spawns']} spawns, "
              f"{time.time() - t0:.0f}s -> {path}")
        if v:
            (out_dir / f"{v:02d}.json").write_text(json.dumps(
                {k: s[k] for k in ("seed", "layout", "fields", "basins", "size") if k in s}, indent=1) + "\n")
    contact(name, sheets, out_dir.parent / f"{stem}.png")


def contact(name, sheets, path, cols=4, cell=520):
    """The sheets side by side, numbered."""
    from PIL import Image, ImageDraw, ImageFont
    rows = math.ceil(len(sheets) / cols)
    img = Image.new("RGB", (cols * cell, rows * cell + 40), (40, 26, 14))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/opentype/urw-base35/P052-Bold.otf", 28)
    except OSError:
        f = ImageFont.load_default()
    d.text((12, 6), name, font=f, fill=(238, 222, 184))
    for k, (v, p, _s) in enumerate(sheets):
        im = Image.open(p).convert("RGB").resize((cell, cell), Image.LANCZOS)
        img.paste(im, ((k % cols) * cell, 40 + (k // cols) * cell))
        d.rectangle([(k % cols) * cell + 6, 46 + (k // cols) * cell, (k % cols) * cell + 70, 86 + (k // cols) * cell],
                    fill=(40, 26, 14))
        d.text(((k % cols) * cell + 14, 50 + (k // cols) * cell), "now" if v == 0 else str(v), font=f,
               fill=(238, 222, 184))
    img.save(path)
    print(f"{name}: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zone", help="zone file (zones/<name>.json or a path)")
    ap.add_argument("out", help="folder for the sheets")
    ap.add_argument("--land", action="append", help="only this land (by name); again for more")
    ap.add_argument("--variants", type=int, default=7, help="new layouts per land (7)")
    ap.add_argument("--seeds", type=int, help="new layouts take seeds after this (default: the land's seed x 10)")
    ap.add_argument("--layout", default="fields", choices=["fields", "basins"],
                    help="fields: the open layout; basins: the old one, new seeds only")
    ap.add_argument("--fields", default="{}", help='the fields block, JSON: {"fill", "ragged", "rocks", "rock", "flat"}')
    ap.add_argument("--basins", type=int, help="basins (fields) per land")
    ap.add_argument("--density", type=float, help="spawns per 10,000 sq yalms (default: the retail pattern's)")
    ap.add_argument("--size", help="the grid, in tiles, for new layouts: WxD (the land's own size)")
    args = ap.parse_args()
    from outdoors import kit as K
    path = Path(args.zone) if Path(args.zone).exists() else HERE / args.zone
    entry = json.loads(path.read_text())["zones"][0]
    for name, settings, types in lands_of(entry):
        if args.land and name not in args.land:
            continue
        walls = (K.config(settings["kit"]).get("walls") or {}).get("families", [])
        seeds = args.seeds if args.seeds is not None else settings.get("seed", 1) * 10
        preview_land(name, settings, types, Path(args.out).expanduser(), args.variants, seeds, args.layout,
                     json.loads(args.fields), args.basins, args.density, walls,
                     [int(x) for x in args.size.split("x")] if args.size else None)


if __name__ == "__main__":
    main()

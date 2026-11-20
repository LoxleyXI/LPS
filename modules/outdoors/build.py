#!/usr/bin/env python3
"""Build an outdoor zone from a zone file (zones/<name>.json): write the DAT and, with --install,
put it into the client's overlay (model, the clone's event/dialog/NPC files, file table, name).
A build also writes the monster SQL where the zone file says; a server's other files (its NPCs, the
way in) come from extensions' "outdoor_built" hooks (modules/__init__.py).

    python3 modules/outdoors/build.py examples/zones/sand_coast.json --out /tmp/411.DAT --install

--client DIR writes only the client's files into DIR, built from the plain game files (no HD
copies), rather than installing them into an overlay:

    zone.DAT          the zone's model
    names.DAT         its NPC name list, when it has monsters (the clone's list with theirs added)
    mobs/<id>.DAT     each monster model of our own, by model id
    manifest.json     zone id, name, the zone it clones (event and dialog files come from there),
                      the files above, and "layout": a fingerprint of where the basins, town
                      spots and monsters fell. The server's files (spawn SQL, NPC places) are
                      made against that layout, so a zone built again must give the same one.

    python3 modules/outdoors/build.py examples/zones/sand_coast.json --client /tmp/sand_coast

A "recolor" block in "outdoor" ({"hue", "saturation", "lightness", "tint", "blend_mode",
"texture_overrides"}) shifts the finished zone's colours as the zones module does.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # LPS root (lps package)
sys.path.insert(0, str(HERE.parent))                 # modules (zones, caves, outdoors)

from xidat.paths import ffxi_dir, overlay as overlay_dir, set_hd, backups_dir   # noqa: E402
import caves                                         # noqa: E402
from caves.kits import _name                         # noqa: E402
import outdoors                                      # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("zone", help="zone file (zones/<name>.json or a path)")
ap.add_argument("--out", help="write the zone DAT here")
ap.add_argument("--install", action="store_true", help="install into the client's overlay")
ap.add_argument("--client", metavar="DIR", help="only the client's files, from plain game files, into DIR")
args = ap.parse_args()
if args.client:
    if args.install:
        sys.exit("--client writes the files to a folder instead of installing them: leave out --install")
    set_hd(False)
    client = Path(args.client)
    client.mkdir(parents=True, exist_ok=True)
    manifest = {"zone_id": None, "name": None, "clone": None, "files": {}, "mobs": {}}
path = Path(args.zone) if Path(args.zone).exists() else HERE / args.zone
entry = json.loads(path.read_text())["zones"][0]
spec = entry["outdoor"]
if "scene" in spec:                                  # a piece of a real zone kept in place (scene.py)
    from outdoors import scene
    source = spec["scene"]["from"]
    out, summary = scene.generate(caves.zone_dat(source).read_bytes(), spec, _name)
    print(f"{entry['name']}: {summary['objects']} of the source's objects kept, {summary['pieces']} pieces of ours; "
          f"effects dropped {summary['effects_dropped']}")
else:
    source = outdoors.kit.config(spec["kit"])["from"]
    out, summary = outdoors.generate(caves.zone_dat(source).read_bytes(), spec, _name)
    print(f"{entry['name']}: {summary['cells']} tiles, {summary['props']} props, {len(summary['basins'])} basins, "
          f"{len(summary['bays'])} bays (layout try {summary['attempts']}); effects moved {summary['effects_moved']}, "
          f"dropped {summary['effects_dropped']}")
recolor = spec.get("recolor")
if recolor:
    # the zones module's colour shift (hue/saturation/lightness/tint, per-texture overrides and
    # the environment's light and fog) over the whole generated zone
    import tempfile
    import zones
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp) / "zone.DAT"
        tmp.write_bytes(out)
        stats = zones.recolor_zone_dat(tmp, tmp, **recolor)
        out = tmp.read_bytes()
    print(f"recolor: {stats}")
from zonekit import budget                           # noqa: E402
print("budget: " + budget.line(out))
if args.out:
    Path(args.out).write_bytes(out)
    print(f"{args.out}: {len(out)} bytes")
if args.client:
    (client / "zone.DAT").write_bytes(out)
    manifest.update(zone_id=entry["zone_id"], name=entry["name"], clone=source)
    manifest["files"]["model"] = "zone.DAT"
spots = list(summary["town"]["spots"]) if summary.get("town") else []
if not args.client:
    # the server's own files from the layout (its NPCs, the way in, doors) are an extension's:
    # its "outdoor_built" hooks (modules/__init__.py) get the zone file's entry, its "outdoor"
    # block, the summary (basins, town spots and campfire, regions, shift) and the zone file
    from modules import run_hooks
    run_hooks("outdoor_built", entry=entry, spec=spec, summary=summary, zone_file=path)
if args.install:
    overlay = overlay_dir(spec.get("overlay", "ffxi-hd"))
    done = outdoors.install(out, entry["zone_id"], source, entry["name"], ffxi_dir(), overlay)
    print(f"installed zone {entry['zone_id']} into {overlay.name}: {done}")
placements = []
mobs_spec = entry.get("mobs")
if mobs_spec:
    # monsters on the level ground (outdoors/mobs.py), with the caves module's server SQL,
    # client models and name list (after the install: the names go into the zone's own list)
    import os
    from spawns import sql as server, models as mobmodels, names
    from outdoors import mobs as omobs
    types = dict(mobs_spec["types"])
    # "pattern": {"name", "density"}: spread as the original field zones spread theirs
    # (spawns/pattern.py, the placements outdoors/preview.py draws); else the camps and gaps of
    # outdoors/mobs.py
    pat = mobs_spec.get("pattern")
    if pat:
        from spawns import pattern as PT
        retail = PT.load(pat.get("name", "retail_fields"))

        def put(land, own, seed):
            return PT.place(PT.land_of(land), own, seed, retail, pat.get("density"))
    else:
        def put(land, own, seed):
            return omobs.place(land, own, seed, mobs_spec.get("gap", 12.0), mobs_spec.get("slope", 0.0))
    placements = put(summary, types, spec.get("seed", 1))
    # the other lands' own monsters (a region's "mobs": {"types"}), on their islands, after the
    # main land's in the same SQL (so its ids stay put)
    for k, (region, land) in enumerate(zip(spec.get("regions", []), summary.get("regions", [])), 1):
        own = {f"r{k}_{key}": t for key, t in (region.get("mobs") or {}).get("types", {}).items()}
        types.update(own)
        placements += put(land, own, region.get("seed", k))
    for i, t in enumerate(types.values()):
        # our own copy of a model, a retail model by its id ("modelid"), or the pool's own
        t["model_id"] = mobs_spec["models_from"] + i if "model" in t else t.get("modelid")
    mobs_spec = dict(mobs_spec, types=types)
    if not args.client:
        sql_path = Path(os.path.expanduser(mobs_spec["sql"]))
        n = server.write_sql(sql_path, entry["zone_id"], mobs_spec, placements, Path(args.zone).name)
        print(f"mobs: {n} spawns -> {sql_path}")
    if args.install:
        overlay = overlay_dir(mobs_spec.get("overlay", "ffxi-hd"))
        modelled = {k: t for k, t in types.items() if "model" in t}
        if modelled:
            print(f"mob models: {mobmodels.build(modelled, ffxi_dir(), overlay)}")
        listed = names.write(overlay, entry["zone_id"],
                             {k: types[p["type"]]["name"] for k, p in enumerate(placements, 1)},
                             backups_dir())
        print(f"mob names: {listed}")
    if args.client:
        from xidat.ftable import load_all, resolve
        from xidat.zone_files import npc_fid
        modelled = {k: t for k, t in types.items() if "model" in t}
        made = mobmodels.export(modelled, ffxi_dir(), client / "mobs")
        manifest["mobs"] = {str(mid): f"mobs/{name}" for mid, name in made.items()}
        clone_names = resolve(npc_fid(source), load_all(ffxi_dir())).read_bytes()
        (client / "names.DAT").write_bytes(names.with_names(
            clone_names, entry["zone_id"], {k: types[p["type"]]["name"] for k, p in enumerate(placements, 1)}))
        manifest["files"]["npc"] = "names.DAT"
        print(f"mob models: {len(made)} -> {client / 'mobs'}; names -> {client / 'names.DAT'}")
if args.client:
    import hashlib

    def plain(v):
        """numpy and tuples as plain JSON values (floats exactly, by repr)"""
        if hasattr(v, "tolist"):
            return v.tolist()
        if isinstance(v, dict):
            return {str(k): plain(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [plain(x) for x in v]
        return v

    layout = {"basins": summary["basins"], "spots": spots, "mobs": placements,
              "regions": [{"basins": r["basins"], "spots": (r.get("town") or {}).get("spots"),
                          "fire": (r.get("town") or {}).get("fire")} for r in summary.get("regions", [])]}
    manifest["layout"] = hashlib.sha256(json.dumps(plain(layout), sort_keys=True, default=repr).encode()).hexdigest()[:16]
    (client / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"layout: {manifest['layout']}")
    print(f"client files: {client}")

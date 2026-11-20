#!/usr/bin/env python3
"""Test build: a generated cave on a base zone's DAT, written to --out (never into the game install).

    python3 modules/caves/build.py --clone 203 --seed 1336 --rooms 5 --out /tmp/410.DAT
    python3 modules/caves/build.py --config examples/zones/ice_cave.json --base <recoloured DAT> --out /tmp/410.DAT

    python3 modules/caves/build.py --scan-kit 111 --out /tmp/beaucedine_draft.json

--config takes a zones.json-style file and uses its first entry's "generate" block; --base is the
DAT to build on (e.g. the zone as LPS already cloned and recoloured it), else the clone's own DAT.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))          # LPS root (lps package)
sys.path.insert(0, str(HERE.parent))                 # modules (zones, caves)

from xidat.paths import ffxi_dir, dats_root          # noqa: E402
from xidat.ftable import load_all, resolve           # noqa: E402
from xidat.zone_files import model_fid               # noqa: E402
import caves                                         # noqa: E402

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--clone", type=int, help="zone whose DAT (textures, lighting) to build on")
ap.add_argument("--config", help="zones.json-style file: its first entry's generate block")
ap.add_argument("--base", help="DAT to build on instead of the clone's own")
ap.add_argument("--seed", type=int, default=1336)
ap.add_argument("--rooms", type=int, default=5)
ap.add_argument("--voxel", type=float, default=3.0)
ap.add_argument("--out", help="write the zone DAT here")
ap.add_argument("--mobs", action="store_true", help="also the config's mobs: server SQL module + client models")
ap.add_argument("--scan-kit", type=int, metavar="ZONE", help="draft a kit from a zone's DAT (prints JSON, or "
                "writes it to --out): how the zone lays each texture and which props it places")
args = ap.parse_args()
if args.scan_kit is not None:
    import json
    draft = caves.kits.scan(caves.zone_dat(args.scan_kit).read_bytes(), args.scan_kit)
    text = json.dumps(draft, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"{args.out}: {len(draft['measured'])} textures, {len(draft['props_found'])} props")
    else:
        print(text)
    sys.exit(0)
if args.config:
    import json
    entry = json.loads(Path(args.config).read_text())["zones"][0]
    if "outdoor" in entry:
        sys.exit("outdoor zones build with modules/outdoors/build.py")
    settings, clone = entry["generate"], entry["clone"]
else:
    settings, clone = {"seed": args.seed, "rooms": args.rooms, "voxel": args.voxel}, args.clone
if args.out:
    base = Path(args.base) if args.base else resolve(model_fid(clone), load_all(ffxi_dir()))
    out = caves.generate_bytes(base.read_bytes(), settings)
    Path(args.out).write_bytes(out)
    print(f"{args.out}: {len(out)} bytes ({base.name} + generated cave, seed {settings.get('seed')})")
if args.mobs:
    import json
    entry = json.loads(Path(args.config).read_text())["zones"][0]
    res = caves.build_mobs(entry, Path(args.config).name, ffxi_dir(), dats_root())
    print(f"mobs: {res['spawns']} spawns -> {res['sql']}; names -> {res['names']}")
    for key, mid, path, stats in res["models"]:
        print(f"  {key}: model {mid} -> {path} ({stats.get('dxt', 0)} textures recoloured, "
              f"{stats.get('vertices_scaled', 0)} vertices scaled)")

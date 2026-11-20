"""Caves: generated cave zones.

A zone entry in zones.json with a "generate" block gets its geometry replaced by a cave generated
from a seed (network.py: rooms of overlapping lobes joined by tunnels, a density field meshed with
surface nets; mesh.py: textures per surface; zonekit/writer.py: our zone DAT writer). The textures are
the cloned zone's own, so nothing from the game is shipped: players regenerate it from the seed.

    {"name": "Ice Cave", "clone": 203, "zone_id": 410,
     "generate": {"seed": 1336, "rooms": 5, "voxel": 3.0, "noise": 4,
                  "textures": {"floor": "delyuka1", "wall": "delkabe1", "ceiling": "delceil1",
                               "bed": "ice01", "shore": "delbase1"}}}

With a kit (kits/<name>.json: a source zone's textures and props, and how that zone lays them) the
block names the kit and keeps only its own choices; see kits.py and README.md:

    "generate": {"kit": "beaucedine", "seed": 1336, "rooms": 6, "extent": 150,
                 "water": {"features": ["lake", "river", "moat"]}}

Test builds (write only where you say, never into the game install):
    python3 modules/caves/build.py --clone 203 --seed 1336 --rooms 5 --out /tmp/410.DAT
"""
from pathlib import Path

from . import kits  # noqa: F401  (caves.kits)

DEFAULT_TEXTURES = {"floor": "delyuka1", "wall": "delkabe1", "ceiling": "delceil1", "bed": "ice01",
                    "shore": "delbase1"}


def generate_bytes(base_dat: bytes, settings: dict) -> bytes:
    """The base zone's DAT with its geometry replaced by the generated cave."""
    from . import mesh, kits
    from zonekit import writer as ffxi_zone
    settings = kits.resolve(settings)
    cave = {k: settings[k] for k in ("seed", "rooms", "voxel", "noise", "depth", "extent") if k in settings}
    geometry = mesh.build(cave, dict(DEFAULT_TEXTURES, **settings.get("textures", {})),
                          water=settings.get("water"), uv=settings.get("uv"),
                          foot=settings.get("foot"), overlay=settings.get("overlay"))
    if settings.get("textures_from"):                  # surfaces textured from another zone
        geometry["extra_textures"] = zone_textures(settings["textures_from"])
    if settings.get("props"):
        add_props(geometry, settings["props"], settings.get("seed", 1))
    return ffxi_zone.build_zone(base_dat, geometry, draw=settings.get("draw"), fog=settings.get("fog"))


def add_props(geometry, spec, seed):
    """Place props from another zone's kit (props.py); solid ones join the collision."""
    import numpy as np
    from . import props
    src = zone_dat(spec["from"])
    names = sorted({m for rule in spec["rules"] for m in rule["models"]})
    kit, textures = props.load_kit(src.read_bytes(), names)
    placements = props.place(geometry["net"], geometry, spec, kit, seed)
    col = geometry["collision"]
    solid = [props.world_triangles(kit, p) for p in placements if p["solid"]]
    solid = [t for t in solid if len(t)]
    if solid:
        tw = np.concatenate(solid)
        base = len(col["positions"])
        col["positions"] = np.concatenate([col["positions"], tw.reshape(-1, 3).astype(np.float32)])
        col["triangles"] = np.concatenate([col["triangles"],
                                           (base + np.arange(len(tw) * 3)).reshape(-1, 3).astype(np.int32)])
        n = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        col["wall"] = np.concatenate([col["wall"], np.abs(n[:, 2]) < 0.5])
    geometry["props"] = props.for_writer(kit, textures, placements)
    geometry["prop_placements"] = placements


def zone_dat(zone_id):
    """A zone's model DAT, preferring the HD overlay copy (as the zones module does)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from xidat.paths import ffxi_dir, hd
    from xidat.ftable import load_all, resolve
    from xidat.zone_files import model_fid
    return hd(resolve(model_fid(zone_id), load_all(ffxi_dir())))


def generate_into(dat_path: Path, settings: dict) -> None:
    """Rewrite a cloned zone DAT in place (the zones module calls this after cloning)."""
    dat_path = Path(dat_path)
    dat_path.write_bytes(generate_bytes(dat_path.read_bytes(), settings))


def zone_textures(zone_id):
    """{name16: raw 0x20 section} for every texture in a zone's DAT (read at build time)."""
    from zonekit import writer as ffxi_zone
    out = {}
    for sec in ffxi_zone.split_sections(zone_dat(zone_id).read_bytes()):
        if sec.kind == ffxi_zone.T_TEXTURE:
            out.setdefault(sec.raw[0x11:0x21], sec.raw)
    return out


def build_mobs(entry, source_json, ffxi, overlay_root):
    """A zone entry's "mobs" block: spawns placed in the generated cave, the server SQL module, and the
    oversized / recoloured client models registered in the overlay. Returns a summary."""
    import os
    from . import mesh, spawns
    from xidat.paths import backups_dir
    from spawns import sql as server, models as mobmodels, names
    from . import kits
    spec = entry["mobs"]
    gen = kits.resolve(entry["generate"])
    types = spec["types"]
    for i, t in enumerate(types.values()):
        t["model_id"] = spec["models_from"] + i
    cave = {k: gen[k] for k in ("seed", "rooms", "voxel", "noise", "depth", "extent") if k in gen}
    geometry = mesh.build(cave, dict(DEFAULT_TEXTURES, **gen.get("textures", {})), water=gen.get("water"),
                          uv=gen.get("uv"), foot=gen.get("foot"), overlay=gen.get("overlay"))
    placements = spawns.place(geometry["net"], spec["spawns"], types, gen.get("seed", 1))
    sql_path = Path(os.path.expanduser(spec["sql"]))
    n = server.write_sql(sql_path, entry["zone_id"], spec, placements, source_json)
    overlay = Path(overlay_root) / spec.get("overlay", "ffxi-hd")
    models = mobmodels.build(types, Path(ffxi), overlay)
    listed = names.write(overlay, entry["zone_id"],
                         {i: types[p["type"]]["name"] for i, p in enumerate(placements, 1)},
                         backups_dir())
    return {"spawns": n, "sql": str(sql_path), "models": models, "names": str(listed)}

"""What a zone DAT spends against the client's limits: printed by every build, so a zone nearing
one is seen before it crashes the game.

    ZoneDef section   <= 8 MiB   the client copies it into a fixed buffer (writer.MAX_ZONEDEF_BYTES)
    collision cells   <= 124,103 grid cells in use (writer.MAX_COLLISION_GROUPS): the most any retail
                                 zone has (Western Altepa Desert); sixteen retail zones pass 60,000
"""
from . import writer as fz


def measure(dat):
    """{"bytes", "zonedef", "zonedef_max", "objects", "collision_cells", "collision_cells_max",
    "collision_triangles", "grid"} for a zone DAT."""
    zd = next((s for s in fz.split_sections(bytes(dat)) if s.kind == fz.T_ZONEDEF), None)
    out = {"bytes": len(dat), "zonedef_max": fz.MAX_ZONEDEF_BYTES, "collision_cells_max": fz.MAX_COLLISION_GROUPS}
    if zd is None:
        return out
    body = bytearray(zd.raw[16:])
    fz.zonedef_decrypt(body, fz.key_tables())
    z = fz.read_zonedef(body)
    c = z.get("collision") or {}
    out.update({"zonedef": len(body), "objects": z["count"], "collision_cells": len(c.get("groups", [])),
                "collision_triangles": sum(m["tris"] for m in c.get("meshes", [])), "grid": c.get("grid")})
    return out


def line(dat):
    """One line for a build's output."""
    m = measure(dat)
    if "zonedef" not in m:
        return f"{m['bytes'] / 1e6:.1f} MB"
    return (f"{m['bytes'] / 1e6:.1f} MB; ZoneDef {m['zonedef'] / 1e6:.2f} of {m['zonedef_max'] / 1e6:.2f} MB "
            f"({100 * m['zonedef'] / m['zonedef_max']:.0f}%), collision cells {m['collision_cells']:,d} of "
            f"{m['collision_cells_max']:,d}, {m['objects']:,d} objects, {m['collision_triangles']:,d} collision triangles")

"""Borrowing models from another zone: named meshes out of its DAT, with the textures they use,
ready to go into ours (the writer's props block, or build_zone's extra textures). Nothing is
shipped: they are read from the player's own copy of that zone at build time."""
import numpy as np

from . import writer as fz


def load_kit(dat, names, tables=None):
    """The named meshes of a zone DAT: {name: {name16, section, textures, tris (k,3,3) local FFXI,
    lo, hi}} plus {texture16: raw texture section} for the textures they use."""
    tables = tables or fz.key_tables()
    sections = fz.split_sections(bytes(dat))
    want = set(names)
    kit, tex_needed = {}, set()
    for s in sections:
        if s.kind != fz.T_MESH:
            continue
        body = bytearray(s.raw[0x10:])
        fz.mesh_decrypt(body, tables)
        m = fz.read_mesh(body)
        name = m["name"].split(b"\0")[0].decode("ascii", "replace").rstrip()
        if name not in want or not m["parts"] or name in kit:
            continue
        tris = []
        for p in m["parts"]:
            t = fz.part_triangles(p, bool(m["config"] & 1))
            if len(t):
                tris.append(p["pos"][t])
            tex_needed.add(bytes(p["texture"]))
        tris = np.concatenate(tris) if tris else np.zeros((0, 3, 3), np.float32)
        pts = tris.reshape(-1, 3)
        kit[name] = {"name16": bytes(m["name"]), "section": s.raw, "tris": tris,
                     "lo": pts.min(0), "hi": pts.max(0)}
    missing = want - set(kit)
    if missing:
        raise ValueError(f"props: no mesh named {sorted(missing)} in the source zone")
    textures = {}
    for s in sections:
        if s.kind == fz.T_TEXTURE and s.raw[0x11:0x21] in tex_needed:
            textures.setdefault(s.raw[0x11:0x21], s.raw)
    return kit, textures

"""More lands in one zone: regions laid from other zones' kits, each on its own island.

A zone file's "outdoor" block may carry "regions": [{...outdoor block..., "offset": [x, z]}]. Each
region is laid by generate() from its own kit's source zone, then put into the main zone: its
land tiles and scatter (the source zone's own meshes) borrowed in under new names, so nothing of
one zone's names meets another's (a texture named the same in two zones would otherwise be
written once and drawn on both), its collision and edge fences merged, its borrowed models
(houses) shared with the main zone's where they're the same.

Names: mesh names take the region's prefix ("x1", "x2"...) and texture names take it after their
8-byte category ("model   gu_wo6c" becomes "model   x1gu_wo6": the client reads the category, and
a texture outside "model" loses its alpha), so a "recolor" block's "texture_overrides" can tint
each region on its own ({"model   x1*": {"hue": 200}}).

Islands lie apart (offset, in yalms): further than the draw distance, nobody sees the next one.
The sky, light, fog and weather are the main zone's (one environment per zone).
"""
import numpy as np

from zonekit import borrow as CP
from zonekit import textures as X
from zonekit import writer as fz


def _strip(name16):
    return name16.split(b"\0")[0].rstrip()


def _new_name(prefix, name16, taken, pad=b"\0"):
    base = (prefix + _strip(name16))[:16]
    new = base
    k = 0
    while new.ljust(16, pad) in taken:
        k += 1
        tag = str(k).encode()
        new = base[:16 - len(tag)] + tag
    out = new.ljust(16, pad)
    taken.add(out)
    return out


def _new_texture_name(prefix, name16, taken):
    """A texture's new name: its first 8 bytes are its category ("model   ", "effect  "...), which
    the client reads (a model texture drawn with its alpha), so the prefix goes in the name half."""
    cat, own = name16[:8], name16[8:].split(b"\0")[0].rstrip()
    base = (prefix + own)[:8]
    new = base
    k = 0
    while (cat + new.ljust(8, b" ")) in taken:
        k += 1
        tag = str(k).encode()
        new = base[:8 - len(tag)] + tag
    out = cat + new.ljust(8, b" ")
    taken.add(out)
    return out


def renamed_mesh(raw, name16, textures, tables):
    """A 0x2E section with its mesh name and its parts' texture names replaced (textures:
    {old16: new16}); everything else as it was."""
    body = bytearray(raw[16:])
    fz.mesh_decrypt(body, tables)
    body[0x10:0x20] = name16
    d = 0x20
    if np.frombuffer(bytes(body[d:d + 4]), "<u4")[0]:
        first, count = np.frombuffer(bytes(body[d + 0x1C:d + 0x24]), "<u4")
        stride = 48 if body[4] & 2 else 36
        p = d + int(first)
        for _ in range(int(count)):
            tex = bytes(body[p:p + 16])
            if tex in textures:
                body[p:p + 16] = textures[tex]
            nv = int(np.frombuffer(bytes(body[p + 16:p + 18]), "<u2")[0])
            p += 20 + nv * stride
            ni = int(np.frombuffer(bytes(body[p:p + 2]), "<u2")[0])
            p += 4 + ni * 2
            p += -p % 4
    fz.mesh_encrypt(body, tables)
    return fz.section_bytes(bytes(raw[:4]), fz.T_MESH, bytes(body))


def renamed_texture(raw, name16):
    out = bytearray(raw)
    out[0x11:0x21] = name16
    return bytes(out)


def merge(geometry, regions, name_of, generate, zone_dat, kit_config, max_texture=None):
    """Lay each region and put it into `geometry` (the main zone's, before build_zone). Returns
    the regions' summaries, in order."""
    tables = fz.key_tables()
    props = geometry["props"]
    have_sections = {bytes(s) for s in props["sections"]}
    taken_mesh = set()
    taken_tex = set(props["textures"])
    summaries = []
    for k, region in enumerate(regions, 1):
        prefix = f"x{k}".encode()
        cfg = kit_config(region["kit"])
        base = zone_dat(cfg["from"]).read_bytes()
        settings = {**region, "_parts": True, "_offset": region.get("offset", [0, 0])}
        if max_texture and "max_texture" not in settings:
            settings["max_texture"] = max_texture
        parts = generate(base, settings, name_of)
        geo, used = parts["geometry"], parts["used"]

        # the region's own source meshes (its tiles and scatter), borrowed under new names
        kit, textures = CP.load_kit(base, sorted(_strip(n).decode("ascii", "replace") for n in used), tables)
        tex_names = {old: _new_texture_name(prefix, old, taken_tex) for old in textures}
        mesh_names = {}
        for m in kit.values():
            # a leading underscore marks a mesh for the client's alpha pass (cut-out leaves and
            # crystals): it stays in front ("_gu_w07_h" becomes "_x1gu_w07_h")
            if m["name16"].startswith(b"_"):
                new16 = b"_" + _new_name(prefix, m["name16"][1:] + b"\0", taken_mesh)[:15]
                taken_mesh.add(new16)
            else:
                new16 = _new_name(prefix, m["name16"], taken_mesh)
            mesh_names[m["name16"]] = new16
            props["sections"].append(renamed_mesh(m["section"], new16, tex_names, tables))
        for old, raw in textures.items():
            if max_texture:
                raw = X.cap(raw, max_texture)
            props["textures"][tex_names[old]] = renamed_texture(raw, tex_names[old])

        # its placed objects (by the new names), and anything it borrowed itself (houses)
        rp = geo["props"]
        for o in rp["objects"]:
            props["objects"].append(dict(o, name=mesh_names.get(o["name"], o["name"])))
        for raw in rp["sections"]:
            if bytes(raw) not in have_sections:
                have_sections.add(bytes(raw))
                props["sections"].append(raw)
        for name16, raw in rp["textures"].items():
            props["textures"].setdefault(name16, raw)

        # what it lays as geometry of its own (paving and the like), and its collision
        geometry["meshes"] += geo["meshes"]
        geometry.setdefault("extra_textures", {}).update(geo.get("extra_textures") or {})
        a, b = geometry["collision"], geo["collision"]
        n = len(a["positions"])
        geometry["collision"] = {
            "positions": np.concatenate([a["positions"], b["positions"]]),
            "triangles": np.concatenate([a["triangles"], b["triangles"] + n]),
            "wall": np.concatenate([a["wall"], b["wall"]]),
            "terrain": np.concatenate([a["terrain"], b["terrain"]]),
        }
        summaries.append(dict(parts["summary"], name=region.get("name", f"region {k}"), prefix=prefix.decode()))
    return summaries

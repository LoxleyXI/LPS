"""Giant crystals: another zone's crystal model (Zi'Tah's lei_cry1_m) laid as our own geometry, so its
colour is ours to set: its texture coloured one hue, saturated and brightened, its vertex colours
pushed past the engine's own light (128) so it reads as glowing, and a faint translucent shell a
little larger round it for a glossy sheen. Parts drawn in other textures (Zi'Tah's forest floor
blended round its foot) are left out.

    "crystals": {"from": 121, "name": "lei_cry1_m", "texture": "li_izu",
                 "hue": 125, "saturation": 0.85, "value": 1.25,     the texture's new colour
                 "bright": 1.8,                                      vertex colour (1 = the engine's own)
                 "shell": {"scale": 1.05, "alpha": 40, "bright": 2.0},
                 "at": [[x, z, scale, yaw], ...]}                    world; standing on the ground there

Returns meshes in the writer's form (cave frame), collision triangles and the texture to add.
"""
import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T


def _colorize(raw, hue, sat, value):
    """A texture section (DXT or paletted) with every colour set to one hue, the given saturation, and
    its brightness scaled: a pink-blue crystal made one clear green."""
    import struct
    from zonekit import textures as X
    b = bytearray(raw)
    h0 = 16

    def paint(rgb):
        # keep each pixel's brightness, scaled; give it the one hue at the given saturation
        v = np.clip(rgb.max(-1) * value, 0.0, 1.0)
        c = v * sat
        hp = (hue % 360) / 60.0
        x = c * (1 - abs(hp % 2 - 1))
        z = np.zeros_like(c)
        r, g, bl = [(c, x, z), (x, c, z), (z, c, x), (z, x, c), (x, z, c), (c, z, x)][int(hp) % 6]
        return np.stack([r, g, bl], -1) + (v - c)[..., None]
    if b[h0] == 0xA1:
        w, h = struct.unpack_from("<II", b, h0 + 0x15)
        kind = bytes(b[h0 + 0x39:h0 + 0x3D]).decode()
        size = struct.unpack_from("<I", b, h0 + 0x3D)[0]
        img = X.decode(bytes(b[h0 + 0x45:h0 + 0x45 + size]), w, h, kind)
        rgb = paint(img[..., :3].astype(np.float64) / 255.0)
        img = np.concatenate([(rgb * 255).round().astype(np.uint8), img[..., 3:]], -1)
        enc = X.encode(img, kind)
        b[h0 + 0x45:h0 + 0x45 + len(enc)] = enc
    else:
        pal = np.frombuffer(bytes(b[h0 + 0x3D:h0 + 0x3D + 1024]), np.uint8).reshape(256, 4).copy()
        rgb = paint(pal[:, [2, 1, 0]].astype(np.float64)[None] / 255.0)[0]
        out = (rgb * 255).round().astype(np.uint8)
        pal[:, 2], pal[:, 1], pal[:, 0] = out[:, 0], out[:, 1], out[:, 2]
        b[h0 + 0x3D:h0 + 0x3D + 1024] = pal.tobytes()
    return bytes(b)


def build(spec, height_at):
    """(meshes, collision (k, 3, 3) cave frame, {texture16: raw}). height_at(x, z) -> ground height."""
    import caves
    dat = caves.zone_dat(spec["from"]).read_bytes()
    tables = fz.key_tables()
    want = spec["name"]
    mesh = None
    for s in fz.split_sections(dat):
        if s.kind != fz.T_MESH:
            continue
        body = bytearray(s.raw[0x10:])
        fz.mesh_decrypt(body, tables)
        m = fz.read_mesh(body)
        if m["name"].split(b"\0")[0].decode("ascii", "replace").rstrip() == want:
            mesh = m
            break
    if mesh is None:
        raise ValueError(f"crystals: no mesh {want!r} in zone {spec['from']}")
    tex_short = spec.get("texture", "li_izu")
    parts = [p for p in mesh["parts"] if bytes(p["texture"]).split()[-1].decode() == tex_short]
    tex16 = bytes(parts[0]["texture"])
    raw = next(s.raw for s in fz.split_sections(dat) if s.kind == fz.T_TEXTURE and s.raw[0x11:0x21] == tex16)
    raw = _colorize(raw, spec.get("hue", 125), spec.get("saturation", 0.85), spec.get("value", 1.25))
    strip = bool(mesh["config"] & 1)
    local = []                                                     # (pos, nrm, uv, tri) FFXI local
    for p in parts:
        t = fz.part_triangles(p, strip)
        if len(t):
            local.append((p["pos"].astype(np.float64), p["nrm"].astype(np.float64), p["uv"].astype(np.float32), t))
    out, solid = [], []
    for x, z, sc, yaw in spec["at"]:
        h = height_at(x, z) - spec.get("sink", 0.6) * sc
        for layer, k, alpha, bright, flags in (("body", 1.0, 128, spec.get("bright", 1.8), 0),
                                               ("shell", spec.get("shell", {}).get("scale", 1.05),
                                                spec.get("shell", {}).get("alpha", 40),
                                                spec.get("shell", {}).get("bright", 2.0), 0x8000)):
            if layer == "shell" and not spec.get("shell"):
                continue
            # the shell is wider, never taller: scaled up from the foot it stood over the tip as a cap
            o = {"rot": (0.0, yaw, 0.0), "scale": (sc * k, sc, sc * k), "pos": (x, -h, z)}
            R = T._rot_zyx(o["rot"]) * np.array(o["scale"])
            for pos, nrm, uv, t in local:
                wp = pos @ R.T + np.array(o["pos"])
                wn = nrm @ T._rot_zyx(o["rot"]).T
                c = np.clip(bright * 128, 0, 255)
                col = np.tile(np.array([c, c, c, alpha], np.uint8), (len(wp), 1))
                tri = t[:, [0, 2, 1]]                              # retail winding, turned for the cave frame
                out.append({"texture": tex16, "flags": flags, "positions": fz.from_ffxi(wp).astype(np.float32),
                            "normals": fz.from_ffxi(wn).astype(np.float32), "uvs": uv, "colours": col,
                            "triangles": tri.astype(np.int32)})
                if layer == "body":
                    solid.append(fz.from_ffxi(wp)[tri.reshape(-1)].reshape(-1, 3, 3))
    return out, (np.concatenate(solid) if solid else np.zeros((0, 3, 3))), {tex16: raw}

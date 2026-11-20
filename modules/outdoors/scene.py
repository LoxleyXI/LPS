"""A scene: a piece of a real zone kept where it is, with our own pieces added to it.

Where outdoors lays a new zone from a source's tiles, a scene keeps a window of the source as it
stands: its land, props and islands at their own positions, so its own placed effects (the sea, the
surf on its beaches) still line up with them and can stay. Everything outside the window goes, and
so do the effect folders not named to stay. Our pieces (architecture.py) are laid on it, and
invisible fences keep players to the part that's walked.

    "outdoor": {"overlay": "ffxi-hd", "draw": 400,
                "scene": {"from": 126,
                          "keep": [x0, x1, z0, z1],          the source's objects inside this stay
                          "drop": ["zil_", "del_"],          ...unless their name starts so
                          "effects": ["umi1"],               placed effect folders that stay
                          "pieces": [{"piece": "jetty", "palette": "qufim_wood", "pos": [x, height, z],
                                      "yaw": 3.14159, "args": {...}}],
                          "fences": [[[x, z], [x, z], ...]]}}  polylines of invisible wall

Frames: FFXI's (x east, z north), heights up.
"""
import math

import numpy as np

from zonekit import writer as fz
from zonekit import tiles as T

from . import architecture as A
from . import effects as E
from .assemble import lay_geometry, laid_meshes, ours
from .kit import terrain


def _bounds(mesh, o):
    """A placed object's world box from all its parts (a part of only snow over the ground still draws)."""
    pts = [p["pos"][fz.part_triangles(p, bool(mesh["config"] & 1))].reshape(-1, 3) for p in mesh["parts"]]
    pts = np.concatenate([p for p in pts if len(p)])
    lo, hi = pts.min(0), pts.max(0)
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = c @ (T._rot_zyx(o["rot"]) * np.array(o["scale"])).T + np.array(o["pos"])
    return w.min(0), w.max(0)


def _fence(line, low=-40.0, high=120.0):
    """Invisible wall along a polyline (x, z), both faces, from `low` to `high` (heights)."""
    quads = []
    for (ax, az), (bx, bz) in zip(line, line[1:]):
        a, b = np.array([ax, -low, az]), np.array([bx, -low, bz])      # FFXI: y down
        a2, b2 = np.array([ax, -high, az]), np.array([bx, -high, bz])
        for tri in ((a, b, b2), (a, b2, a2)):
            quads.append(fz.from_ffxi(np.array(tri)))
            quads.append(fz.from_ffxi(np.array(tri))[[0, 2, 1]])
    return np.array(quads).reshape(-1, 3, 3)


def textures_by_short(dat):
    """{short name: 16-byte name} of a zone's own textures."""
    full = {}
    for sec in fz.split_sections(bytes(dat)):
        if sec.kind == fz.T_TEXTURE:
            full.setdefault(sec.raw[0x11:0x21].decode("ascii", "replace").split()[-1], sec.raw[0x11:0x21])
    return full


def _placed_effects(dat):
    """{(folder kind, folder name)} of the source's placed (not ambient) effect folders."""
    secs = fz.split_sections(bytes(dat))
    paths = dict(fz.walk_tree(secs))
    resources = {s.name for s in secs if s.kind not in (E.EFFECT, fz.T_DIR, fz.T_END)}
    out = set()
    for i, s in enumerate(secs):
        pth = paths.get(i, ())
        if s.kind != E.EFFECT or len(pth) < 3 or pth[1] not in (b"effe", b"mode"):
            continue
        off = E._pos_offset(s.raw[16:], resources)
        if off is None:
            continue
        x, _, z = np.frombuffer(s.raw, "<f4", 3, 16 + off)
        if abs(x) > 1 or abs(z) > 1:
            out.add((pth[1], pth[2]))
    return out


def generate(base_dat, settings, name_of):
    """The zone DAT (bytes) and a summary, for an "outdoor" block with a "scene"."""
    sc = settings["scene"]
    x0, x1, z0, z1 = sc["keep"]
    drop = tuple(sc.get("drop", ()))
    tris, objs, meshes = T.read(base_dat)
    objects, col, used = [], [], set()
    for o in objs:
        n = o["name"]
        x, z = o["pos"][0], o["pos"][2]
        if not (x0 <= x <= x1 and z0 <= z <= z1) or n not in meshes or name_of(n).startswith(drop):
            continue
        if not any(len(fz.part_triangles(p, bool(meshes[n]["config"] & 1))) for p in meshes[n]["parts"]):
            continue
        lo, hi = _bounds(meshes[n], o)
        objects.append({"name": n, "pos": np.array(o["pos"], np.float64), "rot": tuple(o["rot"]),
                        "scale": tuple(o["scale"]), "lo": lo, "hi": hi, "lod": o["lod"]})
        used.add(n)
        if n in tris and max(hi[0] - lo[0], hi[2] - lo[2]) > 2.5:   # tufts and pebbles: walked through
            col.append(ours(T.placed(tris[n], o).reshape(-1, 3, 3), o["scale"]))

    laid, full = {}, textures_by_short(base_dat)
    for piece in sc.get("pieces", []):
        kw = piece.get("args", {})
        mesh = A.build(piece["piece"], piece.get("palette", "ronfaure"), **kw)
        px, h, pz = piece["pos"]
        o = {"rot": (0.0, piece.get("yaw", 0.0), 0.0), "scale": (1.0, 1.0, 1.0), "pos": (px, -h, pz)}
        lay_geometry(mesh, o, full, laid)
        solid = getattr(A, piece["piece"] + "_collision", None)
        if solid:
            keep = {k: kw[k] for k in ("length", "width", "deck") if k in kw}
            col.append(ours(T.placed(A.triangles(solid(**keep)), o).reshape(-1, 3, 3), o["scale"]))

    tris_c = np.concatenate(col)
    n = np.cross(tris_c[:, 1] - tris_c[:, 0], tris_c[:, 2] - tris_c[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    fences = [_fence(line) for line in sc.get("fences", [])]
    fences = np.concatenate(fences) if fences else np.zeros((0, 3, 3))
    tris_c = np.concatenate([tris_c, fences])
    up = np.concatenate([n[:, 2], np.zeros(len(fences))])
    terr = terrain(base_dat)
    keep_fx = {f.encode() for f in sc.get("effects", [])}
    drop_dirs = [f for f in _placed_effects(base_dat) if not (f[0] == b"effe" and f[1] in keep_fx)]
    geometry = {"meshes": laid_meshes(laid),
                "collision": {"positions": tris_c.reshape(-1, 3).astype(np.float32),
                              "triangles": np.arange(len(tris_c) * 3).reshape(-1, 3).astype(np.int32),
                              "wall": np.abs(up) < 0.5,
                              "terrain": np.where(up > 0.6, terr["floor"], terr["wall"]).astype(np.uint32)},
                "props": {"sections": [], "textures": {}, "objects": objects}}
    stats = {}
    out = fz.build_zone(base_dat, geometry, draw=settings.get("draw", 400), keep_meshes=used,
                        report=stats, drop_dirs=tuple(drop_dirs))
    return out, {"objects": len(objects), "pieces": len(sc.get("pieces", [])),
                 "effects_dropped": sorted(b"/".join(f).decode() for f in drop_dirs)}

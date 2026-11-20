"""Kits: what a source zone's textures and props are, and how that zone lays them.

A kit is a JSON file in caves/kits/ that belongs to a source zone, not to any cave built from it:
the facts about its assets live here once, and every cave that uses the kit gets them. A zone's
"generate" block names the kit and says only what is its own (seed, rooms, water features, mobs):

    "generate": {"kit": "beaucedine", "seed": 1336, "rooms": 6, "extent": 150,
                 "water": {"features": ["lake", "river", "moat"]},
                 "props": {"rock": [1, 2], "crystal": [2, 4]}}

Anything the zone sets beats the kit: "surfaces": {"floor": {"repeat": 30}} changes one value, and
the older direct keys ("textures", "uv", "foot", "overlay", "water", "props" with "rules") still work.
"props" picks kit groups by name, with per_lobe as a pair or any of the group's keys as a dict;
leave it out for every group at the kit's defaults, or give [] for none.

scan() drafts a kit from a zone DAT: how its own meshes use each texture (which part of it, how many
yalms a repeat covers, which way it faces) and which props it places, and for an outdoor zone its
land tiles read as shapes (zonekit/tiles.py). See README.md.
"""
import json
from pathlib import Path

import numpy as np

from zonekit.tiles import _rot_zyx

HERE = Path(__file__).resolve().parent
KITS = HERE / "kits"
UV_SURFACES = ("floor", "wall", "ceiling", "shore", "bed")     # surfaces laid with a flat repeat


def load(name):
    path = Path(name) if str(name).endswith(".json") else KITS / f"{name}.json"
    return json.loads(path.read_text())


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(settings):
    """A generate block with its kit expanded into the settings the generator reads (textures, uv,
    foot, overlay, water, textures_from, props with rules). Without a kit it comes back unchanged."""
    if "kit" not in settings:
        return settings
    kit = load(settings["kit"])
    surf = _merge(kit.get("surfaces", {}), settings.get("surfaces"))
    out = {k: v for k, v in settings.items() if k not in ("kit", "surfaces", "props")}
    textures = {s: v["texture"] for s, v in surf.items() if s in UV_SURFACES + ("foot",)}
    out["textures"] = dict(textures, **settings.get("textures", {}))
    out.setdefault("textures_from", kit["from"])
    uv = {s: v["repeat"] for s, v in surf.items() if s in UV_SURFACES and "repeat" in v}
    if uv:
        out["uv"] = dict(uv, **settings.get("uv", {}))
    if "foot" in surf:
        out["foot"] = _merge({k: v for k, v in surf["foot"].items() if k not in ("texture", "note")},
                             settings.get("foot"))
    for key in ("overlay", "water"):
        if key in surf or key in settings:
            if key == "water" and not settings.get("water"):
                continue                                  # water is the zone's choice: no features, no water
            out[key] = _merge({k: v for k, v in surf.get(key, {}).items() if k != "note"}, settings.get(key))
    out["props"] = _props(kit, settings.get("props"))
    if out["props"] is None:
        del out["props"]
    return out


def _props(kit, chosen):
    groups = kit.get("props", {})
    if isinstance(chosen, dict) and "rules" in chosen:
        return chosen                                     # the older direct form
    if chosen is None:
        chosen = {g: {} for g in groups}
    if not chosen:
        return None
    rules = []
    for name, over in chosen.items():
        if name not in groups:
            raise ValueError(f"kit {kit.get('name')}: no prop group {name!r} (has {sorted(groups)})")
        if isinstance(over, list):
            over = {"per_lobe": over}
        rule = {k: v for k, v in _merge(groups[name], over).items() if k != "note"}
        rules.append(rule)
    return {"from": kit["from"], "rules": rules}


# --- scanning a source zone -----------------------------------------------------------------------



def scan(dat, zone_id=None, min_tris=40):
    """A draft kit for a zone DAT: {"from", "measured": {texture: facts}, "props_found": {mesh: facts}}.
    Facts come from the zone's placed meshes in world space (FFXI y is down, so "up" is -y)."""
    from zonekit import writer as fz
    tables = fz.key_tables()
    sections = fz.split_sections(bytes(dat))
    meshes = {}
    zd = None
    for s in sections:
        if s.kind == fz.T_MESH:
            body = bytearray(s.raw[0x10:])
            fz.mesh_decrypt(body, tables)
            m = fz.read_mesh(body)
            meshes.setdefault(bytes(m["name"]), m)
        elif s.kind == fz.T_ZONEDEF and zd is None:
            body = bytearray(s.raw[0x10:])
            fz.zonedef_decrypt(body, tables)
            zd = fz.read_zonedef(body)
    tex = {}
    placed = {}
    for o in (zd or {}).get("objects", []):
        m = meshes.get(o["name"])
        if m is None:
            continue
        placed.setdefault(o["name"], []).append(o)
        if not _is_terrain(m, o):
            continue                                      # props' textures say nothing about laying land
        R = _rot_zyx(o["rot"]) * np.array(o["scale"])
        for p in m["parts"]:
            t = fz.part_triangles(p, bool(m["config"] & 1))
            if not len(t):
                continue
            w = p["pos"][t].astype(np.float64) @ R.T + np.array(o["pos"])
            uv = p["uv"][t].astype(np.float64)
            d = tex.setdefault(bytes(p["texture"]), {"w": [], "uv": [], "meshes": set()})
            d["w"].append(w)
            d["uv"].append(uv)
            d["meshes"].add(o["name"])
    ground = _ground_grid([np.concatenate(d["w"]) for d in tex.values()])
    measured = {}
    for name16, d in tex.items():
        w, uv = np.concatenate(d["w"]), np.concatenate(d["uv"])
        if len(w) < min_tris:
            continue
        measured[_tex_name(name16)] = _texture_facts(w, uv, d["meshes"], ground)
    measured = dict(sorted(measured.items(), key=lambda kv: -kv[1]["triangles"]))
    props = {}
    for name16, objs in placed.items():
        m = meshes[name16]
        pts = np.concatenate([p["pos"] for p in m["parts"]]) if m["parts"] else np.zeros((1, 3))
        size = (pts.max(0) - pts.min(0)) * float(np.median([o["scale"][0] for o in objs]))
        if _is_terrain(m, objs[0]) or len(objs) < 2 or _name(name16).startswith("hitwall"):
            continue                                      # terrain pieces and one-offs aren't props
        props[_name(name16)] = {"placed": len(objs), "size": [round(float(x), 1) for x in size],
                                "cutout": _name(name16).startswith("_"),
                                "textures": sorted({_tex_name(bytes(p["texture"])) for p in m["parts"]})}
    props = dict(sorted(props.items(), key=lambda kv: -kv[1]["placed"]))
    from zonekit import tiles
    draft = {"name": "", "from": zone_id, "surfaces": _draft_surfaces(measured), "props": {},
             "measured": measured, "props_found": props}
    land = tiles.scan(dat, _name)
    if land:                                              # an outdoor zone: its land is a tile kit
        draft["tiles"] = land
    return draft


def _draft_surfaces(measured):
    """First guesses from the measurements: the most used texture of each kind. A person still has
    to look: an atlas (a strip, a second texture in a corner) only shows in the image itself."""
    pick = {}
    for name, f in measured.items():                      # most triangles first
        pick.setdefault(f["suggest"], (name, f))
    out = {}
    if "floor" in pick:
        name, f = pick["floor"]
        out["floor"] = {"texture": name, "repeat": round(max(f["yalms_per_u"], f["yalms_per_v"]) * 2),
                        "note": "draft: floors want a big repeat (the zone's own is small but broken up by its props)"}
        out["shore"] = {"texture": name, "repeat": round(max(f["yalms_per_u"], f["yalms_per_v"]))}
    if "wall" in pick:
        name, f = pick["wall"]
        out["wall"] = {"texture": name, "repeat": round(f["yalms_per_u"] * 1.5), "note": "draft"}
        out["ceiling"] = {"texture": name, "repeat": round(f["yalms_per_u"] * 1.5)}
    if "foot" in pick:
        name, f = pick["foot"]
        v = [0.0, 1.0] if f.get("v_at_ground") == "high" else [1.0, 0.0]
        out["foot"] = {"texture": name, "height": f["yalms_per_v"], "repeat": round(f["yalms_per_u"]),
                       "u_max": 1.0, "v": v,
                       "note": "draft: open the texture; if it is an atlas, narrow v and u_max to the part that belongs at a wall's foot"}
    return out


def _is_terrain(m, o):
    """Land pieces are big; props (rocks, trees, flowers) are small. 30 yalms splits them in the zones
    looked at so far."""
    pts = np.concatenate([p["pos"] for p in m["parts"]]) if m["parts"] else np.zeros((1, 3))
    return float((pts.max(0) - pts.min(0)).max() * max(o["scale"])) > 30


def _name(b):
    return b.split(b"\0")[0].decode("ascii", "replace").rstrip()


def _tex_name(b):
    """A texture's short name, as surfaces name it ("model   gar_frs1" -> "gar_frs1")."""
    parts = _name(b).split()
    return parts[-1] if parts else "(unnamed)"


def _ground_grid(tris, cell=4.0):
    """The zone's ground height (up, in yalms) on an x/z grid, from every upward-facing triangle;
    cells with none take their nearest neighbour's. Returns a lookup for world points."""
    from scipy import ndimage
    w = np.concatenate(tris)
    n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    c = w.mean(1)[np.abs(n[:, 1]) > 0.7]
    lo = c[:, [0, 2]].min(0)
    ij = ((c[:, [0, 2]] - lo) / cell).astype(int)
    shape = ij.max(0) + 1
    g = np.full(shape, np.inf)
    np.minimum.at(g, (ij[:, 0], ij[:, 1]), -c[:, 1])        # the lowest floor in each cell
    _, (ii, jj) = ndimage.distance_transform_edt(~np.isfinite(g), return_indices=True)
    g = g[ii, jj]

    def at(q):
        k = np.clip(((q[:, [0, 2]] - lo) / cell).astype(int), 0, shape - 1)
        return g[k[:, 0], k[:, 1]]
    return at


def _texture_facts(w, uv, meshes, ground):
    """How a texture is laid: facing, the part of it used, and yalms per repeat along u and v."""
    n = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
    area = np.linalg.norm(n, axis=1)
    n = n / (area[:, None] + 1e-12)
    up = np.abs(n[:, 1])                                  # FFXI y is the vertical
    facing = {"floor": float(area[up > 0.7].sum()), "wall": float(area[up <= 0.7].sum())}
    tot = sum(facing.values()) or 1.0
    facing = {k: round(v / tot, 2) for k, v in facing.items()}
    # the uv -> world gradient per triangle: dP/du and dP/dv
    e1, e2 = w[:, 1] - w[:, 0], w[:, 2] - w[:, 0]
    t1, t2 = uv[:, 1] - uv[:, 0], uv[:, 2] - uv[:, 0]
    det = t1[:, 0] * t2[:, 1] - t2[:, 0] * t1[:, 1]
    ok = np.abs(det) > 1e-8
    inv = 1 / det[ok]
    dpdu = (e1[ok] * t2[ok, 1:2] - e2[ok] * t1[ok, 1:2]) * inv[:, None]
    dpdv = (e2[ok] * t1[ok, 0:1] - e1[ok] * t2[ok, 0:1]) * inv[:, None]
    wts = area[ok]
    flat = uv.reshape(-1, 2)
    lo, hi = np.percentile(flat, 1, axis=0), np.percentile(flat, 99, axis=0)
    cen = w.mean(1)
    above = -cen[:, 1] - ground(cen)                      # height over the lowest floor below it
    facts = {"triangles": int(len(w)), "meshes": len(meshes), "facing": facing,
             "u": [round(float(lo[0]), 2), round(float(hi[0]), 2)], "v": [round(float(lo[1]), 2), round(float(hi[1]), 2)],
             "above_ground": [round(float(np.percentile(above, 10)), 1), round(float(np.percentile(above, 90)), 1)],
             "yalms_per_u": round(_wmedian(np.linalg.norm(dpdu, axis=1), wts), 1),
             "yalms_per_v": round(_wmedian(np.linalg.norm(dpdv, axis=1), wts), 1)}
    if facing["wall"] > 0.5:
        # on walls: does v run up the wall, and which end is the ground
        vert = -dpdv[:, 1]                                # world up per unit of v (FFXI y down)
        facts["v_up_wall"] = round(_wmedian(np.abs(vert) / (np.linalg.norm(dpdv, axis=1) + 1e-9), wts), 2)
        facts["v_at_ground"] = "high" if _wmedian(np.sign(vert), wts) < 0 else "low"
    wall_low = (facing["wall"] > 0.5 and facts.get("v_up_wall", 0) > 0.9      # v runs up the wall
                and abs(facts["above_ground"][0]) < 1.5)                      # from the ground itself
    facts["suggest"] = ("floor" if facing["floor"] > 0.7 else
                        "foot" if wall_low else          # a strip up from the ground: see README
                        "wall" if facing["wall"] > 0.5 else "mixed")
    return facts


def _wmedian(x, w):
    if not len(x):
        return 0.0
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return float(x[o][np.searchsorted(c, c[-1] / 2)])

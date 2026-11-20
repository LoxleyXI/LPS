"""Client models for a generated zone's monsters: oversized / recoloured copies made with the LPS mobs module's
recolour (hue, tint, blend, colour ranges, scale, joint scales), registered in an overlay's ROM10.

Each type's "model": {"clone": <model id>, "scale": 2.0, "hue": ..., "tint": ..., "blend_mode": ...};
its new model id is types[key]["model_id"]. Rebuilding reuses the model's existing slot.

A "skin" {"dir": <folder in mobs/skins>, "parts": "parts.json", "materials": {texture name:
recipe.json}} paints the model with our own textures (mobs/skin.py) in place of the recolour (its
scale still applies); a preview of each lands in that folder's check/.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent


def make(t, key, src, dst):
    """One type's model: `src` (the clone's DAT) recoloured or painted, and scaled, into `dst`."""
    import mobs
    spec = t["model"]
    skinned = spec.get("skin")
    recolour = {} if skinned else dict(hue=spec.get("hue", 0), saturation=spec.get("saturation", 0),
                                       lightness=spec.get("lightness", 0), tint=spec.get("tint"),
                                       blend_mode=spec.get("blend_mode", "normal"),
                                       color_ranges=spec.get("color_ranges"))
    stats = mobs.recolor_entity_dat(src, dst, scale=spec.get("scale", 0), joint_scales=None, **recolour)
    if skinned:
        stats["skin"] = _skin(dst, skinned, t.get("name", key))
    return stats


def build(types, ffxi, overlay):
    """Build every type's model into overlay/ROM10 and register it in overlay's FTABLE10/VTABLE10."""
    from xidat.ftable import load_all, resolve_model, model_fid, slot_for, register
    from xidat.paths import hd
    tables = load_all(ffxi)
    rom10 = overlay / "ROM10"
    ft10, vt10 = rom10 / "FTABLE10.DAT", rom10 / "VTABLE10.DAT"
    done = []
    for key, t in types.items():
        src = hd(resolve_model(t["model"]["clone"], tables, ffxi), ffxi, overlay)   # its HD copy, if any
        fid = model_fid(t["model_id"])
        # its own slot on a rebuild; never one another id points at (an overlay file shadows
        # the game's file of that number for every id using it)
        subdir, slot = slot_for(rom10, ffxi / "ROM10", fid)
        dst = rom10 / str(subdir) / f"{slot}.DAT"
        dst.parent.mkdir(parents=True, exist_ok=True)
        stats = make(t, key, src, dst)
        register(ft10, vt10, fid, subdir, slot)
        done.append((key, t["model_id"], f"ROM10/{subdir}/{slot}.DAT", stats))
    return done


def export(types, ffxi, out_dir):
    """Every type's model as <out_dir>/<model id>.DAT, registered nowhere (written to a folder
    rather than installed). Returns {model id: file name}."""
    from xidat.ftable import load_all, resolve_model
    from xidat.paths import hd
    tables = load_all(ffxi)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {}
    for key, t in types.items():
        dst = out_dir / f"{t['model_id']}.DAT"
        make(t, key, hd(resolve_model(t["model"]["clone"], tables, ffxi), ffxi), dst)
        done[t["model_id"]] = dst.name
    return done


def _skin(dat, spec, name):
    """Paint `dat` (an entity DAT, in place) with our own textures: spec's "materials" recipes, on
    the parts named in its "parts" file, both in mobs/skins/<dir>."""
    import json
    from PIL import Image
    from mobs import skin
    here = HERE.parent / "mobs" / "skins" / spec["dir"]
    parts = spec.get("parts")
    if isinstance(parts, str):
        parts = json.loads((here / parts).read_text())["parts"]
    data = dat.read_bytes()
    tris, infos = skin.triangles(data), skin.textures(data)
    pictures = {}
    for tex, recipe in spec["materials"].items():
        recipe = json.loads((here / recipe).read_text()) if isinstance(recipe, str) else recipe
        original = skin.texture_rgba(data, tex) if recipe.get("base") == "original" else None
        everything = [t for ts in tris.values() for t in ts]       # one body: no seams between textures
        pictures[tex] = Image.fromarray(skin.materials(tris[tex], infos[tex].width, recipe, parts, here, original, everything))
    data = skin.paint(data, pictures, parts=parts, check=here / "check")
    dat.write_bytes(data)
    skin.preview(data, here / "check" / f"{name}.preview.png")
    return list(pictures)

#!/usr/bin/env python3
"""Parchment maps of a generated outdoor zone, one sheet per land, in the style of the game's own
area maps: the walkable land pale on aged paper, a dark ink edge, each terrace a shade deeper, a
lettered grid, a title ribbon, a compass rose and italic place names.

    python3 modules/outdoors/maps.py examples/zones/sand_coast.json OUT

The layout is generated again from the zone file (the same seed gives the same land as the
installed zone). Place names come from the zone file's "map" block:
    "map": {"labels": [{"land": "North Field", "text": "Old Watchtower", "at": [x, z]}, ...]}
with each land's camp named "<land> Camp" unless a label says otherwise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))

from zonekit import tiles as T                      # noqa: E402

from outdoors.kit import HOLE                        # noqa: E402

FONTS = Path("/usr/share/fonts/opentype/urw-base35")
INK = (92, 58, 30)
PAPER = (222, 198, 150)
LAND = (238, 222, 184)
SIZE = 1024
GRID = (16, 15)                                       # A-P across, 1-15 down, as the game's maps


def _font(name, px):
    try:
        return ImageFont.truetype(str(FONTS / name), px)
    except OSError:
        return ImageFont.load_default()


def _noise(n, scale, seed):
    """Smooth value noise in 0..1, (n, n)."""
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, n // scale), max(2, n // scale)))
    img = Image.fromarray((small * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)
    return np.asarray(img, np.float32) / 255


def parchment(seed=1):
    """(SIZE, SIZE, 3) aged paper with stains, and burnt ragged edges (RGBA)."""
    n = SIZE
    tone = 0.55 * _noise(n, 160, seed) + 0.3 * _noise(n, 40, seed + 1) + 0.15 * _noise(n, 8, seed + 2)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1) - 0.5
    vignette = np.clip(1 - 1.6 * (xx ** 2 + yy ** 2), 0, 1)
    shade = (0.78 + 0.22 * tone) * (0.82 + 0.18 * vignette)
    img = np.array(PAPER, np.float32)[None, None] * shade[..., None]
    # burnt edge: distance to the sheet's edge, roughened
    edge = np.minimum(np.minimum(xx + 0.5, 0.5 - xx), np.minimum(yy + 0.5, 0.5 - yy))
    rough = edge - 0.022 * _noise(n, 24, seed + 3) - 0.01 * _noise(n, 6, seed + 4)
    char = np.clip((0.03 - rough) / 0.02, 0, 1)
    img = img * (1 - char[..., None]) + np.array([40, 22, 10], np.float32) * char[..., None]
    alpha = np.where(rough < 0.004, 0, 255).astype(np.uint8)
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8), alpha])


def compass(draw, cx, cy, r):
    """A plain eight-point rose with N at the top."""
    for k in range(8):
        a = k * math.pi / 4
        long = r if k % 2 == 0 else r * 0.55
        tip = (cx + math.sin(a) * long, cy - math.cos(a) * long)
        side = r * 0.13
        left = (cx + math.sin(a - math.pi / 2) * side, cy - math.cos(a - math.pi / 2) * side)
        right = (cx + math.sin(a + math.pi / 2) * side, cy - math.cos(a + math.pi / 2) * side)
        draw.polygon([tip, left, (cx, cy)], fill=INK)
        draw.polygon([tip, right, (cx, cy)], outline=INK, fill=PAPER)
    draw.ellipse([cx - r * 0.18, cy - r * 0.18, cx + r * 0.18, cy + r * 0.18], outline=INK, width=2)
    draw.ellipse([cx - r * 0.75, cy - r * 0.75, cx + r * 0.75, cy + r * 0.75], outline=INK, width=1)
    f = _font("P052-Bold.otf", int(r * 0.32))
    draw.text((cx, cy - r - r * 0.05), "N", font=f, fill=INK, anchor="md")


def ribbon(draw, text, cx, cy):
    f = _font("P052-Bold.otf", 30)
    w = draw.textlength(text, font=f) + 70
    h = 44
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    fold = 22
    for side in (-1, 1):                               # the ends folded behind
        ex = x0 if side < 0 else x1
        draw.polygon([(ex, y0 + 10), (ex + side * fold, y0 + 10), (ex + side * (fold - 10), cy + 10),
                      (ex + side * fold, y1 + 10), (ex, y1 + 10)], fill=(200, 176, 128), outline=INK)
    draw.rectangle([x0, y0, x1, y1], fill=(232, 212, 168), outline=INK, width=2)
    draw.line([x0 + 6, y0 + 5, x1 - 6, y0 + 5], fill=INK, width=1)
    draw.line([x0 + 6, y1 - 5, x1 - 6, y1 - 5], fill=INK, width=1)
    draw.text((cx, cy + 1), text.upper(), font=f, fill=INK, anchor="mm")


def extent(land):
    """The land's laid cells as (x0, z0, x1, z1), local."""
    cells = np.array(list(land["placed"]))
    x0, z0 = cells.min(0) * T.TILE
    x1, z1 = (cells.max(0) + 1) * T.TILE
    return x0, z0, x1, z1


def draw_land(name, land, labels, out, seed=1, walls=(), span=None, bare=False, dots=(), legend=(), notes=()):
    """One sheet: land (a summary or region summary with "surface", "placed", "shift"). With
    `span` (yalms across), the scale is set (an atlas keeps every land at one scale); `bare`
    returns the inked land alone, (n, n) RGBA, and its labels in its pixels, for atlas().
    `dots` [(x, z, (r, g, b))] in the land's own coords are drawn over it (monsters, for a
    preview), with `legend` [(text, (r, g, b))] and `notes` (lines of text) in the corner."""
    surf, placed = land["surface"], land["placed"]
    sx, sz = land["shift"][0], land["shift"][2]
    cells = np.array(list(placed))
    x0, z0 = cells.min(0) * T.TILE
    x1, z1 = (cells.max(0) + 1) * T.TILE
    # the area drawn: the land's extent, square, with room for the frame
    cxw, czw = (x0 + x1) / 2, (z0 + z1) / 2
    span = span or max(x1 - x0, z1 - z0) * 1.18
    inner = (SIZE * 0.1, SIZE * 0.12, SIZE * 0.9, SIZE * 0.92)    # map area on the sheet
    side = inner[2] - inner[0]

    def to_px(x, z):                                    # local land coords -> sheet pixels (north up)
        return (inner[0] + ((x - cxw) / span + 0.5) * side, inner[1] + (0.5 - (z - czw) / span) * side)

    # heights over the map area
    n = int(side)
    us = (np.arange(n) + 0.5) / n
    gx = cxw + (us - 0.5) * span
    gz = czw + (0.5 - us) * span
    X, Z = np.meshgrid(gx, gz)
    H = surf.at(X.ravel(), Z.ravel()).reshape(n, n)
    from scipy import ndimage
    land_mask = np.isfinite(H) & (H > HOLE / 2)
    # only what can be walked on: cliff faces and the boundary walls rising out of the land are
    # steep, and the map stops where the ground does (the ink edge is the walkable edge)
    px = span / n
    Hf = np.where(land_mask, H, np.nan)
    gz_, gx_ = np.gradient(np.nan_to_num(Hf, nan=-1e3), px)
    land_mask &= np.hypot(gx_, gz_) < 0.85
    if walls:
        # the land's boundary tiles (its kit's wall family) rise out of it: what of them stands
        # above the land's highest walked ground is the map's edge, their foot is still land
        ci = np.floor(X / T.TILE).astype(int)
        cj = np.floor(Z / T.TILE).astype(int)
        wall_cells = {c for c, v in placed.items() if any(v.get("label", "").startswith(w) for w in walls)}
        in_wall = np.array([(a, b) in wall_cells for a, b in zip(ci.ravel(), cj.ravel())]).reshape(n, n)
        floor = H[land_mask & ~in_wall]
        if floor.size:
            land_mask &= ~(in_wall & (H > np.percentile(floor, 98) + 1.5))
    land_mask = ndimage.binary_opening(land_mask, iterations=2)
    lab, k = ndimage.label(land_mask)
    if k:
        sizes = ndimage.sum(land_mask, lab, range(1, k + 1))
        land_mask = np.isin(lab, 1 + np.nonzero(sizes >= 0.002 * n * n)[0])
    land_mask = ndimage.binary_fill_holes(land_mask) & np.isfinite(H) | land_mask
    # terraces: each level down a shade deeper
    levels = np.unique(np.round(H[land_mask] / 4) * 4)
    top = levels.max() if len(levels) else 0
    shade = np.zeros((n, n), np.float32)
    if len(levels) > 1:
        shade[land_mask] = np.clip((top - H[land_mask]) / max(1.0, top - levels.min()), 0, 1)
    sheet = parchment(seed)
    img = Image.fromarray(sheet, "RGBA")
    ink = Image.new("RGBA", img.size, (0, 0, 0, 0))
    # the land: pale, a little stained, darker where it sits lower
    stain = _noise(n, 30, seed + 7)
    col = np.array(LAND, np.float32)[None, None] * (0.93 + 0.07 * stain[..., None]) * (1 - 0.22 * shade[..., None])
    a = (land_mask * 235).astype(np.uint8)
    land_img = Image.fromarray(np.dstack([np.clip(col, 0, 255).astype(np.uint8), a]), "RGBA")
    m = Image.fromarray((land_mask * 255).astype(np.uint8))
    # ink: a soft brown halo outside the edge, then the edge itself
    halo = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.GaussianBlur(5))
    halo_a = np.asarray(halo, np.float32) * (1 - land_mask) * 0.55
    halo_img = Image.fromarray(np.dstack([np.full((n, n, 3), (150, 105, 60), np.uint8), halo_a.astype(np.uint8)]), "RGBA")
    edge = np.asarray(m.filter(ImageFilter.MaxFilter(3)), bool) & ~np.asarray(m.filter(ImageFilter.MinFilter(3)), bool)
    edge_img = Image.fromarray(np.dstack([np.full((n, n, 3), INK, np.uint8), (edge * 230).astype(np.uint8)]), "RGBA")
    # terrace steps inside the land: where the level changes
    step = np.zeros((n, n), bool)
    q = np.where(land_mask, np.round(np.nan_to_num(H) / 4), -999)    # a terrace step, not every bump
    step[:, 1:] |= (q[:, 1:] != q[:, :-1]) & land_mask[:, 1:] & land_mask[:, :-1]
    step[1:, :] |= (q[1:, :] != q[:-1, :]) & land_mask[1:, :] & land_mask[:-1, :]
    step_img = Image.fromarray(np.dstack([np.full((n, n, 3), INK, np.uint8), (step * 120).astype(np.uint8)]), "RGBA")
    off = (int(inner[0]), int(inner[1]))
    for layer in (halo_img, land_img, step_img, edge_img):
        ink.alpha_composite(layer, off)
    if bare:
        layer = ink.crop((off[0], off[1], off[0] + n, off[1] + n))
        spots = []
        for lab in labels:
            px, py = to_px(lab["at"][0] - sx, lab["at"][1] - sz)
            spots.append((lab, px - off[0], py - off[1]))
        return layer, spots
    img.alpha_composite(ink)
    d = ImageDraw.Draw(img)
    # the grid and its letters
    fl = _font("P052-Roman.otf", 16)
    for k in range(GRID[0] + 1):
        x = inner[0] + k * side / GRID[0]
        d.line([x, inner[1], x, inner[3]], fill=INK + (60,), width=1)
    for k in range(GRID[1] + 1):
        y = inner[1] + k * side / GRID[1]
        d.line([inner[0], y, inner[2], y], fill=INK + (60,), width=1)
    d.rectangle([inner[0], inner[1], inner[2], inner[3]], outline=INK, width=2)
    for k in range(GRID[0]):
        x = inner[0] + (k + 0.5) * side / GRID[0]
        for y in (inner[1] - 12, inner[3] + 12):
            d.text((x, y), chr(65 + k), font=fl, fill=INK, anchor="mm")
    for k in range(GRID[1]):
        y = inner[1] + (k + 0.5) * side / GRID[1]
        for x in (inner[0] - 14, inner[2] + 14):
            d.text((x, y), str(k + 1), font=fl, fill=INK, anchor="mm")
    ribbon(d, name, SIZE / 2, inner[1] + 46)
    compass(d, inner[2] - 78, inner[3] - 92, 58)
    # place names
    fi = _font("P052-Italic.otf", 22)
    for lab in labels:
        x, z = lab["at"][0] - sx, lab["at"][1] - sz
        px, py = to_px(x, z)
        if lab.get("mark", True):
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=INK, width=2)
        d.text((px, py - 10), lab["text"], font=fi, fill=INK, anchor="md")
    for x, z, col in dots:
        px, py = to_px(x, z)
        d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=tuple(col), outline=(40, 25, 10), width=1)
    fs = _font("P052-Roman.otf", 15)
    y = inner[1] + 86
    for text, col in legend:
        d.ellipse([inner[0] + 12, y - 6, inner[0] + 24, y + 6], fill=tuple(col), outline=(40, 25, 10))
        d.text((inner[0] + 30, y), text, font=fs, fill=INK, anchor="lm")
        y += 19
    y = inner[3] - 12 - 18 * len(notes)
    for line in notes:
        d.text((inner[0] + 12, y), line, font=fs, fill=INK, anchor="lm")
        y += 18
    img.save(out)
    return out


def atlas(title, lands, out, seed=9):
    """Every land on one sheet at one scale, their real distances shortened: three in a triangle
    (the first at the top, then west and east below, by where they really lie), else in a row.
    Each is named under its shape. lands: [(name, land, labels, walls)]."""
    span = max(max(e[2] - e[0], e[3] - e[1]) for e in (extent(l) for _, l, _, _ in lands)) * 1.1
    k = len(lands)
    panel = 560
    if k == 3:
        W = Hh = 1400
        # west and east by the lands' real positions (their shift along x)
        rest = sorted(lands[1:], key=lambda e: e[1]["shift"][0])
        order = [lands[0]] + rest
        centres = [(W / 2, 140 + panel / 2), (W / 2 - panel * 0.6, 140 + panel * 1.55), (W / 2 + panel * 0.6, 140 + panel * 1.55)]
    else:
        W, Hh = panel * k + 160, panel + 300
        order = lands
        centres = [(80 + panel * (i + 0.5), 150 + panel / 2) for i in range(k)]
    global SIZE
    keep, SIZE = SIZE, max(W, Hh)
    paper = parchment(seed)
    SIZE = keep
    sheet = Image.fromarray(np.asarray(Image.fromarray(paper, "RGBA").resize((W, Hh))), "RGBA")
    d = ImageDraw.Draw(sheet)
    inner = (70, 70, W - 70, Hh - 70)
    for c in range(1, 16):
        x = inner[0] + c * (inner[2] - inner[0]) / 16
        d.line([x, inner[1], x, inner[3]], fill=INK + (40,), width=1)
    for r in range(1, 16):
        y = inner[1] + r * (inner[3] - inner[1]) / 16
        d.line([inner[0], y, inner[2], y], fill=INK + (40,), width=1)
    d.rectangle(inner, outline=INK, width=2)
    fname = _font("P052-BoldItalic.otf", 30)
    fi = _font("P052-Italic.otf", 22)
    for i, ((name, land, labels, walls), (cx, cy)) in enumerate(zip(order, centres)):
        layer, spots = draw_land(name, land, labels, None, seed=i + 1, walls=walls, span=span, bare=True)
        f = panel / layer.width
        layer = layer.resize((panel, panel), Image.LANCZOS)
        ox, oy = int(cx - panel / 2), int(cy - panel / 2)
        sheet.alpha_composite(layer, (ox, oy))
        for lab, px, py in spots:
            px, py = ox + px * f, oy + py * f
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=INK, width=2)
            d.text((px, py - 10), lab["text"], font=fi, fill=INK, anchor="md")
        box = layer.getchannel("A").point(lambda v: 255 if v > 120 else 0).getbbox()   # the land's own shape
        below = oy + (box[3] if box else panel - 40) + 34
        d.text((ox + ((box[0] + box[2]) / 2 if box else panel / 2), below), name, font=fname, fill=INK, anchor="mm")
    ribbon(d, title, W / 2, 120 if k == 3 else 80)
    compass(d, inner[2] - 90, inner[1] + 110, 60)
    sheet.save(out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("zone", help="zone file (zones/<name>.json or a path)")
    ap.add_argument("out", help="folder for the sheets")
    args = ap.parse_args()
    import caves
    from caves.kits import _name
    import outdoors
    path = Path(args.zone) if Path(args.zone).exists() else HERE / args.zone
    entry = json.loads(path.read_text())["zones"][0]
    spec = entry["outdoor"]
    kit = outdoors.kit.config(spec["kit"])
    _, summary = outdoors.generate(caves.zone_dat(kit["from"]).read_bytes(), spec, _name)
    lands = [(spec.get("land", entry["name"]), summary)] + [(r["name"], r) for r in summary.get("regions", [])]
    kits = [spec["kit"]] + [r["kit"] for r in spec.get("regions", [])]
    walls = [(outdoors.kit.config(k).get("walls") or {}).get("families", []) for k in kits]
    named = entry.get("map", {}).get("labels", [])
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    every = []
    for k, (name, land) in enumerate(lands):
        labels = [dict(lab) for lab in named if lab.get("land") == name]
        fire = (land.get("town") or {}).get("fire")
        if fire and not any(lab.get("camp") for lab in labels):
            labels.append({"text": f"{name} Camp", "at": [fire[0], fire[2]]})
        stem = "".join(c if c.isalnum() else "_" for c in name.lower())
        print(draw_land(name, land, labels, out / f"{stem}.png", seed=k + 1, walls=walls[k]))
        every.append((name, land, labels, walls[k]))
    if len(every) > 1:
        # all of them on one sheet, named for the zone
        import re
        stem = re.sub(r"[^a-z0-9]+", "_", entry["name"].lower()).strip("_")
        print(atlas(entry["name"], every, out / f"{stem}.png"))


if __name__ == "__main__":
    main()

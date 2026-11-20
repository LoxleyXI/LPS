"""
LPS UI Icons: ustats sprite sheet read/write

The character nameplate sprite sheet (ustatshd, ROM/119/51.DAT) contains
both 16x16 and 32x32 icons in a single 256x256 DXT3 texture:

  ┌───────────────────────┬───────────────────────┐
  │  16x16 grid (8x4)    │  32x32 icons (4x2)    │  rows 0-1 (64px)
  │  128 x 64             │  128 x 64             │
  ├───────────────────────┴───────────────────────┤
  │  32x32 icons (8 cols, rows 2-7)               │  rows 2-7 (192px)
  │  256 x 192                                    │
  └───────────────────────────────────────────────┘

Some icons have both a 16x16 and 32x32 version (e.g. linkshell).
These are defined as compound slots with both pixel positions.

The SD sheet (ustats, ROM/280/15.DAT) is a separate 128x64 16x16 grid
that mirrors the small icons. It may not be used by the game client
but is patched for completeness.
"""

import struct
import re
import shutil
from pathlib import Path
from PIL import Image

from lps.constants import FFXI_PATH, OVERLAY_PATH


# ── Sprite sheet definitions ──

SHEETS = {
    'hd': {
        'dat': 'ROM/119/51.DAT',
        'name': 'ustatshd',
        'width': 256, 'height': 256,
        'cols': 8, 'rows': 8,
        'cell': 32,
    },
    'sd': {
        'dat': 'ROM/280/15.DAT',
        'name': 'ustats',
        'width': 128, 'height': 64,
        'cols': 8, 'rows': 4,
        'cell': 16,
    },
}

# Simple slots: (row, col) in the 32x32 grid
SLOT_NAMES = {
    'star':       (0, 3),
    'crest_san':  (0, 4),
    'crest_bas':  (0, 5),
    'crest_win':  (0, 6),
    'gm':         (2, 4),
    'mentor':     (3, 0),
    'flag_y':     (3, 4),
    'flag_w':     (3, 5),
}

# Compound slots: icons with both a 16x16 and 32x32 version in the HD sheet.
# Each entry is { 'small': (x, y), 'large': (x, y) } in pixel coordinates.
COMPOUND_SLOTS = {
    'linkshell':  { 'small': (0, 16), 'large': (0, 64) },
}


# ── DXT3 compression ──

def _rgb565(r, g, b):
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _rgb565_unpack(v):
    r = ((v >> 11) & 0x1F) * 255 // 31
    g = ((v >> 5) & 0x3F) * 255 // 63
    b = (v & 0x1F) * 255 // 31
    return r, g, b


def compress_dxt3_block(pixels_4x4):
    """Compress 16 RGBA pixels to 16 bytes DXT3."""
    alpha_bytes = bytearray(8)
    for i in range(16):
        a = pixels_4x4[i][3] >> 4
        byte_idx = i // 2
        if i % 2 == 0:
            alpha_bytes[byte_idx] |= a
        else:
            alpha_bytes[byte_idx] |= (a << 4)

    min_r = min_g = min_b = 255
    max_r = max_g = max_b = 0
    for r, g, b, a in pixels_4x4:
        min_r, min_g, min_b = min(min_r, r), min(min_g, g), min(min_b, b)
        max_r, max_g, max_b = max(max_r, r), max(max_g, g), max(max_b, b)

    c0 = _rgb565(max_r, max_g, max_b)
    c1 = _rgb565(min_r, min_g, min_b)
    if c0 < c1:
        c0, c1 = c1, c0
        max_r, max_g, max_b, min_r, min_g, min_b = min_r, min_g, min_b, max_r, max_g, max_b

    colors = [
        (max_r, max_g, max_b),
        (min_r, min_g, min_b),
        ((2*max_r+min_r+1)//3, (2*max_g+min_g+1)//3, (2*max_b+min_b+1)//3),
        ((max_r+2*min_r+1)//3, (max_g+2*min_g+1)//3, (max_b+2*min_b+1)//3),
    ]

    indices = 0
    for i in range(16):
        r, g, b, a = pixels_4x4[i]
        best_idx, best_dist = 0, float('inf')
        for j, (cr, cg, cb) in enumerate(colors):
            dist = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
            if dist < best_dist:
                best_dist = dist
                best_idx = j
        indices |= (best_idx << (i * 2))

    return bytes(alpha_bytes) + struct.pack('<HHI', c0, c1, indices)


def decompress_dxt3_block(block):
    """Decompress 16 bytes DXT3 → 16 RGBA tuples."""
    alphas = []
    for i in range(16):
        byte_idx = i // 2
        if i % 2 == 0:
            a4 = block[byte_idx] & 0x0F
        else:
            a4 = (block[byte_idx] >> 4) & 0x0F
        alphas.append(a4 * 17)

    c0 = block[8] | (block[9] << 8)
    c1 = block[10] | (block[11] << 8)
    r0, g0, b0 = _rgb565_unpack(c0)
    r1, g1, b1 = _rgb565_unpack(c1)

    lut = [
        (r0, g0, b0),
        (r1, g1, b1),
        ((2*r0+r1)//3, (2*g0+g1)//3, (2*b0+b1)//3),
        ((r0+2*r1)//3, (g0+2*g1)//3, (b0+2*b1)//3),
    ]

    bits = block[12] | (block[13] << 8) | (block[14] << 16) | (block[15] << 24)
    pixels = []
    for i in range(16):
        ci = (bits >> (i * 2)) & 3
        r, g, b = lut[ci]
        pixels.append((r, g, b, alphas[i]))

    return pixels


def compress_rgba_to_dxt3(rgba_img):
    """Compress a PIL RGBA Image to DXT3 bytes."""
    w, h = rgba_img.size
    pixels = list(rgba_img.getdata())
    result = bytearray()
    for by in range(h // 4):
        for bx in range(w // 4):
            block = []
            for y in range(4):
                for x in range(4):
                    block.append(pixels[(by*4+y) * w + (bx*4+x)])
            result.extend(compress_dxt3_block(block))
    return bytes(result)


def decompress_dxt3_to_rgba(dxt3_data, width, height):
    """Decompress DXT3 bytes to a PIL RGBA Image."""
    pixels = [(0, 0, 0, 0)] * (width * height)
    bp = 0
    for by in range(height // 4):
        for bx in range(width // 4):
            block = dxt3_data[bp:bp+16]
            bp += 16
            block_pixels = decompress_dxt3_block(block)
            for y in range(4):
                for x in range(4):
                    px = bx*4 + x
                    py = by*4 + y
                    if px < width and py < height:
                        pixels[py * width + px] = block_pixels[y*4 + x]

    img = Image.new('RGBA', (width, height))
    img.putdata(pixels)
    return img


# ── DAT container handling ──

ENTRY_HEADER_SIZE = 57
TXD_HEADER_SIZE = 12


def find_texture(data, name):
    """Find a named texture in a DAT, return (txd_offset, width, height, data_size)."""
    pattern = re.compile(b'[135]TXD')
    for m in pattern.finditer(data):
        txd_off = m.start()
        if txd_off < ENTRY_HEADER_SIZE:
            continue
        entry_off = txd_off - ENTRY_HEADER_SIZE
        if data[entry_off] != 0xa1:
            continue
        tex_name = data[entry_off+9:entry_off+17].rstrip(b'\x00').rstrip().decode('ascii', 'replace')
        if tex_name != name:
            continue
        width, height = struct.unpack_from('<II', data, entry_off + 21)
        data_size = struct.unpack_from('<I', data, txd_off + 4)[0]
        return txd_off, width, height, data_size
    return None


def read_sprite_sheet(sheet_key):
    """Read and decompress a sprite sheet from the game DAT."""
    sheet = SHEETS[sheet_key]
    dat_path = FFXI_PATH / sheet['dat']
    if not dat_path.exists():
        raise FileNotFoundError(f"DAT not found: {dat_path}")

    data = dat_path.read_bytes()
    result = find_texture(data, sheet['name'])
    if result is None:
        raise ValueError(f"Texture '{sheet['name']}' not found in {dat_path}")

    txd_off, width, height, data_size = result
    dxt3_data = data[txd_off + TXD_HEADER_SIZE : txd_off + TXD_HEADER_SIZE + data_size]
    return decompress_dxt3_to_rgba(dxt3_data, width, height)


def _write_sheet(sheet_key, sprite):
    """Compress and write a sprite sheet back to the overlay DAT."""
    sheet = SHEETS[sheet_key]
    dxt3_data = compress_rgba_to_dxt3(sprite)

    dat_path = FFXI_PATH / sheet['dat']
    data = bytearray(dat_path.read_bytes())
    result = find_texture(data, sheet['name'])
    if result is None:
        print(f"  Warning: texture '{sheet['name']}' not found, skipping {sheet_key}")
        return None

    txd_off, width, height, data_size = result

    if len(dxt3_data) != data_size:
        print(f"  Warning: DXT3 size mismatch ({len(dxt3_data)} vs {data_size}), skipping {sheet_key}")
        return None

    pixel_start = txd_off + TXD_HEADER_SIZE
    data[pixel_start:pixel_start + len(dxt3_data)] = dxt3_data

    overlay_dat = OVERLAY_PATH / sheet['dat']
    overlay_dat.parent.mkdir(parents=True, exist_ok=True)
    overlay_dat.write_bytes(bytes(data))
    return overlay_dat


def _paste_icon(sprite, icon, x, y, size):
    """Clear a cell to transparent and paste an icon onto the sprite sheet."""
    if icon.size != (size, size):
        icon = icon.resize((size, size), Image.Resampling.LANCZOS)
    clear = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sprite.paste(clear, (x, y))
    sprite.paste(icon, (x, y), icon)


def patch_icon(row, col, icon_path, hd_path=None, sd_path=None):
    """Patch a single icon into the HD sprite sheet (and SD for completeness).

    For simple grid slots, patches at the 32x32 grid position.
    Clears the cell to transparent before pasting to avoid background bleed.

    Args:
        row: Grid row (0-indexed, 32x32 grid)
        col: Grid column (0-indexed)
        icon_path: Path to PNG (scaled to fit)
        hd_path: Optional separate HD icon (32x32)
        sd_path: Optional separate SD icon (16x16)
    """
    results = []

    for sheet_key in ['hd', 'sd']:
        sheet = SHEETS[sheet_key]

        if sheet_key == 'sd' and row >= sheet['rows']:
            continue

        cell = sheet['cell']

        if sheet_key == 'hd' and hd_path:
            icon = Image.open(hd_path).convert('RGBA')
        elif sheet_key == 'sd' and sd_path:
            icon = Image.open(sd_path).convert('RGBA')
        else:
            icon = Image.open(icon_path).convert('RGBA')

        sprite = read_sprite_sheet(sheet_key)
        x, y = col * cell, row * cell
        _paste_icon(sprite, icon, x, y, cell)

        overlay_dat = _write_sheet(sheet_key, sprite)
        if overlay_dat:
            results.append(f"{sheet_key}: {overlay_dat}")

    return results


def patch_compound(slot_name, icon_path):
    """Patch a compound slot (both 16x16 and 32x32 in the HD sheet).

    Compound slots have both a small and large icon at different pixel
    positions within the same sprite sheet. Both are cleared and replaced.

    Args:
        slot_name: Name from COMPOUND_SLOTS (e.g. 'linkshell')
        icon_path: Path to PNG (scaled to both sizes)

    Returns:
        Path to the written overlay DAT, or None on failure.
    """
    if slot_name not in COMPOUND_SLOTS:
        raise ValueError(
            f"Unknown compound slot: '{slot_name}'. "
            f"Known: {', '.join(sorted(COMPOUND_SLOTS.keys()))}"
        )

    positions = COMPOUND_SLOTS[slot_name]
    icon = Image.open(icon_path).convert('RGBA')

    sprite = read_sprite_sheet('hd')

    sx, sy = positions['small']
    _paste_icon(sprite, icon, sx, sy, 16)
    print(f"  Patched 16x16 at ({sx}, {sy})")

    lx, ly = positions['large']
    _paste_icon(sprite, icon, lx, ly, 32)
    print(f"  Patched 32x32 at ({lx}, {ly})")

    overlay_dat = _write_sheet('hd', sprite)
    return overlay_dat


def resolve_slot(slot):
    """Resolve a slot name or (row, col) tuple to (row, col).

    Also accepts compound slot names — returns the 32x32 grid position.
    """
    if isinstance(slot, str):
        if slot in COMPOUND_SLOTS:
            lx, ly = COMPOUND_SLOTS[slot]['large']
            return (ly // 32, lx // 32)
        if slot not in SLOT_NAMES:
            all_names = sorted(list(SLOT_NAMES.keys()) + list(COMPOUND_SLOTS.keys()))
            raise ValueError(f"Unknown slot name: '{slot}'. Known: {', '.join(all_names)}")
        return SLOT_NAMES[slot]
    elif isinstance(slot, (list, tuple)) and len(slot) == 2:
        return tuple(slot)
    else:
        raise ValueError(f"Invalid slot: {slot}. Use a name or [row, col].")


def list_slots():
    """List all known slot names and positions."""
    print("Simple slots (32x32 grid):")
    for name, (row, col) in sorted(SLOT_NAMES.items()):
        print(f"  {name:15s} → row {row}, col {col}")
    print("\nCompound slots (16x16 + 32x32 in HD sheet):")
    for name, pos in sorted(COMPOUND_SLOTS.items()):
        print(f"  {name:15s} → small {pos['small']}, large {pos['large']}")

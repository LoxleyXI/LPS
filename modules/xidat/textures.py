"""Textures in FFXI DATs: their sections, and recolouring them in place.

One copy for every LPS module that recolours (zones, mobs, gear). DXT colours are changed by
shifting their RGB565 endpoints (no decode and re-encode, so nothing is lost); paletted textures
by their palette.

    for sec in parse_sections(data):
        info = parse_texture_info(data, sec)
        if info: adjust_dxt_inplace(buf, info, hue=120)
"""
import struct
from typing import List, Optional, Tuple, NamedTuple


# ---------------------------------------------------------------------------
# Colour space
# ---------------------------------------------------------------------------
def _rgb_to_hsv(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert RGB (0-255) to HSV (h: 0-360, s: 0-1, v: 0-1)."""
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    diff = max_c - min_c

    if diff == 0:
        h = 0
    elif max_c == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_c == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0 if max_c == 0 else diff / max_c
    v = max_c
    return h, s, v


def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (h: 0-360, s: 0-1, v: 0-1) to RGB (0-255)."""
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c

    if h < 60:
        r, g, b = c, x, 0
    elif h < 120:
        r, g, b = x, c, 0
    elif h < 180:
        r, g, b = 0, c, x
    elif h < 240:
        r, g, b = 0, x, c
    elif h < 300:
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return (
        int((r + m) * 255),
        int((g + m) * 255),
        int((b + m) * 255)
    )


# ---------------------------------------------------------------------------
# DAT section parsing
# ---------------------------------------------------------------------------

SECTION_TYPE_TEXTURE = 0x20


class Section(NamedTuple):
    name: bytes       # 4-byte section ID
    type_code: int    # lower 7 bits of metadata
    start: int        # byte offset of section header in DAT
    size: int         # total section size (header + body, 16-byte aligned)
    data_start: int   # byte offset of body (start + 0x10)


class TextureInfo(NamedTuple):
    section: Section
    tex_type: int     # 0x91, 0xA1, 0xB1
    name: str         # 16-char texture name
    width: int
    height: int
    bit_count: int    # bits per pixel
    dxt_type: str     # "1TXD", "3TXD", or "" for paletted
    data_offset: int  # absolute offset of pixel/DXT data in DAT
    data_size: int    # size of pixel/DXT data


def parse_sections(data: bytes) -> List[Section]:
    """Parse all sections from a DAT file."""
    sections = []
    pos = 0
    while pos + 16 <= len(data):
        name = data[pos:pos + 4]
        meta = struct.unpack_from("<I", data, pos + 4)[0]
        type_code = meta & 0x7F
        size = ((meta >> 7) & 0xFFFFF) * 0x10
        if size <= 0:
            break
        sections.append(Section(name, type_code, pos, size, pos + 0x10))
        pos += size
    return sections


def parse_texture_info(data: bytes, section: Section) -> Optional[TextureInfo]:
    """Parse texture header from a 0x20 section."""
    pos = section.data_start
    if pos + 0x31 > len(data):
        return None

    tex_type = data[pos]
    if tex_type not in (0x91, 0xA1, 0xB1):
        return None

    raw_name = data[pos + 1:pos + 0x11]
    name = raw_name.split(b'\x00')[0].decode('ascii', errors='replace').rstrip()

    width = struct.unpack_from("<I", data, pos + 0x15)[0]
    height = struct.unpack_from("<I", data, pos + 0x19)[0]
    bit_count = struct.unpack_from("<H", data, pos + 0x1F)[0]

    if tex_type == 0xA1:
        # DXT layout: +0x39 FourCC, +0x3D size, +0x41 pitch, +0x45 data
        dxt_fourcc_offset = pos + 0x39
        dxt_type = data[dxt_fourcc_offset:dxt_fourcc_offset + 4].decode('ascii', errors='replace')
        dxt_data_size = struct.unpack_from("<I", data, dxt_fourcc_offset + 4)[0]
        dxt_data_offset = pos + 0x45
        return TextureInfo(section, tex_type, name, width, height, bit_count,
                           dxt_type, dxt_data_offset, dxt_data_size)
    else:
        # Paletted: +0x35 format marker, +0x39 extra u32, +0x3D palette/pixels
        data_offset = pos + 0x3D
        if bit_count == 32:
            pixel_data_size = width * height * 4
        else:
            pixel_data_size = 1024 + width * height
        return TextureInfo(section, tex_type, name, width, height, bit_count,
                           "", data_offset, pixel_data_size)


# ---------------------------------------------------------------------------
# RGB565 helpers
# ---------------------------------------------------------------------------

def _rgb565_unpack(value: int) -> Tuple[int, int, int]:
    r = (value >> 11 & 0x1F) * 255 // 31
    g = (value >> 5 & 0x3F) * 255 // 63
    b = (value & 0x1F) * 255 // 31
    return r, g, b


def _rgb565_pack(r: int, g: int, b: int) -> int:
    return ((r * 31 // 255) << 11) | ((g * 63 // 255) << 5) | (b * 31 // 255)


def _parse_color(color: str) -> Tuple[int, int, int, int]:
    """Parse '#RRGGBB' or '#RRGGBBAA' hex color."""
    c = color.lstrip('#')
    r = int(c[0:2], 16)
    g = int(c[2:4], 16)
    b = int(c[4:6], 16)
    a = int(c[6:8], 16) if len(c) >= 8 else 255
    return r, g, b, a


def _blend(base_r, base_g, base_b, tint_r, tint_g, tint_b, alpha, mode):
    """Blend a tint color onto a base color. Same modes as LPS icons."""
    a = alpha / 255.0
    br, bg, bb = base_r / 255.0, base_g / 255.0, base_b / 255.0
    tr, tg, tb = tint_r / 255.0, tint_g / 255.0, tint_b / 255.0

    if mode == 'multiply':
        rr, rg, rb = br * tr, bg * tg, bb * tb
    elif mode == 'screen':
        rr = 1 - (1 - br) * (1 - tr)
        rg = 1 - (1 - bg) * (1 - tg)
        rb = 1 - (1 - bb) * (1 - tb)
    elif mode == 'overlay':
        rr = 2 * br * tr if br < 0.5 else 1 - 2 * (1 - br) * (1 - tr)
        rg = 2 * bg * tg if bg < 0.5 else 1 - 2 * (1 - bg) * (1 - tg)
        rb = 2 * bb * tb if bb < 0.5 else 1 - 2 * (1 - bb) * (1 - tb)
    elif mode == 'add':
        rr, rg, rb = min(1, br + tr), min(1, bg + tg), min(1, bb + tb)
    else:  # normal
        rr, rg, rb = tr, tg, tb

    # Alpha blend with original
    fr = br + (rr - br) * a
    fg = bg + (rg - bg) * a
    fb = bb + (rb - bb) * a
    return int(fr * 255), int(fg * 255), int(fb * 255)


def _hue_in_range(h: float, hue_min: float, hue_max: float) -> bool:
    """Check if a hue value falls within a range (handles wrap-around)."""
    if hue_min <= hue_max:
        return hue_min <= h <= hue_max
    else:
        # Wraps around 360 (e.g. 340-20 means 340-360 + 0-20)
        return h >= hue_min or h <= hue_max


def _adjust_rgb565(value: int, hue: float = 0, saturation: float = 0,
                    lightness: float = 0, tint: str = None,
                    blend_mode: str = 'normal',
                    hue_min: float = None, hue_max: float = None,
                    val_min: float = None, val_max: float = None,
                    sat_min: float = None, sat_max: float = None) -> int:
    """Adjust a single RGB565 color value.

    Args:
        hue: Degrees to rotate hue (0-360)
        saturation: Adjust saturation (-100 to 100, 0 = no change)
        lightness: Adjust brightness (-100 to 100, 0 = no change)
        tint: Hex color '#RRGGBB' or '#RRGGBBAA' to blend
        blend_mode: 'normal', 'multiply', 'screen', 'overlay', 'add'
        hue_min/hue_max: Only adjust colors with hue in range (0-360)
        val_min/val_max: Only adjust colors with brightness in range (0.0-1.0)
        sat_min/sat_max: Only adjust colors with saturation in range (0.0-1.0)
    """
    r, g, b = _rgb565_unpack(value)
    if r == 0 and g == 0 and b == 0:
        return value

    # Check color range filters
    has_filter = (hue_min is not None or val_min is not None or sat_min is not None)
    if has_filter:
        h, s, v = _rgb_to_hsv(r, g, b)
        if hue_min is not None and hue_max is not None:
            if s < 0.05:
                return value
            if not _hue_in_range(h, hue_min, hue_max):
                return value
        if val_min is not None and v < val_min:
            return value
        if val_max is not None and v > val_max:
            return value
        if sat_min is not None and s < sat_min:
            return value
        if sat_max is not None and s > sat_max:
            return value

    # HSV adjustments
    if hue or saturation or lightness:
        h, s, v = _rgb_to_hsv(r, g, b)
        if hue:
            h = (h + hue) % 360
        if saturation:
            s = max(0.0, min(1.0, s * (1.0 + saturation / 100.0)))
        if lightness:
            v = max(0.0, min(1.0, v * (1.0 + lightness / 100.0)))
        r, g, b = _hsv_to_rgb(h, s, v)

    # Tint blend
    if tint:
        tr, tg, tb, ta = _parse_color(tint)
        r, g, b = _blend(r, g, b, tr, tg, tb, ta, blend_mode)

    return _rgb565_pack(r, g, b)


# ---------------------------------------------------------------------------
# DXT in-place hue shift (modify color endpoints only, preserve indices)
# ---------------------------------------------------------------------------

def _adjust_dxt_inplace(data: bytearray, info: TextureInfo, **kwargs) -> None:
    """Adjust DXT texture by modifying RGB565 color endpoints in each block.
    Leaves indices and alpha completely untouched: lossless and format-agnostic."""
    is_dxt1 = info.dxt_type == "1TXD"
    block_size = 8 if is_dxt1 else 16
    color_offset = 0 if is_dxt1 else 8

    pos = info.data_offset
    end = pos + info.data_size

    while pos + block_size <= end:
        cp = pos + color_offset

        c0 = struct.unpack_from("<H", data, cp)[0]
        c1 = struct.unpack_from("<H", data, cp + 2)[0]

        new_c0 = _adjust_rgb565(c0, **kwargs)
        new_c1 = _adjust_rgb565(c1, **kwargs)

        # For DXT1, preserve c0/c1 ordering (controls transparency mode)
        if is_dxt1:
            if c0 > c1 and new_c0 <= new_c1:
                if new_c0 == new_c1:
                    if new_c0 < 0xFFFF:
                        new_c0 += 1
                    else:
                        new_c1 -= 1
                else:
                    new_c0, new_c1 = new_c1, new_c0
                    idx_word = struct.unpack_from("<I", data, cp + 4)[0]
                    idx_word = _swap_dxt1_indices(idx_word)
                    struct.pack_into("<I", data, cp + 4, idx_word)
            elif c0 <= c1 and new_c0 > new_c1:
                new_c0, new_c1 = new_c1, new_c0
                idx_word = struct.unpack_from("<I", data, cp + 4)[0]
                idx_word = _swap_dxt1_indices(idx_word)
                struct.pack_into("<I", data, cp + 4, idx_word)

        struct.pack_into("<H", data, cp, new_c0)
        struct.pack_into("<H", data, cp + 2, new_c1)

        pos += block_size


def _swap_dxt1_indices(idx_word: int) -> int:
    """Swap index 0 and 1 in a 32-bit DXT1 index word (16 x 2-bit indices).
    Index 0 → 1, index 1 → 0, index 2 → 2, index 3 → 3."""
    result = 0
    for i in range(16):
        shift = i * 2
        idx = (idx_word >> shift) & 0x3
        if idx == 0:
            idx = 1
        elif idx == 1:
            idx = 0
        result |= (idx << shift)
    return result


# ---------------------------------------------------------------------------
# Paletted texture hue shift (in-place)
# ---------------------------------------------------------------------------

def _adjust_palette_inplace(data: bytearray, info: TextureInfo, **kwargs) -> None:
    """Adjust a paletted texture's BGRA palette in-place."""
    hue = kwargs.get('hue', 0)
    saturation = kwargs.get('saturation', 0)
    lightness = kwargs.get('lightness', 0)
    tint = kwargs.get('tint')
    blend_mode = kwargs.get('blend_mode', 'normal')
    hue_min = kwargs.get('hue_min')
    hue_max = kwargs.get('hue_max')

    if info.bit_count == 32:
        count = info.width * info.height
    else:
        count = 256  # palette entries only

    pos = info.data_offset
    for i in range(count):
        offset = pos + i * 4
        if offset + 3 >= len(data):
            break
        b, g, r, a = data[offset], data[offset + 1], data[offset + 2], data[offset + 3]
        if a > 0 and (r > 0 or g > 0 or b > 0):
            # Check color range filters
            has_filter = (hue_min is not None or
                          kwargs.get('val_min') is not None or
                          kwargs.get('sat_min') is not None)
            if has_filter:
                ch, cs, cv = _rgb_to_hsv(r, g, b)
                if hue_min is not None and hue_max is not None:
                    if cs < 0.05:
                        continue
                    if not _hue_in_range(ch, hue_min, hue_max):
                        continue
                if kwargs.get('val_min') is not None and cv < kwargs['val_min']:
                    continue
                if kwargs.get('val_max') is not None and cv > kwargs['val_max']:
                    continue
                if kwargs.get('sat_min') is not None and cs < kwargs['sat_min']:
                    continue
                if kwargs.get('sat_max') is not None and cs > kwargs['sat_max']:
                    continue
            # HSV adjustments
            if hue or saturation or lightness:
                h, s, v = _rgb_to_hsv(r, g, b)
                if hue:
                    h = (h + hue) % 360
                if saturation:
                    s = max(0.0, min(1.0, s * (1.0 + saturation / 100.0)))
                if lightness:
                    v = max(0.0, min(1.0, v * (1.0 + lightness / 100.0)))
                r, g, b = _hsv_to_rgb(h, s, v)
            # Tint blend
            if tint:
                tr, tg, tb, ta = _parse_color(tint)
                r, g, b = _blend(r, g, b, tr, tg, tb, ta, blend_mode)
            data[offset] = b
            data[offset + 1] = g
            data[offset + 2] = r


# public names (the gear module's spelling)
adjust_dxt_inplace = _adjust_dxt_inplace
adjust_palette_inplace = _adjust_palette_inplace

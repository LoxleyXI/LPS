"""
LPS Icons - Icon loading, encoding, HQ borders, and tinting

Icon format (BMP2):
- 57-byte header
- 1024-byte BGRA palette (256 colors × 4 bytes, alpha 0-128)
- 1024-byte pixel indices (32×32, bottom-to-top rows)
"""

from pathlib import Path
from PIL import Image
import struct

from .constants import (
    ICONS_PATH, ICON_HEADER_SIZE, ICON_PALETTE_SIZE,
    ICON_SIZE, ICON_LEN_OFFSET, ICON_DATA_OFFSET
)


# HQ border colors: (outer RGBA, inner RGBA)
# Outer = 2px frame, inner = 1px inset border (slightly muted)
HQ_BORDER_COLORS = {
    1: ((183, 183, 183, 255), (159, 159, 159, 255)),  # +1 silver
    2: ((64, 127, 164, 255),  (69, 112, 141, 255)),   # +2 steel blue
    3: ((166, 142, 68, 255),  (134, 119, 77, 255)),   # +3 gold
    4: ((61, 165, 114, 255),  (78, 141, 71, 255)),    # +4 green
}


def load_icon(icon_path: Path) -> Image.Image:
    """
    Load an icon from a PNG file.

    Args:
        icon_path: Path to PNG file (32x32 recommended)

    Returns:
        PIL Image in RGBA mode, resized to 32x32 if needed
    """
    img = Image.open(icon_path).convert('RGBA')
    if img.size != (ICON_SIZE, ICON_SIZE):
        img = img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
    return img


def extract_icon(entry: bytes) -> Image.Image | None:
    """
    Extract icon from an item entry.

    Args:
        entry: Raw item entry bytes (0xC00)

    Returns:
        PIL Image or None if no valid icon
    """
    if len(entry) < ICON_DATA_OFFSET + 4:
        return None

    icon_len = struct.unpack('<I', entry[ICON_LEN_OFFSET:ICON_LEN_OFFSET+4])[0]
    if icon_len == 0:
        return None

    icon_start = ICON_DATA_OFFSET
    icon_end = icon_start + icon_len

    if icon_end > len(entry):
        return None

    icon_data = entry[icon_start:icon_end]
    if len(icon_data) < ICON_HEADER_SIZE + ICON_PALETTE_SIZE + ICON_SIZE * ICON_SIZE:
        return None

    # Parse palette (BGRA, alpha 0-128)
    palette_start = ICON_HEADER_SIZE
    palette = []
    for i in range(256):
        offset = palette_start + i * 4
        b, g, r, a = icon_data[offset:offset+4]
        # Scale alpha from 0-128 to 0-255
        a = min(255, a * 2)
        palette.append((r, g, b, a))

    # Parse pixels (bottom-to-top)
    pixel_start = palette_start + ICON_PALETTE_SIZE
    img = Image.new('RGBA', (ICON_SIZE, ICON_SIZE))
    pixels = img.load()

    for y in range(ICON_SIZE):
        for x in range(ICON_SIZE):
            idx = icon_data[pixel_start + (ICON_SIZE - 1 - y) * ICON_SIZE + x]
            pixels[x, y] = palette[idx]

    return img


def encode_icon(img: Image.Image, name_tag: str = "custom", icon_type: str = "item") -> bytes:
    """
    Encode a PIL image to BMP2 icon format.

    Args:
        img: PIL Image (will be converted to RGBA and resized if needed)
        name_tag: Name tag for header (max 8 chars)
        icon_type: Icon type for header - "item", "weapon", "armor", etc. (max 8 chars)

    Returns:
        Encoded icon bytes (header + palette + pixels)
    """
    # Ensure correct size and mode
    if img.size != (ICON_SIZE, ICON_SIZE):
        img = img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)
    if img.mode != 'RGBA':
        img = img.convert('RGBA')

    # Flip top-to-bottom BEFORE palette conversion (FFXI stores bottom-to-top)
    img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    # Convert to palette mode
    img_p = img.convert('P', palette=Image.Palette.ADAPTIVE, colors=256)
    palette_raw = img_p.getpalette()
    rgba_pixels = list(img.getdata())

    # Build BGRA palette (1024 bytes)
    palette = bytearray(ICON_PALETTE_SIZE)
    for i in range(256):
        if palette_raw and i * 3 + 2 < len(palette_raw):
            r = palette_raw[i * 3 + 0]
            g = palette_raw[i * 3 + 1]
            b = palette_raw[i * 3 + 2]
            palette[i * 4 + 0] = b  # Blue
            palette[i * 4 + 1] = g  # Green
            palette[i * 4 + 2] = r  # Red
            palette[i * 4 + 3] = 128  # Alpha (opaque)

    # Get pixel indices
    pixel_indices = list(img_p.getdata())

    # Set alpha=0 for transparent pixels in the palette
    for px_idx, (r, g, b, a) in enumerate(rgba_pixels):
        if a < 128:
            pal_idx = pixel_indices[px_idx]
            palette[pal_idx * 4 + 3] = 0  # Mark as transparent

    pixels = bytes(pixel_indices)

    # Build header (57 bytes)
    header = bytearray(ICON_HEADER_SIZE)
    header[0] = 0x91  # Magic
    type_bytes = icon_type[:8].encode('ascii', errors='replace').ljust(8, b' ')
    header[1:9] = type_bytes  # Type (8 bytes)
    name = name_tag[:8].encode('ascii', errors='replace').ljust(8, b' ')
    header[9:17] = name  # Name (8 bytes)
    struct.pack_into('<I', header, 17, 0x28)  # Header size offset
    struct.pack_into('<I', header, 21, ICON_SIZE)  # Width
    struct.pack_into('<I', header, 25, ICON_SIZE)  # Height
    struct.pack_into('<H', header, 29, 1)  # Planes
    struct.pack_into('<H', header, 31, 8)  # Bits per pixel
    header[53] = 32  # Row size (single byte, not packed int)

    return bytes(header) + bytes(palette) + pixels


def apply_hq_border(img: Image.Image, hq_level: int) -> Image.Image:
    """
    Apply HQ border to an icon.

    Args:
        img: PIL Image (32x32 RGBA)
        hq_level: 1-4 (white, blue, gold, green)

    Returns:
        Image with 2px HQ border applied
    """
    if hq_level not in HQ_BORDER_COLORS:
        return img

    outer, inner = HQ_BORDER_COLORS[hq_level]
    result = img.copy()
    pixels = result.load()
    s = ICON_SIZE

    # Outer ring (edge pixels)
    for i in range(s):
        pixels[i, 0] = outer
        pixels[i, s - 1] = outer
        pixels[0, i] = outer
        pixels[s - 1, i] = outer

    # Inner ring (1px inset)
    for i in range(1, s - 1):
        pixels[i, 1] = inner
        pixels[i, s - 2] = inner
        pixels[1, i] = inner
        pixels[s - 2, i] = inner

    return result


def apply_tint(img: Image.Image, tint_color: str) -> Image.Image:
    """
    Apply a color tint to an icon.

    The tint is blended with the original pixels using the tint's alpha.
    This allows creating color variations without copying material.

    Args:
        img: PIL Image (32x32 RGBA)
        tint_color: Hex color string like "#ff0000" or "#ff0000aa" (with alpha)

    Returns:
        Tinted image
    """
    # Parse hex color
    color = tint_color.lstrip('#')
    if len(color) == 6:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        a = 255
    elif len(color) == 8:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        a = int(color[6:8], 16)
    else:
        raise ValueError(f"Invalid tint color: {tint_color}")

    result = img.copy().convert('RGBA')
    pixels = result.load()

    # Tint alpha as 0-1
    tint_alpha = a / 255.0

    for y in range(img.size[1]):
        for x in range(img.size[0]):
            orig_r, orig_g, orig_b, orig_a = pixels[x, y]

            # Blend: result = orig * (1 - tint_alpha) + tint * tint_alpha
            new_r = int(orig_r * (1 - tint_alpha) + r * tint_alpha)
            new_g = int(orig_g * (1 - tint_alpha) + g * tint_alpha)
            new_b = int(orig_b * (1 - tint_alpha) + b * tint_alpha)

            pixels[x, y] = (new_r, new_g, new_b, orig_a)

    return result


def _parse_hex_color(tint_color: str) -> tuple[int, int, int, int]:
    """Parse hex color string to RGBA tuple."""
    color = tint_color.lstrip('#')
    if len(color) == 6:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        a = 255
    elif len(color) == 8:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        a = int(color[6:8], 16)
    else:
        raise ValueError(f"Invalid tint color: {tint_color}")
    return r, g, b, a


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


def _blend_colors(r: int, g: int, b: int,
                  tint_r: int, tint_g: int, tint_b: int,
                  alpha: float, mode: str) -> tuple[int, int, int]:
    """
    Blend two colors using the specified blend mode.

    Args:
        r, g, b: Original color (0-255)
        tint_r, tint_g, tint_b: Tint color (0-255)
        alpha: Blend strength (0-1)
        mode: Blend mode (normal, multiply, screen, overlay, add)

    Returns:
        Blended RGB tuple (0-255)
    """
    # Normalize to 0-1
    r1, g1, b1 = r / 255.0, g / 255.0, b / 255.0
    r2, g2, b2 = tint_r / 255.0, tint_g / 255.0, tint_b / 255.0

    if mode == 'multiply':
        # Darken: result = base * blend
        br, bg, bb = r1 * r2, g1 * g2, b1 * b2
    elif mode == 'screen':
        # Lighten: result = 1 - (1 - base) * (1 - blend)
        br = 1 - (1 - r1) * (1 - r2)
        bg = 1 - (1 - g1) * (1 - g2)
        bb = 1 - (1 - b1) * (1 - b2)
    elif mode == 'overlay':
        # Multiply dark, screen light
        br = 2 * r1 * r2 if r1 < 0.5 else 1 - 2 * (1 - r1) * (1 - r2)
        bg = 2 * g1 * g2 if g1 < 0.5 else 1 - 2 * (1 - g1) * (1 - g2)
        bb = 2 * b1 * b2 if b1 < 0.5 else 1 - 2 * (1 - b1) * (1 - b2)
    elif mode == 'add':
        # Add colors (clamped)
        br, bg, bb = min(1, r1 + r2), min(1, g1 + g2), min(1, b1 + b2)
    else:  # normal
        br, bg, bb = r2, g2, b2

    # Apply alpha blending
    final_r = r1 * (1 - alpha) + br * alpha
    final_g = g1 * (1 - alpha) + bg * alpha
    final_b = b1 * (1 - alpha) + bb * alpha

    return (
        int(max(0, min(255, final_r * 255))),
        int(max(0, min(255, final_g * 255))),
        int(max(0, min(255, final_b * 255)))
    )


def _icon_color_in_range(h, s, v, hue_min=None, hue_max=None,
                          val_min=None, val_max=None,
                          sat_min=None, sat_max=None) -> bool:
    """Check if a color falls within the specified HSV ranges."""
    if hue_min is not None and hue_max is not None:
        if s < 0.05:
            return False
        if hue_min <= hue_max:
            if not (hue_min <= h <= hue_max):
                return False
        else:
            if not (h >= hue_min or h <= hue_max):
                return False
    if val_min is not None and v < val_min:
        return False
    if val_max is not None and v > val_max:
        return False
    if sat_min is not None and s < sat_min:
        return False
    if sat_max is not None and s > sat_max:
        return False
    return True


def apply_tint_to_icon_bytes(icon_data: bytes, tint_color: str,
                              blend_mode: str = 'normal',
                              hue_min: float = None, hue_max: float = None,
                              val_min: float = None, val_max: float = None,
                              sat_min: float = None, sat_max: float = None) -> bytes:
    """
    Apply tint directly to icon palette bytes without PIL conversion.

    Args:
        icon_data: Raw icon bytes (header + palette + pixels)
        tint_color: Hex color string like "#ff0000" or "#ff0000aa"
        blend_mode: Blend mode - 'normal', 'multiply', 'screen', 'overlay', 'add'
        hue_min/hue_max: Only tint colors with hue in range (0-360)
        val_min/val_max: Only tint colors with brightness in range (0-1)
        sat_min/sat_max: Only tint colors with saturation in range (0-1)

    Returns:
        Modified icon bytes with tinted palette
    """
    tint_r, tint_g, tint_b, tint_a = _parse_hex_color(tint_color)
    tint_alpha = tint_a / 255.0
    has_filter = any(x is not None for x in [hue_min, hue_max, val_min, val_max, sat_min, sat_max])

    result = bytearray(icon_data)
    palette_start = ICON_HEADER_SIZE

    for i in range(256):
        offset = palette_start + i * 4
        b = result[offset]
        g = result[offset + 1]
        r = result[offset + 2]
        a = result[offset + 3]

        if a > 0:
            if has_filter:
                h, s, v = _rgb_to_hsv(r, g, b)
                if not _icon_color_in_range(h, s, v, hue_min, hue_max,
                                             val_min, val_max, sat_min, sat_max):
                    continue
            new_r, new_g, new_b = _blend_colors(
                r, g, b, tint_r, tint_g, tint_b, tint_alpha, blend_mode
            )
            result[offset] = new_b
            result[offset + 1] = new_g
            result[offset + 2] = new_r

    return bytes(result)


def apply_hue_shift_to_icon_bytes(icon_data: bytes, degrees: float,
                                   hue_min: float = None, hue_max: float = None,
                                   val_min: float = None, val_max: float = None,
                                   sat_min: float = None, sat_max: float = None) -> bytes:
    """
    Shift hue of icon palette colors by specified degrees.

    Args:
        icon_data: Raw icon bytes (header + palette + pixels)
        degrees: Hue shift in degrees (0-360, can be negative)
        hue_min/hue_max: Only shift colors with hue in range (0-360)
        val_min/val_max: Only shift colors with brightness in range (0-1)
        sat_min/sat_max: Only shift colors with saturation in range (0-1)

    Returns:
        Modified icon bytes with shifted palette hues
    """
    has_filter = any(x is not None for x in [hue_min, hue_max, val_min, val_max, sat_min, sat_max])
    result = bytearray(icon_data)
    palette_start = ICON_HEADER_SIZE

    for i in range(256):
        offset = palette_start + i * 4
        b = result[offset]
        g = result[offset + 1]
        r = result[offset + 2]
        a = result[offset + 3]

        if a > 0:
            h, s, v = _rgb_to_hsv(r, g, b)
            if has_filter and not _icon_color_in_range(h, s, v, hue_min, hue_max,
                                                        val_min, val_max, sat_min, sat_max):
                continue
            h = (h + degrees) % 360
            new_r, new_g, new_b = _hsv_to_rgb(h, s, v)

            result[offset] = new_b
            result[offset + 1] = new_g
            result[offset + 2] = new_r

    return bytes(result)



def composite_icon(base: Image.Image, hq_level: int = 0, tint: str = None) -> Image.Image:
    """
    Create a final icon with optional tint and HQ border.

    Processing order:
    1. Apply tint to base icon (if specified)
    2. Apply HQ border on top (NOT tinted)

    Args:
        base: Base icon image
        hq_level: 0 for none, 1-3 for HQ border
        tint: Optional hex color string for tinting

    Returns:
        Composited final icon
    """
    result = base.copy().convert('RGBA')

    # Apply tint first (before border)
    if tint:
        result = apply_tint(result, tint)

    # Apply border last (so it's not tinted)
    if hq_level > 0:
        result = apply_hq_border(result, hq_level)

    return result

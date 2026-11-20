"""
LPS - Loxley Patch System

FFXI DAT patching engine for XIPivot overlays.
Core provides encryption, icon manipulation, and text encoding.
Features are implemented as modules.
"""

from .encryption import read_dat, write_dat, ror5, rol5
from .dmsg import encode_dmsg_entry, decode_dmsg_entry, create_item_text
from .icons import (
    load_icon, apply_hq_border, encode_icon, extract_icon, apply_tint,
    composite_icon, _rgb_to_hsv, _hsv_to_rgb, _blend_colors,
    apply_tint_to_icon_bytes, apply_hue_shift_to_icon_bytes
)
from .constants import FFXI_PATH, OVERLAY_PATH, ENTRY_SIZE
from .ftable import patch_ftable, read_ftable_entry, resolve_ftable_to_path
from .descriptions import build_description, format_mod

__version__ = "2.0.0"

"""
LPS Models - Mob model patching and texture tinting

Model system overview:
- FTABLE.DAT maps file IDs (model IDs) to ROM paths
- VTABLE.DAT tracks version flags for each file ID
- mob_pools.modelid references these file IDs
- Model DATs contain geometry + textures

To create a variant:
1. Find unused FTABLE entry
2. Point it to a new ROM path (e.g., ROM/900/x.DAT)
3. Copy base model and modify textures
4. Create mob_pool entry with new model ID
"""

import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from lps.constants import FFXI_PATH, OVERLAY_PATH
from modules import register_module


# FTABLE/VTABLE paths
FTABLE_PATH = FFXI_PATH / "FTABLE.DAT"
VTABLE_PATH = FFXI_PATH / "VTABLE.DAT"

# Custom model folder (high number to avoid conflicts)
CUSTOM_MODEL_FOLDER = 900


@dataclass
class FTableEntry:
    """Represents an FTABLE entry."""
    file_id: int
    folder: int
    file_num: int

    @property
    def rom_path(self) -> str:
        return f"ROM/{self.folder}/{self.file_num}.DAT"

    @property
    def entry_value(self) -> int:
        """Pack folder/file into 16-bit value."""
        return (self.folder << 8) | self.file_num

    @classmethod
    def from_value(cls, file_id: int, value: int) -> 'FTableEntry':
        folder = (value >> 8) & 0xFF
        file_num = value & 0xFF
        return cls(file_id, folder, file_num)


def read_ftable() -> bytes:
    """Read the FTABLE.DAT file."""
    with open(FTABLE_PATH, 'rb') as f:
        return f.read()


def read_vtable() -> bytes:
    """Read the VTABLE.DAT file."""
    with open(VTABLE_PATH, 'rb') as f:
        return f.read()


def get_ftable_entry(file_id: int) -> FTableEntry:
    """Get the FTABLE entry for a file ID."""
    ftable = read_ftable()
    offset = file_id * 2
    if offset + 2 > len(ftable):
        raise ValueError(f"File ID {file_id} out of range")
    value = struct.unpack('<H', ftable[offset:offset+2])[0]
    return FTableEntry.from_value(file_id, value)


def find_empty_ftable_slots(count: int = 10, min_id: int = 90) -> list[int]:
    """
    Find empty (unused) FTABLE slots.

    Args:
        count: Number of slots to find
        min_id: Minimum file ID to consider

    Returns:
        List of empty file IDs
    """
    ftable = read_ftable()
    empty = []

    for i in range(min_id, len(ftable) // 2):
        value = struct.unpack('<H', ftable[i*2:i*2+2])[0]
        if value == 0:
            empty.append(i)
            if len(empty) >= count:
                break

    return empty


def get_model_dat_path(model_id: int) -> Path:
    """Get the full path to a model DAT file."""
    entry = get_ftable_entry(model_id)
    return FFXI_PATH / entry.rom_path


def read_model_dat(model_id: int) -> bytes:
    """Read a model DAT file."""
    path = get_model_dat_path(model_id)
    with open(path, 'rb') as f:
        return f.read()


def extract_model_textures(model_data: bytes) -> list[tuple[int, int, bytes]]:
    """
    Extract texture data from a model DAT.

    Model DATs have a complex structure. This attempts to find
    texture blocks based on common patterns.

    Returns:
        List of (offset, size, data) tuples for each texture found
    """
    textures = []

    # Look for DXT/texture headers
    # Common texture header patterns in FFXI models
    # This is a simplified implementation

    # TODO: Full model parsing requires understanding the
    # complete model format (vertices, faces, bones, textures)

    return textures


def apply_texture_tint(texture_data: bytes, tint_color: str) -> bytes:
    """
    Apply a color tint to texture data.

    Args:
        texture_data: Raw texture bytes
        tint_color: Hex color like "#ff0000aa"

    Returns:
        Modified texture data
    """
    # Parse tint color
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

    # TODO: Apply tint to texture pixels
    # This requires understanding the texture format (DXT1, DXT3, RGBA, etc.)

    return texture_data


def create_model_variant(
    base_model_id: int,
    new_model_id: int,
    tint: str = None,
    hue_shift: int = 0,
    saturation: float = 1.0,
    brightness: float = 1.0
) -> Path:
    """
    Create a variant of an existing model with modified textures.

    Args:
        base_model_id: Source model ID to copy
        new_model_id: Target model ID (must be unused in FTABLE)
        tint: Optional hex color tint
        hue_shift: Hue rotation in degrees (0-360)
        saturation: Saturation multiplier (0-2, 1=normal)
        brightness: Brightness multiplier (0-2, 1=normal)

    Returns:
        Path to created overlay DAT
    """
    # Read base model
    model_data = read_model_dat(base_model_id)

    # Extract textures
    textures = extract_model_textures(model_data)

    # Apply modifications
    modified_data = bytearray(model_data)

    for offset, size, tex_data in textures:
        if tint:
            tex_data = apply_texture_tint(tex_data, tint)
        # TODO: Apply hue_shift, saturation, brightness
        modified_data[offset:offset+size] = tex_data

    # Write to overlay path
    # Use custom folder to avoid conflicts
    overlay_path = OVERLAY_PATH / f"ROM/{CUSTOM_MODEL_FOLDER}/{new_model_id % 256}.DAT"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)

    with open(overlay_path, 'wb') as f:
        f.write(modified_data)

    print(f"Created model variant: {overlay_path}")
    return overlay_path


def get_model_info(model_id: int) -> dict:
    """Get information about a model."""
    entry = get_ftable_entry(model_id)
    path = get_model_dat_path(model_id)

    info = {
        'model_id': model_id,
        'ftable_entry': entry.entry_value,
        'rom_path': entry.rom_path,
        'full_path': str(path),
        'exists': path.exists(),
    }

    if path.exists():
        info['size'] = path.stat().st_size

    return info


# ============================================================================
# IMPLEMENTATION STATUS
# ============================================================================
#
# What we know:
# - FTABLE.DAT: 109,701 entries × 2 bytes, maps file_id → ROM path
# - Entry format: high_byte=folder, low_byte=file → ROM/{folder}/{file}.DAT
# - 26,625 empty slots available for custom models
# - mob_pools.modelid: 20 bytes, first 2 are model file ID (little endian)
#
# Model DAT structure (observed from ROM/1/46.DAT - goblin):
# - Header: "m_ba" magic
# - Chunks: "RID.", "open", "clos", "effe", "wind", "coll", "end", etc.
# - Color refs: "kr01", "kg01", "kb01" (RGB channels?)
# - Animations embedded
# - Textures format: Unknown, not standard DDS
#
# Challenges:
# 1. Model format is proprietary, needs reverse engineering
# 2. Textures embedded in non-standard format (not DDS)
# 3. FTABLE.DAT not in ROM/, may not work with XIPivot overlay
#
# Possible approaches:
# 1. Noesis plugin - Community has scripts for FFXI models
# 2. Client-side tinting - Ashita/Windower shader hooks
# 3. Full format RE - Parse chunks, find texture data, modify
#
# For now, this module provides:
# - FTABLE reading and slot finding
# - Model info lookup
# - Framework for future texture modification
#
# ============================================================================


register_module('models')

"""
LPS Strings - DMSG resource string patching

DMSG resources contain text strings for various game elements:
- Spell names
- Ability names
- Zone names
- Job names
- Quest names
- Menu text
- etc.
"""

import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from lps.constants import FFXI_PATH, OVERLAY_PATH
from lps.encryption import read_dat, write_dat
from lps.dmsg import encode_dmsg_entry, decode_dmsg_entry
from modules import register_module


@dataclass
class DMSGResource:
    """Definition for a DMSG resource file."""
    name: str
    file_id: int
    rom_path: str
    entry_count: int = 0  # 0 = dynamic

    @property
    def dat_path(self) -> str:
        """Get the DAT path from file ID."""
        # File ID format: 0xYYXX where YY is folder, XX is file
        folder = (self.file_id >> 8) & 0xFF
        file_num = self.file_id & 0xFF
        return f"ROM/{folder}/{file_num}.DAT"


# All DMSG resource definitions
DMSG_RESOURCES = {
    # Spell and ability names
    'spell_names': DMSGResource('spell_names', 0xD996, 'ROM/217/150.DAT'),
    'ability_names': DMSGResource('ability_names', 0xD995, 'ROM/217/149.DAT'),
    'ability_descriptions': DMSGResource('ability_descriptions', 0xD93A, 'ROM/217/58.DAT'),
    'spell_descriptions': DMSGResource('spell_descriptions', 0xD93B, 'ROM/217/59.DAT'),

    # Zone names
    'zone_names': DMSGResource('zone_names', 0xD8A9, 'ROM/216/169.DAT'),
    'zone_names_short': DMSGResource('zone_names_short', 0xD8AA, 'ROM/216/170.DAT'),

    # Job names
    'job_names': DMSGResource('job_names', 0xD8AB, 'ROM/216/171.DAT'),
    'job_names_short': DMSGResource('job_names_short', 0xD8AC, 'ROM/216/172.DAT'),

    # Race names
    'race_names': DMSGResource('race_names', 0xD8AD, 'ROM/216/173.DAT'),

    # Quest/mission names
    'quest_names': DMSGResource('quest_names', 0xD8AE, 'ROM/216/174.DAT'),
    'mission_names': DMSGResource('mission_names', 0xD8B0, 'ROM/216/176.DAT'),

    # Key items
    'key_item_names': DMSGResource('key_item_names', 0xD973, 'ROM/217/115.DAT'),
    'key_item_descriptions': DMSGResource('key_item_descriptions', 0xD974, 'ROM/217/116.DAT'),

    # Status effects
    'status_names': DMSGResource('status_names', 0xD938, 'ROM/217/56.DAT'),
    'status_descriptions': DMSGResource('status_descriptions', 0xD939, 'ROM/217/57.DAT'),

    # Titles
    'title_names': DMSGResource('title_names', 0xD8B2, 'ROM/216/178.DAT'),

    # Unity names
    'unity_names': DMSGResource('unity_names', 0xD8B3, 'ROM/216/179.DAT'),

    # Emotes
    'emote_names': DMSGResource('emote_names', 0xD8B4, 'ROM/216/180.DAT'),

    # Weather
    'weather_names': DMSGResource('weather_names', 0xD8B5, 'ROM/216/181.DAT'),

    # Days/Months
    'day_names': DMSGResource('day_names', 0xD8B6, 'ROM/216/182.DAT'),
    'moon_phase_names': DMSGResource('moon_phase_names', 0xD8B7, 'ROM/216/183.DAT'),

    # Craft names
    'craft_names': DMSGResource('craft_names', 0xD8B8, 'ROM/216/184.DAT'),
    'craft_rank_names': DMSGResource('craft_rank_names', 0xD8B9, 'ROM/216/185.DAT'),

    # Region names
    'region_names': DMSGResource('region_names', 0xD8BA, 'ROM/216/186.DAT'),

    # Nation names
    'nation_names': DMSGResource('nation_names', 0xD8BB, 'ROM/216/187.DAT'),

    # Menu text
    'menu_text': DMSGResource('menu_text', 0xD8BC, 'ROM/216/188.DAT'),

    # System messages
    'system_messages': DMSGResource('system_messages', 0xD8BD, 'ROM/216/189.DAT'),

    # Chat filters
    'chat_filter_names': DMSGResource('chat_filter_names', 0xD8BE, 'ROM/216/190.DAT'),

    # Merit category names
    'merit_category_names': DMSGResource('merit_category_names', 0xD975, 'ROM/217/117.DAT'),
    'merit_names': DMSGResource('merit_names', 0xD976, 'ROM/217/118.DAT'),

    # Job point names
    'job_point_category_names': DMSGResource('job_point_category_names', 0xD977, 'ROM/217/119.DAT'),
    'job_point_names': DMSGResource('job_point_names', 0xD978, 'ROM/217/120.DAT'),

    # Monstrosity
    'monstrosity_species_names': DMSGResource('monstrosity_species_names', 0xD979, 'ROM/217/121.DAT'),
    'monstrosity_ability_names': DMSGResource('monstrosity_ability_names', 0xD97A, 'ROM/217/122.DAT'),

    # Trust names
    'trust_names': DMSGResource('trust_names', 0xD97B, 'ROM/217/123.DAT'),

    # Mount names
    'mount_names': DMSGResource('mount_names', 0xD97C, 'ROM/217/124.DAT'),
}


def read_dmsg_file(resource: DMSGResource) -> list[bytes]:
    """
    Read all entries from a DMSG resource file.

    Args:
        resource: DMSG resource definition

    Returns:
        List of raw entry bytes
    """
    dat_path = FFXI_PATH / resource.dat_path
    data = read_dat(dat_path)

    if len(data) < 4:
        return []

    # DMSG files start with entry count
    entry_count = struct.unpack('<I', data[0:4])[0]
    if entry_count == 0 or entry_count > 100000:
        return []

    # Read offset table
    offsets = []
    for i in range(entry_count):
        offset = struct.unpack('<I', data[4 + i*4:8 + i*4])[0]
        offsets.append(offset)

    # Read entries
    entries = []
    for i in range(entry_count):
        start = offsets[i]
        if i + 1 < entry_count:
            end = offsets[i + 1]
        else:
            end = len(data)
        entries.append(data[start:end])

    return entries


def write_dmsg_file(resource: DMSGResource, entries: list[bytes]):
    """
    Write all entries to a DMSG resource overlay file.

    Args:
        resource: DMSG resource definition
        entries: List of entry bytes
    """
    entry_count = len(entries)

    # Calculate offsets
    header_size = 4 + entry_count * 4
    offsets = []
    current_offset = header_size

    for entry in entries:
        offsets.append(current_offset)
        current_offset += len(entry)

    # Build file
    data = bytearray()
    data.extend(struct.pack('<I', entry_count))

    for offset in offsets:
        data.extend(struct.pack('<I', offset))

    for entry in entries:
        data.extend(entry)

    # Write to overlay
    overlay_path = OVERLAY_PATH / resource.dat_path
    write_dat(overlay_path, bytes(data))


def patch_dmsg_resource(resource_name: str, patches: dict):
    """
    Patch entries in a DMSG resource.

    Args:
        resource_name: Name of the resource (e.g., 'spell_names')
        patches: Dict of entry_index -> entry_data
                 entry_data can be:
                 - str: Simple string (for single-element entries)
                 - list: List of elements for encode_dmsg_entry
                 - bytes: Raw entry bytes
    """
    if resource_name not in DMSG_RESOURCES:
        raise ValueError(f"Unknown DMSG resource: {resource_name}")

    resource = DMSG_RESOURCES[resource_name]

    # Read existing entries
    overlay_path = OVERLAY_PATH / resource.dat_path
    if overlay_path.exists():
        # Read from overlay if it exists
        data = read_dat(overlay_path)
        entry_count = struct.unpack('<I', data[0:4])[0]
        offsets = [struct.unpack('<I', data[4+i*4:8+i*4])[0] for i in range(entry_count)]
        entries = []
        for i in range(entry_count):
            start = offsets[i]
            end = offsets[i+1] if i+1 < entry_count else len(data)
            entries.append(data[start:end])
    else:
        entries = read_dmsg_file(resource)

    # Apply patches
    max_index = max(int(k) for k in patches.keys())
    while len(entries) <= max_index:
        entries.append(b'')

    for index_str, entry_data in patches.items():
        index = int(index_str)

        if isinstance(entry_data, bytes):
            entries[index] = entry_data
        elif isinstance(entry_data, str):
            entries[index] = encode_dmsg_entry([entry_data])
        elif isinstance(entry_data, list):
            entries[index] = encode_dmsg_entry(entry_data)
        else:
            raise ValueError(f"Invalid entry data type: {type(entry_data)}")

    # Write updated file
    write_dmsg_file(resource, entries)
    print(f"Patched {len(patches)} entries in {resource_name}")


def get_dmsg_entry(resource_name: str, index: int) -> list:
    """
    Get a decoded entry from a DMSG resource.

    Args:
        resource_name: Name of the resource
        index: Entry index

    Returns:
        Decoded entry elements
    """
    if resource_name not in DMSG_RESOURCES:
        raise ValueError(f"Unknown DMSG resource: {resource_name}")

    resource = DMSG_RESOURCES[resource_name]
    entries = read_dmsg_file(resource)

    if index >= len(entries):
        return []

    return decode_dmsg_entry(entries[index])


# === Config dispatch ===

def apply_strings_config(config: dict, icons_path=None):
    """Apply string patches from a config dict."""
    for resource_name, patches in config.items():
        if resource_name not in DMSG_RESOURCES:
            print(f"Warning: Unknown string resource '{resource_name}', skipping")
            continue
        patch_dmsg_resource(resource_name, patches)


register_module('strings',
    config_key='strings',
    apply_fn=apply_strings_config,
)

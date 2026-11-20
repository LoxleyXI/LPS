"""
LPS Items Module - Item patching for standard FFXI item types

Patches items within the standard ID ranges (0-28671) by modifying
DAT entries in XIPivot overlay files. Supports template-based creation,
custom text, icons, HQ borders, tinting, and hue shifting.

Item types and their DAT files:
- General (0-4095): ROM/118/106.DAT, text@0x18
- Usable (4096-8191): ROM/118/107.DAT, text@0x1C
- Puppet (8192-8703): ROM/118/110.DAT, text@0x18
- Armor (10240-16383): ROM/118/109.DAT, text@0x2C
- Weapon (16384-23039): ROM/118/108.DAT, text@0x38
- Armor2 (23040-28671): ROM/286/73.DAT, text@0x2C
- General2 (28672+): ROM/286/74.DAT, text@0x18
"""

import struct
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from PIL import Image

from lps.constants import (
    FFXI_PATH, OVERLAY_PATH, ENTRY_SIZE,
    ICON_LEN_OFFSET, ICON_DATA_OFFSET,
    FLAG_RARE, FLAG_EX, FLAG_INSCRIBABLE
)
from lps.encryption import read_dat, write_dat
from lps.dmsg import encode_dmsg_entry
from lps.icons import (
    load_icon, extract_icon, encode_icon, composite_icon,
    apply_tint_to_icon_bytes, apply_hue_shift_to_icon_bytes
)
from modules import register_module


# === Field Definitions ===
# Format: field_name -> (offset, struct_format)

COMMON_FIELDS = {
    'id':      (0x00, '<I'),   # u32 - Item ID
    'flags':   (0x04, '<H'),   # u16 - Item flags
    'stack':   (0x06, '<H'),   # u16 - Stack size
    'kind':    (0x08, '<H'),   # u16 - Item subtype
    'targets': (0x0C, '<H'),   # u16 - Valid targets
}

GENERAL_FIELDS = {
    **COMMON_FIELDS,
    '_unk0A':  (0x0A, '<H'),
    '_unk0E':  (0x0E, '<H'),
    '_unk10':  (0x10, '<H'),
    '_unk12':  (0x12, '<H'),
}

USABLE_FIELDS = {
    **COMMON_FIELDS,
    '_unk0A':  (0x0A, '<H'),
    'cast_time':  (0x0E, '<H'),
    'reuse_delay': (0x10, '<H'),
    '_unk13':  (0x13, '<B'),
    '_unk14':  (0x14, '<H'),
    '_unk16':  (0x16, '<B'),
    '_unk17':  (0x17, '<B'),
    '_unk18':  (0x18, '<B'),
}

ARMOR_FIELDS = {
    **COMMON_FIELDS,
    '_unk0A':  (0x0A, '<H'),
    'level':   (0x0E, '<H'),
    'slots':   (0x10, '<H'),
    'races':   (0x12, '<H'),
    'jobs':    (0x14, '<I'),
    'slvl':    (0x18, '<H'),
    'shield':  (0x1A, '<H'),
    '_unk1C':  (0x1C, '<B'),
    '_unk1D':  (0x1D, '<B'),
    '_unk1E':  (0x1E, '<H'),
    '_unk20':  (0x20, '<I'),
    '_unk24':  (0x24, '<H'),
    'ilvl':    (0x26, '<B'),
    '_unk27':  (0x27, '<B'),
    '_unk28':  (0x28, '<B'),
    '_unk29':  (0x29, '<B'),
    '_unk2A':  (0x2A, '<B'),
    '_unk2B':  (0x2B, '<B'),
}

WEAPON_FIELDS = {
    **COMMON_FIELDS,
    '_unk0A':  (0x0A, '<H'),
    'level':   (0x0E, '<H'),
    'slots':   (0x10, '<H'),
    'races':   (0x12, '<H'),
    'jobs':    (0x14, '<I'),
    'slvl':    (0x18, '<H'),
    'dmg':     (0x1C, '<H'),
    'delay':   (0x1E, '<H'),
    'dps':     (0x20, '<H'),
    'skill':   (0x22, '<H'),
    'emote':   (0x24, '<I'),
    '_unk28':  (0x28, '<B'),
    '_unk29':  (0x29, '<B'),
    '_unk2A':  (0x2A, '<H'),
    '_unk2C':  (0x2C, '<I'),
    '_unk30':  (0x30, '<H'),
    'ilvl':    (0x32, '<B'),
    '_unk33':  (0x33, '<B'),
    '_unk34':  (0x34, '<B'),
    '_unk35':  (0x35, '<B'),
    '_unk36':  (0x36, '<B'),
    '_unk37':  (0x37, '<B'),
}

TYPE_FIELDS = {
    'general':     GENERAL_FIELDS,
    'general2':    GENERAL_FIELDS,
    'usable':      USABLE_FIELDS,
    'puppet':      GENERAL_FIELDS,
    'armor':       ARMOR_FIELDS,
    'armor2':      ARMOR_FIELDS,
    'weapon':      WEAPON_FIELDS,
}


@dataclass
class ItemType:
    """Definition for an item type range."""
    name: str
    id_min: int
    id_max: int
    dat_path: str
    text_offset: int
    fields: dict

    def contains(self, item_id: int) -> bool:
        return self.id_min <= item_id <= self.id_max

    def get_entry_index(self, item_id: int) -> int:
        return item_id - self.id_min


# Standard FFXI item type definitions
ITEM_TYPES = [
    ItemType('general', 0, 4095, 'ROM/118/106.DAT', 0x18, GENERAL_FIELDS),
    ItemType('usable', 4096, 8191, 'ROM/118/107.DAT', 0x1C, USABLE_FIELDS),
    ItemType('puppet', 8192, 8703, 'ROM/118/110.DAT', 0x18, GENERAL_FIELDS),
    ItemType('armor', 10240, 16383, 'ROM/118/109.DAT', 0x2C, ARMOR_FIELDS),
    ItemType('weapon', 16384, 23039, 'ROM/118/108.DAT', 0x38, WEAPON_FIELDS),
    ItemType('armor2', 23040, 28671, 'ROM/286/73.DAT', 0x2C, ARMOR_FIELDS),
    ItemType('general2', 28672, 65535, 'ROM/286/74.DAT', 0x18, GENERAL_FIELDS),
]


def get_item_type(item_id: int) -> Optional[ItemType]:
    """Get the ItemType for a given item ID."""
    for item_type in ITEM_TYPES:
        if item_type.contains(item_id):
            return item_type
    return None


def get_item_type_by_name(name: str) -> Optional[ItemType]:
    """Get an ItemType by its name."""
    for item_type in ITEM_TYPES:
        if item_type.name == name:
            return item_type
    return None


def read_entry(item_id: int) -> tuple[bytes, ItemType]:
    """Read an item entry from the base DAT."""
    item_type = get_item_type(item_id)
    if not item_type:
        raise ValueError(f"Unknown item ID range: {item_id}")

    dat_path = FFXI_PATH / item_type.dat_path
    data = read_dat(dat_path)

    entry_index = item_type.get_entry_index(item_id)
    entry_start = entry_index * ENTRY_SIZE
    entry_end = entry_start + ENTRY_SIZE

    if entry_end > len(data):
        raise ValueError(f"Item {item_id} out of range for {item_type.name}")

    return data[entry_start:entry_end], item_type


def write_entry(item_id: int, entry: bytes):
    """Write a modified item entry to the overlay DAT."""
    item_type = get_item_type(item_id)
    if not item_type:
        raise ValueError(f"Unknown item ID range: {item_id}")

    overlay_dat = OVERLAY_PATH / item_type.dat_path
    if overlay_dat.exists():
        data = read_dat(overlay_dat)
    else:
        base_dat = FFXI_PATH / item_type.dat_path
        data = read_dat(base_dat)

    data = bytearray(data)
    entry_index = item_type.get_entry_index(item_id)
    entry_start = entry_index * ENTRY_SIZE

    if entry_start + ENTRY_SIZE > len(data):
        data.extend(bytes(entry_start + ENTRY_SIZE - len(data)))

    data[entry_start:entry_start + ENTRY_SIZE] = entry

    write_dat(overlay_dat, bytes(data))


def set_field(entry: bytearray, fields: dict, field_name: str, value: int):
    """Set a field value in an entry."""
    if field_name not in fields:
        raise ValueError(f"Unknown field: {field_name}")
    offset, fmt = fields[field_name]
    struct.pack_into(fmt, entry, offset, value)


def get_field(entry: bytes, fields: dict, field_name: str) -> int:
    """Get a field value from an entry."""
    if field_name not in fields:
        raise ValueError(f"Unknown field: {field_name}")
    offset, fmt = fields[field_name]
    return struct.unpack_from(fmt, entry, offset)[0]


def set_text(entry: bytearray, text_offset: int,
             name: str, description: str,
             singular: str = None, plural: str = None):
    """Set the DMSG text in an item entry."""
    full_name = singular or name.lower()
    article = 1 if full_name[0].upper() in 'AEIOU' else 0

    text_elems = [
        name,
        article,
        singular or name.lower(),
        plural or (singular or name.lower()) + 's',
        description
    ]

    text_bytes = encode_dmsg_entry(text_elems)

    text_area_size = ICON_LEN_OFFSET - text_offset
    entry[text_offset:ICON_LEN_OFFSET] = bytes(text_area_size)
    entry[text_offset:text_offset + len(text_bytes)] = text_bytes


def set_icon(entry: bytearray, icon_img: Image.Image, name_tag: str = "custom", icon_type: str = "item"):
    """Set the icon in an item entry."""
    icon_bytes = encode_icon(icon_img, name_tag, icon_type)

    struct.pack_into('<I', entry, ICON_LEN_OFFSET, len(icon_bytes))

    icon_area_size = ENTRY_SIZE - ICON_DATA_OFFSET
    entry[ICON_DATA_OFFSET:] = bytes(icon_area_size)
    entry[ICON_DATA_OFFSET:ICON_DATA_OFFSET + len(icon_bytes)] = icon_bytes


ICON_TYPE_MAP = {
    'general': 'item',
    'general2': 'item',
    'usable': 'item',
    'puppet': 'item',
    'armor': 'armor',
    'armor2': 'armor',
    'weapon': 'weapon',
}


def patch_item(item_id: int, *,
               name: str = None,
               description: str = None,
               singular: str = None,
               plural: str = None,
               template: int = None,
               icon_file: str = None,
               icon_data: str = None,
               hq: int = 0,
               tint: str = None,
               blend_mode: str = 'normal',
               hue_shift: float = None,
               fields: dict = None,
               icons_path: Path = None):
    """Patch a single item.

    Icons can be provided as either:
      icon_file: path to a PNG file (relative to icons_path)
      icon_data: base64-encoded PNG (for embedding in JSON configs)
    """
    if template:
        entry, item_type = read_entry(template)
        entry = bytearray(entry)
        set_field(entry, item_type.fields, 'id', item_id)
    else:
        entry, item_type = read_entry(item_id)
        entry = bytearray(entry)

    current_flags = get_field(entry, item_type.fields, 'flags')
    new_flags = (current_flags | FLAG_RARE | FLAG_EX) & ~FLAG_INSCRIBABLE
    set_field(entry, item_type.fields, 'flags', new_flags)

    if fields:
        for field_name, value in fields.items():
            set_field(entry, item_type.fields, field_name, value)

    if name or description:
        current_name = name or "Unknown"
        current_desc = description or ""
        set_text(entry, item_type.text_offset,
                 current_name, current_desc, singular, plural)

    if icon_file or icon_data or hq > 0:
        if icon_data:
            import base64, io
            png_bytes = base64.b64decode(icon_data)
            base_icon = Image.open(io.BytesIO(png_bytes)).convert('RGBA')
            if base_icon.size != (32, 32):
                base_icon = base_icon.resize((32, 32), Image.Resampling.LANCZOS)
        elif icon_file:
            icon_path = (icons_path or Path('icons')) / icon_file
            base_icon = load_icon(icon_path)
        else:
            base_icon = extract_icon(bytes(entry))

        if base_icon:
            final_icon = composite_icon(base_icon, hq, tint)
            target_type = get_item_type(item_id)
            icon_type = ICON_TYPE_MAP.get(target_type.name, 'item') if target_type else 'item'
            set_icon(entry, final_icon, str(item_id), icon_type)

    elif tint or hue_shift is not None:
        icon_len = struct.unpack('<I', entry[ICON_LEN_OFFSET:ICON_LEN_OFFSET+4])[0]
        if icon_len > 0:
            icon_data = bytes(entry[ICON_DATA_OFFSET:ICON_DATA_OFFSET + icon_len])

            if hue_shift is not None:
                icon_data = apply_hue_shift_to_icon_bytes(icon_data, hue_shift)

            if tint:
                icon_data = apply_tint_to_icon_bytes(icon_data, tint, blend_mode)

            entry[ICON_DATA_OFFSET:ICON_DATA_OFFSET + len(icon_data)] = icon_data

    write_entry(item_id, bytes(entry))
    print(f"Patched item {item_id}: {name or 'unnamed'}")


def patch_items_by_type(type_name: str, items: dict, icons_path: Path = None):
    """Patch multiple items of a specific type."""
    item_type = get_item_type_by_name(type_name)
    if not item_type:
        raise ValueError(f"Unknown item type: {type_name}")

    for item_id_str, config in items.items():
        item_id = int(item_id_str)
        if not item_type.contains(item_id):
            print(f"Warning: Item {item_id} not in {type_name} range, skipping")
            continue

        patch_item(
            item_id,
            name=config.get('name'),
            description=config.get('description'),
            singular=config.get('singular'),
            plural=config.get('plural'),
            template=config.get('template'),
            icon_file=config.get('icon_file'),
            icon_data=config.get('icon_data'),
            hq=config.get('hq', 0),
            tint=config.get('tint'),
            fields=config.get('fields'),
            icons_path=icons_path
        )


# === Config dispatch ===

def apply_items_config(config: dict, icons_path: Path = None):
    """Apply item patches from a config dict."""
    items = config.get('items', {})
    for item_id_str, item_data in items.items():
        item_id = int(item_id_str)
        patch_item(
            item_id,
            name=item_data.get('name'),
            description=item_data.get('description'),
            singular=item_data.get('singular'),
            plural=item_data.get('plural'),
            template=item_data.get('template'),
            icon_file=item_data.get('icon_file'),
            icon_data=item_data.get('icon_data'),
            hq=item_data.get('hq', 0),
            tint=item_data.get('tint'),
            blend_mode=item_data.get('blend_mode', 'normal'),
            hue_shift=item_data.get('hue_shift'),
            fields=item_data.get('fields'),
            icons_path=icons_path
        )


# Register module
register_module('items',
    config_key='item',
    apply_fn=apply_items_config,
)

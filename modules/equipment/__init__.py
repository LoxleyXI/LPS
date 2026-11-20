"""
LPS Items Model - Gear/weapon model recoloring and injection

Two modes:
  Overlay (default) — recolors an existing model DAT via XIPivot overlay.
    Affects ALL items sharing that model ID.

  Inject (inject: true) — creates a new model ID with recolored textures.
    Original model is unaffected. Requires the gearpatch Ashita addon
    loaded in-game to patch the gear tables at runtime.

Configured via items.json.
"""

import json
import shutil
import struct
from pathlib import Path

from xidat.paths import ffxi_dir, hd, overlay as overlay_dir
from mobs import recolor_entity_dat
from xidat.ftable import ensure_rom10, register, next_slot
from modules import register_module


# ── Gear table constants (FFXiMain.dll's gear tables) ──────────────────

RACES = [
    'HumeMale', 'HumeFemale', 'ElvaanMale', 'ElvaanFemale',
    'TaruMale', 'TaruFemale', 'Mithra', 'Galka',
]

ARMOR_SLOTS = ['head', 'body', 'hands', 'legs', 'feet']
WEAPON_SLOTS = ['main', 'sub']
ALL_SLOTS = ['face'] + ARMOR_SLOTS + WEAPON_SLOTS + ['ranged']

# File_id allocation for custom gear (above entity model range)
CUSTOM_GEAR_BASE = 118240
CUSTOM_WEAPON_OFFSET = 5200
CUSTOM_FACE_OFFSET = 7500
CUSTOM_RANGED_OFFSET = 7800

ORIGINAL_G5_COUNT = 64   # armor G5 originals carried over

CUSTOM_MODEL_START = {
    'face': 32, 'head': 672, 'body': 672, 'hands': 672,
    'legs': 672, 'feet': 672, 'main': 1196, 'sub': 1196, 'ranged': 256,
}

SLOT_CAPACITY = {
    'face': 32, 'head': 128, 'body': 128, 'hands': 128,
    'legs': 128, 'feet': 128, 'main': 128, 'sub': 128, 'ranged': 64,
}

# FTABLE target for gear injection
GEAR_FTABLE_TARGET = 126552


def custom_fid(race: str, slot: str, model_offset: int) -> int:
    """Compute file_id for a custom gear model in ROM10."""
    ri = RACES.index(race)
    if slot in ARMOR_SLOTS:
        si = ARMOR_SLOTS.index(slot)
        return CUSTOM_GEAR_BASE + ri * 640 + si * 128 + model_offset
    if slot in WEAPON_SLOTS:
        si = WEAPON_SLOTS.index(slot)
        return (CUSTOM_GEAR_BASE + CUSTOM_WEAPON_OFFSET
                + ri * 256 + si * 128 + model_offset)
    if slot == 'face':
        return CUSTOM_GEAR_BASE + CUSTOM_FACE_OFFSET + ri * 32 + model_offset
    if slot == 'ranged':
        return CUSTOM_GEAR_BASE + CUSTOM_RANGED_OFFSET + ri * 64 + model_offset
    raise ValueError(f'Unknown slot: {slot}')


# ── Gear table data (Atom0s RE, first 8 bytes per race for face G0) ─────

RACE_FACE_G0 = {
    'HumeMale': 7080, 'HumeFemale': 10256, 'ElvaanMale': 13432,
    'ElvaanFemale': 16608, 'TaruMale': 19784, 'TaruFemale': 22952,
    'Mithra': 23184, 'Galka': 26360,
}


# ── FTABLE helpers ──────────────────────────────────────────────────────

def _expand_ftables_for_gear(ffxi: Path):
    """Expand all FTABLEs to support gear injection file_ids."""
    expanded = False
    for idx in range(1, 11):
        if idx == 1:
            ft = ffxi / 'FTABLE.DAT'
            vt = ffxi / 'VTABLE.DAT'
        elif idx == 10:
            ft = ffxi / 'ROM10/FTABLE10.DAT'
            vt = ffxi / 'ROM10/VTABLE10.DAT'
        else:
            ft = ffxi / f'ROM{idx}/FTABLE{idx}.DAT'
            vt = ffxi / f'ROM{idx}/VTABLE{idx}.DAT'

        if not ft.exists():
            if idx == 10:
                # Create ROM10 tables
                ft.parent.mkdir(parents=True, exist_ok=True)
                ft.write_bytes(b'\x00\x00' * GEAR_FTABLE_TARGET)
                vt.write_bytes(b'\x00' * GEAR_FTABLE_TARGET)
                print(f'  Created {ft.name} ({GEAR_FTABLE_TARGET} entries)')
                expanded = True
            continue

        cur = ft.stat().st_size // 2
        if cur >= GEAR_FTABLE_TARGET:
            continue

        add = GEAR_FTABLE_TARGET - cur
        with open(ft, 'ab') as f:
            f.write(b'\x00\x00' * add)
        with open(vt, 'ab') as f:
            f.write(b'\x00' * add)
        print(f'  Expanded {ft.name}: {cur} -> {GEAR_FTABLE_TARGET} entries')
        expanded = True

    return expanded


def _register_gear_fid(ffxi: Path, file_id: int, subdir: int, file_idx: int):
    """Register a gear file_id in ALL FTABLE/VTABLE pairs pointing to ROM10."""
    ft_val = (subdir << 7) | (file_idx & 0x7F)
    vt_val = 10  # ROM10

    for idx in range(1, 11):
        if idx == 1:
            ft = ffxi / 'FTABLE.DAT'
            vt = ffxi / 'VTABLE.DAT'
        elif idx == 10:
            ft = ffxi / 'ROM10/FTABLE10.DAT'
            vt = ffxi / 'ROM10/VTABLE10.DAT'
        else:
            ft = ffxi / f'ROM{idx}/FTABLE{idx}.DAT'
            vt = ffxi / f'ROM{idx}/VTABLE{idx}.DAT'

        if not ft.exists():
            continue

        fdata = bytearray(ft.read_bytes())
        vdata = bytearray(vt.read_bytes())

        if file_id * 2 + 2 > len(fdata):
            continue

        struct.pack_into('<H', fdata, file_id * 2, ft_val)
        vdata[file_id] = vt_val
        ft.write_bytes(bytes(fdata))
        vt.write_bytes(bytes(vdata))


def _resolve_dat_from_ftable(ffxi: Path, file_id: int) -> Path | None:
    """Resolve a file_id to its DAT path."""
    for idx in range(1, 11):
        if idx == 1:
            ft = ffxi / 'FTABLE.DAT'
            vt = ffxi / 'VTABLE.DAT'
        else:
            ft = ffxi / f'ROM{idx}/FTABLE{idx}.DAT'
            vt = ffxi / f'ROM{idx}/VTABLE{idx}.DAT'
        if not ft.exists():
            continue
        fdata = ft.read_bytes()
        vdata = vt.read_bytes()
        if file_id * 2 + 2 > len(fdata) or file_id >= len(vdata):
            continue
        vval = vdata[file_id]
        if vval == 0:
            continue
        fval = struct.unpack_from('<H', fdata, file_id * 2)[0]
        subdir = fval >> 7
        fidx = fval & 0x7F
        rom = 'ROM' if vval == 1 else f'ROM{vval}'
        dat = ffxi / f'{rom}/{subdir}/{fidx}.DAT'
        if dat.exists():
            return dat
    return None


# ── Gear table parsing (from xi_core.py constants) ──────────────────────

# 432 bytes per race, 9 slots × 6 groups × 8 bytes
# We only need to read the original G5 base_fid/count for armor DAT copying

def _read_dll_tables(ffxi: Path) -> bytes | None:
    """Read all 8 race gear tables from FFXiMain.dll (3456 bytes)."""
    dll = ffxi / 'FFXiMain.dll'
    if not dll.exists():
        return None
    with open(dll, 'rb') as f:
        f.seek(0x34C80)
        return f.read(8 * 432)


def _read_armor_g5_entries(ffxi: Path) -> list:
    """Read original armor G5 entries from FFXiMain.dll."""
    data = _read_dll_tables(ffxi)
    if not data:
        return []

    _SLOTS = ['face', 'head', 'body', 'hands', 'legs', 'feet', 'main', 'sub', 'ranged']
    entries = []
    for ri, race in enumerate(RACES):
        for slot in ARMOR_SLOTS:
            si = _SLOTS.index(slot)
            off = ri * 432 + si * 48 + 5 * 8  # G5
            base = struct.unpack_from('<I', data, off)[0]
            count = struct.unpack_from('<I', data, off + 4)[0]
            entries.append((race, slot, base, count))
    return entries


def _resolve_gear_model(ffxi: Path, dll_data: bytes, race: str, slot: str,
                        model_id: int) -> Path | None:
    """Resolve a gear (race, slot, model_id) to its source DAT path.

    Walks the group entries in the DLL gear table exactly as the client does.
    """
    _SLOTS = ['face', 'head', 'body', 'hands', 'legs', 'feet', 'main', 'sub', 'ranged']
    ri = RACES.index(race)
    si = _SLOTS.index(slot)

    # Walk groups to find the file_id for this model_id
    mid_counter = 0
    for g in range(6):
        off = ri * 432 + si * 48 + g * 8
        base_fid = struct.unpack_from('<I', dll_data, off)[0]
        count = struct.unpack_from('<I', dll_data, off + 4)[0]
        if count == 0:
            continue
        if base_fid == 0:
            mid_counter += count
            continue
        if mid_counter <= model_id < mid_counter + count:
            fid = base_fid + (model_id - mid_counter)
            dat = _resolve_dat_from_ftable(ffxi, fid)
            return dat
        mid_counter += count
    return None


def resolve_gear_dat(race: str, slot: str, model_id: int, ffxi: Path = None) -> Path | None:
    """A gear model's DAT: (race, slot, model id) through the client's own gear tables."""
    ffxi = Path(ffxi or ffxi_dir())
    dll_data = _read_dll_tables(ffxi)
    if not dll_data:
        raise FileNotFoundError(f"no FFXiMain.dll in {ffxi}")
    return _resolve_gear_model(ffxi, dll_data, race, slot, model_id)


# ── Main apply functions ────────────────────────────────────────────────

def apply_items(config_path: Path):
    """Process an items.json config file.

    For each item entry:
      Overlay mode (default): recolors and writes to XIPivot overlay
      Inject mode (inject: true): recolors, places in ROM10, registers in FTABLE

    Config format::

        {
          "items": [
            {
              "name": "Emerald Ridill",
              "source_dat": "ROM/30/104.DAT",
              "color_ranges": [
                { "hue_min": 150, "hue_max": 340, "hue": 240, "saturation": 30 }
              ]
            },
            {
              "name": "Fire Ridill",
              "source_dat": "ROM/30/104.DAT",
              "inject": true,
              "race": "HumeMale",
              "slot": "main",
              "tint": "#ff0000",
              "blend_mode": "multiply"
            }
          ]
        }
    """
    config = json.loads(config_path.read_text())
    items = config.get('items', [])
    if not items:
        print("No items defined in config")
        return

    ffxi = ffxi_dir()
    overlay = overlay_dir(ffxi=ffxi)

    inject_items = [e for e in items if e.get('inject')]
    overlay_items = [e for e in items if not e.get('inject')]

    # ── Handle overlay items ────────────────────────────────────────
    for entry in overlay_items:
        _apply_overlay_item(entry, ffxi, overlay)

    # ── Handle inject items ─────────────────────────────────────────
    if inject_items:
        _apply_inject_items(inject_items, ffxi, config_path)

    print(f'\nDone. {len(overlay_items)} overlay + {len(inject_items)} injected.')


def _apply_overlay_item(entry: dict, ffxi: Path, overlay: Path):
    """Recolor a gear model as an XIPivot overlay (existing behaviour)."""
    name = entry['name']
    source_dat = entry['source_dat']
    hue = entry.get('hue', 0)
    sat = entry.get('saturation', 0)
    lit = entry.get('lightness', 0)
    tint = entry.get('tint')
    blend = entry.get('blend_mode', 'normal')
    color_ranges = entry.get('color_ranges')

    src = ffxi / source_dat
    if not src.exists():
        print(f'  ERROR: {src} not found')
        return

    dst = overlay / source_dat
    bak = Path(str(dst) + '.bak')
    if dst.exists() and not bak.exists():
        shutil.copy2(dst, bak)

    print(f'\n=== {name} (overlay) ===')
    print(f'  Source: {source_dat} ({src.stat().st_size:,} bytes)')

    has_adjustments = hue or sat or lit or tint or color_ranges
    if has_adjustments:
        stats = recolor_entity_dat(src, dst,
                                   hue=hue, saturation=sat, lightness=lit,
                                   tint=tint, blend_mode=blend,
                                   color_ranges=color_ranges)
        print(f'  {stats["dxt"]} DXT + {stats["paletted"]} paletted textures')
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    print(f'  Output: {dst}')


def _apply_inject_items(items: list, ffxi: Path, config_path: Path):
    """Recolor gear models and inject as new model IDs in ROM10.

    If 'race' is omitted, injects for ALL 8 races automatically
    using the same model_id per race. Resolves each race's source
    DAT from the DLL gear tables via 'source_model_id'.
    """
    rom10 = ffxi / 'ROM10'

    # Validate required fields
    for entry in items:
        if 'slot' not in entry:
            print(f'  ERROR: inject item "{entry["name"]}" missing "slot"')
            return
        if 'source_dat' not in entry and 'source_model_id' not in entry:
            print(f'  ERROR: inject item "{entry["name"]}" needs '
                  f'"source_dat" or "source_model_id"')
            return

    # Load DLL tables for per-race model resolution
    dll_data = _read_dll_tables(ffxi)

    # Expand FTABLEs for gear injection
    print('\n[*] Expanding FTABLEs for gear injection ...')
    _expand_ftables_for_gear(ffxi)

    # Copy original armor G5 DATs to ROM10 (one-time)
    _copy_armor_originals(ffxi, rom10)

    # Track assigned model_ids
    assigned = {}  # (race, slot) -> list of model_ids

    sql_lines = [
        '-- Generated by LPS gear injection',
        '-- Import into your LSB database',
        '',
    ]

    for entry in items:
        name = entry['name']
        slot = entry['slot']
        hue = entry.get('hue', 0)
        sat = entry.get('saturation', 0)
        lit = entry.get('lightness', 0)
        tint = entry.get('tint')
        blend = entry.get('blend_mode', 'normal')
        color_ranges = entry.get('color_ranges')
        target_mid = entry.get('target_model_id')
        source_mid = entry.get('source_model_id')
        source_dat = entry.get('source_dat')

        if slot not in ALL_SLOTS:
            print(f'  ERROR: Unknown slot "{slot}"')
            continue

        # Determine which races to process
        races = [entry['race']] if 'race' in entry else list(RACES)

        # Auto-assign model_id (shared across all races for this item)
        if target_mid is None:
            base = CUSTOM_MODEL_START[slot]
            capacity = SLOT_CAPACITY[slot]
            n = capacity if slot not in ARMOR_SLOTS else capacity - ORIGINAL_G5_COUNT
            # Find a model_id not used by any race for this slot
            all_used = set()
            for r in races:
                all_used.update(assigned.get((r, slot), []))
            for i in range(n):
                mid = base + i
                if mid not in all_used:
                    target_mid = mid
                    break
            if target_mid is None:
                print(f'  ERROR: No free model_ids for {slot}')
                continue

        # Compute model_offset once (same for all races)
        base_mid = CUSTOM_MODEL_START[slot]
        if slot in ARMOR_SLOTS:
            model_offset = ORIGINAL_G5_COUNT + (target_mid - base_mid)
        else:
            model_offset = target_mid - base_mid

        afterglow = entry.get('afterglow')
        has_adjustments = hue or sat or lit or tint or color_ranges
        all_races_label = 'all races' if len(races) == 8 else ', '.join(races)
        print(f'\n=== {name} (inject: {all_races_label} {slot} mid={target_mid}) ===')

        for race in races:
            if race not in RACES:
                print(f'  ERROR: Unknown race "{race}"')
                continue

            # Resolve source DAT for this race
            if source_mid is not None and dll_data:
                src = _resolve_gear_model(ffxi, dll_data, race, slot, source_mid)
                if not src:
                    print(f'  {race:14s}: source model {source_mid} not found, skipped')
                    continue
                src = hd(src, ffxi)
            elif source_dat:
                src = ffxi / source_dat
                if not src.exists():
                    print(f'  {race:14s}: {source_dat} not found, skipped')
                    continue
                src = hd(src, ffxi)
            else:
                print(f'  {race:14s}: no source specified, skipped')
                continue

            file_id = custom_fid(race, slot, model_offset)

            # Allocate ROM10 slot
            gear_offset = file_id - CUSTOM_GEAR_BASE
            subdir = 200 + gear_offset // 128
            file_idx = gear_offset % 128

            dst_dir = rom10 / str(subdir)
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst = dst_dir / f'{file_idx}.DAT'

            if has_adjustments:
                stats = recolor_entity_dat(src, dst,
                                           hue=hue, saturation=sat, lightness=lit,
                                           tint=tint, blend_mode=blend,
                                           color_ranges=color_ranges)
                print(f'  {race:14s}: {stats["dxt"]} DXT + {stats["paletted"]} pal -> ROM10/{subdir}/{file_idx}.DAT')
            else:
                shutil.copy2(src, dst)
                print(f'  {race:14s}: cloned -> ROM10/{subdir}/{file_idx}.DAT')

            # Apply afterglow particle aura if requested
            afterglow = entry.get('afterglow')
            if afterglow:
                from mobs import _build_afterglow_dat
                weapon_dat = dst.read_bytes()
                ag_r = afterglow.get('r', 1.0)
                ag_g = afterglow.get('g', 1.0)
                ag_b = afterglow.get('b', 1.0)
                glowing_dat = _build_afterglow_dat(weapon_dat, afterglow, ffxi)
                dst.write_bytes(glowing_dat)
                print(f'  {race:14s}: afterglow RGB({ag_r},{ag_g},{ag_b})')

            # Swap particle effects from a different DAT
            effects_source = entry.get('effects_source')
            if effects_source:
                from modules.particles import apply_effects_source
                fx_dat_path = effects_source.get('dat', '')
                fx_src = ffxi / fx_dat_path if fx_dat_path else None
                if fx_src and fx_src.exists():
                    weapon_dat = dst.read_bytes()
                    patched = apply_effects_source(weapon_dat, fx_src.read_bytes())
                    dst.write_bytes(patched)
                    print(f'  {race:14s}: effects from {fx_dat_path}')

            # Apply per-emitter particle overrides
            particle_overrides = entry.get('particle_overrides')
            if particle_overrides:
                from modules.particles import patch_particle_overrides
                weapon_dat = dst.read_bytes()
                patched = patch_particle_overrides(weapon_dat, particle_overrides)
                dst.write_bytes(patched)
                n = len(particle_overrides)
                print(f'  {race:14s}: {n} particle override(s)')

            _register_gear_fid(ffxi, file_id, subdir, file_idx)
            assigned.setdefault((race, slot), []).append(target_mid)

        # SQL
        item_id = entry.get('item_id')
        if item_id:
            sql_lines.append(f'-- {name}')
            sql_lines.append(
                f'UPDATE `item_equipment` SET `MId` = {target_mid} '
                f'WHERE `itemId` = {item_id};')
            sql_lines.append('')
        else:
            sql_lines.append(f'-- {name}: model_id={target_mid} '
                             f'(set item_id in config for auto SQL)')
            sql_lines.append('')

    # Write SQL
    sql_path = config_path.parent / 'items_inject.sql'
    sql_path.write_text('\n'.join(sql_lines))
    print(f'\nSQL: {sql_path}')
    print(f'Requires gearpatch Ashita addon loaded in-game.')


def _copy_armor_originals(ffxi: Path, rom10: Path):
    """Copy original armor G5 DATs to ROM10 (idempotent)."""
    entries = _read_armor_g5_entries(ffxi)
    if not entries:
        print('  WARNING: Could not read armor G5 entries from FFXiMain.dll')
        return

    copied = 0
    already = 0
    for race, slot, orig_base, orig_count in entries:
        for i in range(orig_count):
            old_fid = orig_base + i
            new_fid = custom_fid(race, slot, i)

            # Check if already copied
            gear_offset = new_fid - CUSTOM_GEAR_BASE
            subdir = 200 + gear_offset // 128
            file_idx = gear_offset % 128
            dst = rom10 / str(subdir) / f'{file_idx}.DAT'
            if dst.exists():
                already += 1
                continue

            dat = _resolve_dat_from_ftable(ffxi, old_fid)
            if dat and dat.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(dat, dst)
                _register_gear_fid(ffxi, new_fid, subdir, file_idx)
                copied += 1

    if copied:
        print(f'  Copied {copied} original armor DATs to ROM10')
    if already:
        print(f'  {already} armor DATs already in ROM10 (skipped)')


# === Module registration ===

def setup_cli(subparsers):
    """Add equipment CLI subcommands."""
    im_parser = subparsers.add_parser('equipment', help='Recolor gear models from config')
    im_parser.add_argument('config', nargs='?', default='equipment.json',
                           help='Equipment config JSON (default: equipment.json)')


register_module('equipment',
    cli_setup=setup_cli,
    depends=['zones', 'mobs'],
)

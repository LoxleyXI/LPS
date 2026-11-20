#!/usr/bin/env python3
"""Generate weapons.json for LPS Studio's weapon dropdown.

Scans gear tables for weapon DATs, pulls names from LSB's item_equipment SQL,
checks each for particle effects, and outputs a JSON index.

Run:  python3 gen_weapons.py
"""
import json, struct, os, sys, re
from pathlib import Path

# the gear tables, through LPS's own equipment module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))            # LPS
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'modules'))
os.environ.setdefault('FFXI_DIR',
    str(Path(__file__).resolve().parent / 'game'))
FFXI_DIR = os.environ['FFXI_DIR']

from equipment import resolve_gear_dat   # noqa: E402


def load_weapon_names_from_sql():
    """Load model_id → weapon name from LSB's item_equipment.sql."""
    # Look for item_equipment.sql via LSB_SQL_DIR env var or next to this script
    script_dir = Path(__file__).resolve().parent
    search_paths = []
    if os.environ.get('LSB_SQL_DIR'):
        search_paths.append(Path(os.environ['LSB_SQL_DIR']) / 'item_equipment.sql')
    search_paths.append(script_dir / 'item_equipment.sql')

    sql_path = None
    for p in search_paths:
        if p.exists():
            sql_path = p
            break

    if not sql_path:
        print("Warning: item_equipment.sql not found, using model IDs as names")
        return {}

    mid_to_name = {}
    with open(sql_path) as f:
        for line in f:
            m = re.match(
                r"INSERT INTO `item_equipment` VALUES "
                r"\((\d+),'([^']+)',(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),",
                line
            )
            if not m:
                continue
            model_id = int(m.group(6))
            slot = int(m.group(9))

            # Only weapon slots (MAIN=1, SUB=2, RANGE=4, AMMO=8)
            if (slot & 0x000F) == 0 or model_id <= 0:
                continue

            name = m.group(2).replace('_', ' ').title()
            if model_id not in mid_to_name:
                mid_to_name[model_id] = name

    print(f"Loaded {len(mid_to_name)} weapon names from {sql_path.name}")
    return mid_to_name


KNOWN = load_weapon_names_from_sql()


def align16(v):
    return (v + 0xF) & ~0xF


def count_particle_sections(data: bytes) -> dict:
    """Quick scan for particle-related sections in a DAT."""
    counts = {}
    pos = 0
    while pos + 16 <= len(data):
        meta = struct.unpack_from('<I', data, pos + 4)[0]
        tc = meta & 0x7F
        sz = ((meta >> 7) & 0xFFFFF) * 0x10
        if sz <= 0:
            break
        name = data[pos:pos+4].decode('ascii', errors='replace').strip()
        if tc == 0x05:
            counts.setdefault('generators', []).append(name)
        elif tc == 0x19:
            counts['keyframes'] = counts.get('keyframes', 0) + 1
        elif tc == 0x1F:
            counts['meshes'] = counts.get('meshes', 0) + 1
        elif tc == 0x21:
            counts['sprites'] = counts.get('sprites', 0) + 1
        elif tc == 0x07:
            counts.setdefault('routines', []).append(name)
        pos = align16(pos + sz)
    return counts


def main():
    ffxi = Path(FFXI_DIR)
    weapons = []

    # Scan known model IDs + a broad range
    scan_ids = set(KNOWN.keys())
    for start, end in [(1, 200), (250, 600), (700, 900), (1100, 1200)]:
        for mid in range(start, end):
            scan_ids.add(mid)

    for mid in sorted(scan_ids):
        for slot in ['main', 'sub', 'ranged']:
            try:
                dat_path = resolve_gear_dat('HumeMale', slot, mid)
            except Exception:
                continue
            if not dat_path.exists():
                continue
            try:
                rel = dat_path.relative_to(ffxi)
            except ValueError:
                continue

            data = dat_path.read_bytes()
            counts = count_particle_sections(data)
            gens = counts.get('generators', [])
            name = KNOWN.get(mid, f'Model {mid}')

            entry = {
                'model_id': mid,
                'slot': slot,
                'name': name,
                'dat': f'game/{rel}',
                'size': len(data),
                'generators': len(gens),
                'generator_names': gens,
            }
            if counts.get('keyframes'): entry['keyframes'] = counts['keyframes']
            if counts.get('meshes'): entry['particle_meshes'] = counts['meshes']
            if counts.get('routines'): entry['routines'] = counts['routines']

            weapons.append(entry)
            break

    # Sort: weapons with particles first, then by name
    weapons.sort(key=lambda w: (-w['generators'], w['name']))

    out = Path(__file__).with_name('weapons.json')
    out.write_text(json.dumps(weapons, indent=2, ensure_ascii=False), encoding='utf-8')

    has_fx = sum(1 for w in weapons if w['generators'] > 0)
    print(f'Wrote {len(weapons)} weapons ({has_fx} with particle effects) -> {out}')


if __name__ == '__main__':
    main()

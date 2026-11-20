"""
LPS Zones: recoloured zones

Clones a zone with its textures and lighting recoloured (recolor_zone_dat), or with a generated
cave in place of its geometry; `python -m lps zones zones.json` registers the copies in ROM10.
The file tables, file ids, zone names and texture code it uses are xidat's.

DXT textures are recoloured in place by shifting their RGB565 endpoints: no decode and re-encode,
no quality loss. About 1.5 s per HD zone DAT.
"""

import json
import shutil
import struct
from pathlib import Path
from typing import List, Optional

from xidat.paths import ffxi_dir, hd
from xidat.textures import (  # noqa: F401  (parse_sections etc. are this module's API too)
    Section, TextureInfo, SECTION_TYPE_TEXTURE, parse_sections, parse_texture_info,
    _rgb_to_hsv, _hsv_to_rgb, _parse_color, _blend, _adjust_dxt_inplace, _adjust_palette_inplace,
)
from xidat.ftable import load_all, resolve, ensure_rom10, register, next_slot
from xidat import zone_files


# ---------------------------------------------------------------------------
# Main recolor function
# ---------------------------------------------------------------------------

def _match_texture(name: str, pattern: str) -> bool:
    """Match a texture name against a pattern. Supports trailing * wildcard."""
    name = name.strip().lower()
    pattern = pattern.strip().lower()
    if pattern.endswith('*'):
        return name.startswith(pattern[:-1])
    return name == pattern


def recolor_zone_dat(source_path: Path, output_path: Path,
                     hue: float = 0, saturation: float = 0, lightness: float = 0,
                     tint: str = None, blend_mode: str = 'normal',
                     skip_textures: Optional[List[str]] = None,
                     texture_overrides: Optional[dict] = None) -> dict:
    """Recolor all textures in a zone DAT.

    DXT textures: modifies RGB565 color endpoints in each 4x4 block.
    Paletted textures: modifies palette BGRA entries.
    Indices and alpha are never touched.

    Args:
        source_path: Source zone DAT
        output_path: Output DAT path
        hue: Hue rotation in degrees (0-360)
        saturation: Saturation adjust (-100 to 100)
        lightness: Brightness adjust (-100 to 100)
        tint: Hex color '#RRGGBB' or '#RRGGBBAA'
        blend_mode: 'normal', 'multiply', 'screen', 'overlay', 'add'
        skip_textures: Texture names to leave unchanged
        texture_overrides: Per-texture settings, keyed by name or pattern.
            Supports trailing * wildcard.
            Example: {"suny*": {"hue": 240}, "moon*": {"lightness": -50}}

    Returns:
        Dict with counts: {'dxt': N, 'paletted': N, 'skipped': N, 'total': N}
    """
    skip = set(skip_textures or [])
    overrides = texture_overrides or {}
    data = bytearray(source_path.read_bytes())
    sections = parse_sections(bytes(data))
    default_kwargs = dict(hue=hue, saturation=saturation, lightness=lightness,
                          tint=tint, blend_mode=blend_mode)

    stats = {'dxt': 0, 'paletted': 0, 'skipped': 0, 'total': 0}

    for s in sections:
        if s.type_code != SECTION_TYPE_TEXTURE:
            continue

        info = parse_texture_info(bytes(data), s)
        if info is None:
            continue

        stats['total'] += 1

        if info.name in skip:
            stats['skipped'] += 1
            continue

        # Check for per-texture overrides (merged on top of zone defaults)
        kwargs = dict(default_kwargs)
        for pattern, override in overrides.items():
            if _match_texture(info.name, pattern):
                kwargs = dict(
                    hue=override.get('hue', kwargs['hue']),
                    saturation=override.get('saturation', kwargs['saturation']),
                    lightness=override.get('lightness', kwargs['lightness']),
                    tint=override.get('tint', kwargs['tint']),
                    blend_mode=override.get('blend_mode', kwargs['blend_mode']),
                )
                break

        if not any([kwargs['hue'], kwargs['saturation'], kwargs['lightness'], kwargs['tint']]):
            stats['skipped'] += 1
            continue

        if info.tex_type == 0xA1 and info.dxt_type in ("1TXD", "3TXD"):
            _adjust_dxt_inplace(data, info, **kwargs)
            stats['dxt'] += 1
        elif info.tex_type in (0x91, 0xB1):
            _adjust_palette_inplace(data, info, **kwargs)
            stats['paletted'] += 1

    # Adjust environment lighting (0x2F sections)
    env_kwargs = dict(default_kwargs)
    if 'environment' in overrides:
        env_override = overrides['environment']
        env_kwargs = dict(
            hue=env_override.get('hue', env_kwargs['hue']),
            saturation=env_override.get('saturation', env_kwargs['saturation']),
            lightness=env_override.get('lightness', env_kwargs['lightness']),
            tint=env_override.get('tint', env_kwargs['tint']),
            blend_mode=env_override.get('blend_mode', env_kwargs['blend_mode']),
            fog_tint=env_override.get('fog_tint'),
            fog_end=env_override.get('fog_end'),
            fog_start=env_override.get('fog_start'),
        )

    if any([env_kwargs.get('hue'), env_kwargs.get('saturation'),
            env_kwargs.get('lightness'), env_kwargs.get('tint'),
            env_kwargs.get('fog_tint'), env_kwargs.get('fog_end') is not None,
            env_kwargs.get('fog_start') is not None]):
        env_count = _adjust_environment_inplace(data, env_kwargs)
        stats['environment'] = env_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(bytes(data))

    return stats


# ---------------------------------------------------------------------------
# Environment section (0x2F) color adjustment
# ---------------------------------------------------------------------------

SECTION_TYPE_ENVIRONMENT = 0x2F

# BGRA color offsets within 0x2F data (relative to data_start)
_ENV_COLOR_OFFSETS = {
    'model_sun':     0x0C,
    'model_moon':    0x10,
    'model_ambient': 0x14,
    'model_fog':     0x18,
    'terrain_sun':   0x2C,
    'terrain_moon':  0x30,
    'terrain_ambient': 0x34,
    'terrain_fog':   0x38,
}


def _adjust_bgra(data: bytearray, offset: int, **kwargs):
    """Adjust a single BGRA color in-place."""
    b, g, r, a = data[offset], data[offset+1], data[offset+2], data[offset+3]
    if r == 0 and g == 0 and b == 0:
        return

    hue = kwargs.get('hue', 0)
    sat = kwargs.get('saturation', 0)
    lit = kwargs.get('lightness', 0)
    tint = kwargs.get('tint')
    blend_mode = kwargs.get('blend_mode', 'normal')

    if hue or sat or lit:
        h, s, v = _rgb_to_hsv(r, g, b)
        if hue:
            h = (h + hue) % 360
        if sat:
            s = max(0.0, min(1.0, s * (1.0 + sat / 100.0)))
        if lit:
            v = max(0.0, min(1.0, v * (1.0 + lit / 100.0)))
        r, g, b = _hsv_to_rgb(h, s, v)

    if tint:
        tr, tg, tb, ta = _parse_color(tint)
        r, g, b = _blend(r, g, b, tr, tg, tb, ta, blend_mode)

    data[offset] = b
    data[offset+1] = g
    data[offset+2] = r


def _set_bgra(data: bytearray, offset: int, color: str):
    """Set a BGRA color directly from a hex string."""
    r, g, b, a = _parse_color(color)
    data[offset] = b
    data[offset + 1] = g
    data[offset + 2] = r
    data[offset + 3] = a


def _adjust_environment_inplace(data: bytearray, kwargs: dict) -> int:
    """Adjust all 0x2F environment section colors in-place."""
    fog_tint = kwargs.pop('fog_tint', None)
    sections = parse_sections(bytes(data))
    count = 0
    for s in sections:
        if s.type_code != SECTION_TYPE_ENVIRONMENT:
            continue
        for name, offset in _ENV_COLOR_OFFSETS.items():
            _adjust_bgra(data, s.data_start + offset, **kwargs)
        # Override fog color if specified
        if fog_tint:
            _set_bgra(data, s.data_start + 0x18, fog_tint)  # model fog
            _set_bgra(data, s.data_start + 0x38, fog_tint)  # terrain fog
        # Override fog distances if specified
        fog_end = kwargs.get('fog_end')
        fog_start = kwargs.get('fog_start')
        if fog_end is not None:
            struct.pack_into('<f', data, s.data_start + 0x1C, fog_end)  # model
            struct.pack_into('<f', data, s.data_start + 0x3C, fog_end)  # terrain
        if fog_start is not None:
            struct.pack_into('<f', data, s.data_start + 0x20, fog_start)  # model
            struct.pack_into('<f', data, s.data_start + 0x40, fog_start)  # terrain
        count += 1
    return count


def list_zone_textures(dat_path: Path) -> List[TextureInfo]:
    """List all textures in a zone DAT file."""
    data = dat_path.read_bytes()
    sections = parse_sections(data)
    textures = []
    for s in sections:
        if s.type_code != SECTION_TYPE_TEXTURE:
            continue
        info = parse_texture_info(data, s)
        if info is not None:
            textures.append(info)
    return textures


# ---------------------------------------------------------------------------
# Zone injection from JSON config
# ---------------------------------------------------------------------------

def apply_zones(config_path: Path):
    """Process a zones.json config file.

    For each zone entry:
    1. Resolves source zone's model/dialog/npc/event DATs via FTABLE
    2. Recolors the model DAT (if hue specified)
    3. Clones dialog/npc/event DATs
    4. Registers all four file IDs in FTABLE10/VTABLE10
    5. Places DATs in ROM10/

    Config format::

        {
          "zones": [
            {
              "name": "Green Fields",
              "clone": 115,
              "zone_id": 400,
              "hue": 120
            }
          ]
        }
    """
    config = json.loads(config_path.read_text())
    zones = config.get('zones', [])
    if not zones:
        print("No zones defined in config")
        return

    ffxi = ffxi_dir()
    tables = load_all(ffxi)
    ft10, vt10 = ensure_rom10(ffxi)
    rom10 = ffxi / 'ROM10'
    subdir = 1

    for entry in zones:
        name = entry['name']
        clone_zid = entry['clone']
        zone_id = entry['zone_id']
        hue = entry.get('hue', 0)
        sat = entry.get('saturation', 0)
        lit = entry.get('lightness', 0)
        tint = entry.get('tint')
        blend = entry.get('blend_mode', 'normal')
        tex_overrides = entry.get('textures', {})
        scale = entry.get('scale', 0)
        has_adjustments = hue or sat or lit or tint or tex_overrides
        has_scale = scale and scale != 1.0

        desc = []
        if hue: desc.append(f'hue {hue}°')
        if sat: desc.append(f'sat {sat:+d}%')
        if lit: desc.append(f'lit {lit:+d}%')
        if tint: desc.append(f'tint {tint} ({blend})')
        if has_scale: desc.append(f'scale {scale}x')
        desc_str = ', '.join(desc) if desc else 'clone only'

        print(f'\n=== {name} (zone {zone_id}, clone {clone_zid}, {desc_str}) ===')

        # Resolve source zone DATs (the HD copies when there are any)
        src_dats = {}
        for dtype, fn in zone_files.FIDS.items():
            fid = fn(clone_zid)
            dat_path = resolve(fid, tables)
            if dat_path and dat_path.exists():
                src_dats[dtype] = hd(dat_path, ffxi)
            else:
                print(f'  warning: {dtype} DAT for zone {clone_zid} not found (fid {fid})')

        if 'model' not in src_dats:
            print(f'  ERROR: No model DAT for zone {clone_zid}, skipping')
            continue

        # Process each DAT type
        for dtype in ('model', 'event', 'dialog', 'npc'):
            if dtype not in src_dats:
                continue

            src = src_dats[dtype]
            dst_fid = zone_files.FIDS[dtype](zone_id)
            subdir, slot = next_slot(rom10, subdir)
            dst = rom10 / str(subdir) / f'{slot}.DAT'

            if dtype == 'model' and (has_adjustments or has_scale):
                print(f'  {dtype:8s} → ROM10/{subdir}/{slot}.DAT  ({desc_str})')

                if has_scale:
                    print('           scale: not supported, left at 1')
                stats = recolor_zone_dat(src, dst, hue=hue, saturation=sat,
                                         lightness=lit, tint=tint, blend_mode=blend,
                                         texture_overrides=tex_overrides)
                print(f'           {stats["dxt"]} DXT + {stats["paletted"]} paletted textures')
            else:
                print(f'  {dtype:8s} → ROM10/{subdir}/{slot}.DAT  (clone)')
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

            if dtype == 'model' and entry.get('generate'):
                # a generated cave replaces the clone's geometry (caves module), same textures
                import sys as _sys
                _here = str(Path(__file__).resolve().parent.parent)
                if _here not in _sys.path:
                    _sys.path.insert(0, _here)
                import caves
                caves.generate_into(dst, entry['generate'])
                print(f'           generated cave: seed {entry["generate"].get("seed")}, '
                      f'{entry["generate"].get("rooms", 5)} rooms')

            register(ft10, vt10, dst_fid, subdir, slot)

        print(f'  Registered zone {zone_id} in FTABLE10')

    # Patch zone name tables
    zone_names = {entry['zone_id']: entry['name'] for entry in zones}
    zone_files.write_names(zone_names, ffxi)

    print(f'\nDone. {len(zones)} zone(s) injected.')
    print(f'\nServer-side TODO for each zone:')
    print(f'  1. Apply patches/zone_max_512.patch (MAX_ZONEID >= 512)')
    print(f'  2. INSERT INTO zone_settings (...)')
    print(f'  3. Create scripts/zones/<ZoneName>/')
    print(f'  4. Restart server + client')


# === Module registration ===

try:                                       # the LPS command line's plugin list, when LPS is here
    from modules import register_module
except ImportError:                        # used without the LPS command line: no plugins to register
    def register_module(*args, **kwargs):
        pass


def apply_zones_config(config: dict, icons_path=None):
    """Apply zones from config (expects zones.json content)."""
    # zones module uses its own config format via apply_zones()
    pass


def setup_cli(subparsers):
    """Add zones CLI subcommands."""
    zones_parser = subparsers.add_parser('zones', help='Inject zones from zones.json')
    zones_parser.add_argument('config', nargs='?', default='zones.json',
                              help='Zones config JSON (default: zones.json)')


register_module('zones',
    cli_setup=setup_cli,
)

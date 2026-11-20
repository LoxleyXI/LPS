"""
LPS CLI - Command line interface for Loxley Patch System

Usage:
    python -m lps apply config.json           # Apply patches from config
    python -m lps validate config.json        # Validate config without applying
    python -m lps example output.json         # Create example config

Module commands (loaded dynamically):
    python -m lps zones config.json           # Inject zones (zones module)
    python -m lps mobs config.json            # Inject mobs (mobs module)
    python -m lps equipment config.json       # Recolor gear (equipment module)
"""

import sys
import argparse
from pathlib import Path

from lps.config import apply_config, validate_config, create_example_config
from modules import load_modules, MODULES


def cmd_apply(args):
    """Apply patches from config file(s) or directory."""
    config_path = Path(args.config[0])
    if not config_path.exists():
        print(f"Error: Not found: {config_path}")
        return 1

    extra = [Path(p) for p in args.config[1:]] if len(args.config) > 1 else None
    icons_path = Path(args.icons) if args.icons else None
    apply_config(config_path, icons_path, extra_configs=extra)
    return 0


def cmd_validate(args):
    """Validate config file."""
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        return 1

    errors = validate_config(config_path)
    if errors:
        print("Validation errors:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("Config is valid")
        return 0


def cmd_example(args):
    """Create example config file."""
    output_path = Path(args.output)
    create_example_config(output_path)
    return 0


def cmd_icon_encode(args):
    """Generate icon config from a custom PNG or template item ID.

    Custom PNG: embeds the icon as base64 in the config (your artwork).
    Template ID: outputs hue/tint/hq instructions only (no proprietary data).
    """
    import json

    hq = int(args.hq) if args.hq else 0
    hue = float(args.hue) if args.hue else 0
    tint = args.tint or None

    if args.png:
        # Custom PNG provided - encode as base64
        import base64, io
        from PIL import Image
        from lps.icons import apply_hq_border, apply_tint, ICON_SIZE
        from lps.icons import _rgb_to_hsv, _hsv_to_rgb

        png_path = Path(args.png)
        if not png_path.exists():
            print(f"Error: {png_path} not found")
            return 1

        img = Image.open(png_path).convert('RGBA')
        if img.size != (ICON_SIZE, ICON_SIZE):
            img = img.resize((ICON_SIZE, ICON_SIZE), Image.Resampling.LANCZOS)

        if hq > 0:
            img = apply_hq_border(img, hq)
        if tint:
            img = apply_tint(img, tint)
        if hue:
            pixels = img.load()
            for y in range(img.size[1]):
                for x in range(img.size[0]):
                    r, g, b, a = pixels[x, y]
                    if a == 0: continue
                    h, s, v = _rgb_to_hsv(r, g, b)
                    h = (h + hue) % 360
                    r, g, b = _hsv_to_rgb(h, s, v)
                    pixels[x, y] = (r, g, b, a)

        buf = io.BytesIO()
        img.save(buf, format='PNG')
        b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        entry = {"icon_data": b64}

    elif args.template:
        # Template ID - output instructions only, never base64
        entry = {"template": int(args.template)}
        if hq > 0: entry["hq"] = hq
        if hue: entry["hue_shift"] = hue
        if tint: entry["tint"] = tint
        if args.blend:
            entry["blend_mode"] = args.blend

    else:
        print("Error: provide a PNG file or --template ID")
        return 1

    if args.wrap:
        item_id = args.wrap
        name = args.name or (Path(args.png).stem.replace('_', ' ').replace('-', ' ').title() if args.png else f"Item {item_id}")
        entry["name"] = name
        if args.template and args.png:
            entry["template"] = int(args.template)
        config = {"custom": {"item": {"items": {item_id: entry}}}}
        print(json.dumps(config, indent=2))
    elif args.json:
        print(json.dumps(entry, indent=2))
    else:
        if "icon_data" in entry:
            print(entry["icon_data"])
        else:
            print(json.dumps(entry, indent=2))

    return 0


def cmd_manifest(args):
    """Generate a manifest of config file hashes for launcher distribution."""
    import hashlib
    import json

    config_path = Path(args.path)
    if not config_path.exists():
        print(f"Error: Not found: {config_path}")
        return 1

    # Collect JSON files
    if config_path.is_dir():
        files = sorted(config_path.glob('*.json'))
        # Also include any referenced icon PNGs
        files.extend(sorted(config_path.rglob('*.png')))
    else:
        files = [config_path]

    files = [f for f in files if f.name != 'manifest.json']

    if not files:
        print("No config files found.")
        return 1

    # Hash each file
    manifest = {"version": 1, "files": {}}
    for f in files:
        h = hashlib.sha256()
        with open(f, 'rb') as fp:
            for chunk in iter(lambda: fp.read(8192), b''):
                h.update(chunk)
        rel = str(f.relative_to(config_path)) if config_path.is_dir() else f.name
        manifest["files"][rel] = h.hexdigest()

    # Write manifest
    out = Path(args.output)
    with open(out, 'w') as fp:
        json.dump(manifest, fp, indent=2)

    print(f"Manifest: {out}")
    print(f"  {len(manifest['files'])} files tracked")
    for rel, sha in manifest["files"].items():
        print(f"  {rel}: {sha[:16]}...")

    return 0


# === Module import helper ===

def _import_from(name: str, attr: str):
    """Import an attribute from a module: LPS's own or an extension's."""
    import importlib
    for full in (f'modules.{name}', name):           # LPS's own, then an extension's (by its name)
        try:
            mod = importlib.import_module(full)
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Cannot find '{attr}' in module '{name}'")


# === Module CLI commands ===

def cmd_zones(args):
    """Process zones.json config file."""
    apply_zones = _import_from('zones', 'apply_zones')
    config = Path(args.config)
    if not config.exists():
        print(f"Error: Config not found: {config}")
        return 1
    apply_zones(config)
    return 0


def cmd_mobs(args):
    """Process mobs.json config file."""
    apply_mobs = _import_from('mobs', 'apply_mobs')
    config = Path(args.config)
    if not config.exists():
        print(f"Error: Config not found: {config}")
        return 1
    apply_mobs(config)
    return 0


def cmd_equipment(args):
    """Process equipment model recoloring."""
    apply_items = _import_from('equipment', 'apply_items')
    config = Path(args.config)
    if not config.exists():
        print(f"Error: Config not found: {config}")
        return 1
    apply_items(config)
    return 0


def main():
    # Discover modules
    load_modules()

    parser = argparse.ArgumentParser(
        description='LPS - Loxley Patch System for FFXI DATs'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Core commands
    apply_parser = subparsers.add_parser('apply', help='Apply patches from config')
    apply_parser.add_argument('config', nargs='+', help='Config JSON file(s) or directory')
    apply_parser.add_argument('--icons', help='Icons directory')

    validate_parser = subparsers.add_parser('validate', help='Validate config')
    validate_parser.add_argument('config', help='Config JSON file')

    example_parser = subparsers.add_parser('example', help='Create example config')
    example_parser.add_argument('output', help='Output JSON file')

    manifest_parser = subparsers.add_parser('manifest', help='Generate config manifest for launcher distribution')
    manifest_parser.add_argument('path', help='Config file or directory')
    manifest_parser.add_argument('-o', '--output', default='manifest.json', help='Output manifest path')

    icon_parser = subparsers.add_parser('icon-encode', help='Generate icon config from PNG or template')
    icon_parser.add_argument('png', nargs='?', help='Custom PNG file (embeds as base64)')
    icon_parser.add_argument('--template', help='Template item ID (outputs instructions only, no base64)')
    icon_parser.add_argument('--hq', choices=['1','2','3','4'], help='HQ border (1=silver, 2=blue, 3=gold, 4=green)')
    icon_parser.add_argument('--hue', help='Hue shift in degrees')
    icon_parser.add_argument('--tint', help='Tint color hex (e.g. #ff000080)')
    icon_parser.add_argument('--blend', help='Blend mode for tint (multiply, overlay, screen)')
    icon_parser.add_argument('--json', action='store_true', help='Output as JSON snippet')
    icon_parser.add_argument('--name', help='Item name')
    icon_parser.add_argument('--wrap', metavar='ITEM_ID', help='Output as full LPS config with item ID')

    # Let modules add their subcommands
    for mod_name, mod_info in MODULES.items():
        cli_setup = mod_info.get('cli_setup')
        if cli_setup:
            cli_setup(subparsers)

    args = parser.parse_args()

    commands = {
        'apply': cmd_apply,
        'validate': cmd_validate,
        'example': cmd_example,
        'manifest': cmd_manifest,
        'icon-encode': cmd_icon_encode,
        # Zones module
        'zones': cmd_zones,
        # Mobs module
        'mobs': cmd_mobs,
        # Equipment module
        'equipment': cmd_equipment,
    }

    handler = commands.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())

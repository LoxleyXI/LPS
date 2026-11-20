"""
LPS Config - JSON configuration file processing

Dispatches config sections to registered modules.

Config structure:
{
    "custom": {
        "item": { ... },        -> items module
        "strings": { ... },     -> strings module
    }
}
"""

import json
from pathlib import Path

from modules import load_modules, MODULES


def load_config(config_path: Path) -> dict:
    """Load a JSON config file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def merge_configs(configs: list[dict]) -> dict:
    """Deep-merge multiple configs. Later configs override earlier ones.

    For 'items' dicts, entries are merged by item ID (later wins).
    For lists (like 'ui_icons'), entries are concatenated.
    """
    merged = {}
    for config in configs:
        custom = config.get('custom', {})
        if not merged:
            merged = custom
            continue
        for key, value in custom.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, dict) and isinstance(merged[key], dict):
                # Deep merge dicts (e.g. item.items entries by ID)
                for k2, v2 in value.items():
                    if isinstance(v2, dict) and isinstance(merged[key].get(k2), dict):
                        merged[key][k2] = {**merged[key][k2], **v2}
                    else:
                        merged[key][k2] = v2
            elif isinstance(value, list) and isinstance(merged[key], list):
                merged[key].extend(value)
            else:
                merged[key] = value
    return merged


def resolve_config_paths(config_path: Path) -> list[Path]:
    """Resolve a path to a list of JSON config files.

    Handles:
      - Single file → [file]
      - Directory → sorted list of *.json files in it
    """
    if config_path.is_dir():
        files = sorted(config_path.glob('*.json'))
        if not files:
            print(f"Warning: No .json files found in {config_path}")
        return files
    return [config_path]


def apply_config(config_path: Path, icons_path: Path = None, extra_configs: list[Path] = None):
    """Apply patches from one or more config files.

    Args:
        config_path: Path to a JSON file or directory of JSON files
        icons_path: Optional path to icons directory
        extra_configs: Additional config files to merge
    """
    from lps.constants import ICONS_PATH

    # Collect all config paths
    all_paths = resolve_config_paths(config_path)
    if extra_configs:
        for p in extra_configs:
            all_paths.extend(resolve_config_paths(p))

    if not all_paths:
        print("No config files found.")
        return

    # Load and merge all configs
    configs = []
    for p in all_paths:
        try:
            configs.append(load_config(p))
            print(f"  Loaded: {p}")
        except Exception as e:
            print(f"  Error loading {p}: {e}")

    if not configs:
        return

    custom = merge_configs(configs)

    if icons_path is None:
        config_icons = config_path.parent / 'icons'
        icons_path = config_icons if config_icons.exists() else ICONS_PATH

    # Ensure modules are loaded
    load_modules()

    # Dispatch to registered modules by config key
    for mod_name, mod_info in MODULES.items():
        config_key = mod_info.get('config_key')
        apply_fn = mod_info.get('apply_fn')
        if config_key and apply_fn and config_key in custom:
            apply_fn(custom[config_key], icons_path)

    print(f"\nApplied {len(all_paths)} config(s)")


def validate_config(config_path: Path) -> list[str]:
    """Validate a config file without applying it."""
    errors = []

    try:
        config = load_config(config_path)
    except json.JSONDecodeError as e:
        return [f"Invalid JSON: {e}"]

    custom = config.get('custom', {})

    # Validate items (if items module is loaded)
    load_modules()

    item_config = custom.get('item', {})
    items = item_config.get('items', {})

    if items:
        try:
            from modules.items import get_item_type
            for item_id_str, item_data in items.items():
                try:
                    item_id = int(item_id_str)
                except ValueError:
                    errors.append(f"Invalid item ID: {item_id_str}")
                    continue

                item_type = get_item_type(item_id)
                if not item_type:
                    errors.append(f"Item {item_id} has no valid type range")

                if 'template' in item_data:
                    template_id = item_data['template']
                    template_type = get_item_type(template_id)
                    if not template_type:
                        errors.append(f"Template {template_id} for item {item_id} has no valid type range")

                hq = item_data.get('hq', 0)
                if hq not in [0, 1, 2, 3]:
                    errors.append(f"Item {item_id} has invalid hq level: {hq}")

                tint = item_data.get('tint')
                if tint:
                    color = tint.lstrip('#')
                    if len(color) not in [6, 8]:
                        errors.append(f"Item {item_id} has invalid tint color: {tint}")
        except ImportError:
            pass

    # Validate strings
    strings_config = custom.get('strings', {})
    if strings_config:
        try:
            from modules.strings import DMSG_RESOURCES
            for resource_name in strings_config.keys():
                if resource_name not in DMSG_RESOURCES:
                    errors.append(f"Unknown string resource: {resource_name}")
        except ImportError:
            pass

    return errors


def create_example_config(output_path: Path):
    """Create an example config file."""
    example = {
        "custom": {
            "item": {
                "items": {
                    "22501": {
                        "name": "Iron Blade",
                        "singular": "iron blade",
                        "plural": "iron blades",
                        "description": "DMG:50 Delay:240 STR+2\\nAccuracy+5",
                        "template": 16535,
                        "fields": {"dmg": 50, "delay": 240}
                    }
                }
            },
            "strings": {
                "spell_names": {
                    "100": "Custom Fire"
                }
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(example, f, indent=2)

    print(f"Created example config: {output_path}")

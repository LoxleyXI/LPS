"""
LPS UI Icons Module: Patch individual icons in the character nameplate sprite sheet

The HD sheet (ustatshd, 256x256) contains both 16x16 and 32x32 icons.
Some icons (like the linkshell) have both sizes at different positions.
These are defined as "compound" slots and can be patched by name.

Supports patching by named slot, compound slot, or grid position.
Cells are cleared to transparent before pasting to avoid background bleed.
"""

from pathlib import Path

from modules import register_module
from lps.constants import ICONS_PATH
from .ustats import (
    patch_icon, patch_compound, resolve_slot, list_slots,
    read_sprite_sheet, SHEETS, SLOT_NAMES, COMPOUND_SLOTS,
)


def _resolve_icon_path(icon_ref):
    """Resolve an icon reference to an absolute path.

    Checks: absolute path, relative to cwd, then ICONS_PATH.
    """
    if icon_ref is None:
        return None
    p = Path(icon_ref)
    if p.is_absolute() and p.exists():
        return str(p)
    if p.exists():
        return str(p)
    candidate = ICONS_PATH / icon_ref
    if candidate.exists():
        return str(candidate)
    print(f"  Warning: icon not found: {icon_ref} (checked cwd and {ICONS_PATH})")
    return None


def apply_ui_icons_config(config: dict, icons_path=None):
    """Apply UI icon patches from a config dict."""
    entries = config if isinstance(config, list) else config.get('icons', config.get('ui_icons', []))

    for entry in entries:
        slot = entry.get('slot')
        if slot is None:
            print(f"  Warning: ui_icon entry missing 'slot', skipping")
            continue

        icon = _resolve_icon_path(entry.get('icon'))
        hd = _resolve_icon_path(entry.get('hd'))
        sd = _resolve_icon_path(entry.get('sd'))

        if not icon and not hd:
            print(f"  Warning: ui_icon entry missing 'icon' or 'hd'/'sd', skipping")
            continue

        name = entry.get('name', slot if isinstance(slot, str) else f'{slot}')

        # Compound slots patch both 16x16 and 32x32 in the HD sheet
        if isinstance(slot, str) and slot in COMPOUND_SLOTS:
            print(f"  Patching compound slot '{name}'...")
            result = patch_compound(slot, icon)
            if result:
                print(f"    → {result}")
            continue

        # Simple slot: resolve to grid position
        row, col = resolve_slot(slot)
        print(f"  Patching {name} at ({row},{col})...")
        results = patch_icon(row, col, icon, hd_path=hd, sd_path=sd)
        for r in results:
            print(f"    → {r}")


register_module('ui_icons',
    config_key='ui_icons',
    apply_fn=apply_ui_icons_config,
)

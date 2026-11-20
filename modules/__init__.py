"""
LPS Module System

Modules extend LPS with optional features. Each module registers itself
with CLI subcommands and/or config keys for JSON dispatch.

Where modules come from:
    modules/        shipped with LPS (items, strings, icons, particles, equipment, mobs, models,
                    and the zone builders: zonekit, zones, outdoors, caves, spawns, xidat)
    LPS_MODULES     more folders of modules, os.pathsep-separated: a server's own, kept in its
                    own repo (custom item ranges, its zone files), loaded by their own names
    modules.local/  a folder of your own beside modules/ (gitignored), loaded the same way

A module is a package (a folder with __init__.py) that calls register_module() to add CLI
commands or a config key; packages that don't are libraries the others import by name.

Hooks: an extension can also follow a step of LPS's own with register_hook(event, fn). The zone
builders call run_hooks("outdoor_built", ...) once a zone is generated, which is where a server
writes its own files from the layout (its NPCs, the way in): see modules/outdoors/build.py.

Usage:
    from modules import register_module, load_modules, MODULES

    # In a module's __init__.py:
    register_module('items',
        config_key='item',
        apply_fn=apply_items_config,
        cli_setup=setup_cli,
    )

    # In config.py / __main__.py:
    load_modules()  # discovers and imports all modules
"""

import importlib
import sys
from pathlib import Path
from typing import Callable

MODULES = {}

# Both module directories
MODULES_DIR = Path(__file__).parent
LOCAL_MODULES_DIR = MODULES_DIR.parent / 'modules.local'


def register_module(name: str, *,
                    config_key: str = None,
                    apply_fn: Callable = None,
                    cli_setup: Callable = None,
                    depends: list[str] = None):
    """Register an optional module.

    Args:
        name: Module name (must match directory name)
        config_key: Key in config JSON's "custom" dict to dispatch to
        apply_fn: Called with (config_dict, icons_path) when config key present
        cli_setup: Called with (subparsers) to add CLI subcommands
        depends: List of module names this module depends on
    """
    MODULES[name] = {
        'config_key': config_key,
        'apply_fn': apply_fn,
        'cli_setup': cli_setup,
        'depends': depends or [],
    }


def _discover_dir(modules_dir: Path, package: str):
    """Discover and import modules from a directory."""
    if not modules_dir.is_dir():
        return

    for child in sorted(modules_dir.iterdir()):
        if child.is_dir() and (child / '__init__.py').exists():
            name = child.name
            if name.startswith('_'):
                continue
            try:
                importlib.import_module(f'{package}.{name}')
            except Exception as e:
                print(f"Warning: Failed to load module '{name}': {e}")


def _discover_ext(folder: Path):
    """An extension folder: on the path, each of its packages imported by its own name."""
    if not folder.is_dir():
        print(f"Warning: LPS_MODULES folder not found: {folder}")
        return
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    for child in sorted(folder.iterdir()):
        if child.is_dir() and (child / '__init__.py').exists() and not child.name.startswith('_'):
            try:
                importlib.import_module(child.name)
            except Exception as e:
                print(f"Warning: Failed to load module '{child.name}' from {folder}: {e}")


def extension_dirs():
    """The extension folders: LPS_MODULES's, then modules.local/ if it's there."""
    import os
    dirs = [Path(d).expanduser() for d in os.environ.get('LPS_MODULES', '').split(os.pathsep) if d]
    if LOCAL_MODULES_DIR.is_dir():
        dirs.append(LOCAL_MODULES_DIR)
    return dirs


HOOKS = {}
_extensions_loaded = False


def register_hook(event: str, fn: Callable):
    """Call fn(**context) when LPS reaches `event` (see run_hooks' callers for each context)."""
    HOOKS.setdefault(event, []).append(fn)


def load_extensions():
    """Import the extension folders' packages only (once): enough for their hooks."""
    global _extensions_loaded
    if _extensions_loaded:
        return
    _extensions_loaded = True
    if str(MODULES_DIR) not in sys.path:
        sys.path.insert(1, str(MODULES_DIR))
    for folder in extension_dirs():
        _discover_ext(folder)


def run_hooks(event: str, **context):
    """Run every extension's hooks for `event`; returns how many ran."""
    load_extensions()
    for fn in HOOKS.get(event, []):
        fn(**context)
    return len(HOOKS.get(event, []))


def load_modules():
    """Discover and import modules: LPS's own, then each extension folder's."""
    # LPS's own packages import each other by name (zonekit, xidat, ...)
    if str(MODULES_DIR) not in sys.path:
        sys.path.insert(1, str(MODULES_DIR))
    _discover_dir(MODULES_DIR, 'modules')
    load_extensions()

# LPS - Loxley Patch System

A modular DAT patching system for FFXI. Generates client-side patches using XIPivot overlays without distributing any proprietary game data.

## Architecture

```
lps/          Core engine (encryption, icons, text encoding, FTABLE)
studio/       LPS Studio: web-based equipment editor with 3D preview
modules/      Feature modules
  items/      Item patching (stats, text, icons)
  equipment/  Gear model recolouring, and new gear models (inject)
  particles/  Particle effect editing
  strings/    DMSG string resource patching
  ui_icons/   Character nameplate icon patching
  mobs/       Monster models: recolour, scale, weapons, painted skins
  models/     Mob model variants (new file table entries, tinting)
  zones/      Zone recolouring (textures, lighting, fog)
  outdoors/   Generated outdoor zones, laid from a real zone's own land tiles
  caves/      Generated cave zones
  spawns/     A generated zone's monsters: placement, client models, name list, server SQL
  zonekit/    The zone DAT reader and writer, tiles, borrowed pieces, the client's limits
  xidat/      The client's file tables, file ids, zone names and texture sections
```

### Your own modules

A server keeps what is its own (custom item ranges, its zone files) in its own repo, as a folder
of modules, and points LPS at it:

```bash
LPS_MODULES=/path/to/my-server/lps python -m lps --help
```

Each package in an `LPS_MODULES` folder (several folders: separate them with `:`) is loaded by its
own name, and can register CLI commands and config keys with `register_module()` exactly as LPS's
own modules do. A `modules.local/` folder beside `modules/` (gitignored) is loaded the same way.

An extension can also follow one of LPS's own steps with `register_hook(event, fn)`. A zone build
calls the `outdoor_built` hooks once the zone is generated, with the zone file's entry, its
`outdoor` block, the layout's summary (basins, town spots and campfire, each land) and the zone
file, so a server can write its own files from the layout: its NPCs, where players arrive.

## Requirements

```bash
pip install Pillow numpy scipy      # numpy and scipy for the zone and monster builders
```

## Quick Start

```bash
# Set your FFXI installation path
export FFXI_DIR="/path/to/FINAL FANTASY XI"

# Apply a single config
python -m lps apply config.json

# Apply multiple configs (merged in order, later overrides earlier)
python -m lps apply items.json strings.json ui-icons.json

# Apply all configs from a directory
python -m lps apply configs/

# Validate without applying
python -m lps validate config.json
```

## Config Format

Configs use a JSON structure with module-specific sections under `custom`:

```json
{
  "custom": {
    "item": {
      "items": {
        "22501": {
          "name": "Iron Blade",
          "singular": "iron blade",
          "plural": "iron blades",
          "description": "DMG:50 Delay:240 STR+2\\nAccuracy+5",
          "template": 16535,
          "hq": 1,
          "fields": { "dmg": 50, "delay": 240 }
        }
      }
    },
    "strings": {
      "spell_names": { "100": "Custom Fire" },
      "zone_names": { "400": "Ice Cave" }
    },
    "ui_icons": [
      { "slot": "mentor", "icon": "icons/custom-badge.png" }
    ]
  }
}
```

### Item Templating

Items use a `template` field that references an existing item ID. LPS reads the full entry from the player's retail DAT (all binary fields, icon, unknown bytes) and only overrides what you specify. This means your configs stay minimal and you never need to define fields you don't care about:

```json
{
  "22501": {
    "name": "Emerald Blade",
    "template": 16535,
    "hue_shift": 120,
    "hq": 1,
    "fields": { "dmg": 55 }
  }
}
```

This creates a new item based on item 16535, keeping all its base properties (slots, jobs, races, skill type, icon structure), but with a new name, +1 HQ border, green-shifted icon, and 55 damage.

### Multi-Config Workflow

Split configs into logical files and apply them together:

```
configs/
├── items.json       # Custom items
├── equipment.json   # Model recoloring + particle effects
├── strings.json     # Text overrides
└── icons.json       # Nameplate icons
```

```bash
python -m lps apply configs/
```

Files are merged alphabetically. Later files override earlier entries for the same item ID.

## Module Commands

Item id ranges, text resources and nameplate icon slots are listed in each module's README
(`modules/items/`, `modules/strings/`, `modules/ui_icons/`).

```bash
# Equipment (examples/equipment.json)
python -m lps equipment config.json   # Recolour gear models, or add new ones (inject)

# Monsters and zones
python -m lps mobs mobs.json          # Recoloured, scaled or reskinned monster models
python -m lps zones zones.json        # Recoloured copies of zones, registered as new zones

# Starting out
python -m lps example config.json     # Write an example config

# Icons
python -m lps icon-encode icon.png --json              # Encode PNG to base64 JSON
python -m lps icon-encode icon.png --hq 3 --hue 120   # With HQ border + hue shift
python -m lps icon-encode icon.png --wrap 22501 --template 16535  # Full config

# Distribution
python -m lps manifest configs/ -o manifest.json   # Generate manifest for launchers
```

## Generated Zones

New zones built from the game's own pieces: outdoor land laid from a retail zone's tiles (any
zone with a kit file in `modules/outdoors/kits/`), caves grown from a seed, monsters spread the way
the original field zones spread theirs. A zone file says what to build; see the docstrings of
`modules/outdoors/__init__.py` and `modules/caves/__init__.py`.

```bash
export FFXI_DIR="/path/to/FINAL FANTASY XI"

# build a zone and install it into your XIPivot overlay (and write its monster SQL where it says)
python3 modules/outdoors/build.py examples/zones/sand_coast.json --install

# a generated cave
python3 modules/caves/build.py --config examples/zones/ice_cave.json --out /tmp/cave.DAT

# map previews of new layouts with their monsters, without building anything
python3 modules/outdoors/preview.py examples/zones/sand_coast.json previews/

# parchment maps of a zone's lands
python3 modules/outdoors/maps.py examples/zones/sand_coast.json maps/

# the workbench (pieces and rock in the browser) and the camp editor (/camp: lay a town's camp by hand)
python3 modules/outdoors/workbench/server.py --zones path/to/zone/files
```

A server's own zone files live in its own repo; `--zones` points the camp editor at them.

## LPS Studio

Web-based item editor with 3D model preview. Create items, recolor weapon models, edit particle effects, and export LPS configs, all from the browser.

```bash
# Create a symlink to your FFXI installation
ln -s "/path/to/FINAL FANTASY XI" studio/game

# Generate weapon index (reads the gear tables from your client)
LSB_SQL_DIR=/path/to/sql python3 studio/gen_weapons.py

# Generate item search index
LSB_SQL_DIR=/path/to/sql python3 studio/gen_items.py

# Serve locally
python3 -m http.server 8090 --directory studio
```

Features:
- Multi-item session management (load/save JSON, export SQL)
- Template loading from game DATs (no external requests)
- Item search across 22k+ items
- 3D weapon model viewer with orbit controls
- Live texture recoloring (hue shift, tint, saturation)
- Multi-range color targeting with visual feedback
- Texture overlay with blend modes
- Particle effect editor with per-emitter controls
- Custom particle texture upload
- Icon editing with HQ borders, hue shift, and tinting

## How It Works

LPS reads item/text/texture data from the user's own FFXI installation, applies modifications, and writes the result to an XIPivot overlay directory. The game client loads the overlay files instead of the originals.

DAT files use simple bit rotation encryption (ROL5/ROR5). Item entries are 3072 bytes each with DMSG text encoding and BMP2 paletted icons.

## Server Distribution

LPS is designed for private servers to distribute custom content without distributing proprietary game data. The server hosts JSON config files, and LPS runs locally on each player's machine, reading from their retail installation and writing to a separate overlay directory. The original game files are never modified.

```bash
# Generate a manifest for your launcher
python -m lps manifest configs/ -o manifest.json

# Host configs/ and manifest.json on your web server
# Your launcher downloads changed configs and runs: python -m lps apply configs/
```

See `docs/SERVER_GUIDE.md` for full launcher integration details with example code.

## Documentation

- `docs/SERVER_GUIDE.md` - Server operator and launcher integration guide
- `docs/TECHNICAL.md` - DAT format reference
- `examples/` - Example configs, and example zone files in `examples/zones/`
- Each module has its own `README.md`

## Legal

No proprietary data is included or distributed. Patches are generated from the user's own game files.

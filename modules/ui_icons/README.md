# UI Icons Module

Patches individual icons in the character nameplate sprite sheets (`ustatshd` / `ustats`). Used for GM icons, mentor badges, nation crests, and other nameplate markers.

## Features

- Patch by named slot (`mentor`, `gm`, `star`) or grid position (`2,4`)
- Automatic HD (32x32) and SD (16x16) patching from a single PNG
- Optional separate HD/SD icons for pixel-perfect control
- Extract existing icons for reference
- DXT3 compression (pure Python, no external tools)

## Sprite Sheets

| Version | DAT | Texture | Size | Grid | Cell |
|---------|-----|---------|------|------|------|
| HD | ROM/119/51.DAT | `ustatshd` | 256x256 | 8x8 | 32x32 |
| SD | ROM/280/15.DAT | `ustats` | 128x64 | 8x4 | 16x16 |

Icons are patched through a config (below), applied with `python -m lps apply`.

## Config Format

```json
{
  "custom": {
    "ui_icons": [
      { "slot": "mentor", "icon": "icons/my-badge.png" },
      { "slot": [2, 6], "icon": "icons/custom-badge.png" },
      { "slot": "gm", "hd": "icons/gm-32.png", "sd": "icons/gm-16.png" }
    ]
  }
}
```

## Known Slots

| Name | Position | Description |
|------|----------|-------------|
| star | (0, 3) | Star icon |
| crest_san | (0, 4) | San d'Oria crest |
| crest_bas | (0, 5) | Bastok crest |
| crest_win | (0, 6) | Windurst crest |
| gm | (2, 4) | Game Master icon |
| mentor | (3, 0) | Mentor badge |
| flag_y | (3, 4) | Yellow flag |
| flag_w | (3, 5) | White flag |
| linkshell | small (0, 16), large (0, 64) | Linkshell icon, both sizes (a compound slot: patched in the 16x16 and 32x32 grids at once) |

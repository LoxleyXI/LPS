# Zones Module

Creates new zones by cloning existing ones with hue-shifted, tinted, or recolored textures. Handles FTABLE10/VTABLE10 registration, DAT recoloring, and zone name patching.

## Features

- DXT texture recoloring (modify RGB565 endpoints in-place, no quality loss)
- Paletted texture recoloring (modify palette entries)
- Environment lighting adjustment (sun, moon, ambient, fog colors)
- Per-texture overrides with wildcard matching
- Zone name patching via DMSG
- ~1.5s per HD zone DAT

## Config Format (zones.json)

```json
{
  "zones": [
    {
      "name": "Green Fields",
      "clone": 115,
      "zone_id": 400,
      "hue": 120,
      "saturation": 10,
      "textures": {
        "suny*": { "hue": 240 },
        "environment": { "fog_tint": "#88ff88" }
      }
    }
  ]
}
```

## CLI

```bash
python -m lps zones zones.json                        # Inject zones
```

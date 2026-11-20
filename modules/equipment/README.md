# Equipment Module

Gear/weapon model recoloring and injection. Two modes:

- **Overlay** (default) — recolors an existing model DAT via XIPivot overlay. Affects ALL items sharing that model ID.
- **Inject** — creates a new model ID with recolored textures. Requires the gearpatch Ashita addon.

**Depends on:** zones, mobs

## Features

- Per-race gear model resolution from FFXiMain.dll tables
- Afterglow particle auras on weapons
- Particle effect swapping between weapons
- Per-emitter particle overrides
- Auto model ID assignment
- SQL generation for item_equipment updates

## Config Format (equipment.json)

```json
{
  "items": [
    {
      "name": "Emerald Ridill",
      "source_dat": "ROM/30/104.DAT",
      "color_ranges": [
        { "hue_min": 150, "hue_max": 340, "hue": 240 }
      ]
    },
    {
      "name": "Fire Ridill",
      "source_dat": "ROM/30/104.DAT",
      "inject": true,
      "slot": "main",
      "tint": "#ff0000",
      "blend_mode": "multiply"
    }
  ]
}
```

## CLI

```bash
python -m lps equipment equipment.json
```

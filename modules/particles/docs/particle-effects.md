# Particle Effects

LPS supports swapping, modifying, and combining particle effects on
weapon models. Effects can be previewed in LPS Studio before applying.

## Config format

Particle effect fields are added to entries in `items.json` alongside
existing recolor/afterglow fields:

```json
{
  "items": [
    {
      "name": "Blue Fire Ridill",
      "source_model_id": 259,
      "inject": true,
      "slot": "main",
      "hue": 200,
      "saturation": 30,
      "effects_source": {
        "dat": "ROM/352/107.DAT"
      },
      "particle_overrides": [
        { "id": "gp00", "overrides": { "hue": 200, "scale": 1.2 } },
        { "id": "fr00", "overrides": { "hue": 200 } }
      ]
    }
  ]
}
```

## Effects source

The `effects_source` field copies particle effects from a different
weapon DAT onto the target model. The target keeps its 3D mesh but
gets the source weapon's particle system.

```json
{
  "name": "Ridill with Ragnarok glow",
  "source_model_id": 259,
  "inject": true,
  "slot": "main",
  "effects_source": {
    "dat": "ROM/352/107.DAT"
  }
}
```

The `dat` path is relative to the FFXI game directory.

## Per-emitter overrides

The `particle_overrides` array modifies individual particle generators
within the weapon DAT. Each entry targets a generator by its section ID.

```json
"particle_overrides": [
  {
    "id": "gp00",
    "enabled": true,
    "overrides": {
      "hue": 180,
      "scale": 1.5,
      "speed": 2.0
    }
  },
  {
    "id": "fr00",
    "enabled": false
  }
]
```

### Available overrides

| Field | Type | Description |
|-------|------|-------------|
| `emissionRate` | float | Emission frequency multiplier (2.0 = twice as fast) |
| `lifetime` | float | Particle lifetime multiplier |
| `speed` | float | Velocity multiplier (affects velocity and variance) |
| `scale` | float | Particle size multiplier |
| `spread` | float | Position spread radius multiplier |
| `hue` | int | Hue shift in degrees (0-360) |
| `color` | [r,g,b] | RGB multipliers (e.g. [1.0, 0.5, 0.5] for reddish) |
| `alpha` | float | Opacity multiplier |

Set `"enabled": false` to completely disable a generator.

All values are multipliers (1.0 = no change) except `hue` which is
additive in degrees.

## Combining with other features

Particle effects combine with all existing LPS features. The pipeline
runs in this order:

1. **Recolor**: hue, saturation, lightness, tint, color_ranges
2. **Effects source** - swap particles from another weapon
3. **Afterglow**: wrap with afterglow particle aura
4. **Particle overrides**: patch individual emitters

Example combining everything:

```json
{
  "name": "Custom Elemental Ridill",
  "source_model_id": 259,
  "inject": true,
  "slot": "main",
  "hue": 200,
  "tint": "#3366ff88",
  "blend_mode": "multiply",
  "effects_source": {
    "dat": "ROM/352/107.DAT"
  },
  "afterglow": { "r": 0.2, "g": 0.5, "b": 1.0 },
  "particle_overrides": [
    { "id": "gp00", "overrides": { "hue": 200, "scale": 1.5 } },
    { "id": "fr00", "overrides": { "hue": 200, "speed": 0.8 } }
  ]
}
```

## Using the particle editor

LPS Studio (`studio/`) provides a web-based 3D preview for designing
particle configurations. It exports JSON configs directly.

1. Open LPS Studio in a browser
2. Select a weapon model and effects source
3. Adjust sliders (rate, lifetime, speed, scale, spread, hue, color)
4. Adjust model tinting (hue, saturation, lightness)
5. Click **Export Config**
6. Paste the exported JSON into your `items.json` items array

## Generator IDs

Each weapon DAT contains named particle generators. Common patterns:

| Pattern | Typical effect |
|---------|---------------|
| `gp00`, `gp01` | Glow particles |
| `fr00`, `fr01` | Fire/flame particles |
| `grw0`, `grw1` | Growth/sparkle particles |
| `dis0`, `dis1` | Distortion (heat haze) |
| `bld0`-`bld5` | Blade trail effects |
| `glw0`-`glwf` | Multi-layer glow |

Use the particle editor's Advanced panel to inspect which generators
a weapon has and what each one does.

## Custom particle textures

The particle editor supports replacing particle textures with custom
images. Click any emitter's texture thumbnail to upload a PNG/JPG.
Right-click to revert to the original.

Custom textures are included in the exported config as `custom_texture`
with base64-encoded PNG data. When LPS processes the config, it encodes
the image into the DAT's native texture format (DXT1, DXT3, or paletted)
and replaces the section in-place.

```json
"particle_overrides": [
  {
    "id": "gp00",
    "custom_texture": {
      "width": 64,
      "height": 64,
      "png_base64": "<base64 data>"
    }
  }
]
```

Images are automatically scaled to match the original texture dimensions.

Requires **Pillow** (`pip install Pillow`) for PNG → FFXI texture encoding.

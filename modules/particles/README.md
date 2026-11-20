# Particles Module

Modifies particle generator parameters in weapon/entity DATs. Supports per-emitter overrides for color, scale, speed, emission rate, and more.

## Features

- Per-emitter parameter overrides (color, scale, speed, emission rate, lifetime, spread, alpha)
- Hue shifting on particle colors
- Emitter enable/disable
- Custom texture replacement (PNG → palette/DXT)
- Effects block transplanting between DATs
- DXT1/DXT3 encoding

## Override Format

```json
[
  {
    "id": "gp00",
    "overrides": {
      "hue": 180,
      "scale": 1.5,
      "speed": 2.0,
      "color": [1.0, 0.5, 0.5],
      "alpha": 0.8
    }
  }
]
```

## Tools

- **LPS Studio** (`studio/`) - Web-based 3D editor with live particle preview
- `docs/particle-effects.md` -- Particle effect format documentation

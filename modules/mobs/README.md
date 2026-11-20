# Mobs Module

Creates new mob/NPC models by cloning existing ones with recolored textures. Registers new model IDs via FTABLE10/VTABLE10 and generates SQL for mob_pools/pet_list.

**Depends on:** zones

## Features

- Entity DAT texture recoloring (DXT + paletted)
- Color range targeting (hue/saturation/value filters)
- Uniform and per-joint scaling
- Weapon swapping (main hand + offhand)
- Afterglow particle auras
- Blur (0x5E) distortion effects
- Auto model ID assignment
- SQL generation for mob_pools and pet_list

## Config Format (mobs.json)

```json
{
  "mobs": [
    {
      "name": "Crimson Tiger",
      "clone": 308,
      "hue": 0,
      "saturation": 40,
      "tint": "#ff3300cc",
      "blend_mode": "multiply",
      "scale": 1.5
    }
  ]
}
```

## Painting our own textures

`skin.py` draws a UV template per texture (islands coloured by body part, from a parts.json of
joint groups) and puts a painting back in its place, encoded like the texture it replaces and
stored at half brightness (the client draws entity textures at 2x).

```bash
python3 modules/mobs/skin.py template 308 modules/mobs/skins/tiger/template \
    --parts modules/mobs/skins/tiger/parts.json
```

Paint on the template, never on the game's texture. Then in mobs.json:

```json
{ "name": "Painted Tiger", "clone": 308, "parts": "parts.json",
  "paint": { "tim     tiger": { "base": "paint/full.png", "head": "paint/head.png" } } }
```

`base` covers the whole sheet; each named part is laid over its own islands. A single path in
place of the dict paints the whole texture. Each build writes `check/<texture>.check.png` (the
result with the mesh edges over it) beside the config.

### Materials (no painting, nothing of the game's texture)

`"materials": {"tim     tiger": "rimefang.json"}` builds the texture from layers laid on the
body in 3D: tileable pictures by triplanar projection, flat colours, stripes across an axis,
light baked from the mesh's normals. Layers are limited by part, by facing (up/down) or both.
Small islands (eyes, fangs, claws) are named in parts.json by number from the template's
`.islands.png`. Each build also writes `check/<name>.preview.png`, a render of the result.
See `skins/tiger/rimefang.json`.

## CLI

```bash
python -m lps mobs mobs.json
```

# Caves

Generated cave zones for FFXI, built into LPS. A zone entry in a zones.json-style file with a
`generate` block gets its geometry replaced by a cave grown from a seed, textured and propped from
another zone's own assets, and optionally filled with monsters. Nothing from the game is shipped:
the build reads the player's own client files.

## Making a new zone

1. **Pick the source zone** whose look you want, and draft a kit from it:

       python3 modules/caves/build.py --scan-kit 111 --out /tmp/draft.json

   The draft measures how that zone lays its own textures (see "What the scanner measures") and
   guesses floor, wall and foot. Save it as `kits/<name>.json` and finish it by hand:
   - open each chosen texture and check it is one picture, not an atlas (see "Atlases" below);
   - fill `props` with groups from `props_found` (see "Props");
   - delete `measured` and `props_found` once you are done, or keep them as a record.

2. **Write the zone entry.** It says only what is its own:

   ```json
   "generate": {
     "kit": "beaucedine",
     "seed": 1336, "rooms": 6, "extent": 150, "voxel": 3.0, "noise": 4, "draw": 300,
     "water": {"features": ["lake", "river", "moat"]},
     "props": {"rock": [1, 2], "crystal": [2, 4]}
   }
   ```

   Any kit value can be overridden here: `"surfaces": {"floor": {"repeat": 30}}`. Leave `props` out
   for every kit group at its defaults; `[]` for none.

3. **Build to a scratch file, check it, then install** (never build into the game install):

       python3 modules/caves/build.py --config <zone>.json --base <cloned DAT> --out /tmp/zone.DAT
       (cd modules && python3 -c "from zonekit import writer as fz; print(fz.check_zone(open('/tmp/zone.DAT', 'rb').read())[0])")

   `--mobs` also writes the server SQL module and the oversized client models.

`examples/zones/ice_cave.json` + `kits/beaucedine.json` is the worked example.

## The settings

| Key | Whose | What |
|---|---|---|
| `seed`, `rooms`, `voxel`, `noise`, `depth` | zone | the cave's shape. Same seed, same cave. |
| `extent` | zone | keep every room within ±this many yalms. The base zone's collision grid limits it (Dragon's Aery: ±160, so 150). |
| `draw`, `fog` | zone | draw and fog distances written into the environment section. |
| `water.features` | zone | which pit kinds hold water (`lake`, `river`, `moat`, `rim`); the rest freeze over. |
| `surfaces.*` | kit | texture and repeat (yalms per repeat) per surface: `floor`, `wall`, `ceiling`, `shore`, `bed`, `foot`, `overlay`, `water`. |
| `props` groups | kit | models, where, how many per lobe, scale, solid, sink. |
| `mobs` | zone | types (pool to clone, model to clone + scale/recolour), spawns by room. See `server.py`, `spawns.py`. |

## What the generator decides (and why)

These are rules in `mesh.py` that hold for every cave. Each came from something that looked wrong
in game.

- **Floor or wall** is judged on the smoothed slope, not each triangle's: a stepped slope alternates
  steep and flat triangles, and judging each alone cut floor-textured teeth into the walls. A face
  counts as floor if it is nearly level, or gently sloped and within 1.5 yalms of the ground.
- **Ceilings** only above 12 yalms: low overhangs on a wall stay wall (dark ceiling diamonds otherwise).
- **Walls** take one of two flat projections by which way they face (X or Y). Two projections, two
  meshes, so neither smears.
- **The foot of every wall** (up to `foot.height`) uses the foot texture as a strip measured from the
  real ground under each spot: the lowest open air in that column, not the room's nominal floor
  (snow blocks crept up the walls otherwise). The strip is cut wherever a repeat ends and every other
  repeat is mirrored, so neighbouring triangles always agree and none reads past `u_max`.
- **Pillars, stalagmites and stalactites** wrap their texture around their own axis, a whole number
  of repeats per turn: a flat projection streaks down a round shaft and seams where it changes axis.
- **Under water**, beds and banks keep the floor texture: the water surface gives the effect, and a
  texture change there draws a hard line at the waterline. Only frozen pits get the `bed` texture.
- **The overlay** is a second, alpha-blended pass over the floors whose opacity follows large smooth
  noise, so a repeating floor texture stops reading as a grid.

## Atlases

Many FFXI textures are atlases: more than one picture in one image, and the zone's meshes use only
part of it. `gar_ykk1` (Beaucedine's cliff foot) has snow at both ends of v with rock between, and a
separate brighter strip in the last eighth of u. Its kit entry uses `"v": [0.5, 1.0]` (rock at 0.5,
snow at the ground at 1.0) and `"u_max": 0.86`. The scanner reports the range each texture's own
meshes use (`u`, `v`), which is the first hint; the image is the proof.

## Props

A kit's `props` are named groups:

```json
"rock": {"models": ["gar_iwa01_m"], "where": "edge", "per_lobe": [1, 2], "scale": [0.8, 1.2],
         "solid": true, "sink": 0.05}
```

- `where`: `floor` (anywhere level and dry in a lobe), `edge` (near the walls), `mouth` (at tunnel
  entrances; rarely right, most zones have no archway pieces).
- `solid`: its triangles join the collision.
- `sink`: bed it into the ground by this fraction of its height. Roots and logs are modelled to be half
  buried and float without it.
- Props never go on banks under the water line, on bridges, or against pillars, spikes or each other.
- Names starting `_` are alpha cutouts (plants). Models are anchored at their bottom centre: some sit
  far off their own origin.

`props_found` in a draft lists what the source zone places at least twice, most placed first, with
sizes in yalms. Invisible collision meshes (`hitwall_*`) and land pieces (over 30 yalms) are left out.

## What the scanner measures

For each texture on the source zone's land (not its props), from its placed meshes in world space:

- `facing`: share of its area on floors vs walls.
- `u`, `v`: the part of the texture its meshes use (1st to 99th percentile).
- `yalms_per_u`, `yalms_per_v`: how far one repeat stretches in the world.
- `above_ground`: how high its triangles sit over the lowest floor below them (10th and 90th percentile).
- `v_up_wall`, `v_at_ground`: on walls, whether v runs straight up, and which end is at the ground.
- `suggest`: floor, wall, foot (a wall texture whose v runs up from the ground itself) or mixed.

On Beaucedine the draft picks the same floor, wall and foot as were chosen by eye, with repeats within
a few yalms of the hand-tuned kit. It can't see atlases: check those.

## Outdoor zones: tile kits

FFXI's outdoor land is laid from 40x40-yalm tiles on a grid (centres at 20 mod 40), turned in
quarter turns and sometimes mirrored, with floors in fixed height steps (8 yalms in Valkurm and
Misareaux, 10 in Ronfaure). Each kit is a set of families, each a marching-squares set for one step:
flat, one corner up (inside corner), straight edge, three up (outside corner), sometimes diagonal,
plus junction pieces where a cliff meets a ramp or changes height along its run.

`--scan-kit` reads every tile's shape from its geometry, not its name: the top surface's height
sampled along each side. Two tiles fit side by side when their shared sides have the same profile;
`seams_matching` is the share of the real zone's shared sides that agree within a yalm (Valkurm 98%,
Misareaux 99%, East Ronfaure 96%), so the profiles describe how the kit fits together. `families`
lists each family's step, the cases it has and the ones it lacks: a level map built from a kit must
avoid its missing cases (most kits have no diagonal piece).

The sea is not a mesh: it is drawn by effect generators (`effect umi0-3` textures). Tiles carry
their own texture blends (grass over sand, sand over rock) as alpha layers.

Laying new outdoor zones from these kits is the `outdoors` module (`modules/outdoors/`).

## Client and server files

- Zone DAT: our own writer (`zonekit/writer.py`, shared with outdoor zones). Cipher keys come from the player's FFXiMain.dll at run time.
- Models: the file table's index is 7 bits, so a subdirectory holds slots below 128 (`zones._next_slot`).
- Static mob names come from the zone's name list DAT (`names.py`): the server doesn't send them.
- Mobs: `server.py` writes one SQL file (pools cloned with our model and a scaled hitbox, groups,
  spawn points); `mobmodels.py` builds the oversized / recoloured models into the overlay.

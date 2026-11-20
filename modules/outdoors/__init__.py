"""Outdoor zones laid from a real zone's own land tiles.

FFXI's outdoor land is 40x40-yalm tiles on a grid, in quarter turns and sometimes mirrored, with
floors in fixed height steps. This lays a new zone from a source zone's kit:

    kit.py      the source's tiles as poses (every turn and mirror, at every height it uses them),
                its footstep sounds, and its kit file (kits/<name>.json: levels, step, wall rise,
                families never used as filler, an optional coast, prop rules)
    layout.py   heights on the tile corners: basins, passes with slopes, walls, the coast's bays
    solve.py    a tile pose for every cell: corners must match, sides must agree (constraint
                propagation, then search)
    props.py    the source's scatter, by rules measured from its own placements
    effects.py  the source's ambient effects kept, its placed ones dropped, its sea moved into our bays
    place.py    bridges across its gaps; buildings: houses along roads, or towns on its terraces
                with streets, our own city walls (architecture.py, kits/architecture/) and gates
    assemble.py objects, collision (houses as boxes, fences round the map and its cliff tops),
                our own pieces laid as the zone's geometry, the sea
    zone.py     generate(): all of it into a zone DAT
    workbench/  our pieces in the browser beside any source model (python3 workbench/server.py)

Zone files (zones/<name>.json) are zones.json entries with an "outdoor" block:

    {"zones": [{"name": "Sand Coast", "clone": 103, "zone_id": 411,
                "outdoor": {"kit": "valkurm_dunes", "seed": 411, "size": [20, 16], "basins": 5,
                            "draw": 400, "overlay": "ffxi-hd"}}]}

    python3 modules/outdoors/build.py examples/zones/sand_coast.json --out /tmp/411.DAT [--install]

Nothing is shipped: the tiles, props and effects are the player's own copy of the source zone,
placed anew. The writer, tile reader, textures, borrowing and installing are zonekit's; every
build prints what it spends against the client's limits (zonekit/budget.py).
"""
from .zone import generate  # noqa: F401
from zonekit.install import install  # noqa: F401

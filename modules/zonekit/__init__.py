"""zonekit: what every generated zone shares.

    writer.py    our zone DAT writer (models, textures, ZoneDef: objects, tree, collision)
    tiles.py     reading a zone's models and land tiles (placed triangles, heights, side profiles)
    textures.py  DXT decode, encode and halving (capping a zone's textures)
    borrow.py    models (and their textures) from another zone
    install.py   installing a zone into a client overlay
    budget.py    what a zone spends against the client's limits

The client's limits: the ZoneDef section must fit in 8 MiB (found the hard way), and the collision
grid is kept within what retail zones use (up to 124,103 cells in use). See writer.py.
"""

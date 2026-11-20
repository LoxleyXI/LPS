"""xidat: the client's files, shared by every LPS module that builds them.

    paths       the game folder, Ashita's overlays, the HD copies (and turning those off)
    ftable      file ids to DATs: loading, resolving, registering, free ROM10 slots
    zone_files  a zone's four file ids, and the zone name list
    textures    texture sections in a DAT, and recolouring them in place

No game data and no generators here: just reading and registering files, so the zone, mob and
gear builders all place things the same way.
"""

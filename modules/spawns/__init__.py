"""spawns: a generated zone's monsters, for both generators (caves, outdoors).

    models  their client models: a retail model copied, recoloured or painted (mobs/skin.py) and
            scaled; installed into an overlay (build) or written out as files (export)
    names   the zone's client name list, which names static mobs (with_names, write)
    sql     the server side as one SQL module: mob pools, groups and spawn points

Where they stand is each generator's own (caves/spawns.py, outdoors/mobs.py).
"""

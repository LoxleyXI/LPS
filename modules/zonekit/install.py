"""Installing a zone DAT into a client overlay (e.g. Ashita's ffxi-hd)."""


def install(dat_bytes, zone_id, clone, name, ffxi, overlay):
    """The zone into an overlay (e.g. Ashita's ffxi-hd): our model DAT plus the clone's event, dialog
    and NPC DATs, each registered in the overlay's FTABLE10/VTABLE10 in a slot nothing else uses (a
    rebuild keeps its slots), and the zone's name in the overlay's zone name list."""
    import shutil
    from xidat.ftable import load_all, resolve, slot_for, register
    from xidat.paths import hd
    from xidat.zone_files import FIDS, write_names
    tables = load_all(ffxi)
    rom10 = overlay / "ROM10"
    ft10, vt10 = rom10 / "FTABLE10.DAT", rom10 / "VTABLE10.DAT"
    done = {}
    for kind, fn in FIDS.items():
        fid = fn(zone_id)
        sub, slot = slot_for(rom10, ffxi / "ROM10", fid)
        dst = rom10 / str(sub) / f"{slot}.DAT"
        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind == "model":
            dst.write_bytes(dat_bytes)
        else:
            src = resolve(fn(clone), tables)
            if src is None:
                continue
            shutil.copy2(hd(src, ffxi, overlay), dst)       # its HD copy, if the overlay has one
        register(ft10, vt10, fid, sub, slot)
        done[kind] = f"ROM10/{sub}/{slot}.DAT"
    write_names({zone_id: name}, ffxi, overlay)
    return done

"""A zone's client name list (the per-zone NPC DAT): what the client shows for static mobs, looked up
by entity id. Records are 32 bytes: a 28-byte name then the entity id (0x1000000 + (zone << 12) + n);
record 0 is "none". Static mobs get no name from the server (its mob update mask leaves the name out),
so a mob missing from this list shows as "NPC".
"""
import shutil
import struct
from pathlib import Path

RECORD = 32


def with_names(data, zone_id, names):
    """The name list `data` (a zone's NPC DAT) with {index: display name} set, grown to fit."""
    data = bytearray(data)
    need = (max(names) + 1) * RECORD
    if len(data) < need:
        data += bytes(need - len(data))
    base = 0x1000000 + (zone_id << 12)
    for i, name in names.items():
        raw = name.replace("_", " ").encode("ascii")[:27]
        rec = raw.ljust(28, b"\0") + struct.pack("<I", base + i)
        data[i * RECORD:(i + 1) * RECORD] = rec
    return bytes(data)


def write(overlay, zone_id, names, backup_dir):
    """names: {index: display name}. Rewrites the overlay's name list for the zone in place."""
    from xidat.zone_files import npc_fid
    rom10 = Path(overlay) / "ROM10"
    ft = (rom10 / "FTABLE10.DAT").read_bytes()
    fid = npc_fid(zone_id)
    v = struct.unpack_from("<H", ft, fid * 2)[0]
    path = rom10 / str(v >> 7) / f"{v & 0x7F}.DAT"
    backup = Path(backup_dir) / f"zone{zone_id}_names_original.DAT"
    if not backup.exists():
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
    path.write_bytes(with_names(backup.read_bytes(), zone_id, names))   # always from the original list
    return path

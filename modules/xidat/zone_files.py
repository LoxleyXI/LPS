"""A zone's files: its four file ids, and its name in the client's zone name list.

Every zone has four DATs, found by file id: the model (geometry, textures, collision), the event
scripts, the dialog lines and the NPC name list. Ids for zones under 0x100 are retail's own
formulas; ours (custom zones, 0x100 and up) continue from the model range.

The zone name list is ROM/165/84.DAT, a d_msg table: a 64-byte plain header (+0x14 file size,
+0x1C TOC size, +0x24 data size, +0x28 entry count), then the TOC (u32 offset, u32 size per entry)
and the entries, XOR 0xFF.
"""
import struct
from pathlib import Path

from .paths import ffxi_dir, overlay


def model_fid(zone_id: int) -> int:
    return 0x64 + zone_id if zone_id < 0x100 else 0x147B3 + (zone_id - 0x100)


def event_fid(zone_id: int) -> int:
    return (5820 + zone_id) if zone_id < 0x100 else model_fid(zone_id) + 1100


def dialog_fid(zone_id: int) -> int:
    return (6420 + zone_id) if zone_id < 0x100 else model_fid(zone_id) + 1700


def npc_fid(zone_id: int) -> int:
    return (6720 + zone_id) if zone_id < 0x100 else model_fid(zone_id) + 2600


FIDS = {
    'model':  model_fid,
    'event':  event_fid,
    'dialog': dialog_fid,
    'npc':    npc_fid,
}


def name_entry(name: str) -> bytes:
    """One d_msg zone name entry."""
    name_bytes = name.encode('ascii') + b'\x00'
    while len(name_bytes) % 4 != 0:
        name_bytes += b'\x00'
    # header: u32 (1 element), u32 (offset 12), u32 (type 0, a string)
    entry = bytearray(struct.pack('<III', 1, 12, 0))
    # 28-byte string header (0x01 at byte 0, the rest zeros)
    entry += bytearray(28)
    entry[12] = 1
    entry += name_bytes
    return bytes(entry)


def with_names(dat: bytes, zone_names: dict) -> bytes:
    """The zone name list `dat` with {zone id: name} set, grown to hold the highest id."""
    header = bytearray(dat[:64])
    body = bytearray(b ^ 0xFF for b in dat[64:])

    old_count = struct.unpack_from('<I', header, 0x28)[0]
    old_toc_size = struct.unpack_from('<I', header, 0x1C)[0]

    entries = []
    for i in range(old_count):
        off = struct.unpack_from('<I', body, i * 8)[0]
        sz = struct.unpack_from('<I', body, i * 8 + 4)[0]
        entries.append(bytes(body[old_toc_size + off:old_toc_size + off + sz]))

    empty = name_entry('')
    while len(entries) <= max(zone_names):
        entries.append(empty)
    for zid, name in zone_names.items():
        entries[zid] = name_entry(name)

    new_toc = bytearray()
    new_data = bytearray()
    for entry in entries:
        new_toc += struct.pack('<II', len(new_data), len(entry))
        new_data += entry

    new_header = bytearray(header)
    struct.pack_into('<I', new_header, 0x14, 64 + len(new_toc) + len(new_data))
    struct.pack_into('<I', new_header, 0x1C, len(new_toc))
    struct.pack_into('<I', new_header, 0x24, len(new_data))
    struct.pack_into('<I', new_header, 0x28, len(entries))
    return bytes(new_header) + bytes(b ^ 0xFF for b in (new_toc + new_data))


def write_names(zone_names: dict, ffxi: Path = None, overlay_dir: Path = None):
    """Set zone names in an overlay's zone name list (ffxi-hd's by default), building on the names
    already there, else on the game's own list."""
    ffxi = Path(ffxi or ffxi_dir())
    out_dir = Path(overlay_dir or overlay(ffxi=ffxi))
    dat_path = ffxi / 'ROM/165/84.DAT'
    if (out_dir / 'ROM/165/84.DAT').exists():
        dat_path = out_dir / 'ROM/165/84.DAT'
    if not dat_path.exists():
        print('  warning: ROM/165/84.DAT not found, skipping zone names')
        return
    out_path = out_dir / 'ROM/165/84.DAT'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = with_names(dat_path.read_bytes(), zone_names)
    out_path.write_bytes(data)
    n = struct.unpack_from('<I', data, 0x28)[0]
    print(f'\n  Patched zone names → {out_path.name} ({n} entries)')

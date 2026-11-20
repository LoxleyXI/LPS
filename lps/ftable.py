"""
LPS FTABLE - FTABLE/VTABLE read/write utilities

FTABLE.DAT maps file IDs to ROM paths (2 bytes per entry).
VTABLE.DAT tracks which ROM directory each file ID belongs to (1 byte per entry).

Entry format:
    FTABLE: u16 = (subdir << 7) | (file_index & 0x7F)
    VTABLE: u8  = ROM index (1 = ROM/, 2 = ROM2/, ..., 10 = ROM10/)
"""

import struct
from pathlib import Path


def patch_ftable(ft_path: Path, vt_path: Path,
                 file_id: int, subdir: int, file_index: int,
                 rom_index: int = 10):
    """Write a file_id entry into an FTABLE/VTABLE pair.

    Args:
        ft_path: Path to FTABLE DAT
        vt_path: Path to VTABLE DAT
        file_id: The file ID to register
        subdir: ROM subdirectory number
        file_index: File number within subdirectory (0-127)
        rom_index: ROM directory index (default 10 = ROM10/)
    """
    fdata = bytearray(ft_path.read_bytes())
    vdata = bytearray(vt_path.read_bytes())
    struct.pack_into('<H', fdata, file_id * 2, (subdir << 7) | (file_index & 0x7F))
    vdata[file_id] = rom_index
    ft_path.write_bytes(bytes(fdata))
    vt_path.write_bytes(bytes(vdata))


def read_ftable_entry(ft_path: Path, vt_path: Path, file_id: int) -> tuple[int, int, int] | None:
    """Read a single FTABLE/VTABLE entry.

    Returns:
        (subdir, file_index, rom_index) or None if unregistered
    """
    fdata = ft_path.read_bytes()
    vdata = vt_path.read_bytes()

    if file_id * 2 + 2 > len(fdata) or file_id >= len(vdata):
        return None

    rom_index = vdata[file_id]
    if rom_index == 0:
        return None

    fval = struct.unpack_from('<H', fdata, file_id * 2)[0]
    subdir = fval >> 7
    file_index = fval & 0x7F

    return subdir, file_index, rom_index


def resolve_ftable_to_path(ffxi_path: Path, ft_path: Path, vt_path: Path,
                           file_id: int) -> Path | None:
    """Resolve a file_id to its full DAT path.

    Returns:
        Path to the DAT file, or None if unregistered
    """
    entry = read_ftable_entry(ft_path, vt_path, file_id)
    if entry is None:
        return None

    subdir, file_index, rom_index = entry
    rom = 'ROM' if rom_index == 1 else f'ROM{rom_index}'
    return ffxi_path / f'{rom}/{subdir}/{file_index}.DAT'

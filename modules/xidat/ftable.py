"""The client's file tables: which DAT a file id is.

FTABLE.DAT / VTABLE.DAT and each ROMn's pair: VTABLE says which ROM folder owns an id (0 = none),
FTABLE where in it ((dir << 7) | file, so at most 128 files a folder). A higher ROM number's entry
wins. Our files go in ROM10, an overlay's own tables registering them.

    tables = load_all(ffxi)
    path = resolve(fid, tables, ffxi)
    sub, slot = slot_for(overlay / "ROM10", ffxi / "ROM10", fid)
    register(ft10, vt10, fid, sub, slot)
"""
import struct
from pathlib import Path
from typing import Optional, Tuple

from .paths import ffxi_dir


def load_all(ffxi: Path) -> dict:
    """Load all FTABLE/VTABLE pairs into {rom_idx: (fdata, vdata)}."""
    tables = {}
    for idx in range(1, 11):
        if idx == 1:
            ft = ffxi / 'FTABLE.DAT'
            vt = ffxi / 'VTABLE.DAT'
        else:
            ft = ffxi / f'ROM{idx}/FTABLE{idx}.DAT'
            vt = ffxi / f'ROM{idx}/VTABLE{idx}.DAT'
        if ft.exists() and vt.exists():
            tables[idx] = (bytearray(ft.read_bytes()), bytearray(vt.read_bytes()))
    return tables


def resolve(file_id: int, tables: dict, ffxi: Path = None) -> Optional[Path]:
    """Resolve a file_id to a DAT path, or None if unregistered."""
    for idx in sorted(tables):
        fdata, vdata = tables[idx]
        if file_id * 2 + 2 > len(fdata) or file_id >= len(vdata):
            continue
        vval = vdata[file_id]
        if vval == 0:
            continue
        fval = struct.unpack_from('<H', fdata, file_id * 2)[0]
        subdir = fval >> 7
        fidx = fval & 0x7F
        rom = 'ROM' if vval == 1 else f'ROM{vval}'
        return Path(ffxi or ffxi_dir()) / f'{rom}/{subdir}/{fidx}.DAT'
    return None


def ensure_rom10(ffxi: Path, entity_models: bool = False) -> Tuple[Path, Path]:
    """Create ROM10/FTABLE10.DAT and VTABLE10.DAT if they don't exist.

    If entity_models=True, ensures tables are large enough for custom
    model IDs (15000+, file_id 113239+, requires 118240 entries).
    Zone injection only needs the base size (109701 entries).
    """
    rom10 = ffxi / 'ROM10'
    ft10 = rom10 / 'FTABLE10.DAT'
    vt10 = rom10 / 'VTABLE10.DAT'

    base_ft_size = (ffxi / 'FTABLE.DAT').stat().st_size
    base_vt_size = (ffxi / 'VTABLE.DAT').stat().st_size

    # Entity models need 118240 entries minimum
    if entity_models:
        min_ft = 118240 * 2  # uint16 per entry
        min_vt = 118240      # uint8 per entry
        ft_size = max(base_ft_size, min_ft)
        vt_size = max(base_vt_size, min_vt)
    else:
        ft_size = base_ft_size
        vt_size = base_vt_size

    rom10.mkdir(parents=True, exist_ok=True)
    if not ft10.exists():
        ft10.write_bytes(b'\x00' * ft_size)
    elif ft10.stat().st_size < ft_size:
        # Expand existing table
        data = bytearray(ft10.read_bytes())
        data += b'\x00' * (ft_size - len(data))
        ft10.write_bytes(bytes(data))
    if not vt10.exists():
        vt10.write_bytes(b'\x00' * vt_size)
    elif vt10.stat().st_size < vt_size:
        data = bytearray(vt10.read_bytes())
        data += b'\x00' * (vt_size - len(data))
        vt10.write_bytes(bytes(data))

    return ft10, vt10


def register(ft_path: Path, vt_path: Path,
                  file_id: int, subdir: int, fidx: int):
    """Register a file_id in FTABLE10/VTABLE10."""
    fdata = bytearray(ft_path.read_bytes())
    vdata = bytearray(vt_path.read_bytes())
    struct.pack_into('<H', fdata, file_id * 2, (subdir << 7) | fidx)
    vdata[file_id] = 10
    ft_path.write_bytes(bytes(fdata))
    vt_path.write_bytes(bytes(vdata))


def next_slot(rom10: Path, subdir: int = 1) -> tuple[int, int]:
    """Find next free (subdir, file_index) in ROM10.

    FTABLE encodes file_index as 7 bits (0-127 per subdir), so this
    auto-bumps to the next subdir when a directory fills up.
    Returns (subdir, slot) with slot always < 128.
    """
    while True:
        d = rom10 / str(subdir)
        d.mkdir(parents=True, exist_ok=True)
        used = {int(f.stem) for f in d.glob('*.DAT') if f.stem.isdigit()}
        for slot in range(128):
            if slot not in used:
                return subdir, slot
        subdir += 1


def free_slot(overlay_rom10: Path, game_rom10: Path, skip_fid: int = -1) -> tuple[int, int]:
    """A (subdir, slot) that is safe to put a new file in, for an overlay's ROM10.

    next_slot only looks for a missing file, but an overlay's FTABLE10 also points ids at the
    game's own ROM10 files, and a file in the overlay shadows the game's file of the same number
    for every id that uses it. So: no file in the overlay or the game's ROM10 at that number, and
    no id (other than skip_fid) pointing at it."""
    fdata = (overlay_rom10 / 'FTABLE10.DAT').read_bytes()
    vdata = (overlay_rom10 / 'VTABLE10.DAT').read_bytes()
    taken = set()
    for f in range(len(vdata)):
        if vdata[f] == 10 and f != skip_fid:
            v = struct.unpack_from('<H', fdata, f * 2)[0]
            taken.add((v >> 7, v & 0x7F))
    for sub in range(1, 64):
        for slot in range(128):
            if (sub, slot) in taken:
                continue
            if (overlay_rom10 / str(sub) / f'{slot}.DAT').exists() or (game_rom10 / str(sub) / f'{slot}.DAT').exists():
                continue
            return sub, slot
    raise RuntimeError('no free ROM10 slot')


def slot_for(overlay_rom10: Path, game_rom10: Path, fid: int) -> tuple[int, int]:
    """Where to (re)write file id fid in an overlay: its current slot if nothing else points
    there, else a safe free one."""
    fdata = (overlay_rom10 / 'FTABLE10.DAT').read_bytes()
    vdata = (overlay_rom10 / 'VTABLE10.DAT').read_bytes()
    if fid < len(vdata) and vdata[fid]:
        v = struct.unpack_from('<H', fdata, fid * 2)[0]
        cur = (v >> 7, v & 0x7F)
        shared = any(vdata[f] == 10 and f != fid and struct.unpack_from('<H', fdata, f * 2)[0] == v
                     for f in range(len(vdata)))
        if not shared:
            return cur
    return free_slot(overlay_rom10, game_rom10, fid)


# ---------------------------------------------------------------------------
# Entity models (mobs, NPCs): model id -> file id
# ---------------------------------------------------------------------------

MODEL_RANGES = [
    (0,    1499,  1300),
    (1500, 2999, 50295),
    (3000, 3499, 96907),
    (3500, None, 98239),
]


def model_fid(model_id: int) -> int:
    """The file id of an entity model (the model id a mob pool or NPC look names)."""
    for start, end, offset in MODEL_RANGES:
        if end is None or start <= model_id <= end:
            return model_id + offset
    raise ValueError(f'modelid {model_id} out of range')


def resolve_model(model_id: int, tables: dict, ffxi: Path = None) -> Optional[Path]:
    """An entity model's DAT, or None if unregistered."""
    return resolve(model_fid(model_id), tables, ffxi)

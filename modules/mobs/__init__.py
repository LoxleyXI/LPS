"""
LPS Mobs - Entity model recoloring and injection

Creates new mob/NPC models by cloning existing ones with hue-shifted,
tinted, or otherwise recolored textures. Registers new model IDs via
FTABLE10/VTABLE10 and outputs SQL for mob_pools/pet_list.

Entity DATs use the same 0x20 DXT/paletted texture sections as zone DATs,
so the same recolor functions apply directly.

Configured via mobs.json.
"""

import json
import shutil
import struct
from pathlib import Path

from xidat.paths import ffxi_dir, hd, overlay
from xidat.textures import (
    parse_sections, parse_texture_info, _adjust_dxt_inplace, _adjust_palette_inplace,
)
from xidat.ftable import (
    load_all, resolve_model, model_fid, ensure_rom10, next_slot, register, slot_for,
)
try:                                       # the LPS command line's plugin list, when LPS is here
    from modules import register_module
except ImportError:                        # used without the LPS command line: no plugins to register
    def register_module(*args, **kwargs):
        pass
from PIL import Image


# ---------------------------------------------------------------------------
# Custom model ids
# ---------------------------------------------------------------------------

MODEL_CUSTOM_START = 15000
MODEL_CUSTOM_END = 20000


def modelid_blob(modelid: int) -> str:
    """Build the 20-byte mob_pools.modelid blob as hex string."""
    raw = struct.pack('<HH', 0, modelid) + bytes(16)
    return '0x' + raw.hex().upper()


def _next_free_modelid(tables: dict) -> int:
    """Find the first unused custom model ID."""
    for mid in range(MODEL_CUSTOM_START, MODEL_CUSTOM_END):
        fid = model_fid(mid)
        found = False
        for idx in sorted(tables):
            fdata, vdata = tables[idx]
            if fid < len(vdata) and vdata[fid] != 0:
                found = True
                break
        if not found:
            return mid
    raise RuntimeError(f'No free model IDs in range {MODEL_CUSTOM_START}-{MODEL_CUSTOM_END}')


# ---------------------------------------------------------------------------
# Afterglow — relic weapon particle aura (portable to any weapon)
# ---------------------------------------------------------------------------

# The afterglow aura block is extracted from an AG weapon DAT (Ragnarok AG: AFTERGLOW_TEMPLATE).
# Structure: container_hdr(32) + aura_front(12688) + [weapon DAT] + aura_tail(336)
# The aura contains ParticleGenerators, KeyFrames, ParticleMeshes, and Textures.
# Color is controlled by tinting the particle palette and float multipliers.

_AFTERGLOW_CACHE = {}

# The template: Ragnarok's afterglow weapon (gear model 546), read from the player's own client.
# Every offset here and in _build_afterglow_dat (the cut, its palettes, its particle generators)
# was measured from that DAT, so it is the one template this works with.
AFTERGLOW_TEMPLATE = 546
AFTERGLOW_SIZE = 26320              # its file: container header, aura, the weapon, the aura's tail


def _load_afterglow_template(ffxi_dir: Path):
    """The afterglow aura blocks, from the template weapon's DAT (found through the gear tables)."""
    if 'data' in _AFTERGLOW_CACHE:
        return _AFTERGLOW_CACHE['data']
    from equipment import resolve_gear_dat
    ag_path = resolve_gear_dat('HumeMale', 'main', AFTERGLOW_TEMPLATE, ffxi_dir)
    if not ag_path or not ag_path.exists():
        raise FileNotFoundError(f'afterglow: gear model {AFTERGLOW_TEMPLATE} not found in this client')
    ag_data = ag_path.read_bytes()
    if len(ag_data) != AFTERGLOW_SIZE or ag_data[32:36] != b'aura' or ag_data[25984:25988] != b'aura':
        raise ValueError(f'afterglow: {ag_path} is not the afterglow DAT these offsets were measured '
                         f'from ({len(ag_data)} bytes, {AFTERGLOW_SIZE} expected)')
    result = {
        'container_hdr': ag_data[0:32],
        'aura_front': ag_data[32:12720],
        'aura_tail': ag_data[25984:],
    }
    _AFTERGLOW_CACHE['data'] = result
    return result


def _build_afterglow_dat(weapon_dat: bytes, afterglow_opts: dict, ffxi_dir: Path) -> bytes:
    """Wrap a weapon DAT with the afterglow particle aura.

    afterglow_opts keys:
        r, g, b: float 0.0-1.0, color multipliers (default 1.0 = white)
    """
    template = _load_afterglow_template(ffxi_dir)
    container_hdr = template['container_hdr']
    aura_front = bytearray(template['aura_front'])
    aura_tail = template['aura_tail']

    r = afterglow_opts.get('r', 1.0)
    g = afterglow_opts.get('g', 1.0)
    b = afterglow_opts.get('b', 1.0)

    # Tint particle palettes (pi1 and flmy textures within aura_front)
    pi1_palette = (2304 - 32) + 16 + 57
    flmy_palette = (7504 - 32) + 16 + 57
    for pal_off in [pi1_palette, flmy_palette]:
        for i in range(256):
            po = pal_off + i * 4
            if po + 4 > len(aura_front):
                break
            aura_front[po]   = min(255, int(aura_front[po]   * b))
            aura_front[po+1] = min(255, int(aura_front[po+1] * g))
            aura_front[po+2] = min(255, int(aura_front[po+2] * r))

    # Tint particle generator float multipliers (RGB at offsets 48 and 64 in each generator)
    for gen_data_start in [32 + 16, (608 - 32) + 16]:
        for float_base in [48, 64]:
            off = gen_data_start + float_base
            if off + 12 <= len(aura_front):
                struct.pack_into('<fff', aura_front, off, r, g, b)

    return container_hdr + bytes(aura_front) + weapon_dat + aura_tail


# ---------------------------------------------------------------------------
# Blur (0x5E) section — avatar-style distortion aura
# ---------------------------------------------------------------------------

def _build_blur_section(r: int, g: int, b: int,
                        alpha: int = 20, radius: float = 120.0,
                        layers: int = 11, frequency: int = 8,
                        falloff: int = 0x0190, ring_count: int = 8) -> bytes:
    """Build a 0x5E blur section with the given color and parameters.

    Parameters:
        r, g, b: Ring color (0-255).
        alpha: Distortion strength per ring (higher = more blur). 5-50 useful.
        radius: Outer blur radius (larger = softer transitions). 60-180 useful.
        layers: Number of distortion layers (more = blurrier). Avatars use 3-11.
        frequency: Oscillation speed (higher = faster shimmer). Avatars use 4-8.
        falloff: Edge blur falloff (higher = subtler edges). Avatars use 200-413.
        ring_count: Number of gradient rings (default 8).
    """
    inner_radius = radius * 0.4

    body = bytearray()
    body += struct.pack('<BBH', layers, frequency, falloff)
    body += struct.pack('BBBB', 128, 128, 128, 0)   # base color 1
    body += struct.pack('BBBB', 128, 128, 128, 0)   # base color 2
    body += struct.pack('<f', inner_radius)

    for i in range(ring_count):
        t = i / max(1, ring_count - 1)
        a = max(4, int(alpha * (1 - t * 0.5)))
        rad = inner_radius + (radius - inner_radius) * t
        body += struct.pack('BBBB', r, g, b, a)
        body += struct.pack('<f', rad)

    # Final ring at radius 0
    body += struct.pack('BBBB', r, g, b, max(1, alpha // 2))
    body += struct.pack('<f', 0.0)
    # Tail marker
    body += struct.pack('BBBB', 0, 0, 128, 63)
    body += struct.pack('<f', 0.0)

    while len(body) % 16 != 0:
        body += b'\x00'

    padded_size = len(body) + 16
    meta = ((padded_size // 16) << 7) | 0x5E
    header = b'blur' + struct.pack('<I', meta) + b'\x00' * 8
    return header + bytes(body)


def _inject_blur_section(data: bytearray, blur_bytes: bytes) -> bytearray:
    """Insert a 0x5E section into an entity DAT before its final END section."""
    sections = parse_sections(bytes(data))
    # Find the last END (type 0x00) section
    last_end = None
    for s in sections:
        if s.type_code == 0x00:
            last_end = s
    if last_end is None:
        raise ValueError('No END section found in DAT')
    pos = last_end.start
    return bytearray(bytes(data[:pos]) + blur_bytes + bytes(data[pos:]))


# ---------------------------------------------------------------------------
# Entity DAT recoloring
# ---------------------------------------------------------------------------

def recolor_entity_dat(source: Path, output: Path,
                       hue: float = 0, saturation: float = 0, lightness: float = 0,
                       tint: str = None, blend_mode: str = 'normal',
                       color_ranges: list = None,
                       scale: float = 0,
                       joint_scales: dict = None,
                       weapon_source: Path = None,
                       weapon_recolor: dict = None,
                       offhand_source: Path = None,
                       offhand_recolor: dict = None,
                       blur: dict = None) -> dict:
    """Recolor and/or scale all textures in an entity DAT.

    color_ranges: List of dicts, each with hue_min/hue_max to target specific
    color ranges. Each range can have its own hue/saturation/lightness/tint.
    If provided, the top-level adjustments are ignored in favour of ranges.

    scale: Uniform scale factor (e.g. 2.0 = double size, 0.5 = half).
    Scales skeleton bone translations, mesh vertex positions, and animation
    translation keyframes. Best results with grounded mobs (tigers, beetles,
    wyrms). Floating mobs (bombs, snolls) will float proportionally higher.
    0 or 1.0 = no scaling.
    """
    raw = source.read_bytes()
    if weapon_source:
        weapon_dat = weapon_source.read_bytes()
        if weapon_recolor:
            # Recolor the weapon DAT before swapping
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.DAT', delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _recolor_weapon_dat(weapon_source, tmp_path, weapon_recolor)
                weapon_dat = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)
        raw = swap_weapon(raw, weapon_dat)
    if offhand_source:
        offhand_dat = offhand_source.read_bytes()
        if offhand_recolor:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.DAT', delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _recolor_weapon_dat(offhand_source, tmp_path, offhand_recolor)
                offhand_dat = tmp_path.read_bytes()
            finally:
                tmp_path.unlink(missing_ok=True)
        raw = swap_weapon(raw, offhand_dat, block_name='rng_')
    data = bytearray(raw)
    sections = parse_sections(bytes(data))

    stats = {'dxt': 0, 'paletted': 0, 'total': 0,
             'joints_scaled': 0, 'vertices_scaled': 0, 'anims_scaled': 0}

    # Build list of adjustment passes
    if color_ranges:
        passes = []
        for cr in color_ranges:
            passes.append(dict(
                hue=cr.get('hue', 0),
                saturation=cr.get('saturation', 0),
                lightness=cr.get('lightness', 0),
                tint=cr.get('tint'),
                blend_mode=cr.get('blend_mode', 'normal'),
                hue_min=cr.get('hue_min'),
                hue_max=cr.get('hue_max'),
                val_min=cr.get('val_min'),
                val_max=cr.get('val_max'),
                sat_min=cr.get('sat_min'),
                sat_max=cr.get('sat_max'),
            ))
    else:
        passes = [dict(hue=hue, saturation=saturation, lightness=lightness,
                       tint=tint, blend_mode=blend_mode)]

    do_scale = scale and scale != 1.0

    for s in sections:
        if s.type_code == 0x20:
            info = parse_texture_info(bytes(data), s)
            if info is None:
                continue
            stats['total'] += 1
            for kw in passes:
                if info.tex_type == 0xA1 and info.dxt_type in ('1TXD', '3TXD'):
                    _adjust_dxt_inplace(data, info, **kw)
                elif info.tex_type in (0x91, 0xB1):
                    _adjust_palette_inplace(data, info, **kw)
            if info.tex_type == 0xA1:
                stats['dxt'] += 1
            else:
                stats['paletted'] += 1

        elif s.type_code == 0x29 and do_scale:
            stats['joints_scaled'] += _scale_skeleton_inplace(data, s, scale)

        elif s.type_code == 0x2A and do_scale:
            stats['vertices_scaled'] += _scale_mesh_inplace(data, s, scale)

        elif s.type_code == 0x2B and do_scale:
            stats['anims_scaled'] += _scale_animation_inplace(data, s, scale)

    # Per-joint scaling (body part scaling)
    if joint_scales:
        js = _scale_joints_inplace(data, sections, joint_scales)
        stats['part_joints'] = js['joints']
        stats['part_vertices'] = js['vertices']

    # Blur (0x5E) — avatar-style distortion aura
    if blur:
        blur_section = _build_blur_section(
            r=blur.get('r', 200), g=blur.get('g', 200), b=blur.get('b', 255),
            alpha=blur.get('alpha', 20),
            radius=blur.get('radius', 120.0),
            layers=blur.get('layers', 11),
            frequency=blur.get('frequency', 8),
            falloff=blur.get('falloff', 0x0190),
        )
        data = _inject_blur_section(data, blur_section)
        stats['blur'] = True

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(data))
    return stats


# ---------------------------------------------------------------------------
# Weapon swapping (splice weapon block from one DAT into another)
# ---------------------------------------------------------------------------

def _parse_blocks(data: bytes) -> list:
    """Parse a mob DAT into sub-model blocks delimited by HDR (0x01) sections."""
    sections = []
    pos = 0
    while pos < len(data) - 16:
        name = data[pos:pos + 4].decode('ascii', errors='replace')
        meta = struct.unpack_from('<I', data, pos + 4)[0]
        tc = meta & 0x7F
        sz = ((meta >> 7) & 0xFFFFF) * 0x10
        if sz <= 0:
            break
        sections.append({'name': name.strip(), 'type': tc, 'pos': pos, 'size': sz})
        pos += sz

    blocks = []
    current = None
    for s in sections:
        if s['type'] == 0x01:  # HDR
            if current:
                current['end'] = s['pos']
                blocks.append(current)
            current = {'name': s['name'], 'start': s['pos'], 'sections': [s]}
        elif current:
            current['sections'].append(s)
    if current:
        last = current['sections'][-1]
        current['end'] = last['pos'] + last['size']
        blocks.append(current)
    return blocks


def _find_weapon_block(blocks: list) -> dict | None:
    """Find the weapon block: has mesh sections, not 'mdl_' or 'skl_'."""
    for b in blocks:
        has_mesh = any(s['type'] == 0x2A for s in b['sections'])
        if has_mesh and b['name'] not in ('mdl_', 'skl_'):
            return b
    return None


def _recolor_weapon_dat(source: Path, output: Path, opts: dict):
    """Recolor and/or scale a weapon DAT (gear or mob)."""
    data = bytearray(source.read_bytes())
    sections = parse_sections(bytes(data))

    # Recolor textures
    for s in sections:
        if s.type_code != 0x20:
            continue
        info = parse_texture_info(bytes(data), s)
        if info is None:
            continue
        kw = dict(
            hue=opts.get('hue', 0),
            saturation=opts.get('saturation', 0),
            lightness=opts.get('lightness', 0),
            tint=opts.get('tint'),
            blend_mode=opts.get('blend_mode', 'normal'),
        )
        cr = opts.get('color_ranges')
        if cr:
            for cr_entry in cr:
                cr_kw = dict(
                    hue=cr_entry.get('hue', 0),
                    saturation=cr_entry.get('saturation', 0),
                    lightness=cr_entry.get('lightness', 0),
                    tint=cr_entry.get('tint'),
                    blend_mode=cr_entry.get('blend_mode', 'normal'),
                    hue_min=cr_entry.get('hue_min'), hue_max=cr_entry.get('hue_max'),
                    val_min=cr_entry.get('val_min'), val_max=cr_entry.get('val_max'),
                    sat_min=cr_entry.get('sat_min'), sat_max=cr_entry.get('sat_max'),
                )
                if info.tex_type == 0xA1 and info.dxt_type in ('1TXD', '3TXD'):
                    _adjust_dxt_inplace(data, info, **cr_kw)
                elif info.tex_type in (0x91, 0xB1):
                    _adjust_palette_inplace(data, info, **cr_kw)
        else:
            if info.tex_type == 0xA1 and info.dxt_type in ('1TXD', '3TXD'):
                _adjust_dxt_inplace(data, info, **kw)
            elif info.tex_type in (0x91, 0xB1):
                _adjust_palette_inplace(data, info, **kw)

    # Scale weapon mesh vertices
    wep_scale = opts.get('scale', 0)
    if wep_scale and wep_scale != 1.0:
        for s in sections:
            if s.type_code == 0x2A:
                _scale_mesh_inplace(data, s, wep_scale)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(bytes(data))


def swap_weapon(target_data: bytes, source_data: bytes | bytes, block_name: str = None) -> bytes:
    """Replace a weapon block in target_data with one from source_data.

    block_name: if set, find the target block by HDR name (e.g. 'rng_'
    for offhand). If None, finds the first weapon block.

    If the target doesn't have the requested block, inserts the source
    block after the existing weapon block (for adding offhand to mobs
    that don't have one).

    Remaps the source weapon's joint array to match the target mob's
    weapon joint, so gear weapons (joint 5) work on mobs (joint 33 etc).
    """
    if isinstance(source_data, (bytes, bytearray)):
        src_raw = source_data
    else:
        src_raw = source_data

    target_blocks = _parse_blocks(target_data)
    source_blocks = _parse_blocks(bytes(src_raw))

    # Find target block
    target_wep = None
    if block_name:
        target_wep = next((b for b in target_blocks if b['name'] == block_name), None)
    if not target_wep:
        target_wep = _find_weapon_block(target_blocks)

    source_wep = _find_weapon_block(source_blocks)
    if not source_wep:
        raise ValueError('Source DAT has no weapon block')

    # Find the target joint from the MATCHED block (main or offhand)
    if target_wep:
        target_joint = _get_weapon_joint(target_data, target_wep)
    else:
        # Inserting new block — use main weapon joint as fallback
        main_wep = _find_weapon_block(target_blocks)
        target_joint = _get_weapon_joint(target_data, main_wep) if main_wep else None

    # Extract source weapon block and remap its joint
    src_bytes = bytearray(src_raw[source_wep['start']:source_wep['end']])
    if target_joint is not None:
        _remap_weapon_joint(src_bytes, target_joint)

    # Ensure the block ends with an END section (gear DATs lack them)
    has_end = any(s['type'] == 0x00 for s in source_wep['sections'])
    if not has_end:
        end_section = b'end \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
        # Set meta: type=0x00, size=0x10 → (0x10//0x10)<<7 | 0x00 = 0x80
        end_section = b'end ' + struct.pack('<I', 0x80) + b'\x00' * 8
        src_bytes += end_section

    result = bytearray()
    if target_wep:
        # Replace existing block
        result += target_data[:target_wep['start']]
        result += src_bytes
        result += target_data[target_wep['end']:]
    elif main_wep:
        # Insert after main weapon block (adding offhand)
        insert_pos = main_wep['end']
        result += target_data[:insert_pos]
        result += src_bytes
        result += target_data[insert_pos:]
    else:
        raise ValueError('Target DAT has no weapon block')

    return bytes(result)


def _get_weapon_joint(data: bytes, block: dict) -> int | None:
    """Read the joint array value from a weapon block's mesh section."""
    for s in block['sections']:
        if s['type'] != 0x2A:
            continue
        ds = s['pos'] + 0x10
        flags3 = data[ds + 2]
        if not (flags3 & 0x80):
            return None  # no joint array
        off = ds + 6
        struct.unpack_from('<I', data, off)[0]; off += 6
        ja_off = struct.unpack_from('<I', data, off)[0] * 2; off += 4
        nj = struct.unpack_from('<H', data, off)[0]
        if nj > 0:
            return struct.unpack_from('<H', data, ds + ja_off)[0]
    return None


def _remap_weapon_joint(block_bytes: bytearray, target_joint: int):
    """Overwrite all mesh joint arrays in a weapon block to target_joint."""
    pos = 0
    while pos < len(block_bytes) - 16:
        meta = struct.unpack_from('<I', block_bytes, pos + 4)[0]
        tc = meta & 0x7F
        sz = ((meta >> 7) & 0xFFFFF) * 0x10
        if sz <= 0:
            break
        if tc == 0x2A:
            ds = pos + 0x10
            flags3 = block_bytes[ds + 2]
            if flags3 & 0x80:  # has joint array
                off = ds + 6
                struct.unpack_from('<I', block_bytes, off)[0]; off += 6
                ja_off = struct.unpack_from('<I', block_bytes, off)[0] * 2; off += 4
                nj = struct.unpack_from('<H', block_bytes, off)[0]
                for i in range(nj):
                    struct.pack_into('<H', block_bytes, ds + ja_off + i * 2, target_joint)
        pos += sz


# ---------------------------------------------------------------------------
# Entity DAT scaling (skeleton + mesh vertex positions)
# ---------------------------------------------------------------------------

SECTION_SKELETON = 0x29
SECTION_MESH = 0x2A


# ---------------------------------------------------------------------------
# Per-joint scaling (scale specific body parts)
# ---------------------------------------------------------------------------

def _get_joint_array(data: bytes, ds: int) -> list:
    """Read the joint mapping array from a mesh section (0x2A)."""
    flags3 = data[ds + 0x02]
    use_joint_array = (flags3 & 0x80) != 0

    off = ds + 6
    struct.unpack_from('<I', data, off)[0]; off += 6
    joint_array_off = struct.unpack_from('<I', data, off)[0] * 2; off += 4
    num_joints = struct.unpack_from('<H', data, off)[0]; off += 2
    vc_off = struct.unpack_from('<I', data, off)[0] * 2; off += 6
    vjm_off = struct.unpack_from('<I', data, off)[0] * 2; off += 6
    vd_off = struct.unpack_from('<I', data, off)[0] * 2

    # Read joint array
    joint_array = []
    if use_joint_array:
        for i in range(num_joints):
            joint_array.append(struct.unpack_from('<H', data, ds + joint_array_off + i * 2)[0])

    single_count = struct.unpack_from('<H', data, ds + vc_off)[0]
    double_count = struct.unpack_from('<H', data, ds + vc_off + 2)[0]

    # Read vertex-to-joint mapping for single-joint vertices
    vert_joints = []
    p = ds + vjm_off
    for i in range(single_count):
        ref = struct.unpack_from('<H', data, p)[0]
        joint_idx = ref & 0x7F
        if use_joint_array and joint_idx < len(joint_array):
            joint_idx = joint_array[joint_idx]
        vert_joints.append(joint_idx)
        p += 4  # 2 refs per vertex (ref0 + ref1)

    # Double-joint vertices
    for i in range(double_count):
        ref0 = struct.unpack_from('<H', data, p)[0]
        ref1 = struct.unpack_from('<H', data, p + 2)[0]
        j0 = ref0 & 0x7F
        j1 = ref1 & 0x7F
        if use_joint_array:
            j0 = joint_array[j0] if j0 < len(joint_array) else j0
            j1 = joint_array[j1] if j1 < len(joint_array) else j1
        vert_joints.append((j0, j1))
        p += 4

    return vert_joints, single_count, double_count, vd_off


def _scale_joints_inplace(data: bytearray, sections: list, joint_scales: dict) -> dict:
    """Scale specific joints' bone translations and their attached vertices.

    joint_scales: {joint_index: scale_factor, ...}
    All joints in the set + their descendants are scaled.

    Returns stats dict.
    """
    stats = {'joints': 0, 'vertices': 0}

    # Find skeleton section to get hierarchy + expand joint set to include descendants
    skel_section = next((s for s in sections if s.type_code == 0x29), None)
    if not skel_section:
        return stats

    ds = skel_section.data_start
    num_joints = data[ds + 0x02]
    joints_start = ds + 0x04
    JOINT_SIZE = 30
    TRANS_OFFSET = 18

    # Build parent map
    parents = []
    for i in range(num_joints):
        parents.append(data[joints_start + i * JOINT_SIZE])

    # Expand: include all descendants of specified joints
    expanded = {}
    for joint_idx, scale in joint_scales.items():
        expanded[joint_idx] = scale
        # Add all descendants
        changed = True
        while changed:
            changed = False
            for i in range(num_joints):
                if i not in expanded and parents[i] in expanded:
                    expanded[i] = expanded[parents[i]]
                    changed = True

    stats['joints'] = len(expanded)

    # Scale vertices attached to expanded joints (all mesh sections)
    for mesh_section in (s for s in sections if s.type_code == 0x2A):
        mds = mesh_section.data_start
        flags3 = data[mds + 0x02]
        cloth = (flags3 & 0x01) != 0
        has_normals = not cloth

        try:
            vert_joints, single_count, double_count, vd_off = _get_joint_array(bytes(data), mds)
        except (struct.error, IndexError):
            continue

        vertex_base = mds + vd_off

        # Scale single-joint vertices
        stride_single = 24 if has_normals else 12
        for i in range(single_count):
            if i < len(vert_joints) and vert_joints[i] in expanded:
                s = expanded[vert_joints[i]]
                base = vertex_base + i * stride_single
                for j in range(3):
                    off = base + j * 4
                    val = struct.unpack_from('<f', data, off)[0]
                    struct.pack_into('<f', data, off, val * s)
                stats['vertices'] += 1

        # Scale double-joint vertices
        stride_double = 56 if has_normals else 32
        double_base = vertex_base + single_count * stride_single
        for i in range(double_count):
            idx = single_count + i
            if idx < len(vert_joints):
                j0, j1 = vert_joints[idx] if isinstance(vert_joints[idx], tuple) else (vert_joints[idx], vert_joints[idx])
                if j0 in expanded or j1 in expanded:
                    s = expanded.get(j0, expanded.get(j1, 1.0))
                    base = double_base + i * stride_double
                    for j in range(6):
                        off = base + j * 4
                        val = struct.unpack_from('<f', data, off)[0]
                        struct.pack_into('<f', data, off, val * s)
                    stats['vertices'] += 1

    return stats


def _scale_skeleton_inplace(data: bytearray, section, scale: float) -> int:
    """Scale all bone translations in a skeleton section (0x29).

    Joint layout (30 bytes each):
      parent(u8) + pad(u8) + rotation(4×f32) + translation(3×f32)
    Translation starts at byte 18 within each joint.
    """
    num_joints = data[section.data_start + 0x02]
    joints_start = section.data_start + 0x04
    JOINT_SIZE = 30  # 2 + 16 + 12
    TRANS_OFFSET = 18  # 2 + 16

    for i in range(num_joints):
        base = joints_start + i * JOINT_SIZE + TRANS_OFFSET
        for j in range(3):
            off = base + j * 4
            val = struct.unpack_from('<f', data, off)[0]
            struct.pack_into('<f', data, off, val * scale)

    return num_joints


def _scale_mesh_inplace(data: bytearray, section, scale: float) -> int:
    """Scale all vertex positions in a mesh section (0x2A).

    Parses the mesh header to find vertex data offset and counts,
    then scales position floats in-place. Normals are NOT scaled.

    Single-joint vertex: pos(3×f32) + normal(3×f32) = 24 bytes
    Double-joint vertex: p0x,p1x,p0y,p1y,p0z,p1z,w0,w1,n0x,n1x,... = 56 bytes
    """
    ds = section.data_start

    flags3 = data[ds + 0x02]
    cloth_effect = (flags3 & 0x01) != 0
    has_normals = not cloth_effect

    # Parse header offsets (same layout as xi_export.py parse_mesh)
    # Skip: u8,u8,flags3(u8),u8,symmetric(u8),u8 = 6 bytes
    # Then: instruction_offset(u32), u8, u8, joint_array_offset(u32), num_joints(u16),
    #       vertex_counts_offset(u32), u16, vertex_joint_mapping_offset(u32), u16,
    #       vertex_data_offset(u32), u16
    off = ds + 6
    _instr_off = struct.unpack_from('<I', data, off)[0] * 2;  off += 4
    off += 2  # u8, u8
    _joint_off = struct.unpack_from('<I', data, off)[0] * 2;  off += 4
    _num_joints = struct.unpack_from('<H', data, off)[0];     off += 2
    vc_off = struct.unpack_from('<I', data, off)[0] * 2;      off += 4
    off += 2  # u16
    _vjm_off = struct.unpack_from('<I', data, off)[0] * 2;    off += 4
    off += 2  # u16
    vd_off = struct.unpack_from('<I', data, off)[0] * 2;      off += 4

    # Read vertex counts
    single_count = struct.unpack_from('<H', data, ds + vc_off)[0]
    double_count = struct.unpack_from('<H', data, ds + vc_off + 2)[0]

    vertex_base = ds + vd_off
    scaled = 0

    # Scale single-joint vertices: 3 floats position, then 3 floats normal
    stride_single = 24 if has_normals else 12
    for i in range(single_count):
        base = vertex_base + i * stride_single
        for j in range(3):  # x, y, z only — skip normals
            foff = base + j * 4
            val = struct.unpack_from('<f', data, foff)[0]
            struct.pack_into('<f', data, foff, val * scale)
        scaled += 1

    # Scale double-joint vertices: interleaved p0x,p1x,p0y,p1y,p0z,p1z,w0,w1,...
    # Scale first 6 floats (positions), skip weights (2) and normals (6)
    stride_double = 56 if has_normals else 32
    double_base = vertex_base + single_count * stride_single
    for i in range(double_count):
        base = double_base + i * stride_double
        for j in range(6):  # p0x, p1x, p0y, p1y, p0z, p1z
            foff = base + j * 4
            val = struct.unpack_from('<f', data, foff)[0]
            struct.pack_into('<f', data, foff, val * scale)
        scaled += 1

    return scaled


def _scale_animation_inplace(data: bytearray, section, scale: float) -> int:
    """Scale translation keyframes in an animation section (0x2B).

    Animation track layout per joint (84 bytes):
      joint_index (i32)
      rotation:    4 offsets (i32) + 4 const_values (f32) = 32 bytes
      translation: 3 offsets (i32) + 3 const_values (f32) = 24 bytes
      scale:       3 offsets (i32) + 3 const_values (f32) = 24 bytes

    Root joint (index 0) is NOT scaled — it controls world-space
    positioning (e.g. float height for bombs). Only child joint
    translations are scaled.
    """
    ds = section.data_start
    # Header: u16, num_joints(u16), num_frames(u16), keyframe_duration(f32)
    num_joints = struct.unpack_from('<H', data, ds + 2)[0]
    num_frames = struct.unpack_from('<H', data, ds + 4)[0]
    keyframe_data_off = ds + 10  # after header (2+2+2+4=10)

    scaled_channels = 0
    track_cursor = keyframe_data_off

    for _j in range(num_joints):
        # joint_index (i32)
        track_cursor += 4

        # Skip rotation: 4 offsets (i32) + 4 const values (f32) = 32 bytes
        track_cursor += 32

        # Translation: 3 offsets (i32) + 3 const values (f32) = 24 bytes
        trans_start = track_cursor

        for ch in range(3):  # x, y, z
            offset_pos = trans_start + ch * 4
            const_pos = trans_start + 12 + ch * 4

            offset_val = struct.unpack_from('<i', data, offset_pos)[0]

            if offset_val < 0:
                continue
            elif offset_val == 0:
                val = struct.unpack_from('<f', data, const_pos)[0]
                struct.pack_into('<f', data, const_pos, val * scale)
                scaled_channels += 1
            else:
                arr_base = keyframe_data_off + offset_val * 4
                for frame in range(num_frames):
                    foff = arr_base + frame * 4
                    if foff + 4 <= len(data):
                        val = struct.unpack_from('<f', data, foff)[0]
                        struct.pack_into('<f', data, foff, val * scale)
                val = struct.unpack_from('<f', data, const_pos)[0]
                struct.pack_into('<f', data, const_pos, val * scale)
                scaled_channels += 1

        track_cursor += 24

        # Skip scale: 3 offsets (i32) + 3 const values (f32) = 24 bytes
        track_cursor += 24

    return scaled_channels


# ---------------------------------------------------------------------------
# Mob injection from JSON config
# ---------------------------------------------------------------------------

def apply_mobs(config_path: Path):
    """Process a mobs.json config file.

    For each mob entry:
    1. Resolves source model ID to DAT via FTABLE
    2. Recolors textures
    3. Places in ROM10 and registers in FTABLE10/VTABLE10
    4. Outputs SQL for mob_pools and pet_list

    Config format::

        {
          "mobs": [
            {
              "name": "Crimson Tiger",
              "clone": 308,
              "hue": 0,
              "saturation": 40,
              "tint": "#ff3300cc",
              "blend_mode": "multiply"
            }
          ]
        }
    """
    config = json.loads(config_path.read_text())
    mobs = config.get('mobs', [])
    if not mobs:
        print("No mobs defined in config")
        return

    ffxi = ffxi_dir()
    tables = load_all(ffxi)
    hd_overlay = overlay(ffxi=ffxi)
    ft10, vt10 = ensure_rom10(ffxi, entity_models=True)
    rom10 = ffxi / 'ROM10'
    subdir = 1
    # An overlay with its own FTABLE10 (Ashita's ffxi-hd) hides the game's: the client only sees
    # what the overlay registers, so install there, in a slot nothing in either ROM10 uses
    overlay10 = hd_overlay / 'ROM10'
    if (overlay10 / 'FTABLE10.DAT').exists():
        ft10, vt10 = overlay10 / 'FTABLE10.DAT', overlay10 / 'VTABLE10.DAT'
        tables[10] = (bytearray(ft10.read_bytes()), bytearray(vt10.read_bytes()))
        print(f'Installing into the overlay: {overlay10}')
    else:
        overlay10 = None

    # Check if base FTABLE is expanded (required for entity models)
    base_ft_size = (ffxi / 'FTABLE.DAT').stat().st_size
    if base_ft_size < 118240 * 2:
        print(f'WARNING: Base FTABLE.DAT has {base_ft_size // 2} entries (need 118240)')
        print(f'  Grow the base file tables (FTABLE.DAT / VTABLE.DAT and ROM2-9) to 118,240 entries first.')
        print(f'  Custom models will be invisible without expanded base FTABLEs.')
        print()

    sql_lines = [
        '-- Generated by LPS mob recoloring',
        '-- Import into your LSB database',
        '',
    ]

    for entry in mobs:
        name = entry['name']
        clone_mid = entry['clone']
        hue = entry.get('hue', 0)
        sat = entry.get('saturation', 0)
        lit = entry.get('lightness', 0)
        tint = entry.get('tint')
        blend = entry.get('blend_mode', 'normal')
        color_ranges = entry.get('color_ranges')
        model_id = entry.get('model_id')
        scale = entry.get('scale', 0)
        weapon_mid = entry.get('weapon')  # model ID to take weapon from
        joint_scales_raw = entry.get('joint_scales')
        # Parse joint_scales: {"5-7": 3.0} → {5: 3.0, 6: 3.0, 7: 3.0}
        joint_scales = None
        if joint_scales_raw:
            joint_scales = {}
            for key, val in joint_scales_raw.items():
                if '-' in key:
                    lo, hi = key.split('-')
                    for j in range(int(lo), int(hi) + 1):
                        joint_scales[j] = val
                else:
                    joint_scales[int(key)] = val

        # Resolve weapon sources (main hand + offhand)
        def _resolve_weapon_entry(entry_key):
            """Resolve a weapon config to (dat_path, recolor_opts)."""
            val = entry.get(entry_key)
            if val is None:
                return None, None

            # Can be an int (mob model ID), a string (gear DAT path),
            # or a dict with model/gear_model/path + recolor options
            if isinstance(val, int):
                dat = resolve_model(val, tables, ffxi)
                if not dat:
                    print(f'  WARNING: {entry_key} model {val} not found')
                return dat, None
            elif isinstance(val, str):
                dat = ffxi / val
                if not dat.exists():
                    hd = hd_overlay / val
                    dat = hd if hd.exists() else None
                if not dat:
                    print(f'  WARNING: {entry_key} {val} not found')
                return dat, None
            elif isinstance(val, dict):
                recolor = {}
                for k in ('hue', 'saturation', 'lightness', 'tint', 'blend_mode', 'color_ranges'):
                    if k in val:
                        recolor[k] = val[k]

                # Resolve DAT source
                dat = None
                if 'model' in val:
                    dat = resolve_model(val['model'], tables, ffxi)
                elif 'gear_model' in val:
                    try:
                        from equipment import resolve_gear_dat
                        dat = resolve_gear_dat('HumeMale', 'main', val['gear_model'], ffxi)
                    except Exception as e:
                        print(f'  WARNING: {entry_key} gear_model {val["gear_model"]}: {e}')
                elif 'path' in val:
                    dat = ffxi / val['path']
                    if not dat.exists():
                        hd = hd_overlay / val['path']
                        dat = hd if hd.exists() else None

                if not dat:
                    print(f'  WARNING: {entry_key} source not found')
                return dat, recolor or None
            return None, None

        blur = entry.get('blur')

        weapon_src, weapon_recolor = _resolve_weapon_entry('weapon')
        offhand_src, offhand_recolor = _resolve_weapon_entry('offhand')

        # Auto-assign model ID
        if model_id is None:
            model_id = _next_free_modelid(tables)

        print(f'\n=== {name} (clone mid={clone_mid} → mid={model_id}) ===')

        # Resolve source DAT (prefer HD)
        src_dat = resolve_model(clone_mid, tables, ffxi)
        if not src_dat or not src_dat.exists():
            print(f'  ERROR: Model {clone_mid} DAT not found')
            continue

        src_dat = hd(src_dat, ffxi)

        print(f'  Source: {src_dat.name} ({src_dat.stat().st_size:,} bytes)')

        # Place in ROM10
        if overlay10:
            subdir, slot = slot_for(overlay10, rom10, model_fid(model_id))
            dst = overlay10 / str(subdir) / f'{slot}.DAT'
        else:
            subdir, slot = next_slot(rom10, subdir)
            dst = rom10 / str(subdir) / f'{slot}.DAT'

        # Describe adjustments
        desc = []
        if hue: desc.append(f'hue {hue}°')
        if sat: desc.append(f'sat {sat:+d}%')
        if lit: desc.append(f'lit {lit:+d}%')
        if tint: desc.append(f'tint {tint} ({blend})')
        if color_ranges: desc.append(f'{len(color_ranges)} color range(s)')
        if scale and scale != 1.0: desc.append(f'scale {scale}x')
        if joint_scales: desc.append(f'{len(joint_scales)} joints scaled')
        if weapon_src: desc.append(f'weapon swap')
        if offhand_src: desc.append(f'offhand swap')
        if blur: desc.append(f'blur RGB({blur.get("r",200)},{blur.get("g",200)},{blur.get("b",255)})')
        desc_str = ', '.join(desc) if desc else 'clone only'

        has_adjustments = hue or sat or lit or tint or color_ranges or (scale and scale != 1.0) or joint_scales or weapon_src or offhand_src or blur
        if has_adjustments:
            print(f'  Recoloring: {desc_str}')
            stats = recolor_entity_dat(src_dat, dst,
                                        hue=hue, saturation=sat, lightness=lit,
                                        tint=tint, blend_mode=blend,
                                        color_ranges=color_ranges,
                                        scale=scale,
                                        joint_scales=joint_scales,
                                        weapon_source=weapon_src,
                                        weapon_recolor=weapon_recolor,
                                        offhand_source=offhand_src,
                                        offhand_recolor=offhand_recolor,
                                        blur=blur)
            print(f'  {stats["dxt"]} DXT + {stats["paletted"]} paletted textures')
            if stats.get('joints_scaled'):
                print(f'  Scaled: {stats["joints_scaled"]} joints + {stats["vertices_scaled"]} vertices + {stats["anims_scaled"]} anim channels ({scale}x)')
            if stats.get('part_joints'):
                print(f'  Parts: {stats["part_joints"]} joints + {stats["part_vertices"]} vertices scaled')
            if stats.get('blur'):
                print(f'  Blur: 0x5E section injected')
        else:
            print(f'  Cloning (no recolor)')
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_dat, dst)

        # Our own textures over its own: "paint" {"texture name": png or {part: png}} (paintings on
        # the UV template), or "materials" {"texture name": spec or spec.json} (materials laid on
        # the body in 3D); paths relative to the config; "parts" a parts.json (see skin.py)
        if entry.get('paint') or entry.get('materials'):
            from . import skin
            here = config_path.parent
            def _rel(p):
                return {k: _rel(v) for k, v in p.items()} if isinstance(p, dict) else here / p
            def _missing(p):
                return [m for v in p.values() for m in _missing(v)] if isinstance(p, dict) else [p] if not p.exists() else []
            parts = entry.get('parts')
            if isinstance(parts, str):
                parts = json.loads((here / parts).read_text())['parts']
            pictures = {tex: _rel(p) for tex, p in (entry.get('paint') or {}).items()}
            missing = _missing(pictures)
            if missing:
                print(f'  SKIPPED: no picture at {", ".join(str(m) for m in missing)}')
                dst.unlink()
                continue
            data = dst.read_bytes()
            if entry.get('materials'):
                tris, infos = skin.triangles(data), skin.textures(data)
                for tex, spec in entry['materials'].items():
                    if isinstance(spec, str):
                        spec = json.loads((here / spec).read_text())
                    n = entry.get('paint_size') or infos[tex].width
                    original = skin.texture_rgba(data, tex) if spec.get('base') == 'original' else None
                    everything = [t for ts in tris.values() for t in ts]
                    pictures[tex] = Image.fromarray(skin.materials(tris[tex], n, spec, parts, here, original, everything))
            data = skin.paint(data, pictures, size=entry.get('paint_size'), parts=parts,
                              check=here / 'check')
            dst.write_bytes(data)
            skin.preview(data, here / 'check' / f"{name.replace(' ', '_')}.preview.png")
            print(f'  Painted: {", ".join(repr(t) for t in pictures)} (check: {here / "check"})')

        # Register in FTABLE10
        fid = model_fid(model_id)
        register(ft10, vt10, fid, subdir, slot)
        print(f'  Registered: mid={model_id} fid={fid} → ROM10/{subdir}/{slot}.DAT')

        # Mark as used in our tables dict so next auto-assign skips it
        if 10 not in tables:
            tables[10] = (bytearray(ft10.read_bytes()), bytearray(vt10.read_bytes()))
        else:
            tables[10] = (bytearray(ft10.read_bytes()), bytearray(vt10.read_bytes()))

        # Generate SQL — clone stats from source mob, scale hitbox
        internal_name = name.replace(' ', '_')
        pool_id = model_id + 10000
        pet_id = model_id + 200
        blob = modelid_blob(model_id)
        clone_blob = modelid_blob(clone_mid)
        hitbox_scale = scale if (scale and scale != 1.0) else 1.0

        sql_lines.extend([
            f'-- {name} (modelid {model_id}, cloned from {clone_mid})',
            f"INSERT INTO `mob_pools`",
            f"  SELECT {pool_id}, '{internal_name}', '{internal_name}',",
            f"    familyid, {blob},",
            f"    mJob, sJob, cmbSkill, cmbDelay, cmbDmgMult,",
            f"    behavior, aggro, true_detection, links, mobType, immunity,",
            f"    name_prefix, flag, entityFlags, animationsub,",
            f"    hasSpellScript, spellList, namevis, roamflag,",
            f"    skill_list_id, resist_id,",
            f"    modelSize, LEAST(ROUND(modelHitboxSize * {hitbox_scale:.2f}), 255)",
            f"  FROM `mob_pools` WHERE `modelid` = {clone_blob}",
            f"  LIMIT 1;",
            f"",
            f"INSERT INTO `pet_list` VALUES (",
            f"  {pet_id}, '{internal_name}', {pool_id},",
            f"  28, 75, 3600, 0, 3",
            f");",
            f"",
        ])

    # Write SQL
    sql_path = config_path.parent / 'mobs_inject.sql'
    sql_path.write_text('\n'.join(sql_lines))

    print(f'\nDone. {len(mobs)} mob(s) injected.')
    print(f'SQL: {sql_path}')
    print(f'\nServer-side TODO:')
    print(f'  1. Import {sql_path.name} into database')
    print(f'  2. Create spawn commands using setModelId({model_id})')
    print(f'  3. Restart server + client')


# === Module registration ===

def setup_cli(subparsers):
    """Add mobs CLI subcommands."""
    mobs_parser = subparsers.add_parser('mobs', help='Inject recolored mobs from mobs.json')
    mobs_parser.add_argument('config', nargs='?', default='mobs.json',
                             help='Mobs config JSON (default: mobs.json)')


register_module('mobs',
    cli_setup=setup_cli,
    depends=['zones'],
)

"""Our own paintings on an entity's mesh: a UV template out, a painted texture back in.

template(dat, out) draws, for each texture an entity DAT's meshes use, the mesh's UV triangles at
the size we paint at: each island filled in the colour of the body part its joints belong to (a
parts.json names the joint groups), the edges in black, on a dark ground; and a sheet per part
with only that part lit. Paint on those (by hand or by an image model, given the template and
nothing of the game's own texture), and paint() puts the picture, or the parts laid together,
back under the same texture name, encoded DXT1 or DXT3 like the section it replaces.

The painted picture is spread a few pixels past every island's edge before it is encoded, so
filtering and DXT's 4x4 blocks pick up paint and not the ground around it.

Mesh instructions (0x2A, from its instruction offset): 0x8010 render properties, 0x8000 a
16-byte texture name, 0x5453 a triangle strip (count, three vertex ids, three UVs, then one
vertex id and UV per further triangle), 0x0054 a triangle list (count, per triangle three ids
then three UVs), 0x0043 and 0x4353 vertex coloured lists and strips (no UVs), 0xFFFF the end.
A symmetric mesh draws each triangle a second time mirrored, with the same UVs: both sides share
the paint.
"""
import struct
import sys
from pathlib import Path

if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # modules

import numpy as np
from PIL import Image, ImageDraw

from xidat.textures import parse_sections, parse_texture_info
from zonekit import textures as tx

GROUND = (40, 40, 44, 255)


def _vertex_joints(data, ds):
    """The joint each vertex hangs from (its first, for two-joint vertices)."""
    flags3 = data[ds + 2]
    use_array = flags3 & 0x80
    o = ds + 6
    o += 4 + 2
    ja_off = struct.unpack_from('<I', data, o)[0] * 2; o += 4
    n_joints = struct.unpack_from('<H', data, o)[0]; o += 2
    vc_off = struct.unpack_from('<I', data, o)[0] * 2; o += 6
    vjm_off = struct.unpack_from('<I', data, o)[0] * 2
    array = [struct.unpack_from('<H', data, ds + ja_off + i * 2)[0] for i in range(n_joints)]
    single, double = struct.unpack_from('<HH', data, ds + vc_off)
    out = []
    for i in range(single + double):
        j = struct.unpack_from('<H', data, ds + vjm_off + i * 4)[0] & 0x7F
        out.append(array[j] if use_array and j < len(array) else j)
    return out


def _quat(x, y, z, w):
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def skeleton(data):
    """[(rotation 3x3, translation 3)] per joint in bind pose, from the first 0x29 section. The
    skeleton stands with +x forward, -y up and +z/-z the two sides."""
    for s in parse_sections(data):
        if s.type_code == 0x29:
            ds = s.data_start
            out = []
            for i in range(data[ds + 2]):
                b = ds + 4 + i * 30
                parent = data[b]
                r = _quat(*struct.unpack_from('<4f', data, b + 2))
                t = np.array(struct.unpack_from('<3f', data, b + 18))
                if parent == i:
                    out.append((r, t))
                else:
                    pr, pt = out[parent]
                    out.append((pr @ r, pr @ t + pt))
            return out
    return []


def _bind_pose(data, ds, bones):
    """(positions, normals) per vertex of a mesh section, in the skeleton's bind pose."""
    flags3 = data[ds + 2]
    use_array, normals = flags3 & 0x80, not flags3 & 0x01
    o = ds + 12
    ja_off = struct.unpack_from('<I', data, o)[0] * 2
    n_joints = struct.unpack_from('<H', data, o + 4)[0]
    vc_off = struct.unpack_from('<I', data, o + 6)[0] * 2
    vjm_off = struct.unpack_from('<I', data, o + 12)[0] * 2
    vd_off = struct.unpack_from('<I', data, o + 18)[0] * 2
    array = [struct.unpack_from('<H', data, ds + ja_off + i * 2)[0] for i in range(n_joints)]
    single, double = struct.unpack_from('<HH', data, ds + vc_off)

    def bone(ref):
        j = ref & 0x7F
        j = array[j] if use_array and j < len(array) else j
        return bones[j] if j < len(bones) else (np.eye(3), np.zeros(3))

    pos, nrm = [], []
    p = ds + vd_off
    for i in range(single):
        r, t = bone(struct.unpack_from('<H', data, ds + vjm_off + i * 4)[0])
        v = np.array(struct.unpack_from('<3f', data, p))
        n = np.array(struct.unpack_from('<3f', data, p + 12)) if normals else np.array([0, -1.0, 0])
        p += 24 if normals else 12
        pos.append(r @ v + t)
        nrm.append(r @ n)
    for i in range(double):
        k = single + i
        r0, t0 = bone(struct.unpack_from('<H', data, ds + vjm_off + k * 4)[0])
        r1, t1 = bone(struct.unpack_from('<H', data, ds + vjm_off + k * 4 + 2)[0])
        f = struct.unpack_from('<8f', data, p)
        v0, v1, w0, w1 = np.array(f[0:6:2]), np.array(f[1:6:2]), f[6], f[7]
        if normals:
            g = struct.unpack_from('<6f', data, p + 32)
            n = w0 * (r0 @ np.array(g[0::2])) + w1 * (r1 @ np.array(g[1::2]))
        else:
            n = np.array([0, -1.0, 0])
        p += 56 if normals else 32
        pos.append(w0 * (r0 @ v0 + t0) + w1 * (r1 @ v1 + t1))
        nrm.append(n)
    nrm = np.array(nrm)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
    return np.array(pos), nrm


def triangles(data):
    """{texture name: [(uv0, uv1, uv2, joint, positions 3x3, normals 3x3), ...]} over every mesh
    section in the DAT, positions and normals in the skeleton's bind pose."""
    out = {}
    bones = skeleton(data)
    for s in parse_sections(data):
        if s.type_code != 0x2A:
            continue
        ds = s.data_start
        joints = _vertex_joints(data, ds)
        P, N = _bind_pose(data, ds, bones)
        o = ds + 6
        instr = ds + struct.unpack_from('<I', data, o)[0] * 2
        name = None
        p = instr

        def u16():
            nonlocal p
            v = struct.unpack_from('<H', data, p)[0]
            p += 2
            return v

        def uv():
            nonlocal p
            v = struct.unpack_from('<ff', data, p)
            p += 8
            return v

        def joint(*ids):
            js = [joints[i] for i in ids if i < len(joints)]
            return max(set(js), key=js.count) if js else 0

        while True:
            op = u16()
            if op == 0xFFFF:
                break
            if op == 0x8010:
                p += 44
            elif op == 0x8000:
                name = data[p:p + 16].split(b'\0')[0].decode('ascii', 'replace').rstrip()
                p += 16
            elif op == 0x5453:
                n = u16()
                ids = [u16(), u16(), u16()]
                uvs = [uv(), uv(), uv()]
                for _ in range(1, n):
                    ids.append(u16())
                    uvs.append(uv())
                tris = out.setdefault(name, [])
                for k in range(len(ids) - 2):
                    tris.append((uvs[k], uvs[k + 1], uvs[k + 2], joint(*ids[k:k + 3]),
                                 P[ids[k:k + 3]], N[ids[k:k + 3]]))
            elif op == 0x0054:
                n = u16()
                tris = out.setdefault(name, [])
                for _ in range(n):
                    ids = [u16(), u16(), u16()]
                    tris.append((uv(), uv(), uv(), joint(*ids), P[ids], N[ids]))
            elif op == 0x0043:
                n = u16()
                p += n * 10
            elif op == 0x4353:
                n = u16()
                p += 6 + 4 + (n - 1) * 2
            else:
                raise ValueError(f'unknown mesh opcode 0x{op:04X} in {s.name!r}')
    return out


def textures(data):
    """{texture name: TextureInfo} for every 0x20 section."""
    out = {}
    for s in parse_sections(data):
        if s.type_code == 0x20:
            info = parse_texture_info(data, s)
            if info:
                out[info.name] = info
    return out


def texture_rgba(data, name):
    """(h, w, 4) uint8 of a DXT texture in the DAT, as stored (the client shows it twice as
    bright)."""
    info = textures(data)[name]
    return tx.decode(data[info.data_offset:info.data_offset + info.data_size], info.width,
                     info.height, info.dxt_type)


# distinct, light, nameable: the template's legend reads "the head is the orange"
COLOURS = [
    ('red', (235, 110, 110)), ('orange', (245, 175, 90)), ('yellow', (240, 230, 110)),
    ('green', (130, 220, 120)), ('cyan', (110, 220, 230)), ('blue', (120, 150, 240)),
    ('purple', (180, 130, 235)), ('pink', (240, 140, 200)), ('white', (235, 235, 235)),
    ('brown', (170, 120, 80)),
]
DIM = (70, 70, 76, 255)


def _px(uv, size):
    # entity UVs stay within 0..1 (1.0 is the far edge, not a wrap back to 0)
    return (min(max(uv[0], 0.0), 1.0) * size, min(max(uv[1], 0.0), 1.0) * size)


def _poly(t, size):
    return [_px(t[0], size), _px(t[1], size), _px(t[2], size)]


def coverage(tris, size):
    """(size, size) bool: where these triangles sample the texture."""
    img = Image.new('L', (size, size), 0)
    d = ImageDraw.Draw(img)
    for t in tris:
        d.polygon(_poly(t, size), fill=255, outline=255)
    return np.array(img) > 0


def islands(tris, n=1024):
    """The UV island each triangle lies in (numbered from 1, left to right, top to bottom)."""
    from scipy import ndimage
    labels, _ = ndimage.label(coverage(tris, n))
    out = []
    for t in tris:
        x, y = np.mean(_poly(t, n), 0)
        out.append(int(labels[min(int(y), n - 1), min(int(x), n - 1)]))
    # renumber by reading order of each island's top-left
    seen = {}
    for lab in sorted({i for i in out if i}, key=lambda i: tuple(np.argwhere(labels == i).min(0))):
        seen[lab] = len(seen) + 1
    return [seen.get(i, 0) for i in out]


def split(tris, parts):
    """{part: triangles}. A part is a list of joints, or {"joints": [...], "islands": [...]}
    (islands as numbered on the template's .islands.png; they win over joints). Triangles no
    part names go to 'other'."""
    parts = {k: v if isinstance(v, dict) else {'joints': v} for k, v in (parts or {}).items()}
    by_joint = {j: part for part, v in parts.items() for j in v.get('joints', [])}
    by_island = {i: part for part, v in parts.items() for i in v.get('islands', [])}
    isl = islands(tris) if by_island else [0] * len(tris)
    out = {part: [] for part in parts}
    for t, i in zip(tris, isl):
        out.setdefault(by_island.get(i) or by_joint.get(t[3], 'other'), []).append(t)
    return {k: v for k, v in out.items() if v}


def template(dat, out_dir, size=1024, parts=None):
    """Per texture, <texture>.png: every island filled in its part's colour, edges in black on a
    dark ground; and <texture>.<part>.png per part: that part's islands alone in colour, the rest
    dimmed, for painting a part at a time. Returns {texture: {part: colour name}}."""
    data = Path(dat).read_bytes()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    legend = {}
    line = max(1, size // 512)
    for name, tris in triangles(data).items():
        if name is None:
            continue
        groups = split(tris, parts)
        colour = {part: COLOURS[k % len(COLOURS)] for k, part in enumerate(groups)}
        stem = name.replace(' ', '_')

        def sheet(lit):
            img = Image.new('RGBA', (size, size), GROUND)
            d = ImageDraw.Draw(img)
            for part, ts in groups.items():
                fill = colour[part][1] + (255,) if lit in (None, part) else DIM
                for t in ts:
                    d.polygon(_poly(t, size), fill=fill)
            for t in tris:
                poly = _poly(t, size)
                d.line(poly + [poly[0]], fill=(0, 0, 0, 255), width=line)
            return img

        sheet(None).save(out_dir / f'{stem}.png')
        # every island numbered, for naming small ones (eyes, fangs, claws) in parts.json
        num = sheet(None)
        d = ImageDraw.Draw(num)
        isl = islands(tris, size)
        for i in sorted(set(isl) - {0}):
            pts = np.concatenate([_poly(t, size) for t, k in zip(tris, isl) if k == i])
            x, y = pts.mean(0)
            d.rectangle([x - 11, y - 8, x + 11, y + 8], fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
            d.text((x, y), str(i), fill=(0, 0, 0, 255), anchor='mm')
        num.save(out_dir / f'{stem}.islands.png')
        if len(groups) > 1:
            for part in groups:
                sheet(part).save(out_dir / f'{stem}.{part.replace(" ", "_")}.png')
        legend[name] = {part: colour[part][0] for part in groups}
    return legend


def _spread(rgba, mask, px=8):
    """Paint carried out from the islands into the ground around them, a pixel at a time."""
    img = rgba.astype(np.float32)
    have = mask.copy()
    for _ in range(px):
        acc = np.zeros_like(img)
        cnt = np.zeros(have.shape, np.float32)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            acc += np.roll(np.roll(img * have[..., None], dy, 0), dx, 1)
            cnt += np.roll(np.roll(have, dy, 0), dx, 1)
        grow = ~have & (cnt > 0)
        img[grow] = acc[grow] / cnt[grow][:, None]
        have |= grow
    return img.astype(np.uint8)


def _erode(mask, px):
    out = mask.copy()
    for _ in range(px):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            out &= np.roll(np.roll(mask, dy, 0), dx, 1)
        mask = out.copy()
    return out


def _load(pic, n):
    img = pic if isinstance(pic, Image.Image) else Image.open(pic)
    return np.array(img.convert('RGBA').resize((n, n), Image.LANCZOS))


def compose(tris, pictures, n, parts=None):
    """One (n, n, 4) picture from a single path, or from {part: path} (an optional 'base' first,
    then each part laid over its own islands)."""
    if not isinstance(pictures, dict):
        return _load(pictures, n)
    out = _load(pictures['base'], n) if 'base' in pictures else np.zeros((n, n, 4), np.uint8)
    for part, ts in split(tris, parts).items():
        if part in pictures:
            m = coverage(ts, n)
            out[m] = _load(pictures[part], n)[m]
    return out


def paint(data, pictures, size=None, parts=None, gain=0.5, check=None):
    """data with each named texture replaced: {texture: picture or {part: picture}}. Pictures
    may be any size; they are fitted to `size` (default: the texture's own).

    gain: the client draws entity textures at twice their stored brightness (MODULATE2X), so
    paint is stored at half by default. check: a folder to write each result as the client
    shows it, with the mesh's edges over it, to see the painting sits on its islands."""
    data = bytes(data)
    tris = triangles(data)
    infos = textures(data)
    swaps = {}
    for name, pics in pictures.items():
        info = infos.get(name)
        if info is None:
            raise KeyError(f'no texture {name!r} in this DAT (it has {sorted(infos)})')
        if info.tex_type != 0xA1 or info.dxt_type not in ('1TXD', '3TXD'):
            raise ValueError(f'{name!r} is not DXT1/DXT3 (paletted textures not handled yet)')
        n = size or info.width
        ts = tris.get(name, [])
        rgba = compose(ts, pics, n, parts)
        if not isinstance(pics, Image.Image):
            rgba[..., 3] = 255  # paintings are opaque; materials() may cut feather tips
        mask = coverage(ts, n)
        if mask.any():
            # an island's outermost pixels are where a template's outline survives; paint them
            # again from inside
            edge = max(1, n // 256)
            rgba = _spread(rgba, _erode(mask, edge), px=max(4, n // 64) + edge)
        if check:
            seen = Image.fromarray(rgba).convert('RGBA')
            d = ImageDraw.Draw(seen)
            for t in ts:
                poly = _poly(t, n)
                d.line(poly + [poly[0]], fill=(0, 0, 0, 160), width=1)
            Path(check).mkdir(parents=True, exist_ok=True)
            seen.save(Path(check) / f'{name.replace(" ", "_")}.check.png')
        rgba = rgba.astype(np.float32)
        rgba[..., :3] *= gain
        swaps[info.section.start] = (info, tx.encode(rgba, info.dxt_type), n)

    out = bytearray()
    for s in parse_sections(data):
        raw = data[s.start:s.start + s.size]
        if s.start in swaps:
            info, blocks, n = swaps[s.start]
            head = bytearray(data[s.data_start:s.data_start + 0x45])
            step = 8 if info.dxt_type == '1TXD' else 16
            struct.pack_into('<II', head, 0x15, n, n)
            struct.pack_into('<II', head, 0x3D, len(blocks), (n // 4) * step)
            body = bytes(head) + blocks
            body += b'\0' * (-len(body) % 16)
            meta = struct.unpack_from('<I', raw, 4)[0]
            meta = (meta & 0x7F) | (((16 + len(body)) // 16) << 7)
            raw = raw[:4] + struct.pack('<I', meta) + raw[8:16] + body
        out += raw
    return bytes(out)


def _raster(tri2d, w, h):
    """Pixels inside a 2D triangle: (ys, xs, barycentric weights (k, 3))."""
    (x0, y0), (x1, y1), (x2, y2) = tri2d
    lo_x, hi_x = max(int(np.floor(min(x0, x1, x2))), 0), min(int(np.ceil(max(x0, x1, x2))), w - 1)
    lo_y, hi_y = max(int(np.floor(min(y0, y1, y2))), 0), min(int(np.ceil(max(y0, y1, y2))), h - 1)
    if lo_x > hi_x or lo_y > hi_y:
        return None
    ys, xs = np.mgrid[lo_y:hi_y + 1, lo_x:hi_x + 1]
    px, py = xs.ravel() + 0.5, ys.ravel() + 0.5
    d = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(d) < 1e-12:
        return None
    a = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / d
    b = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / d
    c = 1 - a - b
    eps = -1e-4
    inside = (a >= eps) & (b >= eps) & (c >= eps)
    if not inside.any():
        return None
    return ys.ravel()[inside], xs.ravel()[inside], np.stack([a, b, c], 1)[inside]


def surface(tris, n):
    """Per texel of an (n, n) texture: where it sits on the body. Returns (mask, positions
    (n, n, 3), normals (n, n, 3), joints (n, n), shared (n, n)), each texel taken from the
    triangle that covers it (the last one drawn, where islands overlap). shared marks texels
    that two places on the body use (a front and a hind paw, say): whatever is painted there
    shows in both."""
    mask = np.zeros((n, n), bool)
    shared = np.zeros((n, n), bool)
    pos = np.zeros((n, n, 3), np.float32)
    nrm = np.zeros((n, n, 3), np.float32)
    jnt = np.full((n, n), -1, np.int32)
    for t in tris:
        hit = _raster(_poly(t, n), n, n)
        if hit is None:
            continue
        ys, xs, w = hit
        p = w @ t[4]
        shared[ys, xs] |= mask[ys, xs] & (np.linalg.norm(pos[ys, xs] - p, axis=1) > 0.05)
        mask[ys, xs] = True
        pos[ys, xs] = p
        nn = w @ t[5]
        nrm[ys, xs] = nn / np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1e-6)
        jnt[ys, xs] = t[3]
    # texels just outside an island (seen by filtering) take their nearest neighbour's place
    for _ in range(4):
        grow = ~mask
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            src = np.roll(np.roll(mask, dy, 0), dx, 1) & grow
            if src.any():
                pos[src] = np.roll(np.roll(pos, dy, 0), dx, 1)[src]
                nrm[src] = np.roll(np.roll(nrm, dy, 0), dx, 1)[src]
                jnt[src] = np.roll(np.roll(jnt, dy, 0), dx, 1)[src]
                shared[src] = np.roll(np.roll(shared, dy, 0), dx, 1)[src]
                mask = mask | src
                grow = ~mask
    return mask, pos, nrm, jnt, shared


def occlusion(tris, n, rays=48, reach=0.1, ground_reach=0.02, seed=0, blockers=None):
    """(n, n) ambient light per texel, 0 (shut in) to 1 (open sky): how much of the sky each
    point of the body sees past the rest of the body (both sides) and the ground under its feet,
    within reach and ground_reach (fractions of the model's length). Worked out at the
    triangles' corners and spread across them. `blockers`: the whole model's triangles, when
    `tris` are only one texture's (a neck must see the body it joins, or a seam shows)."""
    P = np.array([t[4] for t in tris], np.float64)
    B = np.array([t[4] for t in blockers], np.float64) if blockers else P
    size = np.ptp(B.reshape(-1, 3), 0).max()
    reach, ground_reach = reach * size, ground_reach * size
    N = np.array([t[5] for t in tris], np.float64)
    both = np.concatenate([B, B * np.array([1, 1, -1])])
    v0, e1, e2 = both[:, 0], both[:, 1] - both[:, 0], both[:, 2] - both[:, 0]
    ground = B[..., 1].max()  # +y is down; the lowest point is where the feet stand
    rng = np.random.default_rng(seed)
    # cosine-weighted directions about +z, turned onto each normal below
    u1, u2 = rng.random(rays), rng.random(rays)
    r, phi = np.sqrt(u1), 2 * np.pi * u2
    local = np.stack([r * np.cos(phi), r * np.sin(phi), np.sqrt(1 - u1)], 1)

    def corner(p, nrm):
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
        a = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        t1 = np.cross(nrm, a); t1 /= np.linalg.norm(t1)
        t2 = np.cross(nrm, t1)
        d = local @ np.stack([t1, t2, nrm])
        o = p + nrm * 0.01
        pv = np.cross(d[:, None, :], e2[None])                       # (rays, tris, 3)
        det = np.einsum('tk,rtk->rt', e1, pv)
        inv = 1 / np.where(np.abs(det) < 1e-12, np.inf, det)
        tv = o - v0
        uu = np.einsum('tk,rtk->rt', tv, pv) * inv
        qv = np.cross(tv, e1)
        vv = np.einsum('rk,tk->rt', d, qv) * inv
        tt = np.einsum('tk,tk->t', e2, qv)[None] * inv
        hit = (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (tt > 1e-4) & (tt < reach)
        blocked = hit.any(1)
        down = d[:, 1] > 1e-6
        blocked |= down & ((ground - o[1]) / np.where(down, d[:, 1], 1) < ground_reach)
        return 1 - blocked.mean()

    cache = {}
    ao = np.zeros(P.shape[:2])
    for i in range(len(P)):
        for k in range(3):
            key = tuple(np.round(P[i, k], 3)) + tuple(np.round(N[i, k], 2))
            if key not in cache:
                cache[key] = corner(P[i, k], N[i, k])
            ao[i, k] = cache[key]

    img = np.ones((n, n), np.float32)
    have = np.zeros((n, n), bool)
    for t, a in zip(tris, ao):
        hit = _raster(_poly(t, n), n, n)
        if hit is None:
            continue
        ys, xs, w = hit
        img[ys, xs] = w @ a
        have[ys, xs] = True
    # soften the triangle facets, inside the islands only
    from scipy import ndimage
    sig = n / 160
    num = ndimage.gaussian_filter(img * have, sig)
    den = ndimage.gaussian_filter(have.astype(np.float32), sig)
    return np.where(have, num / np.maximum(den, 1e-6), img)

def _corners_to_texels(tris, values, n, blur=160):
    """(n, n) from a value per triangle corner, spread across each triangle and softened
    inside the islands."""
    from scipy import ndimage
    img = np.zeros((n, n), np.float32)
    have = np.zeros((n, n), bool)
    for t, v in zip(tris, values):
        hit = _raster(_poly(t, n), n, n)
        if hit is None:
            continue
        ys, xs, w = hit
        img[ys, xs] = w @ v
        have[ys, xs] = True
    sig = n / blur
    num = ndimage.gaussian_filter(img * have, sig)
    den = ndimage.gaussian_filter(have.astype(np.float32), sig)
    return np.where(have, num / np.maximum(den, 1e-6), img)


def curvature(tris, n):
    """(n, n) surface curvature per texel: above 0 on convex shapes (ridges, shoulders,
    knuckles), below 0 in creases and hollows, about 1 for a sharp edge. From how the normals
    turn between each corner and the corners it shares edges with."""
    P = np.array([t[4] for t in tris], np.float64)
    N = np.array([t[5] for t in tris], np.float64)
    key = lambda p: tuple(np.round(p, 3))
    total, count = {}, {}
    for i in range(len(P)):
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                dp = P[i, b] - P[i, a]
                L2 = dp @ dp
                if L2 < 1e-10:
                    continue
                k = (N[i, b] - N[i, a]) @ dp / L2
                total[key(P[i, a])] = total.get(key(P[i, a]), 0.0) + k
                count[key(P[i, a])] = count.get(key(P[i, a]), 0) + 1
    vals = np.array([[total[key(p)] / count[key(p)] if key(p) in count else 0.0 for p in tri] for tri in P])
    scale = np.percentile(np.abs(vals), 90) or 1.0
    return _corners_to_texels(tris, np.clip(vals / scale, -1, 1), n)

def flow(tris, n, along=(-1.0, 0.8, 0.0)):
    """(n, n, 2) unit directions in texture space (x, y in texels) that fur lies along: the 3D
    direction `along` (default back and down; +x is forward, +y down) laid onto each triangle
    and carried into its UVs."""
    out = np.zeros((n, n, 2), np.float32)
    f3 = np.array(along, np.float64)
    for t in tris:
        P = np.asarray(t[4], np.float64)
        uv = np.array([_px(c, n) for c in t[:3]], np.float64)
        e = np.stack([P[1] - P[0], P[2] - P[0]], 1)            # 3x2
        d = np.stack([uv[1] - uv[0], uv[2] - uv[0]], 1)        # 2x2 (texels)
        if abs(np.linalg.det(d)) < 1e-9:
            continue
        J = e @ np.linalg.inv(d)                               # 3D change per texel step
        nrm = np.cross(e[:, 0], e[:, 1])
        nrm /= max(np.linalg.norm(nrm), 1e-12)
        f = f3 - (f3 @ nrm) * nrm
        ab, *_ = np.linalg.lstsq(J, f, rcond=None)
        L = np.linalg.norm(ab)
        if L < 1e-12:
            continue
        hit = _raster(_poly(t, n), n, n)
        if hit is None:
            continue
        ys, xs, _ = hit
        out[ys, xs] = ab / L
    # one direction per triangle jumps at every edge: smooth it across the islands
    from scipy import ndimage
    have = (np.abs(out).sum(-1) > 0).astype(np.float32)
    sig = n / 100
    den = np.maximum(ndimage.gaussian_filter(have, sig), 1e-6)
    sm = np.stack([ndimage.gaussian_filter(out[..., k], sig) / den for k in (0, 1)], -1)
    sm /= np.maximum(np.linalg.norm(sm, axis=-1, keepdims=True), 1e-6)
    return np.where(have[..., None] > 0, sm, out).astype(np.float32)


def fur(tris, n, length=0.04, along=(-1.0, 0.8, 0.0), clump=0.012, stroke=0.004, seed=0):
    """(n, n) fur in 0..1 around 0.5: strokes `stroke` of the texture's width across, smeared
    along the flow over `length` of it, under soft clumps `clump` of its width across."""
    from scipy import ndimage
    rng = np.random.default_rng(seed)
    field = flow(tris, n, along)
    noise = ndimage.gaussian_filter(rng.random((n, n)).astype(np.float32), max(0.7, stroke * n))
    steps = max(2, int(length * n))
    ys, xs = np.mgrid[0:n, 0:n].astype(np.float32)
    acc = noise.copy()
    wsum = np.ones((n, n), np.float32)
    for sign in (1, -1):
        y, x = ys.copy(), xs.copy()
        for k in range(1, steps + 1):
            iy, ix = np.clip(y.astype(int), 0, n - 1), np.clip(x.astype(int), 0, n - 1)
            d = field[iy, ix]
            x += sign * d[..., 0]
            y += sign * d[..., 1]
            w = 1 - k / (steps + 1)  # strands fade towards their ends
            acc += w * noise[np.clip(y.astype(int), 0, n - 1), np.clip(x.astype(int), 0, n - 1)]
            wsum += w
    strands = acc / wsum
    strands = (strands - strands.mean()) / max(strands.std(), 1e-6)
    lumps = ndimage.gaussian_filter(rng.random((n, n)).astype(np.float32), clump * n)
    lumps = (lumps - lumps.mean()) / max(lumps.std(), 1e-6)
    return 0.5 + 0.12 * strands + 0.1 * lumps

def free_edges(tris):
    """[(triangle index, corner a, corner b)] for edges no other triangle shares in 3D: the
    edges of wings, feather cards and frills. Edges on the mirror plane (z = 0) join the other
    side, so they don't count."""
    key = lambda p: tuple(np.round(p, 3))
    count = {}
    for i, t in enumerate(tris):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = frozenset((key(t[4][a]), key(t[4][b])))
            count[e] = count.get(e, 0) + 1
    out = []
    for i, t in enumerate(tris):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            pa, pb = t[4][a], t[4][b]
            if abs(pa[2]) < 1e-3 and abs(pb[2]) < 1e-3:
                continue
            if count[frozenset((key(pa), key(pb)))] == 1:
                out.append((i, a, b))
    return out


def cutouts(tris, n, P, owner, parts, depth=0.03, width=0.04, along=(-1.0, 0.2, 0.4), seed=0):
    """(n, n) alpha: 0 where feather tips are cut into the free edges of the named parts, 255
    elsewhere. depth: how far the notches reach in from the edge, width: how far apart the tips
    are (both in the model's units). Only edges facing downstream along `along` (where feathers
    end) are cut, only within the triangle the edge belongs to, and never deeper than half of it."""
    alpha = np.full((n, n), 255, np.uint8)
    want = np.isin(owner, parts)
    rng = np.random.default_rng(seed)
    for i, a, b in free_edges(tris):
        t = tris[i]
        pa, pb = np.asarray(t[4][a]), np.asarray(t[4][b])
        pc = np.asarray(t[4][3 - a - b])
        e = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
        out_dir = (pa + pb) / 2 - pc
        out_dir = out_dir - (out_dir @ e) * e
        h = np.linalg.norm(out_dir)
        if h < 1e-9:
            continue
        fl = np.array(along, np.float64)
        fl = fl / np.linalg.norm(fl)
        # mirrored triangles share these texels: count the edge if it faces downstream either way
        facing = max(out_dir @ fl, (out_dir * [1, 1, -1]) @ fl) / h
        if facing < 0.35:
            continue
        edge_depth = min(depth, 0.5 * h)
        hit = _raster(_poly(t, n), n, n)
        if hit is None:
            continue
        ys, xs, _ = hit
        keep = want[ys, xs]
        ys, xs = ys[keep], xs[keep]
        if not len(ys):
            continue
        q = P[ys, xs]
        d = pb - pa
        L = np.linalg.norm(d)
        if L < 1e-9:
            continue
        d = d / L
        s = np.clip((q - pa) @ d, 0, L)
        dist = np.linalg.norm(q - (pa + s[:, None] * d), axis=1)
        near = dist < edge_depth
        if not near.any():
            continue
        # V-shaped notches between rounded tips, a little uneven
        phase = rng.uniform(0, 1)
        f = (s / width + phase) % 1.0
        tooth = 1 - np.abs(f - 0.5) * 2
        reach = edge_depth * (0.25 + 0.75 * tooth ** 0.7) * (0.85 + 0.3 * rng.random())
        cut = near & (dist < reach)
        alpha[ys[cut], xs[cut]] = 0
    return alpha

def cells(P, size, seed=0):
    """Borders between random cells about `size` across in 3D: 0 at a border, rising to 1 in a
    cell's middle. Laid on the body, they read as scales or plates."""
    q = P / size
    base = np.floor(q).astype(np.int64)
    f1 = np.full(P.shape[:-1], np.inf)
    f2 = np.full(P.shape[:-1], np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = base + np.array([dx, dy, dz])
                h = (c[..., 0] * 73856093) ^ (c[..., 1] * 19349663) ^ (c[..., 2] * 83492791) ^ seed
                jit = np.stack([((h >> k) & 1023) / 1023.0 for k in (0, 10, 20)], -1)
                dist = np.linalg.norm(q - (c + jit), axis=-1)
                f2 = np.minimum(f2, np.maximum(f1, dist))
                f1 = np.minimum(f1, dist)
    return np.clip((f2 - f1) * 2.5, 0, 1).astype(np.float32)

def colour_mask(rgb, sat_max=0.3, val_min=0.1, hue=None, smooth=0.005, min_area=0.001, fill=0.002, sat_min=0.0):
    """(n, n) bool: where a texture (n, n, 3 in 0..1, as the client shows it) is greyer than
    sat_max but no greyer than sat_min, and brighter than val_min (and within hue [from, to]
    degrees, if given: a colour, with sat_min keeping greys out), as clean
    regions: smoothed by `smooth` of its width first, specks under `min_area` of it dropped and
    holes under `fill` of it filled."""
    from scipy import ndimage
    n = rgb.shape[0]
    a = ndimage.gaussian_filter(rgb, (smooth * n, smooth * n, 0))
    mx, mn = a.max(-1), a.min(-1)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    m = (sat <= sat_max) & (sat >= sat_min) & (mx >= val_min)
    if hue:
        d = mx - mn + 1e-6
        r, g, b = a[..., 0], a[..., 1], a[..., 2]
        h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 60
        lo, hi = hue
        m &= ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))
    m = ndimage.binary_opening(m, iterations=2)
    m = ndimage.binary_closing(m, iterations=3)
    lab, k = ndimage.label(m)
    if k:
        sizes = ndimage.sum(m, lab, range(1, k + 1))
        m = np.isin(lab, 1 + np.nonzero(sizes >= min_area * n * n)[0])
    holes = ndimage.binary_fill_holes(m) & ~m
    hl, hk = ndimage.label(holes)
    if hk:
        hs = ndimage.sum(holes, hl, range(1, hk + 1))
        m |= np.isin(hl, 1 + np.nonzero(hs < fill * n * n)[0])
    return m


def _nearest_point(P, size, seed=0):
    """Distance from each 3D point to the nearest of a scatter of points about `size` apart."""
    q = P / size
    base = np.floor(q).astype(np.int64)
    f1 = np.full(P.shape[:-1], np.inf)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = base + np.array([dx, dy, dz])
                h = (c[..., 0] * 73856093) ^ (c[..., 1] * 19349663) ^ (c[..., 2] * 83492791) ^ seed
                jit = np.stack([((h >> k) & 1023) / 1023.0 for k in (0, 10, 20)], -1)
                f1 = np.minimum(f1, np.linalg.norm(q - (c + jit), axis=-1))
    return f1 * size

def _smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _noise(P, freq, seed=0):
    """Smooth noise in -1..1 over 3D positions: a few sine waves in random directions."""
    rng = np.random.default_rng(seed)
    out = np.zeros(P.shape[:-1], np.float32)
    for _ in range(4):
        k = rng.normal(size=3)
        k *= freq / np.linalg.norm(k)
        out += np.sin(P @ k + rng.uniform(0, 2 * np.pi))
    return out / 2


def _triplanar(img, P, N, scale):
    """Sample a tileable picture onto 3D points from three sides, blended by the normal."""
    h, w = img.shape[:2]

    def sample(u, v):
        x, y = (u / scale * w) % w, (v / scale * h) % h
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        fx, fy = (x - x0)[..., None], (y - y0)[..., None]
        x0, y0 = x0 % w, y0 % h
        x1, y1 = (x0 + 1) % w, (y0 + 1) % h
        return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x1] * fx * (1 - fy) +
                img[y1, x0] * (1 - fx) * fy + img[y1, x1] * fx * fy)

    wt = np.abs(N) ** 4
    wt /= np.maximum(wt.sum(-1, keepdims=True), 1e-6)
    return (sample(P[..., 2], P[..., 1]) * wt[..., 0:1] +
            sample(P[..., 0], P[..., 2]) * wt[..., 1:2] +
            sample(P[..., 0], P[..., 1]) * wt[..., 2:3])


def _colour(hexstr):
    h = hexstr.lstrip('#')
    return np.array([int(h[k:k + 2], 16) for k in (0, 2, 4)], np.float32)


AXES = {'x': 0, 'y': 1, 'z': 2}


def materials(tris, n, spec, parts=None, here=Path('.'), original=None, everything=None):
    """An (n, n, 4) texture built from materials laid over the body in 3D. Nothing of the
    original texture is used unless spec "base" is "original" (pass it as `original`, the
    texture as stored): then ours goes over it, and "masks" {name: {"sat_max", "val_min",
    "hue", "include", "exclude"}} pick regions of it by colour (its metal, say) that layers can be limited
    to with "mask", and to a band in from the mask's edge with "edge": [from, to], or dot with
    "rivets" {"inset", "spacing", "radius"}. A layer's "gloss" adds a highlight from the light.

    spec: {"layers": [...], "shade": 0..1}. Each layer is "image" (a tileable picture, laid on by
    triplanar projection, "scale" world units per repeat, "tint" its average colour and
    "contrast" how strongly its light and dark show), "colorize" (the texture beneath recoloured,
    its light and dark kept; "gain" scales it) or
    "color", and is limited by any of: "parts" (names from parts), "facing" ("up"/"down", how
    sharply with "soft": [from, to]), "stripes" ({"axis", "every", "width", "warp", "taper",
    "seed", "soft", "rough"}: bands across that axis, thinner on the underside with taper, soft
    and rough at their edges; left off texels two
    places on the body share, where they could only line up with one). "opacity" scales it.
    A picture's "soften" blurs it over that fraction of the tile first, keeping clumps and losing
    grain. The skeleton stands with +x forward and -y up. "shade" bakes light from above into it,
    "occlusion" shadow into creases and under the body (both 0..1), "curvature" lightens
    convex shapes and darkens hollows by up to that fraction. "cutouts" {"parts", "depth",
    "width"} cuts feather tips into those parts' free edges (alpha)."""
    mask, P, N, J, shared = surface(tris, n)
    up = -N[..., 1]
    owner = np.full((n, n), '', object)
    for part, ts in split(tris, parts).items():
        owner[coverage(ts, n)] = part
    img = np.full((n, n, 3), 128, np.float32)
    keep = spec.get('base') == 'original' and original is not None
    if keep:
        # the clone's own texture under ours, as the client shows it (twice as stored)
        orig = np.array(Image.fromarray(original).resize((n, n), Image.LANCZOS)).astype(np.float32)
        img = np.clip(orig[..., :3] * 2, 0, 255)
    painted = np.zeros((n, n), np.float32)
    masks = {}
    for name, m in spec.get('masks', {}).items():
        if not keep:
            raise ValueError('masks read the original texture: set "base": "original"')
        masks[name] = colour_mask(img / 255, m.get('sat_max', 0.3), m.get('val_min', 0.1), m.get('hue'),
                                  m.get('smooth', 0.005), m.get('min_area', 0.001), m.get('fill', 0.002),
                                  m.get('sat_min', 0.0))
        if 'include' in m:
            masks[name] |= np.isin(owner, m['include'])
        if 'exclude' in m:
            masks[name] &= ~np.isin(owner, m['exclude'])
    for layer in spec.get('layers', []):
        if 'scales' in layer:
            sc = layer['scales'] if isinstance(layer['scales'], dict) else {}
            v = cells(P, sc.get('size', 0.03), sc.get('seed', 0))
            rel = (v - 0.6) * layer.get('contrast', 1.0)
            col = _colour(layer.get('tint', '#808080')) * (1 + rel[..., None])
        elif 'fur' in layer:
            fu = layer['fur'] if isinstance(layer['fur'], dict) else {}
            v = fur(tris, n, fu.get('length', 0.04), fu.get('along', (-1.0, 0.8, 0.0)),
                    fu.get('clump', 0.012), fu.get('stroke', 0.004), fu.get('seed', 0))
            rel = (v - 0.5) * 2 * layer.get('contrast', 1.0)
            col = _colour(layer.get('tint', '#808080')) * (1 + rel[..., None])
        elif 'image' in layer:
            pic = np.array(Image.open(here / layer['image']).convert('RGB'), np.float32)
            if layer.get('soften'):
                # keep the picture's large clumps, lose its grain: a blur over that fraction of
                # the tile, wrapping so it still tiles
                from scipy import ndimage
                sig = layer['soften'] * pic.shape[0]
                pic = ndimage.gaussian_filter(pic, (sig, sig, 0), mode='wrap')
            col = _triplanar(pic, P, N, layer.get('scale', 1.0))
            if 'tint' in layer:
                # the tint is the layer's average colour; the picture adds its light and dark
                # around it, "contrast" times as strongly
                rel = col / max(float(pic.mean()), 1.0) - 1
                col = _colour(layer['tint']) * (1 + layer.get('contrast', 1.0) * rel)
        elif 'colorize' in layer:
            # the texture beneath in a new colour: its light and dark kept, its hue replaced
            # ("gain" brightens or darkens the result)
            lum = img @ np.array([0.3, 0.59, 0.11], np.float32)
            col = _colour(layer['colorize']) * (lum / 128.0 * layer.get('gain', 1.0))[..., None]
        else:
            col = np.broadcast_to(_colour(layer['color']), img.shape).astype(np.float32)
        a = np.full((n, n), float(layer.get('opacity', 1.0)), np.float32)
        if 'parts' in layer:
            a *= np.isin(owner, layer['parts'])
        if 'mask' in layer:
            from scipy import ndimage
            m = masks[layer['mask']]
            a *= m
            band = layer.get('edge')
            if band:
                # a band [from, to] (fractions of the width) in from the mask's edge: seams, bevels
                dt = ndimage.distance_transform_edt(m) / n
                a *= _smooth(band[0] - 0.002, band[0], dt) * (1 - _smooth(band[1], band[1] + 0.002, dt))
            rv = layer.get('rivets')
            if rv:
                dt = ndimage.distance_transform_edt(m) / n
                inset, size = rv.get('inset', 0.012), rv.get('size', 0.004)
                on_line = np.abs(dt - inset) < size
                dd = _nearest_point(P, rv.get('spacing', 0.06), rv.get('seed', 0))
                r3 = rv.get('radius', 0.012)
                dome = np.clip(1 - (dd / r3) ** 2, 0, 1)
                a *= on_line * (dd < r3)
                col = col * (0.55 + 0.6 * dome[..., None])
        if 'facing' in layer:
            lo, hi = layer.get('soft', [0.0, 0.5])
            a *= _smooth(lo, hi, up if layer['facing'] == 'up' else -up)
        st = layer.get('stripes')
        if st:
            ax = P[..., AXES[st.get('axis', 'x')]]
            f = (ax + st.get('warp', 0.1) * _noise(P, 6.0, st.get('seed', 0))) / st.get('every', 0.3)
            dist = np.abs((f % 1.0) - 0.5) * 2  # 1 at a stripe's centre line, 0 halfway between
            dist = dist + st.get('rough', 0.0) * _noise(P, 40.0, st.get('seed', 0) + 2)
            width = st.get('width', 0.35) * (_smooth(-0.45, 0.5, up) if st.get('taper') else 1)
            width = width * (1 + 0.5 * _noise(P, 9.0, st.get('seed', 0) + 1))
            # a stripe too thin to read is a crack: let it go instead
            soft = st.get('soft', 0.08)
            a *= _smooth(1 - width - soft, 1 - width + soft, dist) * _smooth(0.08, 0.16, width) * ~shared
        if layer.get('gloss'):
            light = np.array([0.35, -1.0, 0.25])
            light /= np.linalg.norm(light)
            col = col + 255 * layer['gloss'] * np.clip(N @ light, 0, 1)[..., None] ** layer.get('power', 8)
        img = img * (1 - a[..., None]) + col * a[..., None]
        painted = np.maximum(painted, a)
    shade = np.ones((n, n), np.float32)
    ak = spec.get('occlusion', 0)
    if ak:
        shade *= (1 - ak) + ak * occlusion(tris, n, blockers=everything)
    ck = spec.get('curvature', 0)
    if ck:
        shade *= 1 + ck * curvature(tris, n)
    k = spec.get('shade', 0.6)
    if k:
        light = np.array([0.35, -1.0, 0.25])
        light /= np.linalg.norm(light)
        lit = 0.45 + 0.75 * np.clip(N @ light, 0, 1) + 0.15 * up
        shade *= (1 - k) + k * lit
    # the original is shaded already: only our paint takes the bake
    w = painted if keep else 1.0
    img *= (1 + w * (shade - 1))[..., None]
    alpha = np.full((n, n), 255, np.float32)
    if keep:
        alpha = orig[..., 3].copy()
    cu = spec.get('cutouts')
    if cu:
        alpha = cutouts(tris, n, P, owner, cu['parts'], cu.get('depth', 0.03), cu.get('width', 0.04),
                        cu.get('along', (-1.0, 0.2, 0.4)), cu.get('seed', 0)).astype(np.float32)
    out = np.concatenate([np.clip(img, 0, 255), alpha[..., None]], -1)
    return out.astype(np.uint8)


def preview(data, out, texture=None, size=520, gain=2.0):
    """A software render of the model in bind pose, both sides, wearing its texture as stored
    (times `gain`, as the client draws it): a side view and a three-quarter view side by side."""
    tris_by = triangles(bytes(data))
    infos = textures(bytes(data))
    def yaw(deg):
        a = np.radians(deg)
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def pitch(deg):
        a = np.radians(deg)
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])

    # side; front three-quarter from a little above
    views = [np.eye(3), pitch(-20) @ yaw(-40)]
    light = np.array([0.3, -1.0, 0.4])
    light /= np.linalg.norm(light)
    all_p = np.concatenate([t[4] for ts in tris_by.values() for t in ts])
    centre = (all_p.max(0) + all_p.min(0)) / 2
    centre[2] = 0  # the mirror plane
    radius = np.linalg.norm(all_p - centre, axis=1).max()
    panels = []
    for rot in views:
        img = np.full((size, size, 3), 60, np.float32)
        zbuf = np.full((size, size), np.inf)
        for name, tris in tris_by.items():
            info = infos.get(name)
            if info is not None and info.tex_type == 0xA1:
                tex = tx.decode(data[info.data_offset:info.data_offset + info.data_size],
                                info.width, info.height, info.dxt_type).astype(np.float32)
                tex[..., :3] *= gain
            else:
                tex = np.full((4, 4, 4), 160, np.float32)
            th, tw = tex.shape[:2]
            for mirror in (1, -1):
                for t in tris:
                    P = (t[4] - centre) * np.array([1, 1, mirror])
                    N = t[5] * np.array([1, 1, mirror])
                    V = P @ rot.T
                    s2 = size * 0.45 / radius
                    tri2d = [(size / 2 + v[0] * s2, size / 2 + v[1] * s2) for v in V]
                    hit = _raster(tri2d, size, size)
                    if hit is None:
                        continue
                    ys, xs, w = hit
                    uv = w @ np.array(t[:3])
                    u = (uv[:, 0] * tw).astype(int).clip(0, tw - 1)
                    v = (uv[:, 1] * th).astype(int).clip(0, th - 1)
                    z = w @ V[:, 2]
                    near = (z < zbuf[ys, xs]) & (tex[v, u, 3] >= 128)  # alpha test, as the client
                    ys, xs, w, z, u, v = ys[near], xs[near], w[near], z[near], u[near], v[near]
                    zbuf[ys, xs] = z
                    n = w @ N
                    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-6)
                    lit = 0.55 + 0.45 * np.abs(n @ light)
                    img[ys, xs] = tex[v, u, :3] * lit[:, None]
        panels.append(np.clip(img, 0, 255).astype(np.uint8))
    Image.fromarray(np.concatenate(panels, 1)).save(out)


if __name__ == '__main__':
    # python3 modules/mobs/skin.py template <model id> <out dir> [--parts parts.json] [--size N]
    import argparse
    import json
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # modules
    from xidat.paths import ffxi_dir, hd
    from xidat.ftable import load_all, resolve_model

    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['template'])
    ap.add_argument('model', type=int)
    ap.add_argument('out')
    ap.add_argument('--parts', help='JSON with {"parts": {name: [joints]}}')
    ap.add_argument('--size', type=int, default=1024)
    a = ap.parse_args()
    src = hd(resolve_model(a.model, load_all(ffxi_dir()), ffxi_dir()))
    parts = json.loads(Path(a.parts).read_text())['parts'] if a.parts else None
    for tex, legend in template(src, a.out, a.size, parts).items():
        print(f'{tex!r}: ' + ', '.join(f'{p} = {c}' for p, c in legend.items()))

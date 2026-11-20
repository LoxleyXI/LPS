"""Our zone DAT writer: a base zone's DAT with its geometry replaced by generated meshes.

build_zone(base_dat, geometry) keeps what makes the base zone the kind of zone it is (weather and
environment, textures, sounds, fight resources, zone-line volumes, point-light slots, directory
tree) and replaces its geometry:
  * the base zone's 0x2E meshes under "mode" go; ours come in as one 0x2E per 64-yalm tile,
  * a fresh 0x1C ZoneDef: one placement per tile, a quadtree over them, one culling table that
    lists every placement (visible from anywhere), one collision transform per placement, and a
    collision grid built from our collision mesh,
  * the "effe" tree goes: its generators autorun on zone load at the old arena's positions
    (several sit inside our entrance room). Textures we use from it are copied into "mode".

Format facts, learned from the retail DATs (zones 203, 206, 245) and community notes:
  * Sections: 16-byte header, 4-char name, u32 meta = type (low 7 bits) | size/16 << 7.
  * 0x2E and 0x1C bodies are encrypted with key tables that live in FFXiMain.dll; we read the DLL
    from the user's install at run time and never store the tables.
  * FFXI world axes: X east, Y DOWN (negative is up), Z north. Our cave frame is x east, y north,
    z up, so (x, y, z) -> (x, -z, y), a proper rotation (det +1).
  * Front faces: in every retail render and collision triangle the stored normal points opposite
    (b-a)x(c-a). Our input winds (b-a)x(c-a) into the air, so we write (a, c, b).
  * Collision: retail rock (walls, floors, ceilings alike) is single sided, stone terrain, no
    hitWall bit, in camera-blocking (flags 0) meshes; the hitWall bit only ever appears in flags 1
    (camera transparent) meshes. Each grid group's u32 is (row << 23) | (col << 14) | pair count.
"""
import math
import struct
from pathlib import Path

import numpy as np

from xidat.paths import ffxi_dir  # noqa: F401  (zonekit.writer.ffxi_dir, as before)

# ---------------------------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------------------------

T_END, T_DIR, T_ZONEDEF, T_TEXTURE, T_MESH, T_RID = 0x00, 0x01, 0x1C, 0x20, 0x2E, 0x36
T_ENV = 0x2F                                   # environment: lighting, fog, draw distance (unencrypted)


class Sec:
    """One DAT section: its raw bytes (header included) plus parsed name/type."""
    __slots__ = ("raw", "name", "kind")

    def __init__(self, raw):
        self.raw = bytes(raw)
        self.name = self.raw[:4]
        self.kind = struct.unpack_from("<I", self.raw, 4)[0] & 0x7F

    def __repr__(self):
        return f"Sec({self.name!r}, 0x{self.kind:02X}, {len(self.raw)})"


def split_sections(dat):
    out, pos = [], 0
    while pos + 16 <= len(dat):
        meta = struct.unpack_from("<I", dat, pos + 4)[0]
        size = ((meta >> 7) & 0xFFFFF) * 16
        if size <= 0:
            raise ValueError(f"bad section size at 0x{pos:X}")
        out.append(Sec(dat[pos:pos + size]))
        pos += size
    if pos != len(dat):
        raise ValueError(f"trailing {len(dat) - pos} bytes after the last section")
    return out


def section_bytes(name, kind, body):
    """A section from a 4-byte name, a type and its body (everything after the 16-byte header).
    The body is zero padded to a multiple of 16."""
    body = bytes(body) + b"\0" * (-len(body) % 16)
    size = 16 + len(body)
    if size // 16 >= 1 << 20:
        raise ValueError("section too large for the 20-bit size field")
    head = bytes(name)[:4].ljust(4, b"\0") + struct.pack("<I", kind | ((size // 16) << 7)) + b"\0" * 8
    return head + body


def walk_tree(sections):
    """Yield (index, path) where path is the tuple of enclosing directory names."""
    stack = []
    for i, s in enumerate(sections):
        if s.kind == T_END:
            if stack:
                stack.pop()
            yield i, tuple(stack)
            continue
        yield i, tuple(stack)
        if s.kind == T_DIR:
            stack.append(s.name)


# ---------------------------------------------------------------------------------------------
# key tables and the two ciphers
# ---------------------------------------------------------------------------------------------

_KEY_SIGS = (bytes.fromhex("E2E506A9"), bytes.fromhex("B8C5F784"))
_key_cache = {}


def key_tables(dll=None):
    """(table_a, table_b): the two 256-byte tables, found in FFXiMain.dll by their first bytes."""
    dll = Path(dll) if dll else ffxi_dir() / "FFXiMain.dll"
    if dll not in _key_cache:
        blob = dll.read_bytes()
        tabs = []
        for sig in _KEY_SIGS:
            at = blob.find(sig, 0x30000)
            if at < 0:
                raise ValueError(f"zone key table {sig.hex()} not found in {dll}")
            tabs.append(blob[at:at + 256])
        _key_cache[dll] = tuple(tabs)
    return _key_cache[dll]


def _mesh_stream(body, ta):
    """0x2E layer 1: a keyed byte stream XORed over body[8:total]. Self inverse."""
    meta = struct.unpack_from("<I", body, 0)[0]
    if (meta >> 24) < 5:
        return
    total = meta & 0xFFFFFF
    key = ta[body[5] ^ 0xF0]
    step = 0
    for p in range(8, total):
        spread = (key & 0xFF) * 0x101
        step += 1
        key += step
        body[p] ^= (spread >> (key & 7)) & 0xFF
        step += 1
        key += step


def _mesh_shuffle(body, tb):
    """0x2E layer 2: 8-byte blocks of the first half swapped with the second half when the key
    bit says so (only when the u16 at +6 is 0xFFFF). Self inverse."""
    if struct.unpack_from("<H", body, 6)[0] != 0xFFFF:
        return
    total = struct.unpack_from("<I", body, 0)[0] & 0xFFFFFF
    half = ((total - 8) & ~0xF) // 2
    k1 = body[5] ^ 0xF0
    k2 = tb[k1]
    for off in range(8, 8 + half, 8):
        if k2 & 1:
            a, b = off, off + half
            body[a:a + 8], body[b:b + 8] = body[b:b + 8], body[a:a + 8]
        k1 += 9
        k2 += k1


def mesh_decrypt(body, tables):
    """In place on a 0x2E section body (bytes after the 16-byte header)."""
    _mesh_stream(body, tables[0])
    _mesh_shuffle(body, tables[1])


def mesh_encrypt(body, tables):
    _mesh_shuffle(body, tables[1])
    _mesh_stream(body, tables[0])


def _def_stream(body, ta):
    """0x1C layer 1: runs of 16..23 bytes inverted (XOR 0xFF) when the key's low bit is set."""
    length = struct.unpack_from("<I", body, 0)[0] & 0xFFFFFF
    key = ta[body[7] ^ 0xFF]
    length = min(length, len(body) - 8)
    p, done, step = 8, 0, 0
    while done < length:
        run = ((key >> 4) & 7) + 16
        if key & 1 and done + run < length:
            for q in range(p, p + run):
                body[q] ^= 0xFF
        p += run
        done += run
        step += 1
        key += step


def _def_names(body):
    count = struct.unpack_from("<I", body, 4)[0] & 0xFFFFFF
    for i in range(count):
        at = 0x20 + i * 0x64
        body[at:at + 16] = bytes(c ^ 0x55 for c in body[at:at + 16])


def zonedef_encrypted(body):
    return (struct.unpack_from("<I", body, 0)[0] >> 24) > 0x1A


def zonedef_decrypt(body, tables):
    if zonedef_encrypted(body):
        _def_stream(body, tables[0])
        _def_names(body)


def zonedef_encrypt(body, tables):
    if zonedef_encrypted(body):
        _def_names(body)
        _def_stream(body, tables[0])


def roundtrip_check(dat, tables=None):
    """Decrypt and re-encrypt every 0x2E and 0x1C section; returns (n_checked, n_exact)."""
    tables = tables or key_tables()
    n = ok = 0
    for s in split_sections(dat):
        if s.kind not in (T_MESH, T_ZONEDEF):
            continue
        body = bytearray(s.raw[16:])
        if s.kind == T_MESH:
            mesh_decrypt(body, tables)
            mesh_encrypt(body, tables)
        else:
            zonedef_decrypt(body, tables)
            zonedef_encrypt(body, tables)
        n += 1
        ok += bytes(body) == s.raw[16:]
    return n, ok


# ---------------------------------------------------------------------------------------------
# readers (used to learn from real zones and to check our own output)
# ---------------------------------------------------------------------------------------------

def read_mesh(body):
    """A decrypted 0x2E body -> {name, config, parts: [{texture, flags, pos, nrm, col, uv, idx}]}.
    Returns parts == [] for collision-only stubs (meshCount0 == 0)."""
    config = body[4]
    name = bytes(body[0x10:0x20])
    d = 0x20
    count0 = struct.unpack_from("<I", body, d)[0]
    out = {"name": name, "config": config, "author": bytes(body[8:0x10]), "parts": [],
           "bbox": struct.unpack_from("<6f", body, d + 4)}
    if count0 == 0:
        return out
    first, count1 = struct.unpack_from("<II", body, d + 0x1C)
    p = d + first
    blend = bool(config & 2)
    stride = 48 if blend else 36
    for _ in range(count1):
        tex = bytes(body[p:p + 16])
        nv, flags = struct.unpack_from("<HH", body, p + 16)
        p += 20
        v = np.frombuffer(bytes(body[p:p + nv * stride]), dtype=np.uint8).reshape(nv, stride)
        p += nv * stride
        f = v.view(np.float32) if stride % 4 == 0 else None
        if blend:
            pos, nrm, col, uv = f[:, 0:3], f[:, 6:9], v[:, 36:40], f[:, 10:12]
        else:
            pos, nrm, col, uv = f[:, 0:3], f[:, 3:6], v[:, 24:28], f[:, 7:9]
        ni, unk = struct.unpack_from("<HH", body, p)
        p += 4
        idx = np.frombuffer(bytes(body[p:p + ni * 2]), dtype=np.uint16).astype(np.int64)
        p += ni * 2
        p += -p % 4
        out["parts"].append({"texture": tex, "flags": flags, "pos": pos.copy(), "nrm": nrm.copy(),
                             "col": col.copy(), "uv": uv.copy(), "idx": idx, "unk": unk})
    return out


def part_triangles(part, strip):
    """Index triples of a sub-mesh (tri list, or strip with alternating winding)."""
    idx = part["idx"]
    if not strip:
        return idx[:len(idx) // 3 * 3].reshape(-1, 3)
    tris = []
    for k in range(len(idx) - 2):
        a, b, c = idx[k], idx[k + 1], idx[k + 2]
        if a == b or b == c or a == c:
            continue
        tris.append((a, b, c) if k % 2 == 0 else (b, a, c))
    return np.array(tris, dtype=np.int64).reshape(-1, 3)


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def read_zonedef(body):
    """A decrypted 0x1C body -> dict of its parts (offsets are body relative, like the file)."""
    n = _u32(body, 4) & 0xFFFFFF
    z = {"count": n, "meta": _u32(body, 0), "keyword": _u32(body, 4),
         "coll": _u32(body, 8), "blocks": tuple(body[0xC:0x10]), "tree": _u32(body, 0x10),
         "cull": _u32(body, 0x14), "lights": _u32(body, 0x18), "unk1c": _u32(body, 0x1C)}
    objs = []
    for i in range(n):
        r = 0x20 + i * 0x64
        objs.append({"name": bytes(body[r:r + 16]),
                     "pos": struct.unpack_from("<3f", body, r + 0x10),
                     "rot": struct.unpack_from("<3f", body, r + 0x1C),
                     "scale": struct.unpack_from("<3f", body, r + 0x28),
                     "effect": bytes(body[r + 0x34:r + 0x38]),
                     "lod": struct.unpack_from("<3f", body, r + 0x38),
                     "flags": bytes(body[r + 0x44:r + 0x48]),
                     "cull_link": _u32(body, r + 0x48), "env": bytes(body[r + 0x4C:r + 0x50]),
                     "file_link": _u32(body, r + 0x50),
                     "lights": struct.unpack_from("<4I", body, r + 0x54)})
    z["objects"] = objs
    tables = []
    if z["cull"]:
        p = z["cull"]
        for _ in range(_u32(body, p)):
            p += 4
            c = _u32(body, p)
            tables.append((p, list(struct.unpack_from(f"<{c}I", body, p + 4))))
            p += 4 * c
        z["cull_end"] = p + 4
    z["tables"] = tables
    nodes = []
    seen = set()

    def node(off, depth):
        if off in seen or off + 0x80 > len(body):
            return
        seen.add(off)
        corners = np.array(struct.unpack_from("<24f", body, off)).reshape(8, 3)
        ref, cnt = struct.unpack_from("<II", body, off + 0x60)
        kids = struct.unpack_from("<4I", body, off + 0x68)
        tail = struct.unpack_from("<2I", body, off + 0x78)
        members = list(struct.unpack_from(f"<{cnt}I", body, ref)) if ref and cnt and ref + 4 * cnt <= len(body) else []
        nodes.append({"off": off, "depth": depth, "corners": corners, "ref": ref, "count": cnt,
                      "kids": kids, "tail": tail, "members": members})
        for k in kids:
            if k:
                node(k, depth + 1)
    if z["tree"]:
        node(z["tree"], 0)
    z["nodes"] = nodes
    if z["coll"]:
        z["collision"] = read_collision(body, z["coll"], z["blocks"])
    return z


def read_collision(body, cb, blocks):
    h = struct.unpack_from("<8I", body, cb)
    c = {"header": h, "n_meshes": h[0], "first_mesh": h[1], "n_groups": h[2], "groups_off": h[3],
         "map_off": h[4], "tf_off": h[5], "idx_count": h[6]}
    meshes = []
    p = h[1]
    for _ in range(h[0]):
        po, no, io, tc, fl = struct.unpack_from("<IIIHH", body, p)
        meshes.append({"off": p, "pos_off": po, "nrm_off": no, "idx_off": io, "tris": tc, "flags": fl})
        p = io + tc * 8
    c["meshes"] = meshes
    c["mesh_end"] = p
    groups = []
    p = h[3]
    for _ in range(h[2]):
        cf = _u32(body, p)
        pairs = [struct.unpack_from("<II", body, p + 4 + 8 * j) for j in range(cf & 0x7FF)]
        groups.append({"off": p, "flags": cf, "pairs": pairs})
        p += 4 + 8 * len(pairs) + 4
    c["groups"] = groups
    c["groups_end"] = p
    zbx, zbz, bw, bl = blocks
    cols, rows = zbx * (bw // 4), zbz * (bl // 4)
    c["grid"] = (cols, rows)
    c["map"] = list(struct.unpack_from(f"<{cols * rows}I", body, h[4]))
    c["map_end"] = h[4] + 4 * cols * rows
    return c


def collision_mesh_arrays(body, m):
    nv = (m["nrm_off"] - m["pos_off"]) // 12
    nn = (m["idx_off"] - m["nrm_off"]) // 12
    pos = np.frombuffer(bytes(body[m["pos_off"]:m["pos_off"] + nv * 12]), dtype=np.float32).reshape(-1, 3)
    nrm = np.frombuffer(bytes(body[m["nrm_off"]:m["nrm_off"] + nn * 12]), dtype=np.float32).reshape(-1, 3)
    raw = np.frombuffer(bytes(body[m["idx_off"]:m["idx_off"] + m["tris"] * 8]), dtype=np.uint16).reshape(-1, 4)
    return pos, nrm, raw


# ---------------------------------------------------------------------------------------------
# axes
# ---------------------------------------------------------------------------------------------

def to_ffxi(v):
    """Cave frame (x east, y north, z up) -> FFXI (X east, Y down, Z north). A proper rotation."""
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], -1)


def from_ffxi(v):
    v = np.asarray(v, dtype=np.float64)
    return np.stack([v[..., 0], v[..., 2], -v[..., 1]], -1)


# ---------------------------------------------------------------------------------------------
# texture names
# ---------------------------------------------------------------------------------------------

def texture_sections(sections):
    """{16-byte texture name: section index} for every 0x20 section (first occurrence wins)."""
    out = {}
    for i, s in enumerate(sections):
        if s.kind == T_TEXTURE and len(s.raw) >= 0x21:
            out.setdefault(s.raw[0x11:0x21], i)
    return out


def resolve_texture(name, available):
    """A mesh's texture (short 'delyuka1' / 'ice01', or the full 16-char form) -> 16-byte name."""
    if isinstance(name, bytes):
        name = name.decode("ascii")
    name = str(name)
    if len(name) == 16 and name.encode("ascii") in available:
        return name.encode("ascii")
    short = name.strip().split()[-1] if name.strip() else ""
    hits = [full for full in available if full[8:16].decode("ascii", "replace").strip() == short]
    if len(hits) == 1:
        return hits[0]
    listing = ", ".join(sorted(repr(f.decode("ascii", "replace")) for f in available))
    if not hits:
        raise ValueError(f"texture {name!r} is not in the base DAT; available: {listing}")
    raise ValueError(f"texture {name!r} is ambiguous in the base DAT: {[h.decode() for h in hits]}")


# ---------------------------------------------------------------------------------------------
# building blocks of the new zone
# ---------------------------------------------------------------------------------------------

OBJ_SIZE, NODE_SIZE, XFORM_SIZE = 0x64, 0x80, 0xC0
LOD = (10.0, 100.0, 1000.0)                 # every retail placement we looked at uses these
OBJ_FLAGS = b"\x00\x00\x00\x01"             # the common retail value for terrain rock
XFORM_MISC = 2                              # +0xA4 on retail placements that carry geometry
TERRAIN_STONE = 5
MAX_COLLISION_GROUPS = 124103               # collision grid cells in use: the most any retail zone has
                                            # (Western Altepa Desert, 125; sixteen retail zones pass 60,000)


def _bbox6(p):
    lo, hi = p.min(0), p.max(0)
    return (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])


def mesh_section_body(name16, parts, key, author=b"lps\0\0\0\0\0", stub_box=(0.0,) * 6):
    """A decrypted 0x2E body. parts: [(texture16, pos, nrm, bgra, uv, tris, flags)] in the object's
    local FFXI frame, tris already wound the retail way; flags 0x8000 = alpha blended (water).
    Empty parts -> a collision-only stub."""
    out = bytearray(0x20)
    out[4:8] = bytes([0, key & 0xFF, 0xFF, 0xFF])          # tri list, not blended, key, layer 2 on
    out[8:16] = author[:8].ljust(8, b"\0")
    out[16:32] = name16
    if not parts:
        out += struct.pack("<I6f", 0, *stub_box)          # like retail 'grid': bbox, no parts
    else:
        allp = np.concatenate([p[1] for p in parts])
        box = _bbox6(allp)
        ntri = sum(len(p[5]) for p in parts)
        out += struct.pack("<I6fII6fI", 1, *box, 0x40, len(parts), *box, ntri)
        for tex, pos, nrm, col, uv, tris, pflags in parts:
            nv = len(pos)
            if nv > 0xFFFF or 3 * len(tris) > 0xFFFF:
                raise ValueError("sub-mesh over the u16 limits (split it first)")
            out += tex + struct.pack("<HH", nv, pflags)
            v = np.zeros(nv, dtype=[("p", "<f4", 3), ("n", "<f4", 3), ("c", "u1", 4), ("t", "<f4", 2)])
            v["p"], v["n"], v["c"], v["t"] = pos, nrm, col, uv
            out += v.tobytes()
            out += struct.pack("<HH", 3 * len(tris), 0)
            out += tris.astype("<u2").tobytes()
            out += b"\0" * (-len(out) % 4)
    total = len(out)
    struct.pack_into("<I", out, 0, total | (5 << 24))
    out += b"\0" * (-len(out) % 16)
    return out


def double_sided(raw, tables=None):
    """A 0x2E mesh section made visible from both sides: every triangle gets a back face (winding
    reversed, normals flipped, same texture and colours). For models only ever seen from one side
    where they came from (San d'Oria's wall from Ronfaure: its inner faces were never made).
    Same name and key; returns the section unchanged if it has nothing to draw."""
    tables = tables or key_tables()
    body = bytearray(raw[16:])
    mesh_decrypt(body, tables)
    m = read_mesh(body)
    if not m["parts"]:
        return raw
    strip = bool(m["config"] & 1)
    parts = []
    for p in m["parts"]:
        t = part_triangles(p, strip)
        if not len(t):
            continue
        nv = len(p["pos"])
        pos = np.concatenate([p["pos"], p["pos"]])
        nrm = np.concatenate([p["nrm"], -p["nrm"]])
        col = np.concatenate([p["col"], p["col"]])
        uv = np.concatenate([p["uv"], p["uv"]])
        tris = np.concatenate([t, t[:, ::-1] + nv])
        for chunk in _split_render(pos, nrm, col, uv, tris, max_tris=0xFFFF // 3):
            parts.append((p["texture"], *[c.astype(np.float32) if k in (0, 1, 3) else c for k, c in enumerate(chunk[:4])],
                          chunk[4], p["flags"]))
    new = mesh_section_body(m["name"], parts, body[5], author=m["author"])
    return section_bytes(bytes(raw[:4]), T_MESH, _encrypted_mesh(new, tables))


class _Tile:
    def __init__(self, key):
        self.key = key
        self.parts = {}          # texture16 -> list of (pos, nrm, col, uv, tris) chunks (world)
        self.col = []            # list of (tri_world (k,3,3), normals (k,3), terrain (k,), wall (k,))
        self.centre = None


def _split_render(pos, nrm, col, uv, tris, max_tris=20000):
    """Yield (pos, nrm, col, uv, tris) chunks with compact local indices, within the u16 limits."""
    for s in range(0, len(tris), max_tris):
        t = tris[s:s + max_tris]
        used, inv = np.unique(t, return_inverse=True)
        yield pos[used], nrm[used], col[used], uv[used], inv.reshape(-1, 3)


def _collision_meshes(tri_w, nrm_w, terrain, wall, centre, max_tris=512, cell=16.0,
                      camera_transparent_walls=False):
    """Split one object's collision into small meshes (local to centre). Yields dicts with local
    vertex/normal pools, raw u16 quads, world tris (for grid bucketing) and header flags."""
    kinds = [(np.ones(len(tri_w), bool), 0)]
    if camera_transparent_walls:
        kinds = [(~wall, 0), (wall, 1)]
    for sel_kind, flags in kinds:
        if not sel_kind.any():
            continue
        cen = tri_w[sel_kind].mean(1)
        sub = np.floor(cen[:, [0, 2]] / cell).astype(np.int64)
        idx_kind = np.nonzero(sel_kind)[0]
        order = np.lexsort((sub[:, 1], sub[:, 0]))
        keys = sub[order]
        breaks = np.nonzero(np.any(np.diff(keys, axis=0) != 0, axis=1))[0] + 1
        for grp in np.split(idx_kind[order], breaks):
            for s in range(0, len(grp), max_tris):
                g = grp[s:s + max_tris]
                local = (tri_w[g] - centre).astype(np.float32)
                verts, vinv = np.unique(local.reshape(-1, 3), axis=0, return_inverse=True)
                # exact: the client takes a triangle's plane from its stored normal, so rounding them
                # (tried, to save ZoneDef space) left lips everywhere that caught the player
                nq = np.round(nrm_w[g], 5).astype(np.float32)
                norms, ninv = np.unique(nq, axis=0, return_inverse=True)
                if len(verts) >= 0x1000 or len(norms) >= 0x1000:
                    raise ValueError("collision mesh pool too large; lower max_tris")
                vi = vinv.reshape(-1, 3)
                t = terrain[g]
                q = np.zeros((len(g), 4), dtype=np.uint32)
                # retail winding: stored normal == -(b-a)x(c-a), so write (a, c, b) of an air-facing tri
                q[:, 0] = vi[:, 0] | ((t & 1) << 15)
                q[:, 1] = vi[:, 2] | (((t >> 1) & 1) << 15)
                q[:, 2] = vi[:, 1] | (((t >> 2) & 1) << 15)
                q[:, 3] = ninv.reshape(-1) | (((t >> 3) & 1) << 15)
                if flags:
                    q[:, 2] |= 0x4000
                yield {"verts": verts, "norms": norms, "quads": q.astype(np.uint16), "flags": flags,
                       "world": tri_w[g]}


FIT_GRID = False                           # never shrink the block grid below the base zone's: the client
                                           # drops every entity (the base zone's NPC and event files place
                                           # them all over its map) into the collision grid by position,
                                           # unchecked, and one outside it crashed the client entering a
                                           # fitted city (ffximain+0x16663B, Sep 2026)
MAX_ZONEDEF_BYTES = 8 * 1024 * 1024 - 16  # the ZoneDef section must fit in 8 MiB: the client copies it into
                                          # a fixed buffer. Every generated city zone over it crashed on zone-in (reading or
                                          # writing past a buffer in FFXiMain: the collision walk, a section copy,
                                          # the cull tables); 8,372,848 bytes loaded (Sep 2026)



def build_zone(base_dat, geometry, tile=64.0, drop_dirs=(b"effe",), double_sided=False,
               camera_transparent_walls=False, terrain=TERRAIN_STONE, tables=None, report=None,
               draw=None, fog=None, keep_meshes=None):
    """The base zone DAT with its geometry replaced by `geometry` (mesh.build() output). Returns bytes.

    geometry["meshes"]: [{texture, positions, normals, uvs, colours (BGRA u8), triangles}] in the cave
    frame, faces wound so (b-a)x(c-a) points into the air; geometry["collision"]: {positions,
    triangles, wall}. Optional geometry["zone_lines"]: {"z5n0": {"pos": (x, y, z) cave frame,
    "rot": raw FFXI rotation, "size": raw FFXI half extents}} moves the base zone's own volumes.

    draw: zone draw distance (yalms): written into every environment (0x2F) section, and our tiles'
    detail thresholds are pushed past it (a tile has one detail level, so it would vanish at the
    middle threshold, 100, whatever the draw distance). fog: fog far distance (defaults to draw).

    double_sided: also emit every collision triangle back to back (retail never does; option kept
    for an in-game A/B). camera_transparent_walls: put col_wall triangles in flags=1 meshes with the
    hitWall bit, the retail invisible-barrier combination (the camera then passes through them).

    keep_meshes: names (16 bytes) of the base zone's own meshes to keep, for placed objects that
    reuse them (outdoor.py lays the base zone's land tiles anew); the rest of its meshes go.
    """
    tables = tables or key_tables()
    sections = split_sections(bytes(base_dat))
    stats = report if report is not None else {}

    # --- locate the pieces of the base zone --------------------------------------------------
    paths = dict(walk_tree(sections))
    zd_idx = [i for i, s in enumerate(sections) if s.kind == T_ZONEDEF]
    if len(zd_idx) != 1:
        raise ValueError(f"expected one 0x1C ZoneDef section, found {len(zd_idx)}")
    zd_idx = zd_idx[0]
    mode_path = paths[zd_idx]
    base_zd = bytearray(sections[zd_idx].raw[16:])
    zonedef_decrypt(base_zd, tables)
    base = read_zonedef(base_zd)
    first_mesh = next((s for i, s in enumerate(sections) if s.kind == T_MESH and paths[i] == mode_path), None)
    key = first_mesh.raw[16 + 5] if first_mesh else 0x80

    available = texture_sections(sections)
    # textures from another zone (geometry["extra_textures"]: {name16: raw 0x20 section}), for
    # surfaces the base zone has nothing for (e.g. Beaucedine's cliff face on a Dragon's Aery base)
    extra = {k: v for k, v in (geometry.get("extra_textures") or {}).items() if k not in available}
    for k in extra:
        available[k] = ("extra", extra[k])

    # --- geometry into FFXI space, cut into tiles ----------------------------------------------
    meshes = geometry["meshes"]
    coll = geometry["collision"]
    cpos = to_ffxi(coll["positions"])
    ctri = np.asarray(coll["triangles"], dtype=np.int64)
    cwall = np.asarray(coll.get("wall", np.zeros(len(ctri), bool)), dtype=bool)
    ctw = cpos[ctri]                                           # (m,3,3) world tris
    g = np.cross(ctw[:, 1] - ctw[:, 0], ctw[:, 2] - ctw[:, 0])  # air-facing (proper rotation)
    area = np.linalg.norm(g, axis=1)
    keep = area > 1e-6
    stats["collision_dropped_degenerate"] = int((~keep).sum())
    ctw, g, area, cwall = ctw[keep], g[keep], area[keep], cwall[keep]
    cn = g / area[:, None]
    cterr = np.full(len(ctw), terrain, dtype=np.uint32)
    if "terrain" in coll:
        cterr = np.asarray(coll["terrain"], dtype=np.uint32)[keep]

    all_pts = [ctw.reshape(-1, 3)] + [to_ffxi(m["positions"]) for m in meshes]
    lo = np.min([p.min(0) for p in all_pts], 0)

    def tile_of(xz):
        return [tuple(k) for k in np.floor((xz - lo[[0, 2]]) / tile).astype(np.int64)]

    tiles = {}

    def get_tile(k):
        if k not in tiles:
            tiles[k] = _Tile(k)
        return tiles[k]

    for m in meshes:
        tex = resolve_texture(m["texture"], available)
        pos = to_ffxi(m["positions"])
        nrm = to_ffxi(m["normals"])
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        col = np.asarray(m["colours"], dtype=np.uint8)
        uv = np.asarray(m["uvs"], dtype=np.float32)
        tri = np.asarray(m["triangles"], dtype=np.int64)
        tri = tri[:, [0, 2, 1]]                              # retail winding: n == -(b-a)x(c-a)
        keys = tile_of(pos[tri].mean(1)[:, [0, 2]])
        by = {}
        for i, k in enumerate(keys):
            by.setdefault(k, []).append(i)
        for k, ids in by.items():
            get_tile(k).parts.setdefault((tex, int(m.get("flags", 0))), []).append((pos, nrm, col, uv, tri[ids]))

    ckeys = tile_of(ctw.mean(1)[:, [0, 2]])
    by = {}
    for i, k in enumerate(ckeys):
        by.setdefault(k, []).append(i)
    for k, ids in by.items():
        ids = np.array(ids)
        get_tile(k).col.append((ctw[ids], cn[ids], cterr[ids], cwall[ids]))
        if double_sided:
            get_tile(k).col.append((ctw[ids][:, [0, 2, 1]], -cn[ids], cterr[ids], cwall[ids]))

    # --- per tile: render parts, centre, 0x2E section, collision meshes ------------------------
    objects = []
    mesh_sections = []
    for n, k in enumerate(sorted(tiles)):
        t = tiles[k]
        pts = [c[0].reshape(-1, 3) for c in t.col]
        for chunks in t.parts.values():
            for pos, _, _, _, tri in chunks:
                pts.append(pos[np.unique(tri)])
        pts = np.concatenate(pts)
        wlo, whi = pts.min(0), pts.max(0)
        centre = np.round((wlo + whi) / 2, 3)
        parts = []
        for (tex, pflags), chunks in t.parts.items():
            pos = np.concatenate([c[0] for c in chunks])
            nrm = np.concatenate([c[1] for c in chunks])
            col = np.concatenate([c[2] for c in chunks])
            uv = np.concatenate([c[3] for c in chunks])
            off = np.cumsum([0] + [len(c[0]) for c in chunks[:-1]])
            tri = np.concatenate([c[4] + o for c, o in zip(chunks, off)])
            for p, nn, cc, u, tt in _split_render(pos, nrm, col, uv, tri):
                parts.append((tex, (p - centre).astype(np.float32), nn.astype(np.float32), cc, u, tt, pflags))
        name = f"cave{n:04d}".encode().ljust(16, b" ")
        stub = _bbox6(pts - centre)
        body = mesh_section_body(name, parts, key, stub_box=stub)
        mesh_sections.append(section_bytes(f"c{n:03x}".encode(), T_MESH, _encrypted_mesh(body, tables)))
        cols = []
        if t.col:
            tw = np.concatenate([c[0] for c in t.col])
            tn = np.concatenate([c[1] for c in t.col])
            tt = np.concatenate([c[2] for c in t.col])
            tw_wall = np.concatenate([c[3] for c in t.col])
            cols = list(_collision_meshes(tw, tn, tt, tw_wall, centre,
                                          camera_transparent_walls=camera_transparent_walls))
        ylo = min([c["world"][..., 1].min() for c in cols] + [wlo[1]])
        yhi = max([c["world"][..., 1].max() for c in cols] + [whi[1]])
        objects.append({"name": name, "pos": centre, "lo": wlo, "hi": whi, "col": cols,
                        "ycull": (float(ylo) - 5.0, float(yhi) + 5.0)})
    # props from another zone's kit (props.py): placed objects, no collision meshes of their own
    # (solid ones were merged into geometry["collision"] before tiling)
    prop_block = geometry.get("props") or {}
    for po in prop_block.get("objects", []):
        objects.append({"name": po["name"], "pos": np.asarray(po["pos"], dtype=np.float64),
                        "lo": np.asarray(po["lo"]), "hi": np.asarray(po["hi"]), "col": [],
                        "ycull": (float(po["lo"][1]) - 5.0, float(po["hi"][1]) + 5.0),
                        "rot": po["rot"], "scale": po["scale"], "lod": po.get("lod")})
    stats["props"] = len(prop_block.get("objects", []))
    stats["objects"] = len(objects)
    stats["render_triangles"] = sum(len(m["triangles"]) for m in meshes)
    stats["collision_triangles"] = sum(len(c["quads"]) for o in objects for c in o["col"])
    stats["collision_meshes"] = sum(len(o["col"]) for o in objects)

    # the base zone's own trigger volumes (zone lines) stay; report where they sit against the cave
    glo = np.min([o["lo"] for o in objects], 0)
    ghi = np.max([o["hi"] for o in objects], 0)
    for s in sections:
        if s.kind == T_RID:
            for sid, centre in rid_entries(s.raw):
                moved = sid in geometry.get("zone_lines", {})
                inside = bool(np.all(centre[[0, 2]] >= glo[[0, 2]]) and np.all(centre[[0, 2]] <= ghi[[0, 2]]))
                stats.setdefault("zone_lines", []).append(
                    (sid, tuple(round(float(v), 1) for v in from_ffxi(centre)), "moved" if moved else ("INSIDE cave bounds" if inside else "outside")))

    # --- the new ZoneDef --------------------------------------------------------------------
    far = max(float(draw or 0), 1000.0)                     # one detail level: keep it all past the draw
    lod = (LOD[0], far * 5, far * 5) if draw else LOD
    zd_body = _zonedef_body(base, base_zd, objects, stats, lod)
    zonedef_encrypt(zd_body, tables)
    zd_section = section_bytes(sections[zd_idx].name, T_ZONEDEF, zd_body)

    # --- textures the meshes use that do not live beside them in `mode` -----------------------
    used = {tex for m in meshes for tex in [resolve_texture(m["texture"], available)]}
    in_mode = {s.raw[0x11:0x21] for i, s in enumerate(sections) if s.kind == T_TEXTURE and paths[i] == mode_path}
    copies = [available[t][1] if isinstance(available[t], tuple) else sections[available[t]].raw
              for t in sorted(used - in_mode)]
    stats["textures_copied_into_mode"] = [t[0x11:0x21].decode("ascii", "replace") for t in copies]

    # --- reassemble --------------------------------------------------------------------------
    out = []
    drop_depth = None
    stack = []
    for i, s in enumerate(sections):
        if drop_depth is not None:
            if s.kind == T_DIR:
                stack.append(s.name)
            elif s.kind == T_END:
                stack.pop()
                if len(stack) == drop_depth:
                    drop_depth = None
            continue
        if s.kind == T_DIR:
            # drop_dirs: a top-level directory's name, or a path below the root as a tuple
            # ((b"mode", b"taki") drops mode/taki and everything in it)
            if (s.name in drop_dirs and len(stack) == 1) or tuple(stack[1:]) + (s.name,) in drop_dirs:
                stats.setdefault("dropped_dirs", []).append(b"/".join(tuple(stack[1:]) + (s.name,)).decode("ascii", "replace"))
                drop_depth = len(stack)
                stack.append(s.name)
                continue
            stack.append(s.name)
            out.append(s.raw)
            continue
        if s.kind == T_END:
            if tuple(stack) == mode_path:
                out.extend(copies)
                # present = really in the base zone or already copied (textures merely available
                # from another zone don't count, or the props' own would be skipped)
                have = {t for t, v in available.items() if not isinstance(v, tuple)} | {c[0x11:0x21] for c in copies}
                for name16, raw in sorted(prop_block.get("textures", {}).items()):
                    if name16 not in have:                   # the props' own textures, once each
                        out.append(raw)
                        have.add(name16)
                out.extend(mesh_sections)
                out.extend(prop_block.get("sections", []))
            stack.pop()
            out.append(s.raw)
            continue
        if i == zd_idx:
            out.append(zd_section)
        elif s.kind == T_MESH and paths[i] == mode_path:
            if keep_meshes:
                mb = bytearray(s.raw[16:])
                mesh_decrypt(mb, tables)
                if bytes(mb[0x10:0x20]) in keep_meshes:
                    out.append(s.raw)
            continue
        elif s.kind == T_RID and geometry.get("zone_lines"):
            out.append(_move_zone_lines(s.raw, geometry["zone_lines"]))
        elif s.kind == T_ENV and draw:
            out.append(_env_distances(s.raw, float(draw), float(fog or draw)))
        else:
            out.append(s.raw)
    data = b"".join(out)
    stats["bytes"] = len(data)
    return data


def _env_distances(raw, draw, fog):
    """An environment (0x2F) section with its draw distance and fog far distances set. Unencrypted;
    offsets from the data start: fog far model +0x1C, fog far terrain +0x3C, draw distance +0x58.
    Fog far 0 means fog off: left as it is."""
    r = bytearray(raw)
    d = 0x10
    for off in (0x1C, 0x3C):
        if struct.unpack_from("<f", r, d + off)[0] > 0:
            struct.pack_into("<f", r, d + off, fog)
    struct.pack_into("<f", r, d + 0x58, draw)
    return bytes(r)


def _encrypted_mesh(body, tables):
    body = bytearray(body)
    mesh_encrypt(body, tables)
    return body


def rid_entries(raw):
    """[(source id, FFXI centre)] of a 0x36 section's volumes."""
    ds = 16
    if raw[ds:ds + 3] != b"RID":
        return []
    first = _u32(raw, ds + 0x10)
    out = []
    for i in range(_u32(raw, ds + first)):
        e = ds + first + 16 + i * 0x40
        out.append((bytes(raw[e + 0x24:e + 0x28]).decode("ascii", "replace"),
                    np.array(struct.unpack_from("<3f", raw, e))))
    return out


def _move_zone_lines(raw, moves):
    raw = bytearray(raw)
    ds = 16
    first = _u32(raw, ds + 0x10)
    n = _u32(raw, ds + first)
    for i in range(n):
        e = ds + first + 16 + i * 0x40
        sid = bytes(raw[e + 0x24:e + 0x28]).decode("ascii", "replace")
        if sid in moves:
            mv = moves[sid]
            struct.pack_into("<3f", raw, e, *to_ffxi(mv["pos"]))
            if "rot" in mv:
                struct.pack_into("<3f", raw, e + 0x0C, *mv["rot"])
            if "size" in mv:
                struct.pack_into("<3f", raw, e + 0x18, *mv["size"])
    return bytes(raw)


def _zonedef_body(base, base_zd, objects, stats, lod=LOD):
    """Serialize header, objects, one culling table, quadtree, point lights and collision."""
    n = len(objects)
    zbx, zbz, bw, bl = base["blocks"]
    if FIT_GRID:
        # the block grid (and with it the collision cell map, 4 bytes a 4-yalm cell) sized to this
        # zone's own extent round the origin, not the base zone's: retail zones each declare their
        # own, and the cell map counts against the ZoneDef's 8 MiB
        ext = np.concatenate([np.concatenate([o["lo"][None], o["hi"][None]]) for o in objects])
        ax, az = float(np.abs(ext[:, 0]).max()) + 8.0, float(np.abs(ext[:, 2]).max()) + 8.0
        zbx = min(255, 2 * int(math.ceil(ax / bw)))
        zbz = min(255, 2 * int(math.ceil(az / bl)))
        base = dict(base, blocks=(zbx, zbz, bw, bl))
    else:
        # never smaller than the base zone's, but grown to take geometry that reaches past it (more
        # lands laid beside the first: outdoors/regions.py); the base zone's entities stay inside
        ext = np.concatenate([np.concatenate([o["lo"][None], o["hi"][None]]) for o in objects])
        ax, az = float(np.abs(ext[:, 0]).max()) + 8.0, float(np.abs(ext[:, 2]).max()) + 8.0
        zbx = max(zbx, min(254, 2 * int(math.ceil(ax / bw))))
        zbz = max(zbz, min(254, 2 * int(math.ceil(az / bl))))
        base = dict(base, blocks=(zbx, zbz, bw, bl))
    sub_x, sub_z = bw // 4, bl // 4
    cell_w, cell_l = bw / sub_x, bl / sub_z
    half_x, half_z = zbx * bw / 2.0, zbz * bl / 2.0
    cols, rows = zbx * sub_x, zbz * sub_z
    allw = np.concatenate([np.concatenate([o["lo"][None], o["hi"][None]]) for o in objects])
    if np.any(np.abs(allw[:, 0]) >= half_x) or np.any(np.abs(allw[:, 2]) >= half_z):
        raise ValueError(f"geometry leaves the collision grid (+-{half_x} x +-{half_z} yalms)")

    body = bytearray(0x20)
    obj_off = 0x20
    cull_off = obj_off + n * OBJ_SIZE
    table_link = cull_off + 4                     # objects/transforms point at the table's count
    tree_off = cull_off + 8 + 4 * n

    # quadtree, as retail builds it: its leaves are the zone's own blocks (bw x bl yalms on the block
    # grid the header declares, centred on the origin), all at one depth (6 for a 60 x 50 grid), each
    # object in the block its position falls in; children in (-x-z, +x-z, -x+z, +x+z) order, empty
    # ones left out. The client assumes this shape: a flat tree sized by the biggest object (a sea
    # surface spans the zone) put 1,330 objects in a leaf and it read past a buffer on zone-in; an
    # adaptive tree with 85-yalm leaves made it write past one (a generated city, Sep 2026).
    lo = allw.min(0)
    hi = allw.max(0)
    # flat, as every zone we've made has had: sized by the biggest object (a sea surface makes it
    # one level). Retail builds its block grid at depth 6; a deeper tree never proved needed
    side = max(hi[0] - lo[0], hi[2] - lo[2], 1.0)
    obj_span = max(max(o["hi"][0] - o["lo"][0], o["hi"][2] - o["lo"][2]) for o in objects)
    depth = max(1, int(math.ceil(math.log2(max(side / max(obj_span, 1.0), 1.0)))))
    depth = min(depth, int(math.ceil(math.log2(max(zbx, zbz)))))
    span = 2 ** depth
    w = side * 1.0001 / span                  # the tree's cells: its objects' spread in 2**depth
    bx = np.array([int((o["pos"][0] - lo[0]) // w) for o in objects]).clip(0, span - 1)
    bz = np.array([int((o["pos"][2] - lo[2]) // w) for o in objects]).clip(0, span - 1)
    nodes = []                                    # (box_lo, box_hi, members, children indices)

    def build(i0, j0, size, members, level):
        me = len(nodes)
        nodes.append(None)
        if level == depth:
            blo = np.min([objects[i]["lo"] for i in members], 0) - 0.5
            bhi = np.max([objects[i]["hi"] for i in members], 0) + 0.5
            nodes[me] = (blo, bhi, sorted(members), [])
            return me
        h = size // 2
        quads = [[], [], [], []]
        for i in members:
            quads[(bx[i] >= i0 + h) + 2 * (bz[i] >= j0 + h)].append(i)
        kids = []
        for q, mem in enumerate(quads):
            if mem:
                kids.append(build(i0 + h * (q & 1), j0 + h * (q >> 1), h, mem, level + 1))
        blo = np.min([nodes[c][0] for c in kids], 0)
        bhi = np.max([nodes[c][1] for c in kids], 0)
        nodes[me] = (blo, bhi, [], kids)
        return me

    build(0, 0, span, list(range(n)), 0)
    node_off = [tree_off + i * NODE_SIZE for i in range(len(nodes))]
    lists_off = tree_off + len(nodes) * NODE_SIZE
    leaf_ref = {}
    p = lists_off
    for i, (_, _, members, _) in enumerate(nodes):
        if members:
            leaf_ref[i] = p
            p += 4 * len(members)
    lights = bytes(base_zd[base["lights"]:base["coll"]]) if base["lights"] and base["coll"] > base["lights"] else b""
    lights_off = p
    coll_off = lights_off + len(lights)
    stats["tree_nodes"] = len(nodes)
    stats["tree_depth"] = depth

    # objects
    objs = bytearray()
    for o in objects:
        r = bytearray(OBJ_SIZE)
        r[0:16] = o["name"]
        struct.pack_into("<3f3f3f", r, 0x10, *o["pos"], *o.get("rot", (0, 0, 0)), *o.get("scale", (1, 1, 1)))
        struct.pack_into("<3f", r, 0x38, *(o.get("lod") or lod))
        r[0x44:0x48] = OBJ_FLAGS
        struct.pack_into("<I", r, 0x48, table_link)
        objs += r
    # one culling table holding every object: visible from any camera floor
    cull = struct.pack("<II", 1, n) + struct.pack(f"<{n}I", *range(n))
    # tree nodes + leaf lists
    tree = bytearray()
    for i, (blo, bhi, members, kids) in enumerate(nodes):
        x0, y0, z0 = blo
        x1, y1, z1 = bhi
        corners = [(x0, y0, z1), (x1, y0, z1), (x1, y0, z0), (x0, y0, z0),
                   (x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)]
        rec = bytearray(NODE_SIZE)
        struct.pack_into("<24f", rec, 0, *[c for xyz in corners for c in xyz])
        struct.pack_into("<II", rec, 0x60, leaf_ref.get(i, 0), len(members))
        struct.pack_into("<4I", rec, 0x68, *([node_off[c] for c in kids] + [0] * (4 - len(kids))))
        tree += rec
    for i, (_, _, members, _) in enumerate(nodes):
        if members:
            tree += struct.pack(f"<{len(members)}I", *members)

    # collision: header, meshes, transforms, groups, map
    cmeshes = []
    for oi, o in enumerate(objects):
        for c in o["col"]:
            cmeshes.append((oi, c))
    mesh_blob = bytearray()
    mesh_at = []
    p = coll_off + 0x20
    for oi, c in cmeshes:
        mesh_at.append(p)
        pos_o = p + 16
        nrm_o = pos_o + 12 * len(c["verts"])
        idx_o = nrm_o + 12 * len(c["norms"])
        blob = struct.pack("<IIIHH", pos_o, nrm_o, idx_o, len(c["quads"]), c["flags"])
        blob += c["verts"].astype("<f4").tobytes() + c["norms"].astype("<f4").tobytes()
        blob += c["quads"].astype("<u2").tobytes()
        if len(c["quads"]) > 0xFFFF:
            raise ValueError("collision mesh over 65535 triangles")
        mesh_blob += blob
        p += len(blob)
    xf_off = p
    xforms = bytearray()
    for o in objects:
        rec = bytearray(XFORM_SIZE)
        tx, ty, tz = o["pos"]
        struct.pack_into("<16f", rec, 0x00, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, tx, ty, tz, 1)
        struct.pack_into("<16f", rec, 0x40, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -tx, -ty, -tz, 1)
        struct.pack_into("<9f", rec, 0x80, 1, 0, 0, 0, 1, 0, 0, 0, 1)
        struct.pack_into("<II", rec, 0xA4, XFORM_MISC, table_link)
        struct.pack_into("<2f", rec, 0xB4, *o["ycull"])
        xforms += rec
    groups_off = xf_off + len(xforms)

    cells = {}
    for mi, (oi, c) in enumerate(cmeshes):
        w = c["world"]
        tlo = w.min(1) - 1.0
        thi = w.max(1) + 1.0
        c0 = np.clip(np.floor((tlo[:, 0] + half_x) / cell_w).astype(int), 0, cols - 1)
        c1 = np.clip(np.floor((thi[:, 0] + half_x) / cell_w).astype(int), 0, cols - 1)
        r0 = np.clip(np.floor((tlo[:, 2] + half_z) / cell_l).astype(int), 0, rows - 1)
        r1 = np.clip(np.floor((thi[:, 2] + half_z) / cell_l).astype(int), 0, rows - 1)
        hit = set()
        for a0, a1, b0, b1 in zip(c0, c1, r0, r1):
            for r in range(b0, b1 + 1):
                for cc in range(a0, a1 + 1):
                    hit.add(r * cols + cc)
        for cell in hit:
            cells.setdefault(cell, []).append(mi)
    groups = bytearray()
    cell_map = np.zeros(rows * cols, dtype="<u4")
    for cell in sorted(cells):
        r, cc = divmod(cell, cols)
        mis = cells[cell]
        if len(mis) > 0x7FF:
            raise ValueError("too many collision meshes in one grid cell")
        cell_map[cell] = groups_off + len(groups)
        groups += struct.pack("<I", (r << 23) | (cc << 14) | len(mis))
        for mi in mis:
            oi = cmeshes[mi][0]
            groups += struct.pack("<II", xf_off + oi * XFORM_SIZE, mesh_at[mi])
        groups += b"\0\0\0\0"
    if len(cells) > MAX_COLLISION_GROUPS:
        # past anything retail loads, so untried. (A generated zone with 79,373 once crashed on zone-in,
        # but retail zones load with up to 124,103, so that crash had another cause.)
        raise ValueError(f"{len(cells)} collision grid cells in use: over {MAX_COLLISION_GROUPS}, more than any "
                         f"retail zone has; make the zone smaller")
    map_off = groups_off + len(groups)
    chead = struct.pack("<8I", len(cmeshes), coll_off + 0x20, len(cells), groups_off, map_off, xf_off, n, 0)
    stats["collision_groups"] = len(cells)
    stats["grid"] = (cols, rows, cell_w)

    body += objs + cull + tree + lights + chead + mesh_blob + xforms + groups + cell_map.tobytes()
    assert len(body) == map_off + 4 * rows * cols
    stats["zonedef_bytes"] = len(body)
    if len(body) > MAX_ZONEDEF_BYTES:
        raise ValueError(f"ZoneDef is {len(body):,d} bytes, over the {MAX_ZONEDEF_BYTES:,d} the client holds: "
                         f"collision meshes {(xf_off - coll_off) / 1e6:.2f} MB, transforms {len(xforms) / 1e6:.2f} MB, "
                         f"grid groups {len(groups) / 1e6:.2f} MB, cell map {4 * rows * cols / 1e6:.2f} MB")
    body += b"\0" * (-len(body) % 16)
    struct.pack_into("<I", body, 0x00, (base["meta"] & 0xFF000000) | len(body))
    struct.pack_into("<I", body, 0x04, (base["keyword"] & 0xFF000000) | n)
    struct.pack_into("<I", body, 0x08, coll_off)
    body[0x0C:0x10] = bytes(base["blocks"])
    struct.pack_into("<IIII", body, 0x10, tree_off, cull_off, lights_off if lights else 0, base["unk1c"])
    return body


def geometry_from_npz(path):
    """The sample format (sample_cave.npz) -> the dict mesh.build() returns."""
    d = np.load(path)
    meshes = [{"texture": str(d[f"m{i}_texture"]), "positions": d[f"m{i}_positions"],
               "normals": d[f"m{i}_normals"], "uvs": d[f"m{i}_uvs"], "colours": d[f"m{i}_colours"],
               "triangles": d[f"m{i}_triangles"]} for i in range(int(d["n_meshes"]))]
    return {"meshes": meshes, "collision": {"positions": d["col_positions"], "triangles": d["col_triangles"],
                                            "wall": d["col_wall"]}}


# ---------------------------------------------------------------------------------------------
# checks on a finished DAT (ours or retail): structural invariants + decoded geometry
# ---------------------------------------------------------------------------------------------

def check_zone(dat, tables=None):
    """Decode a zone DAT and verify the invariants the client relies on. Returns
    (problems: list of str, info: dict with decoded world geometry in the cave frame)."""
    tables = tables or key_tables()
    bad = []
    sections = split_sections(dat)
    depth = 0
    for s in sections:
        depth += (s.kind == T_DIR) - (s.kind == T_END)
        if depth < 0:
            bad.append("directory end without a start")
    if depth != 0:
        bad.append(f"unbalanced directories ({depth})")
    paths = dict(walk_tree(sections))
    textures = texture_sections(sections)
    zdi = [i for i, s in enumerate(sections) if s.kind == T_ZONEDEF]
    if len(zdi) != 1:
        return bad + [f"{len(zdi)} ZoneDef sections"], {}
    zdi = zdi[0]
    mode = paths[zdi]
    b = bytearray(sections[zdi].raw[16:])
    zonedef_decrypt(b, tables)
    z = read_zonedef(b)
    n = z["count"]
    if (z["meta"] & 0xFFFFFF) != len(b):
        bad.append("ZoneDef meta length != body length")

    mesh_by_name = {}
    for i, s in enumerate(sections):
        if s.kind == T_MESH and paths[i] == mode:
            mb = bytearray(s.raw[16:])
            mesh_decrypt(mb, tables)
            m = read_mesh(mb)
            if (struct.unpack_from("<I", mb, 0)[0] & 0xFFFFFF) > len(mb):
                bad.append(f"mesh {m['name']} meta length past the section")
            mesh_by_name[m["name"]] = m
    info = {"objects": n, "mode_meshes": len(mesh_by_name)}

    # objects and render geometry
    rtri, rnrm_ok, rnrm_bad, unit_bad = [], 0, 0, 0
    obj_box = []
    for i, o in enumerate(z["objects"]):
        m = mesh_by_name.get(o["name"])
        if m is None:
            bad.append(f"object {i} {o['name']} has no 0x2E mesh in its directory")
            obj_box.append(None)
            continue
        if any(abs(r) > 1e-6 for r in o["rot"]) or any(abs(s - 1) > 1e-6 for s in o["scale"]):
            obj_box.append(None)                          # retail placements: skip the geometry checks
            continue
        pos = np.array(o["pos"])
        x0, x1, y0, y1, z0, z1 = m["bbox"]
        obj_box.append((np.array([x0, y0, z0]) + pos, np.array([x1, y1, z1]) + pos))
        for p in m["parts"]:
            if p["texture"] not in textures:
                bad.append(f"{o['name']} uses missing texture {p['texture']}")
            t = part_triangles(p, m["config"] & 1)
            w = p["pos"][t].astype(np.float64) + pos
            rtri.append(w)
            g = np.cross(w[:, 1] - w[:, 0], w[:, 2] - w[:, 0])
            d = np.einsum("ij,ij->i", g, p["nrm"][t].sum(1))
            rnrm_ok += int((d < 0).sum())
            rnrm_bad += int((d > 0).sum())
            unit_bad += int((np.abs(np.linalg.norm(p["nrm"], axis=1) - 1) > 1e-3).sum())
            lo, hi = p["pos"].min(0), p["pos"].max(0)
            if np.any(lo < np.array([x0, y0, z0]) - 1e-3) or np.any(hi > np.array([x1, y1, z1]) + 1e-3):
                bad.append(f"{o['name']} vertices outside its header bbox")
            if p["idx"].max(initial=0) >= len(p["pos"]):
                bad.append(f"{o['name']} index out of range")
    if unit_bad:
        bad.append(f"{unit_bad} non-unit render normals")
    info["render_winding_retail"] = rnrm_ok
    info["render_winding_other"] = rnrm_bad

    # culling tables and links
    table_offs = {t[0] for t in z["tables"]}
    for tpos, members in z["tables"]:
        if any(m >= n for m in members):
            bad.append("culling table lists a missing object")
    for i, o in enumerate(z["objects"]):
        if o["cull_link"] and o["cull_link"] not in table_offs:
            bad.append(f"object {i} culling link points at no table")

    # space tree
    seen = {}
    leaf_depths = set()
    by_off = {nd["off"]: nd for nd in z["nodes"]}
    for nd in z["nodes"]:
        for k in nd["kids"]:
            if k:
                if k not in by_off:
                    bad.append("tree child offset points at no node")
                    continue
                c = by_off[k]["corners"]
                if np.any(c.min(0) < nd["corners"].min(0) - 1e-3) or np.any(c.max(0) > nd["corners"].max(0) + 1e-3):
                    bad.append("tree child box outside its parent")
        if nd["members"]:
            leaf_depths.add(nd["depth"])
            if nd["ref"] + 4 * nd["count"] > len(b):
                bad.append("leaf list past the section")
            for m in nd["members"]:
                seen[m] = seen.get(m, 0) + 1
                box = obj_box[m] if m < len(obj_box) else None
                if box is not None:
                    lo, hi = nd["corners"].min(0), nd["corners"].max(0)
                    if np.any(box[0] < lo - 1e-3) or np.any(box[1] > hi + 1e-3):
                        bad.append(f"object {m} geometry outside its leaf box")
    missing = [i for i in range(n) if i not in seen]
    if missing:
        bad.append(f"{len(missing)} objects in no tree leaf")
    info["tree_nodes"] = len(z["nodes"])
    info["leaf_depths"] = sorted(leaf_depths)

    # collision
    c = z.get("collision")
    ctri, cnorm, cwall = [], [], []
    if c:
        ntf = (c["groups_off"] - c["tf_off"]) / XFORM_SIZE
        if c["idx_count"] != n:
            bad.append(f"collision idx_count {c['idx_count']} != object count {n}")
        if ntf != n:
            bad.append(f"{ntf} collision transforms for {n} objects")
        if c["mesh_end"] != c["tf_off"]:
            bad.append("collision mesh chain does not end at the transforms")
        if c["groups_end"] != c["map_off"]:
            bad.append("collision groups do not end at the map")
        if c["map_end"] > len(b) or len(b) - c["map_end"] >= 16:
            bad.append("collision map does not end the section")
        mesh_at = {m["off"]: m for m in c["meshes"]}
        grp_at = {g["off"]: g for g in c["groups"]}
        cols, rows = c["grid"]
        for cell, off in enumerate(c["map"]):
            if off and off not in grp_at:
                bad.append(f"map cell {cell} points at no group")
            elif off:
                f = grp_at[off]["flags"]
                if (f >> 23, (f >> 14) & 0x1FF) != divmod(cell, cols) and z is not None:
                    info["group_cell_mismatch"] = info.get("group_cell_mismatch", 0) + 1
        for i in range(n):
            a8 = _u32(b, c["tf_off"] + i * XFORM_SIZE + 0xA8)
            if a8 != z["objects"][i]["cull_link"]:
                bad.append(f"transform {i} culling group != object link")
                break
        done = set()
        rel_ok = rel_bad = 0
        ycull_bad = 0
        for g in c["groups"]:
            for tf, mo in g["pairs"]:
                if (tf - c["tf_off"]) % XFORM_SIZE or not (c["tf_off"] <= tf < c["groups_off"]):
                    bad.append("group pair transform is not a transform record")
                    continue
                if mo not in mesh_at:
                    bad.append("group pair mesh is not a mesh")
                    continue
                if (tf, mo) in done:
                    continue
                done.add((tf, mo))
                M = np.array(struct.unpack_from("<16f", b, tf)).reshape(4, 4)
                inv = np.array(struct.unpack_from("<16f", b, tf + 0x40)).reshape(4, 4)
                pos, nrm, raw = collision_mesh_arrays(b, mesh_at[mo])
                ia, ib, ic, idn = raw[:, 0] & 0x7FFF, raw[:, 1] & 0x3FFF, raw[:, 2] & 0x3FFF, raw[:, 3] & 0x7FFF
                if max(ia.max(), ib.max(), ic.max()) >= len(pos) or idn.max() >= len(nrm):
                    bad.append("collision index out of range")
                    continue
                w = pos.astype(np.float64) @ M[:3, :3] + M[3, :3]
                nw = nrm.astype(np.float64) @ inv[:3, :3].T
                nw /= np.linalg.norm(nw, axis=1, keepdims=True) + 1e-12
                t = np.stack([w[ia], w[ib], w[ic]], 1)
                nn = nw[idn]
                gg = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
                d = np.einsum("ij,ij->i", gg, nn)
                rel_ok += int((d < 0).sum())
                rel_bad += int((d >= 0).sum())
                ylo, yhi = struct.unpack_from("<2f", b, tf + 0xB4)
                if t[..., 1].min() < ylo or t[..., 1].max() > yhi:
                    ycull_bad += 1
                ctri.append(t[:, [0, 2, 1]])                 # back to air-facing (b-a)x(c-a)
                cnorm.append(nn)
                cwall.append(((raw[:, 2] & 0x4000) != 0))
        info["collision_winding_retail"] = rel_ok
        info["collision_winding_other"] = rel_bad
        info["collision_meshes_with_ycull_miss"] = ycull_bad
        info["collision_groups"] = len(c["groups"])
        info["collision_meshes"] = len(c["meshes"])
    if rtri:
        rt = np.concatenate(rtri)
        info["render_triangles"] = len(rt)
        info["render_bounds_cave"] = (from_ffxi(rt.reshape(-1, 3)).min(0), from_ffxi(rt.reshape(-1, 3)).max(0))
    if ctri:
        ct = np.concatenate(ctri)
        info["collision_triangles"] = len(ct)
        info["collision_bounds_cave"] = (from_ffxi(ct.reshape(-1, 3)).min(0), from_ffxi(ct.reshape(-1, 3)).max(0))
        info["collision_walls"] = int(np.concatenate(cwall).sum())
        info["collision_world_tris_cave"] = from_ffxi(ct)
    # every texture a mesh draws with must be in the zone (a blank name is an untextured retail part)
    present = {s.raw[0x11:0x21] for s in sections if s.kind == T_TEXTURE}
    wanted = set()
    for s in sections:
        if s.kind == T_MESH:
            b = bytearray(s.raw[0x10:])
            mesh_decrypt(b, tables)
            for p in read_mesh(b)["parts"]:
                wanted.add(bytes(p["texture"]))
    for t in sorted(wanted - present):
        if t.strip(b" \0"):
            bad.append(f"mesh texture {t.decode('ascii', 'replace')!r} is not in the zone")
    return bad, info


def main(argv=None):
    """python writer.py roundtrip|check <dat>...  |  build <base dat> <geometry.npz> <out dat>"""
    import sys
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args.pop(0) if args else "help"
    if cmd == "roundtrip":
        for a in args:
            print(a, "0x2E/0x1C sections (checked, byte exact):", roundtrip_check(Path(a).read_bytes()))
    elif cmd == "check":
        for a in args:
            problems, info = check_zone(Path(a).read_bytes())
            info.pop("collision_world_tris_cave", None)
            print(a, "problems:", problems or "none")
            for k, v in info.items():
                print(f"  {k}: {v}")
    elif cmd == "build":
        rep = {}
        out = build_zone(Path(args[0]).read_bytes(), geometry_from_npz(args[1]), report=rep)
        Path(args[2]).write_bytes(out)
        for k, v in rep.items():
            print(f"  {k}: {v}")
    else:
        print(main.__doc__)


if __name__ == "__main__":
    main()

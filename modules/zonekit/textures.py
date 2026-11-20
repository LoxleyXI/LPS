"""Borrowed textures at half size: a zone DAT has a size the client won't load past, and a zone
that borrows from others carries their textures on top of its own.

halve(raw) takes a 0x20 texture section (DXT1 or DXT3, the kinds zones use), decodes it, averages
each 2x2 of pixels and encodes it again in the same kind: a quarter of the bytes. Paletted
textures, and any below `least` pixels on their longer side, come back as they were.

Layout (from the section's data start): 0x00 kind (0xA1 = DXT), 0x01 name (16), 0x11 a bitmap
header (0x15 width, 0x19 height), 0x39 FourCC ("1TXD" / "3TXD"), 0x3D data size, 0x41 pitch,
0x45 the blocks, then a few bytes of padding.
"""
import struct

import numpy as np

from . import writer as fz


def _565(c):
    c = c.astype(np.uint32)
    return np.stack([((c >> 11) & 31) * 255 // 31, ((c >> 5) & 63) * 255 // 63, (c & 31) * 255 // 31], -1)


def _to565(rgb):
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint32)
    return ((rgb[..., 0] * 31 + 127) // 255 << 11) | ((rgb[..., 1] * 63 + 127) // 255 << 5) | ((rgb[..., 2] * 31 + 127) // 255)


def decode(data, w, h, kind):
    """(h, w, 4) uint8 RGBA from DXT1/DXT3 blocks."""
    bw, bh = w // 4, h // 4
    step = 8 if kind == "1TXD" else 16
    blk = np.frombuffer(bytes(data[:bw * bh * step]), np.uint8).reshape(bh * bw, step)
    col = blk[:, -8:]
    c0 = col[:, 0].astype(np.uint16) | (col[:, 1].astype(np.uint16) << 8)
    c1 = col[:, 2].astype(np.uint16) | (col[:, 3].astype(np.uint16) << 8)
    idx = np.unpackbits(col[:, 4:8, None], axis=2, bitorder="little").reshape(-1, 4, 4, 2)
    idx = (idx[..., 0] | (idx[..., 1] << 1)).reshape(-1, 16)
    a, b = _565(c0).astype(np.float32), _565(c1).astype(np.float32)
    four = (c0 > c1) | (kind != "1TXD")
    pal = np.stack([a, b, np.where(four[:, None], (2 * a + b) / 3, (a + b) / 2),
                    np.where(four[:, None], (a + 2 * b) / 3, 0)], 1)
    rgb = np.take_along_axis(pal, idx[..., None].repeat(3, 2), 1)
    if kind == "1TXD":
        alpha = np.where(~four[:, None] & (idx == 3), 0, 255)
    else:
        al = blk[:, :8]
        nib = np.stack([al & 15, al >> 4], -1).reshape(-1, 16)
        alpha = nib * 17
    px = np.concatenate([rgb, alpha[..., None]], -1).reshape(bh, bw, 4, 4, 4)
    return px.transpose(0, 2, 1, 3, 4).reshape(h, w, 4).astype(np.uint8)


def encode(img, kind):
    """DXT1/DXT3 blocks for (h, w, 4) RGBA: endpoints at the block's colour extremes along its
    brightest-to-darkest line, each pixel the nearest of the four."""
    h, w = img.shape[:2]
    px = img.reshape(h // 4, 4, w // 4, 4, 4).transpose(0, 2, 1, 3, 4).reshape(-1, 16, 4).astype(np.float32)
    rgb, alpha = px[..., :3], px[..., 3]
    lum = rgb @ np.array([0.3, 0.59, 0.11], np.float32)
    hi = np.take_along_axis(rgb, lum.argmax(1)[:, None, None].repeat(3, 2), 1)[:, 0]
    lo = np.take_along_axis(rgb, lum.argmin(1)[:, None, None].repeat(3, 2), 1)[:, 0]
    c0, c1 = _to565(hi), _to565(lo)
    holes = (alpha < 128).any(1) if kind == "1TXD" else np.zeros(len(px), bool)
    # four colours need c0 > c1; three (and a hole) need c0 <= c1
    swap = np.where(holes, c0 > c1, c0 < c1)
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)
    four = ~holes & (c0 != c1)
    a, b = _565(c0).astype(np.float32), _565(c1).astype(np.float32)
    pal = np.stack([a, b, np.where(four[:, None], (2 * a + b) / 3, (a + b) / 2),
                    np.where(four[:, None], (a + 2 * b) / 3, 1e9)], 1)
    d = ((rgb[:, :, None, :] - pal[:, None, :, :]) ** 2).sum(-1)
    idx = d.argmin(2)
    if kind == "1TXD":
        idx = np.where(holes[:, None] & (alpha < 128), 3, idx)
    bits = np.zeros((len(px), 4), np.uint8)
    for k in range(16):
        bits[:, k // 4] |= (idx[:, k].astype(np.uint8) & 3) << (2 * (k % 4))
    colour = np.concatenate([np.stack([c0 & 255, c0 >> 8, c1 & 255, c1 >> 8], 1).astype(np.uint8), bits], 1)
    if kind == "1TXD":
        return colour.tobytes()
    nib = np.clip(np.rint(alpha / 17), 0, 15).astype(np.uint8)
    al = (nib[:, 0::2] | (nib[:, 1::2] << 4))
    return np.concatenate([al, colour], 1).tobytes()


def halve(raw, least=512):
    """A texture section at half its width and height (or as it was: see the module doc)."""
    b = bytes(raw[16:])
    if b[0] != 0xA1:
        return raw
    w, h = struct.unpack_from("<II", b, 0x15)
    kind = b[0x39:0x3D].decode("ascii", "replace")
    if kind not in ("1TXD", "3TXD") or max(w, h) < least or w < 8 or h < 8:
        return raw
    size = struct.unpack_from("<I", b, 0x3D)[0]
    img = decode(b[0x45:0x45 + size], w, h, kind).astype(np.float32)
    small = img.reshape(h // 2, 2, w // 2, 2, 4).mean((1, 3))
    data = encode(small, kind)
    head = bytearray(b[:0x45])
    struct.pack_into("<II", head, 0x15, w // 2, h // 2)
    step = 8 if kind == "1TXD" else 16
    struct.pack_into("<II", head, 0x3D, len(data), (w // 2 // 4) * step)
    body = bytes(head) + data + b[0x45 + size:]
    return fz.section_bytes(bytes(raw[:4]), fz.T_TEXTURE, body)


def cap(dat, most=512):
    """A whole zone DAT with every DXT texture bigger than `most` on its longer side halved until
    it fits. The HD texture pack's 1024s are most of a zone file's size."""
    out = []
    for sec in fz.split_sections(bytes(dat)):
        raw = sec.raw
        if sec.kind == fz.T_TEXTURE:
            while True:
                b = raw[16:]
                if b[0] != 0xA1:
                    break
                w, h = struct.unpack_from("<II", b, 0x15)
                if max(w, h) <= most:
                    break
                smaller = halve(raw, least=most + 1)
                if smaller is raw:
                    break
                raw = smaller
        out.append(raw)
    return b"".join(out)

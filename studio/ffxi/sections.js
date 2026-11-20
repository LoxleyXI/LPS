// FFXI DAT section parser and texture decoder
// Shared by the level editor and particle editor.

export class Reader {
  constructor(view, pos = 0) { this.dv = view; this.pos = pos; }
  u8()  { return this.dv.getUint8(this.pos++); }
  u16() { const v = this.dv.getUint16(this.pos, true); this.pos += 2; return v; }
  i32() { const v = this.dv.getInt32(this.pos, true); this.pos += 4; return v; }
  u32() { const v = this.dv.getUint32(this.pos, true); this.pos += 4; return v; }
  f32() { const v = this.dv.getFloat32(this.pos, true); this.pos += 4; return v; }
  str(n) {
    let s = '';
    for (let i = 0; i < n; i++) { const c = this.dv.getUint8(this.pos + i); if (c === 0) break; s += String.fromCharCode(c); }
    this.pos += n; return s.trim();
  }
}

const align16 = (v) => (v + 0xF) & ~0xF;

export function parseSections(dv) {
  const sections = [];
  let pos = 0;
  while (pos + 16 <= dv.byteLength) {
    const meta = dv.getUint32(pos + 4, true);
    const typeCode = meta & 0x7F;
    const size = ((meta >>> 7) & 0xFFFFF) * 0x10;
    if (size <= 0) break;
    let id = '';
    for (let i = 0; i < 4; i++) { const c = dv.getUint8(pos + i); if (c) id += String.fromCharCode(c); }
    id = id.trim();
    sections.push({ id, typeCode, start: pos, size, dataStart: pos + 0x10 });
    pos = align16(pos + size);
  }
  return sections;
}

// Type code names
export const TYPE_NAMES = {
  0x00:'END', 0x01:'HDR', 0x05:'Particle', 0x06:'Route', 0x07:'EffectRoutine',
  0x19:'KeyFrame', 0x1C:'ZoneDef', 0x1F:'ParticleMesh', 0x20:'Texture',
  0x21:'SpriteSheet', 0x25:'WeightedMesh', 0x29:'Skeleton', 0x2A:'Mesh',
  0x2B:'Animation', 0x2E:'ZoneMesh', 0x2F:'Environment', 0x3D:'Sound',
  0x45:'Info', 0x54:'WeaponTrace', 0x5D:'BumpMap', 0x5E:'Blur',
};

// ── Texture decoding ───────────────────────────────────────────────────────

export function parseTexture(bytes, dv, section) {
  const ds = section.dataStart;
  const texType = bytes[ds];
  const name = String.fromCharCode(...bytes.slice(ds + 1, ds + 17)).replace(/\0.*/, '').trim();
  const width = dv.getUint32(ds + 21, true);
  const height = dv.getUint32(ds + 25, true);
  const bpp = dv.getUint16(ds + 31, true);

  if (texType === 0xA1) {
    // DXT texture
    const dxtOff = ds + 17 + dv.getUint32(ds + 17, true);
    const magic = String.fromCharCode(...bytes.slice(dxtOff, dxtOff + 4));
    const dxtLen = dv.getUint32(dxtOff + 4, true);
    const dxtData = bytes.slice(dxtOff + 12, dxtOff + 12 + dxtLen);
    let rgba;
    if (magic === '1TXD') rgba = decodeDxt1(dxtData, width, height);
    else if (magic === '3TXD') rgba = decodeDxt3(dxtData, width, height);
    else return null;
    return { name, width, height, texType, dxtType: magic, rgba };
  } else if (texType === 0xB1 || texType === 0x91) {
    // Paletted texture
    const headerSize = dv.getUint32(ds + 17, true);
    const palOff = ds + 17 + headerSize;
    const palette = [];
    for (let i = 0; i < 256; i++) {
      const po = palOff + i * 4;
      palette.push({ b: bytes[po], g: bytes[po+1], r: bytes[po+2], a: bytes[po+3] });
    }
    const pixOff = palOff + 1024;
    const rgba = new Uint8Array(width * height * 4);
    for (let i = 0; i < width * height; i++) {
      const idx = bytes[pixOff + i];
      const c = palette[idx];
      rgba[i*4] = c.r; rgba[i*4+1] = c.g; rgba[i*4+2] = c.b;
      rgba[i*4+3] = Math.min(255, c.a * 2);
    }
    return { name, width, height, texType, palette, rgba };
  }
  return null;
}

function decodeDxt1(data, w, h) {
  const rgba = new Uint8Array(w * h * 4);
  const bw = w >> 2, bh = h >> 2;
  let bp = 0;
  for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) {
    const c0 = data[bp] | (data[bp+1] << 8); bp += 2;
    const c1 = data[bp] | (data[bp+1] << 8); bp += 2;
    const r0 = ((c0>>11)&31)*255/31|0, g0 = ((c0>>5)&63)*255/63|0, b0 = (c0&31)*255/31|0;
    const r1 = ((c1>>11)&31)*255/31|0, g1 = ((c1>>5)&63)*255/63|0, b1 = (c1&31)*255/31|0;
    const lut = [
      [r0,g0,b0,255], [r1,g1,b1,255],
      c0>c1 ? [(2*r0+r1)/3|0,(2*g0+g1)/3|0,(2*b0+b1)/3|0,255] : [(r0+r1)>>1,(g0+g1)>>1,(b0+b1)>>1,255],
      c0>c1 ? [(r0+2*r1)/3|0,(g0+2*g1)/3|0,(b0+2*b1)/3|0,255] : [0,0,0,0],
    ];
    const bits = data[bp]|(data[bp+1]<<8)|(data[bp+2]<<16)|(data[bp+3]<<24); bp += 4;
    for (let py = 0; py < 4; py++) for (let px = 0; px < 4; px++) {
      const ci = (bits >> ((py*4+px)*2)) & 3;
      const x = bx*4+px, y = by*4+py;
      if (x < w && y < h) {
        const o = (y*w+x)*4;
        rgba[o]=lut[ci][0]; rgba[o+1]=lut[ci][1]; rgba[o+2]=lut[ci][2]; rgba[o+3]=lut[ci][3];
      }
    }
  }
  return rgba;
}

function decodeDxt3(data, w, h) {
  const rgba = new Uint8Array(w * h * 4);
  const bw = w >> 2, bh = h >> 2;
  let bp = 0;
  for (let by = 0; by < bh; by++) for (let bx = 0; bx < bw; bx++) {
    const alphas = [];
    for (let i = 0; i < 8; i++) alphas.push(data[bp++]);
    const c0 = data[bp]|(data[bp+1]<<8); bp += 2;
    const c1 = data[bp]|(data[bp+1]<<8); bp += 2;
    const r0 = ((c0>>11)&31)*255/31|0, g0 = ((c0>>5)&63)*255/63|0, b0 = (c0&31)*255/31|0;
    const r1 = ((c1>>11)&31)*255/31|0, g1 = ((c1>>5)&63)*255/63|0, b1 = (c1&31)*255/31|0;
    const lut = [
      [r0,g0,b0], [r1,g1,b1],
      [(2*r0+r1)/3|0,(2*g0+g1)/3|0,(2*b0+b1)/3|0],
      [(r0+2*r1)/3|0,(g0+2*g1)/3|0,(b0+2*b1)/3|0],
    ];
    const bits = data[bp]|(data[bp+1]<<8)|(data[bp+2]<<16)|(data[bp+3]<<24); bp += 4;
    for (let py = 0; py < 4; py++) for (let px = 0; px < 4; px++) {
      const ci = (bits >> ((py*4+px)*2)) & 3;
      const ai = py * 2 + (px >> 1);
      const a4 = (px & 1) ? (alphas[ai] >> 4) : (alphas[ai] & 0xF);
      const x = bx*4+px, y = by*4+py;
      if (x < w && y < h) {
        const o = (y*w+x)*4;
        rgba[o]=lut[ci][0]; rgba[o+1]=lut[ci][1]; rgba[o+2]=lut[ci][2]; rgba[o+3]=a4*17;
      }
    }
  }
  return rgba;
}

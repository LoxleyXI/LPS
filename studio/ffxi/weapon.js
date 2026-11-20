// FFXI weapon DAT parser
// Extracts 3D mesh geometry for three.js rendering.
// Ports parse_mesh from xi_export.py (entity mesh 0x2A instruction stream).

import { Reader, parseSections } from './sections.js';

// ── Parse weapon mesh (0x2A section) ────────────────────────────────────────
// Weapon DATs have simpler meshes than full entity models. Typically single-joint
// (no skeleton needed for static display). We extract triangles with positions,
// normals, UVs, and vertex colors.

export function parseWeaponMesh(bytes, dv, section) {
  const ds = section.dataStart;
  const r = new Reader(dv, ds);

  r.u8(); r.u8();
  const flags3 = r.u8();
  const clothEffect = (flags3 & 0x01) !== 0;
  const useJointArray = (flags3 & 0x80) !== 0;
  const hasNormals = !clothEffect;

  r.u8();
  const symmetric = r.u8() === 0x01;
  r.u8();

  const instructionOffset = 2 * r.u32();
  r.u8(); r.u8();
  const jointArrayOffset = 2 * r.u32();
  const numJoints = r.u16();
  const vertexCountsOffset = 2 * r.u32();
  r.u16();
  const vertexJointMappingOffset = 2 * r.u32();
  r.u16();
  const vertexDataOffset = 2 * r.u32();
  r.u16();
  r.u32(); r.u16();

  // Skip cloth header if present
  if (clothEffect) {
    r.u16(); r.u16(); r.u32(); r.u16(); r.u16(); r.u16();
    r.f32(); r.f32(); r.f32(); r.f32(); r.f32();
  }

  // Joint array
  r.pos = ds + jointArrayOffset;
  const jointArray = [];
  for (let i = 0; i < numJoints; i++) jointArray.push(r.u16());

  // Vertex counts
  r.pos = ds + vertexCountsOffset;
  const singleJointCount = r.u16();
  const doubleJointCount = r.u16();

  // Read vertex positions and normals
  const positions = [];
  const normals = [];

  r.pos = ds + vertexDataOffset;
  // Single-joint vertices: 3f pos + 3f normal = 24 bytes
  for (let i = 0; i < singleJointCount; i++) {
    positions.push([r.f32(), r.f32(), r.f32()]);
    normals.push(hasNormals ? [r.f32(), r.f32(), r.f32()] : [0, 0, 1]);
  }
  // Double-joint vertices: interleaved
  for (let i = 0; i < doubleJointCount; i++) {
    const px0 = r.f32(), px1 = r.f32();
    const py0 = r.f32(), py1 = r.f32();
    const pz0 = r.f32(), pz1 = r.f32();
    const w0 = r.f32(), w1 = r.f32();
    let nx = 0, ny = 0, nz = 1;
    if (hasNormals) {
      nx = r.f32(); r.f32(); // nx0, nx1
      ny = r.f32(); r.f32();
      nz = r.f32(); r.f32();
    }
    // Use weight-blended position (both joints' positions mixed)
    positions.push([px0 * w0 + px1 * w1, py0 * w0 + py1 * w1, pz0 * w0 + pz1 * w1]);
    normals.push([nx, ny, nz]);
  }

  // Parse instruction stream to build triangles
  r.pos = ds + instructionOffset;
  let currentTexture = 'untextured';
  const primitives = [];

  for (let safety = 0; safety < 10000; safety++) {
    if (r.pos + 2 > dv.byteLength) break;
    const opcode = r.u16();
    if (opcode === 0xFFFF) break;

    if (opcode === 0x8010) {
      // Render properties: 44 bytes (4×u8 + 2×f32 + 4×u8 + f32 + 2×u32 + u16 + f32 + u16 + 2×f32)
      r.u8(); r.u8(); r.u8(); r.u8();  // 4 bytes
      r.f32(); r.f32();                 // 8 bytes
      r.u8(); r.u8(); r.u8(); r.u8();  // 4 bytes
      r.f32();                          // 4 bytes
      r.u32(); r.u32();                // 8 bytes
      r.u16();                          // 2 bytes
      r.f32();                          // 4 bytes
      r.u16();                          // 2 bytes
      r.f32(); r.f32();                // 8 bytes
      continue;
    }
    if (opcode === 0x8000) {
      // Texture reference: 16-char string
      currentTexture = r.str(16) || 'untextured';
      continue;
    }
    if (opcode === 0x5453) {
      // Triangle strip
      const numTris = r.u16();
      const strip = [];
      for (let i = 0; i < 3; i++) strip.push({ vi: r.u16(), u: 0, v: 0 });
      strip[0].u = r.f32(); strip[0].v = r.f32();
      strip[1].u = r.f32(); strip[1].v = r.f32();
      strip[2].u = r.f32(); strip[2].v = r.f32();
      for (let i = 1; i < numTris; i++) {
        strip.push({ vi: r.u16(), u: r.f32(), v: r.f32() });
      }
      // Convert strip to triangles
      const tris = [];
      for (let i = 0; i < strip.length - 2; i++) {
        if (i % 2 === 0) tris.push(strip[i], strip[i+1], strip[i+2]);
        else tris.push(strip[i+1], strip[i], strip[i+2]);
      }
      primitives.push({ texture: currentTexture, corners: tris, mirrored: false });
      if (symmetric) primitives.push({ texture: currentTexture, corners: tris, mirrored: true });
      continue;
    }
    if (opcode === 0x0054) {
      // Triangle list
      const numTris = r.u16();
      const corners = [];
      for (let i = 0; i < numTris; i++) {
        const v0 = r.u16(), v1 = r.u16(), v2 = r.u16();
        const u0 = r.f32(), v0u = r.f32();
        const u1 = r.f32(), v1u = r.f32();
        const u2 = r.f32(), v2u = r.f32();
        corners.push({vi:v0,u:u0,v:v0u}, {vi:v1,u:u1,v:v1u}, {vi:v2,u:u2,v:v2u});
      }
      primitives.push({ texture: currentTexture, corners, mirrored: false });
      if (symmetric) primitives.push({ texture: currentTexture, corners, mirrored: true });
      continue;
    }
    if (opcode === 0x0043) {
      // Triangle list with vertex colors
      const numTris = r.u16();
      const corners = [];
      for (let i = 0; i < numTris; i++) {
        const v0 = r.u16(), v1 = r.u16(), v2 = r.u16();
        const b = r.u8(), g = r.u8(), rv = r.u8(), a = r.u8();
        const color = [rv/255, g/255, b/255, a/255];
        corners.push({vi:v0,u:0,v:0,color}, {vi:v1,u:0,v:0,color}, {vi:v2,u:0,v:0,color});
      }
      primitives.push({ texture: 'vertex_color', corners, mirrored: false });
      if (symmetric) primitives.push({ texture: 'vertex_color', corners, mirrored: true });
      continue;
    }
    if (opcode === 0x4353) {
      // Triangle strip with vertex colors
      const numTris = r.u16();
      const strip = [];
      for (let i = 0; i < 3; i++) strip.push({ vi: r.u16(), u: 0, v: 0 });
      const b = r.u8(), g = r.u8(), rv = r.u8(), a = r.u8();
      const color = [rv/255, g/255, b/255, a/255];
      strip.forEach(c => c.color = color);
      for (let i = 1; i < numTris; i++) strip.push({ vi: r.u16(), u: 0, v: 0, color });
      const tris = [];
      for (let i = 0; i < strip.length - 2; i++) {
        if (i % 2 === 0) tris.push(strip[i], strip[i+1], strip[i+2]);
        else tris.push(strip[i+1], strip[i], strip[i+2]);
      }
      primitives.push({ texture: 'vertex_color', corners: tris, mirrored: false });
      if (symmetric) primitives.push({ texture: 'vertex_color', corners: tris, mirrored: true });
      continue;
    }
    // Unknown opcode. Bail
    break;
  }

  return { positions, normals, primitives, symmetric };
}

// ── Build flat arrays for three.js BufferGeometry ───────────────────────────

export function buildGeometry(mesh, textures) {
  // Group by texture, output flat pos/normal/uv arrays per material
  const groups = {};

  for (const prim of mesh.primitives) {
    const key = prim.texture;
    if (!groups[key]) groups[key] = { pos: [], norm: [], uv: [] };
    const g = groups[key];

    for (const corner of prim.corners) {
      const vi = corner.vi;
      if (vi >= mesh.positions.length) continue;
      let [px, py, pz] = mesh.positions[vi];
      let [nx, ny, nz] = mesh.normals[vi];

      if (prim.mirrored) { px = -px; nx = -nx; }

      g.pos.push(px, py, pz);
      g.norm.push(nx, ny, nz);
      g.uv.push(corner.u, corner.v);
    }
  }

  return groups;
}

// ── Parse full weapon DAT ───────────────────────────────────────────────────

export function parseWeaponDat(datBuffer) {
  const bytes = new Uint8Array(datBuffer);
  const dv = new DataView(datBuffer);
  const sections = parseSections(dv);

  // Find mesh and texture sections
  const meshSections = sections.filter(s => s.typeCode === 0x2A);
  const texSections = sections.filter(s => s.typeCode === 0x20);

  let mesh = null;
  if (meshSections.length > 0) {
    // Use the first mesh section (weapon mesh)
    mesh = parseWeaponMesh(bytes, dv, meshSections[0]);
  }

  return { mesh, sections };
}

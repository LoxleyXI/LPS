// LPS Studio: Equipment editor with live 3D preview
// Weapon model viewer, texture recoloring, and particle effect editing.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { parseTexture, parseSections } from './ffxi/sections.js';
import { parseAllEffects } from './ffxi/effects.js';
import { parseWeaponDat, buildGeometry } from './ffxi/weapon.js';
import { ParticleSystem } from './ffxi/particle_runtime.js';

// ── Three.js setup ──────────────────────────────────────────────────────────

const canvas = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x2a2a4a);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.position.set(0, 0.5, 2);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
controls.target.set(0, 0, 0);

// Lighting: bright ambient + 3-point directional
const ambient = new THREE.AmbientLight(0x8888aa, 2.0);
scene.add(ambient);
const sun = new THREE.DirectionalLight(0xffeedd, 1.5);
sun.position.set(2, 3, 1);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x8888cc, 1.0);
fill.position.set(-2, 1, 2);
scene.add(fill);
const back = new THREE.DirectionalLight(0x666688, 0.8);
back.position.set(0, -1, -2);
scene.add(back);

// Grid
const grid = new THREE.GridHelper(4, 20, 0x333355, 0x2a2a4a);
scene.add(grid);

let weaponGroup = new THREE.Group();
scene.add(weaponGroup);

const particleSystem = new ParticleSystem(scene, camera);

// ── Distortion post-processing ─────────────────────────────────────────────

const DistortionShader = {
  uniforms: {
    tDiffuse: { value: null },
    uTime: { value: 0 },
    uParticleCount: { value: 0 },
    uParticles: { value: new Array(16).fill(null).map(() => new THREE.Vector4()) },
  },
  vertexShader: `
    varying vec2 vUv;
    void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
  `,
  fragmentShader: `
    uniform sampler2D tDiffuse;
    uniform float uTime;
    uniform int uParticleCount;
    uniform vec4 uParticles[16]; // xy=screenUV, z=radius, w=strength

    varying vec2 vUv;

    void main() {
      vec2 distortion = vec2(0.0);
      float aspect = 1.0;

      for (int i = 0; i < 16; i++) {
        if (i >= uParticleCount) break;
        vec2 center = uParticles[i].xy;
        float radius = uParticles[i].z;
        float strength = uParticles[i].w;

        vec2 diff = vUv - center;
        float dist = length(diff);
        if (dist < 0.001) continue;

        // Heat-haze: sinusoidal offset modulated by distance falloff
        float falloff = exp(-dist * dist / (2.0 * radius * radius));
        float wave = sin(dist * 40.0 - uTime * 3.0) * 0.5 + 0.5;
        distortion += normalize(diff) * strength * falloff * wave * 0.03;
      }

      gl_FragColor = texture2D(tDiffuse, vUv + distortion);
    }
  `,
};

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const distortionPass = new ShaderPass(DistortionShader);
composer.addPass(distortionPass);

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== w || canvas.height !== h) {
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
}

const _projVec = new THREE.Vector3();

function updateDistortionUniforms() {
  const dp = particleSystem.getDistortionParticles();
  const count = Math.min(dp.length, 16);
  distortionPass.uniforms.uParticleCount.value = count;
  distortionPass.uniforms.uTime.value += 0.016;

  for (let i = 0; i < count; i++) {
    const p = dp[i];
    _projVec.copy(p.worldPos).project(camera);
    const sx = (_projVec.x + 1) * 0.5;
    const sy = (_projVec.y + 1) * 0.5;
    if (_projVec.z > 1 || _projVec.z < -1) { distortionPass.uniforms.uParticles.value[i].set(0,0,0,0); continue; }
    const camDist = camera.position.distanceTo(p.worldPos);
    const screenScale = camDist > 0.01 ? (1.0 / camDist) * 0.15 : 0.3;
    distortionPass.uniforms.uParticles.value[i].set(sx, sy, p.scale * screenScale, p.alpha * 0.8);
  }
  for (let i = count; i < 16; i++) {
    distortionPass.uniforms.uParticles.value[i].set(0, 0, 0, 0);
  }
}

function animate() {
  requestAnimationFrame(animate);
  resize();
  controls.update();
  particleSystem.update();
  updateDistortionUniforms();
  composer.render();
}
animate();

// ── State ───────────────────────────────────────────────────────────────────

let weapons = [];
let currentDat = null;
let currentEffects = null;
let currentTextures = {};

// ── Load weapons.json ───────────────────────────────────────────────────────

const modelSearch = document.getElementById('model-search');
const modelDropdown = document.getElementById('model-dropdown');
const fxSearch = document.getElementById('fx-search');
const fxDropdown = document.getElementById('fx-dropdown');

let currentModelDat = null;
let currentFxDat = null;

async function loadWeaponList() {
  const resp = await fetch('weapons.json');
  weapons = await resp.json();
}

/** Classify generator names into effect categories */
function classifyEffects(w) {
  const tags = [];
  const names = (w.generator_names || []).map(n => n.toLowerCase());
  const has = (pat) => names.some(n => n.includes(pat));

  if (has('glw') || has('grw')) tags.push({ label: 'glow', cls: 'glow' });
  if (has('flm') || has('fir') || has('frm')) tags.push({ label: 'fire', cls: 'fire' });
  if (has('bld') || has('blr')) tags.push({ label: 'blade', cls: 'spark' });
  if (has('dis')) tags.push({ label: 'distort', cls: 'blend' });
  if (tags.length === 0 && w.generators > 0) tags.push({ label: `${w.generators} FX`, cls: 'fx' });
  return tags;
}

function renderDropdownItems(dropdown, query, fxOnly, onSelect) {
  let filtered = fxOnly ? weapons.filter(w => w.generators > 0) : weapons;
  if (query) {
    const q = query.toLowerCase();
    filtered = filtered.filter(w =>
      w.name.toLowerCase().includes(q) ||
      String(w.model_id).includes(q) ||
      (w.generator_names || []).some(n => n.toLowerCase().includes(q))
    );
  }

  dropdown.innerHTML = '';
  const max = Math.min(filtered.length, 80);
  for (let i = 0; i < max; i++) {
    const w = filtered[i];
    const item = document.createElement('div');
    item.className = 'weapon-item';

    const badges = classifyEffects(w).map(
      t => `<span class="fx-badge ${t.cls}">${t.label}</span>`
    ).join('');

    item.innerHTML = `<span class="w-name">${w.name}</span>${badges}<span class="w-id">${w.model_id}</span>`;
    item.addEventListener('click', () => onSelect(w));
    dropdown.appendChild(item);
  }
  if (filtered.length > max) {
    const more = document.createElement('div');
    more.className = 'weapon-item';
    more.innerHTML = `<span class="w-name" style="color:#667">...${filtered.length - max} more</span>`;
    dropdown.appendChild(more);
  }
}

function setupDropdown(input, dropdown, fxFilter, onSelect) {
  input.addEventListener('focus', () => {
    renderDropdownItems(dropdown, input.value, fxFilter(), onSelect);
    dropdown.classList.add('open');
  });
  input.addEventListener('input', () => {
    renderDropdownItems(dropdown, input.value, fxFilter(), onSelect);
    dropdown.classList.add('open');
  });
}

// Model dropdown: all weapons
setupDropdown(modelSearch, modelDropdown, () => false, (w) => {
  modelSearch.value = w.name;
  modelDropdown.classList.remove('open');
  currentModelDat = w.dat;
  if (!currentFxDat && w.generators > 0) {
    currentFxDat = w.dat;
    fxSearch.value = w.name;
  }
  reloadScene();
});

// Effects dropdown: only weapons with FX
setupDropdown(fxSearch, fxDropdown, () => true, (w) => {
  fxSearch.value = w.name;
  fxDropdown.classList.remove('open');
  currentFxDat = w.dat;
  if (!currentModelDat) {
    currentModelDat = w.dat;
    modelSearch.value = w.name;
  }
  reloadScene();
});

document.addEventListener('click', (e) => {
  if (!e.target.closest('.search-wrap')) {
    modelDropdown.classList.remove('open');
    fxDropdown.classList.remove('open');
  }
});

// ── Load & display weapon ───────────────────────────────────────────────────

let _loadedModelPath = null;
let _loadedFxPath = null;
let _fxBuffer = null;

async function reloadScene() {
  if (!currentModelDat && !currentFxDat) return;

  document.getElementById('weapon-info').style.display = 'none';
  document.getElementById('generators-panel').style.display = 'none';
  document.getElementById('textures-panel').style.display = 'none';
  document.getElementById('routines-panel').style.display = 'none';
  document.getElementById('keyframes-panel').style.display = 'none';

  // Load model DAT (only if changed)
  if (currentModelDat && currentModelDat !== _loadedModelPath) {
    const resp = await fetch(currentModelDat);
    currentDat = await resp.arrayBuffer();
    _loadedModelPath = currentModelDat;
  }

  // Load effects DAT (only if changed)
  const fxPath = currentFxDat || currentModelDat;
  if (fxPath !== _loadedFxPath) {
    if (fxPath === currentModelDat) {
      _fxBuffer = currentDat;
    } else {
      const resp = await fetch(fxPath);
      _fxBuffer = await resp.arrayBuffer();
    }
    _loadedFxPath = fxPath;
  }

  // Parse weapon mesh from model DAT
  const { mesh, sections } = parseWeaponDat(currentDat);

  // Parse model textures from the model DAT
  const modelEffects = parseAllEffects(currentDat);
  currentTextures = {};
  for (const tex of modelEffects.textures) {
    currentTextures[tex.name] = tex;
  }

  // Parse particle effects from effects DAT (may be different)
  currentEffects = (_fxBuffer === currentDat) ? modelEffects : parseAllEffects(_fxBuffer);

  // Build 3D mesh
  weaponGroup.clear();
  if (mesh) {
    const groups = buildGeometry(mesh, currentTextures);
    let totalVerts = 0;

    for (const [texName, g] of Object.entries(groups)) {
      if (g.pos.length === 0) continue;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(g.pos, 3));
      geo.setAttribute('normal', new THREE.Float32BufferAttribute(g.norm, 3));
      geo.setAttribute('uv', new THREE.Float32BufferAttribute(g.uv, 2));
      geo.computeBoundingSphere();
      totalVerts += g.pos.length / 3;

      let material;
      const tex = currentTextures[texName];
      if (tex && tex.rgba) {
        const dataTex = new THREE.DataTexture(tex.rgba, tex.width, tex.height, THREE.RGBAFormat);
        dataTex.needsUpdate = true;
        dataTex.magFilter = THREE.NearestFilter;
        dataTex.minFilter = THREE.NearestFilter;
        dataTex.wrapS = THREE.RepeatWrapping;
        dataTex.wrapT = THREE.RepeatWrapping;
        material = new THREE.MeshStandardMaterial({ map: dataTex, side: THREE.DoubleSide, metalness: 0.3, roughness: 0.6 });
      } else {
        material = new THREE.MeshStandardMaterial({ color: 0x888899, side: THREE.DoubleSide, metalness: 0.3, roughness: 0.6 });
      }

      const meshObj = new THREE.Mesh(geo, material);
      meshObj.rotation.x = -Math.PI / 2;
      weaponGroup.add(meshObj);
    }

    // Frame camera on weapon
    const box = new THREE.Box3().setFromObject(weaponGroup);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);
    controls.target.copy(center);
    camera.position.copy(center).add(new THREE.Vector3(0, maxDim * 0.3, maxDim * 1.5));

    document.getElementById('mesh-stats').textContent =
      `${totalVerts} vertices, ${Object.keys(groups).length} materials, ${currentEffects.textures.length} textures`;
    document.getElementById('weapon-info').style.display = 'block';
  }

  // Compute weapon bounds for particle spread
  if (weaponGroup.children.length > 0) {
    const box = new THREE.Box3().setFromObject(weaponGroup);
    particleSystem.weaponBounds = {
      min: { x: box.min.x, y: box.min.y, z: box.min.z },
      max: { x: box.max.x, y: box.max.y, z: box.max.z },
    };
  }

  // Start particle effects
  reloadParticles();

  // Render panels
  renderGeneratorsPanel(currentEffects.generators);
  renderTexturesPanel(currentEffects.textures);
  renderRoutinesPanel(currentEffects.effectRoutines);
  renderKeyframesPanel(currentEffects.keyframes);
}

// ── Generators panel ────────────────────────────────────────────────────────

function renderGeneratorsPanel(generators) {
  const panel = document.getElementById('generators-panel');
  const list = document.getElementById('generators-list');
  list.innerHTML = '';

  if (generators.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  for (const gen of generators) {
    const block = document.createElement('div');
    block.className = 'gen-block';

    const h = gen.header;
    const tags = [];
    if (h.autoRun) tags.push('AutoRun');
    if (h.continuousSingleton) tags.push('Singleton');
    if (h.batched) tags.push('Batched');
    const tagStr = tags.length ? ` [${tags.join(', ')}]` : '';

    block.innerHTML = `<h4>"${gen.id}"${tagStr}</h4>`;

    const fields = document.createElement('div');
    fields.className = 'gen-header-fields';
    const hf = (name, val, highlight = false) =>
      `<div class="field"><span class="field-name">${name}:</span> <span class="field-val${highlight ? ' highlight' : ''}">${val}</span></div>`;

    fields.innerHTML = [
      hf('Attach', h.attachType),
      hf('Joint', `${h.attachJoint0}/${h.attachJoint1}`),
      hf('Emission', `${h.particlesPerEmission}/frame, every ${h.framesPerEmission} frames`),
      hf('Variance', h.emissionVariance),
      hf('PosScale', `${h.actorPosScale} (${h.actorPosScaleAmount.toFixed(2)})`),
      hf('SizeScale', `${h.actorSizeScale} (${h.actorSizeScaleAmount.toFixed(2)})`),
      hf('Color1', `${h.colorR1.toFixed(2)}, ${h.colorG1.toFixed(2)}, ${h.colorB1.toFixed(2)}`, true),
      hf('Color2', `${h.colorR2.toFixed(2)}, ${h.colorG2.toFixed(2)}, ${h.colorB2.toFixed(2)}`, true),
      h.environmentId ? hf('EnvId', h.environmentId) : '',
    ].join('');
    block.appendChild(fields);

    const c1 = `rgb(${(h.colorR1*255)|0},${(h.colorG1*255)|0},${(h.colorB1*255)|0})`;
    const c2 = `rgb(${(h.colorR2*255)|0},${(h.colorG2*255)|0},${(h.colorB2*255)|0})`;
    const swatchHtml = `<span class="color-swatch" style="background:${c1}"></span><span class="color-swatch" style="background:${c2}"></span>`;
    fields.querySelector('.highlight')?.insertAdjacentHTML('afterend', ' ' + swatchHtml);

    for (const [secNum, sec] of Object.entries(gen.sections)) {
      if (sec.opcodes.length === 0) continue;
      const secDiv = document.createElement('div');
      const hdr = document.createElement('div');
      hdr.className = 'gen-section-hdr';
      hdr.textContent = `${sec.label} (${sec.opcodes.length} ops)`;
      const body = document.createElement('div');
      body.style.display = 'block';
      hdr.addEventListener('click', () => { body.style.display = body.style.display === 'none' ? 'block' : 'none'; });
      secDiv.appendChild(hdr);

      for (const op of sec.opcodes) {
        const row = document.createElement('div');
        row.className = 'opcode-row';

        let paramsHtml = '';
        if (op.decoded) {
          const parts = [];
          for (const [k, v] of Object.entries(op.decoded)) {
            if (k === 'r' || k === 'g' || k === 'b' || k === 'a') {
              parts.push(`${k}=${typeof v === 'number' ? v.toFixed(3) : v}`);
            } else {
              parts.push(`${k}=${v}`);
            }
          }
          paramsHtml = `<span class="op-decoded">${parts.join(' ')}</span>`;
        } else if (op.floats) {
          paramsHtml = `<span class="op-params">${op.floats.map(f => f.toFixed(4)).join(', ')}</span>`;
        } else if (op.hex) {
          paramsHtml = `<span class="op-params">${op.hex}</span>`;
        }

        let swatchStr = '';
        if (op.decoded && op.decoded.r !== undefined && op.decoded.g !== undefined && op.decoded.b !== undefined) {
          const cr = (Math.min(1, op.decoded.r) * 255) | 0;
          const cg = (Math.min(1, op.decoded.g) * 255) | 0;
          const cb = (Math.min(1, op.decoded.b) * 255) | 0;
          swatchStr = `<span class="color-swatch" style="background:rgb(${cr},${cg},${cb})"></span> `;
        }

        row.innerHTML = `<span class="op-badge">${op.op}</span><span class="op-name">${op.name}</span>${swatchStr}${paramsHtml}`;
        body.appendChild(row);
      }

      secDiv.appendChild(body);
      block.appendChild(secDiv);
    }

    list.appendChild(block);
  }
}

// ── Textures panel ──────────────────────────────────────────────────────────

function renderTexturesPanel(textures) {
  const panel = document.getElementById('textures-panel');
  const list = document.getElementById('textures-list');
  list.innerHTML = '';

  if (textures.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  for (const tex of textures) {
    if (!tex.rgba) continue;
    const div = document.createElement('div');
    div.className = 'tex-thumb';

    const canvas = document.createElement('canvas');
    const thumbSize = 64;
    canvas.width = thumbSize; canvas.height = thumbSize;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(tex.width, tex.height);
    imgData.data.set(tex.rgba);
    const tmp = document.createElement('canvas');
    tmp.width = tex.width; tmp.height = tex.height;
    tmp.getContext('2d').putImageData(imgData, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(tmp, 0, 0, thumbSize, thumbSize);
    div.appendChild(canvas);

    const label = document.createElement('div');
    label.className = 'tex-thumb-label';
    const typeStr = tex.texType === 0xA1 ? tex.dxtType : tex.texType === 0xB1 ? 'PAL' : `0x${tex.texType.toString(16)}`;
    label.textContent = `${tex.name} ${tex.width}x${tex.height} ${typeStr}`;
    div.appendChild(label);

    if (tex.palette) {
      const palCanvas = document.createElement('canvas');
      palCanvas.width = 128; palCanvas.height = 16;
      const pctx = palCanvas.getContext('2d');
      for (let i = 0; i < 256; i++) {
        const c = tex.palette[i];
        pctx.fillStyle = `rgba(${c.r},${c.g},${c.b},${Math.min(255, c.a*2)/255})`;
        pctx.fillRect((i % 128) * 1, Math.floor(i / 128) * 8, 1, 8);
      }
      const palDiv = document.createElement('div');
      palDiv.className = 'tex-thumb-label';
      palDiv.textContent = 'Palette:';
      div.appendChild(palDiv);
      div.appendChild(palCanvas);
    }

    list.appendChild(div);
  }
}

// ── Effect Routines panel ───────────────────────────────────────────────────

function renderRoutinesPanel(routines) {
  const panel = document.getElementById('routines-panel');
  const list = document.getElementById('routines-list');
  list.innerHTML = '';
  if (routines.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  for (const rt of routines) {
    const div = document.createElement('div');
    div.className = 'routine-item';
    div.innerHTML = `<span class="routine-id">${rt.id}</span> <span style="color:#666">offset ${rt.start}, ${rt.size}B</span>`;
    list.appendChild(div);
  }
}

// ── KeyFrames panel ─────────────────────────────────────────────────────────

function renderKeyframesPanel(keyframes) {
  const panel = document.getElementById('keyframes-panel');
  const list = document.getElementById('keyframes-list');
  list.innerHTML = '';
  if (keyframes.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';
  for (const kf of keyframes) {
    const div = document.createElement('div');
    div.className = 'kf-block';
    const pairsStr = kf.pairs.map(p => `(${p.time.toFixed(3)}, ${p.value.toFixed(4)})`).join(' ');
    div.innerHTML = `<span class="kf-id">${kf.id}</span> <span class="kf-pairs">${pairsStr}</span>`;
    list.appendChild(div);
  }
}

// ── Particle controls ───────────────────────────────────────────────────────

let currentRoutineState = 'unsheathed';

function reloadParticles() {
  if (!currentEffects) return;
  particleSystem.loadAll(currentEffects);
  applyRoutineState(currentRoutineState);
  buildEditorPanel();
  buildStatePanel();
}

function buildStatePanel() {
  const panel = document.getElementById('state-panel');
  const container = document.getElementById('state-buttons');
  const info = document.getElementById('routine-info');
  container.innerHTML = '';
  info.textContent = '';

  if (!currentEffects) { panel.style.display = 'none'; return; }
  const routines = currentEffects.effectRoutines || [];
  if (routines.length === 0 && currentEffects.generators.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  addStateBtn(container, 'Unsheathed', 'unsheathed');
  addStateBtn(container, 'Sheathed', 'sheathed');

  const routineInfo = routines
    .filter(r => r.activates.length > 0)
    .map(r => `${r.id}${r.isLoop ? ' (loop)' : ''}: ${r.activates.map(a => a.genId).join(', ')}`)
    .join(' | ');
  info.textContent = routineInfo;
}

function addStateBtn(container, label, stateId) {
  const btn = document.createElement('button');
  btn.className = 'state-btn' + (currentRoutineState === stateId ? ' active' : '');
  btn.textContent = label;
  btn.dataset.state = stateId;
  btn.addEventListener('click', () => setRoutineState(stateId));
  container.appendChild(btn);
}

function setRoutineState(stateId) {
  currentRoutineState = stateId;
  reloadParticles();
}

function applyRoutineState(stateId) {
  if (!currentEffects) return;
  const emitters = particleSystem.emitters;

  if (stateId === 'sheathed') {
    for (const e of emitters) e.enabled = e.autoRun;
    return;
  }

  if (stateId === 'unsheathed') {
    const SKIP = new Set(['PointLight', 'Audio', 'LensFlare', 'Null']);
    for (const e of emitters) e.enabled = !SKIP.has(e.linkedDataType);
    return;
  }
}

// ── Particle editor ────────────────────────────────────────────────────────

function buildEditorPanel() {
  const panel = document.getElementById('editor-panel');
  const container = document.getElementById('emitter-editors');
  container.innerHTML = '';

  const emitters = particleSystem.emitters;
  if (emitters.length === 0) { panel.style.display = 'none'; return; }
  panel.style.display = 'block';

  for (let ei = 0; ei < emitters.length; ei++) {
    const em = emitters[ei];
    const block = document.createElement('div');
    block.className = 'emitter-block';

    const typeBadge = em.linkedDataType !== 'StaticMesh' ? ` <span class="em-type">${em.linkedDataType}</span>` : '';
    let texPreviewHtml = `<canvas class="em-tex-preview" data-ei="${ei}" width="28" height="28" title="Click: replace | Right-click: revert"></canvas>`;

    block.innerHTML = `
      <div class="emitter-header">
        <input type="checkbox" data-ei="${ei}" class="em-toggle" ${em.enabled ? 'checked' : ''}>
        ${texPreviewHtml}
        <span class="em-name">${em.gen.id}</span>${typeBadge}
      </div>
      ${editorSlider(ei, 'emissionRate', 'Rate', 0.1, 5, 1, 0.1)}
      ${editorSlider(ei, 'lifetime', 'Life', 0.1, 5, 1, 0.1)}
      ${editorSlider(ei, 'speed', 'Speed', 0, 5, 1, 0.1)}
      ${editorSlider(ei, 'scale', 'Scale', 0.1, 5, 1, 0.1)}
      ${editorSlider(ei, 'spread', 'Spread', 0, 5, 1, 0.1)}
      ${editorSlider(ei, 'hue', 'Hue', 0, 360, 0, 1)}
      <div class="em-color-row">
        <label>Color</label>
        <input type="color" data-ei="${ei}" class="em-color" value="#ffffff">
        <input type="range" data-ei="${ei}" class="em-alpha" min="0" max="200" value="100" style="flex:1">
        <span class="em-val" id="em-alpha-val-${ei}">1.0</span>
      </div>
    `;
    container.appendChild(block);

    if (em.rawTex) {
      const canvas = block.querySelector('.em-tex-preview');
      const ctx = canvas.getContext('2d');
      const tmp = document.createElement('canvas');
      tmp.width = em.rawTex.width; tmp.height = em.rawTex.height;
      const tmpCtx = tmp.getContext('2d');
      const imgData = tmpCtx.createImageData(em.rawTex.width, em.rawTex.height);
      imgData.data.set(em.rawTex.rgba);
      tmpCtx.putImageData(imgData, 0, 0);
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, 28, 28);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(tmp, 0, 0, 28, 28);
    }
  }

  // Wire up events
  container.querySelectorAll('.em-toggle').forEach(el => {
    el.addEventListener('change', () => { emitters[el.dataset.ei].enabled = el.checked; });
  });

  container.querySelectorAll('.em-slider').forEach(el => {
    el.addEventListener('input', () => {
      const em = emitters[el.dataset.ei];
      const val = parseFloat(el.value);
      em.overrides[el.dataset.key] = val;
      el.nextElementSibling.textContent = el.dataset.key === 'hue' ? `${val}°` : val.toFixed(1);
    });
  });

  container.querySelectorAll('.em-color').forEach(el => {
    el.addEventListener('input', () => {
      const em = emitters[el.dataset.ei];
      const hex = el.value;
      em.overrides.colorR = parseInt(hex.slice(1, 3), 16) / 128;
      em.overrides.colorG = parseInt(hex.slice(3, 5), 16) / 128;
      em.overrides.colorB = parseInt(hex.slice(5, 7), 16) / 128;
    });
  });

  container.querySelectorAll('.em-alpha').forEach(el => {
    el.addEventListener('input', () => {
      const em = emitters[el.dataset.ei];
      const val = parseFloat(el.value) / 100;
      em.overrides.colorA = val;
      document.getElementById(`em-alpha-val-${el.dataset.ei}`).textContent = val.toFixed(1);
    });
  });

  // Snapshot original textures for revert
  for (const em of emitters) {
    if (!em._origTex && em.rawTex) {
      em._origTex = { rgba: em.rawTex.rgba, width: em.rawTex.width, height: em.rawTex.height, name: em.rawTex.name };
    }
  }

  // Click texture preview to replace with custom image
  const fileInput = document.getElementById('tex-upload');
  let pendingEmitterIdx = -1;

  container.querySelectorAll('.em-tex-preview').forEach(canvas => {
    canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const ei = parseInt(canvas.dataset.ei);
      const em = emitters[ei];
      if (!em._origTex) return;

      const orig = em._origTex;
      const tex = new THREE.DataTexture(orig.rgba, orig.width, orig.height, THREE.RGBAFormat);
      tex.needsUpdate = true;
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearFilter;

      if (em.texture) em.texture.dispose();
      em.texture = tex;
      em.mat.map = tex;
      em.mat.color.set(0xffffff);
      em.mat.needsUpdate = true;
      for (const m of em.meshPool) { m.material.map = tex; m.material.color.set(0xffffff); m.material.needsUpdate = true; }

      em.rawTex = { ...orig };
      delete em.customTexture;

      const pctx = canvas.getContext('2d');
      const tmp = document.createElement('canvas');
      tmp.width = orig.width; tmp.height = orig.height;
      const tmpCtx = tmp.getContext('2d');
      const imgData = tmpCtx.createImageData(orig.width, orig.height);
      imgData.data.set(orig.rgba);
      tmpCtx.putImageData(imgData, 0, 0);
      pctx.fillStyle = '#111';
      pctx.fillRect(0, 0, 28, 28);
      pctx.imageSmoothingEnabled = false;
      pctx.drawImage(tmp, 0, 0, 28, 28);
    });

    canvas.addEventListener('click', () => {
      pendingEmitterIdx = parseInt(canvas.dataset.ei);
      fileInput.value = '';
      fileInput.click();
    });
  });

  fileInput.addEventListener('change', () => {
    if (pendingEmitterIdx < 0 || !fileInput.files[0]) return;
    const em = emitters[pendingEmitterIdx];
    const file = fileInput.files[0];
    const img = new Image();
    img.onload = () => {
      const tw = em.rawTex?.width || 64;
      const th = em.rawTex?.height || 64;

      const tmp = document.createElement('canvas');
      tmp.width = tw; tmp.height = th;
      const ctx = tmp.getContext('2d');
      ctx.clearRect(0, 0, tw, th);
      ctx.drawImage(img, 0, 0, tw, th);
      const rgba = new Uint8Array(ctx.getImageData(0, 0, tw, th).data);

      const tex = new THREE.DataTexture(rgba, tw, th, THREE.RGBAFormat);
      tex.needsUpdate = true;
      tex.magFilter = THREE.LinearFilter;
      tex.minFilter = THREE.LinearFilter;

      if (em.texture) em.texture.dispose();
      em.texture = tex;
      em.mat.map = tex;
      em.mat.color.set(0xffffff);
      em.mat.needsUpdate = true;
      for (const m of em.meshPool) { m.material.map = tex; m.material.color.set(0xffffff); m.material.needsUpdate = true; }

      em.rawTex = { rgba, width: tw, height: th, name: 'custom' };
      em.customTexture = { rgba, width: tw, height: th, dataUrl: tmp.toDataURL('image/png') };

      const preview = container.querySelector(`.em-tex-preview[data-ei="${pendingEmitterIdx}"]`);
      if (preview) {
        const pctx = preview.getContext('2d');
        pctx.fillStyle = '#111';
        pctx.fillRect(0, 0, 28, 28);
        pctx.imageSmoothingEnabled = false;
        pctx.drawImage(tmp, 0, 0, 28, 28);
      }

      URL.revokeObjectURL(img.src);
    };
    img.src = URL.createObjectURL(file);
  });
}

function editorSlider(ei, key, label, min, max, def, step) {
  const display = key === 'hue' ? `${def}°` : def.toFixed(1);
  return `<div class="em-slider-row">
    <label>${label}</label>
    <input type="range" class="em-slider" data-ei="${ei}" data-key="${key}"
      min="${min}" max="${max}" value="${def}" step="${step}">
    <span class="em-val">${display}</span>
  </div>`;
}

// ── Export ──────────────────────────────────────────────────────────────────

document.getElementById('export-btn').addEventListener('click', () => {
  const emitters = particleSystem.emitters;
  const modelWeapon = weapons.find(w => w.dat === currentModelDat);
  const fxWeapon = weapons.find(w => w.dat === currentFxDat);

  const emitterOverrides = [];
  for (const em of emitters) {
    const ov = em.overrides;
    const nonDefault = {};
    if (ov.emissionRate !== 1) nonDefault.emissionRate = ov.emissionRate;
    if (ov.lifetime !== 1) nonDefault.lifetime = ov.lifetime;
    if (ov.speed !== 1) nonDefault.speed = ov.speed;
    if (ov.scale !== 1) nonDefault.scale = ov.scale;
    if (ov.spread !== 1) nonDefault.spread = ov.spread;
    if (ov.hue !== 0) nonDefault.hue = ov.hue;
    if (ov.colorR !== 1 || ov.colorG !== 1 || ov.colorB !== 1) {
      nonDefault.color = [ov.colorR, ov.colorG, ov.colorB];
    }
    if (ov.colorA !== 1) nonDefault.alpha = ov.colorA;

    const hasCustomTex = !!em.customTexture;

    if (!em.enabled || Object.keys(nonDefault).length > 0 || hasCustomTex) {
      const entry = {
        id: em.gen.id,
        type: em.linkedDataType,
        enabled: em.enabled,
        ...(Object.keys(nonDefault).length > 0 ? { overrides: nonDefault } : {}),
      };
      if (hasCustomTex) {
        const b64 = em.customTexture.dataUrl.split(',')[1];
        entry.custom_texture = {
          width: em.customTexture.width,
          height: em.customTexture.height,
          png_base64: b64,
        };
      }
      emitterOverrides.push(entry);
    }
  }

  const modelHue = parseInt(document.getElementById('hue').value) || 0;
  const modelSat = parseInt(document.getElementById('sat').value) || 0;
  const modelLit = parseInt(document.getElementById('lit').value) || 0;
  const modelTintHex = document.getElementById('tint-color').value;
  const modelTintAlpha = parseInt(document.getElementById('tint-alpha').value) || 0;
  const hasTint = modelTintAlpha > 0;

  const exported = {
    _comment: "Generated by LPS Studio",
    name: `${modelWeapon?.name || 'Unknown'}${fxWeapon && fxWeapon !== modelWeapon ? ' + ' + fxWeapon.name + ' FX' : ''}`,
    source_model_id: modelWeapon?.model_id,
    slot: modelWeapon?.slot || 'main',
    ...(modelHue ? { hue: modelHue } : {}),
    ...(modelSat ? { saturation: modelSat } : {}),
    ...(modelLit ? { lightness: modelLit } : {}),
    ...(hasTint ? { tint: modelTintHex + Math.round(modelTintAlpha).toString(16).padStart(2, '0'), blend_mode: 'multiply' } : {}),
    ...(currentFxDat && currentFxDat !== currentModelDat ? {
      effects_source: {
        model_id: fxWeapon?.model_id,
        dat: currentFxDat?.replace('game/', ''),
      },
    } : {}),
    ...(emitterOverrides.length > 0 ? { particle_overrides: emitterOverrides } : {}),
  };

  const json = JSON.stringify(exported, null, 2);
  navigator.clipboard?.writeText(json);

  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  const safeName = (modelWeapon?.name || 'weapon').replace(/[^a-zA-Z0-9]/g, '_').toLowerCase();
  a.download = `${safeName}_config.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// ── Advanced toggle ────────────────────────────────────────────────────────

document.getElementById('advanced-toggle').addEventListener('click', function() {
  const body = document.getElementById('advanced-body');
  const open = body.style.display !== 'none';
  body.style.display = open ? 'none' : 'block';
  this.classList.toggle('open', !open);
});

// ── Tinting controls ────────────────────────────────────────────────────────

function setupTintControls() {
  const hueSlider = document.getElementById('hue');
  const satSlider = document.getElementById('sat');
  const litSlider = document.getElementById('lit');
  const tintColor = document.getElementById('tint-color');
  const tintAlpha = document.getElementById('tint-alpha');

  const update = () => {
    document.getElementById('hue-val').textContent = hueSlider.value;
    document.getElementById('sat-val').textContent = satSlider.value;
    document.getElementById('lit-val').textContent = litSlider.value;
    document.getElementById('tint-val').textContent = tintAlpha.value;

    const h = parseInt(hueSlider.value) / 360;
    const s = parseInt(satSlider.value) / 100;
    const l = parseInt(litSlider.value) / 100;
    const alpha = parseInt(tintAlpha.value) / 255;
    const hex = tintColor.value;
    const tr = parseInt(hex.slice(1,3), 16) / 255;
    const tg = parseInt(hex.slice(3,5), 16) / 255;
    const tb = parseInt(hex.slice(5,7), 16) / 255;

    weaponGroup.traverse(obj => {
      if (!obj.isMesh || !obj.material || !obj.material.map) return;
      const tex = obj.material.map;
      const origKey = '_origImage';
      if (!tex[origKey]) tex[origKey] = tex.image.data.slice();
      const orig = tex[origKey];
      const data = tex.image.data;

      for (let i = 0; i < data.length; i += 4) {
        let r = orig[i] / 255, g = orig[i+1] / 255, b = orig[i+2] / 255;

        const max = Math.max(r, g, b), min = Math.min(r, g, b);
        let lum = (max + min) / 2, sat2 = 0, hue2 = 0;
        if (max !== min) {
          const d = max - min;
          sat2 = lum > 0.5 ? d / (2 - max - min) : d / (max + min);
          if (max === r) hue2 = ((g - b) / d + (g < b ? 6 : 0)) / 6;
          else if (max === g) hue2 = ((b - r) / d + 2) / 6;
          else hue2 = ((r - g) / d + 4) / 6;
        }

        hue2 = (hue2 + h) % 1; if (hue2 < 0) hue2 += 1;
        sat2 = Math.max(0, Math.min(1, sat2 + s));
        lum = Math.max(0, Math.min(1, lum + l));

        let rr, gg, bb;
        if (sat2 === 0) { rr = gg = bb = lum; }
        else {
          const hue2rgb = (p, q, t) => { if (t<0) t+=1; if (t>1) t-=1; if (t<1/6) return p+(q-p)*6*t; if (t<1/2) return q; if (t<2/3) return p+(q-p)*(2/3-t)*6; return p; };
          const q2 = lum < 0.5 ? lum * (1 + sat2) : lum + sat2 - lum * sat2;
          const p2 = 2 * lum - q2;
          rr = hue2rgb(p2, q2, hue2 + 1/3);
          gg = hue2rgb(p2, q2, hue2);
          bb = hue2rgb(p2, q2, hue2 - 1/3);
        }

        if (alpha > 0) {
          rr = rr * (1 - alpha) + tr * alpha;
          gg = gg * (1 - alpha) + tg * alpha;
          bb = bb * (1 - alpha) + tb * alpha;
        }

        data[i]   = Math.max(0, Math.min(255, (rr * 255) | 0));
        data[i+1] = Math.max(0, Math.min(255, (gg * 255) | 0));
        data[i+2] = Math.max(0, Math.min(255, (bb * 255) | 0));
      }
      tex.needsUpdate = true;
      obj.material.color.set(0xffffff);
    });
  };

  [hueSlider, satSlider, litSlider, tintColor, tintAlpha].forEach(el =>
    el.addEventListener('input', update));
}

// ── Init ────────────────────────────────────────────────────────────────────

setupTintControls();
loadWeaponList();

// ── Tab switching ──────────────────────────────────────────────────────────

const itemsFrame = document.getElementById('items-frame');
const equipmentView = document.getElementById('equipment-view');

document.querySelectorAll('.tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');

    const isItems = tab.dataset.tab === 'items';
    equipmentView.style.display = isItems ? 'none' : 'flex';
    itemsFrame.style.display = isItems ? 'block' : 'none';
    document.getElementById('viewport-controls').style.display = isItems ? 'none' : '';
  });
});

// ── Viewport lighting controls ─────────────────────────────────────────────

document.getElementById('ambient-slider').addEventListener('input', (e) => {
  const v = parseInt(e.target.value) / 100;
  ambient.intensity = v;
  sun.intensity = v * 0.8;
  fill.intensity = v * 0.33;
});

document.getElementById('bg-color').addEventListener('input', (e) => {
  const hex = parseInt(e.target.value.slice(1), 16);
  renderer.setClearColor(hex);
});

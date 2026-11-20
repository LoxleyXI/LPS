# Server Operator Guide

How to use LPS to distribute custom content to your players without distributing any proprietary game data.

## How It Works

1. **You** create JSON configs describing your custom items, strings, icons, etc.
2. **You** host those JSON files on your web server alongside a manifest
3. **Players** download the JSON configs via your launcher
4. **LPS** runs locally on the player's machine, reads from their retail installation, and writes overlay files to a separate directory. The original game files are never modified.

No DATs are distributed. The JSON configs contain only your custom data (item names, stats, icon modifications, etc). LPS reads the player's own game installation and applies the patches locally.

## Setup

### 1. Create Your Configs

Use LPS Studio or hand-author JSON configs:

```
configs/
  items.json          # Custom items (stats, text, icons)
  equipment.json      # Model recoloring, particle effects
  strings.json        # Custom spell/zone/ability names
  ui_icons.json       # Nameplate icon replacements
```

See `examples/` for config format reference.

### 2. Test Locally

```bash
# Set your FFXI path
export FFXI_DIR="/path/to/FINAL FANTASY XI"

# Apply all configs
python -m lps apply configs/

# Verify in-game
```

### 3. Generate a Manifest

The manifest is a JSON file listing all your configs with their SHA256 hashes. Launchers use this to detect which configs have changed and need re-downloading.

```bash
python -m lps manifest configs/ -o manifest.json
```

Output:
```json
{
  "version": 1,
  "files": {
    "items.json": "a1b2c3d4...",
    "equipment.json": "e5f6a7b8...",
    "strings.json": "c9d0e1f2..."
  }
}
```

### 4. Host on Your Web Server

Upload the configs and manifest to your server:

```
https://yourserver.com/patches/
  manifest.json
  items.json
  equipment.json
  strings.json
  ui_icons.json
```

Any static file host works (nginx, Apache, S3, GitHub Pages, etc).

## Launcher Integration

Your launcher needs to:

1. **Fetch** `manifest.json` from your server
2. **Compare** hashes against locally cached configs
3. **Download** any changed config files
4. **Run** `python -m lps apply` on the downloaded configs

### Example Flow

```
Launcher starts
  → GET https://yourserver.com/patches/manifest.json
  → Compare with local manifest.json
  → Download changed files to local configs/ directory
  → Run: python -m lps apply configs/
  → Launch game
```

### Example Launcher Code (Python)

```python
import hashlib
import json
import urllib.request
from pathlib import Path

SERVER_URL = "https://yourserver.com/patches"
LOCAL_DIR = Path("patches")

def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def sync_patches():
    LOCAL_DIR.mkdir(exist_ok=True)

    # Fetch remote manifest
    with urllib.request.urlopen(f"{SERVER_URL}/manifest.json") as resp:
        remote = json.loads(resp.read())

    # Load local manifest
    local_manifest = LOCAL_DIR / "manifest.json"
    local = json.loads(local_manifest.read_text()) if local_manifest.exists() else {"files": {}}

    # Download changed files
    for filename, remote_hash in remote["files"].items():
        local_path = LOCAL_DIR / filename
        local_hash = local["files"].get(filename)

        if local_hash == remote_hash and local_path.exists():
            continue  # unchanged

        print(f"  Downloading {filename}...")
        urllib.request.urlretrieve(f"{SERVER_URL}/{filename}", local_path)

    # Save manifest
    local_manifest.write_text(json.dumps(remote, indent=2))

    # Apply patches
    import subprocess
    subprocess.run(["python", "-m", "lps", "apply", str(LOCAL_DIR)])
```

### Example Launcher Code (JavaScript/Electron)

```javascript
const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');
const { execSync } = require('child_process');

const SERVER_URL = 'https://yourserver.com/patches';
const LOCAL_DIR = './patches';

async function fetchJson(url) {
  return new Promise((resolve, reject) => {
    https.get(url, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => resolve(JSON.parse(data)));
    }).on('error', reject);
  });
}

function hashFile(filepath) {
  const data = fs.readFileSync(filepath);
  return crypto.createHash('sha256').update(data).digest('hex');
}

async function syncPatches() {
  fs.mkdirSync(LOCAL_DIR, { recursive: true });

  const remote = await fetchJson(`${SERVER_URL}/manifest.json`);
  const localManifest = path.join(LOCAL_DIR, 'manifest.json');
  const local = fs.existsSync(localManifest)
    ? JSON.parse(fs.readFileSync(localManifest))
    : { files: {} };

  for (const [filename, remoteHash] of Object.entries(remote.files)) {
    const localPath = path.join(LOCAL_DIR, filename);
    if (local.files[filename] === remoteHash && fs.existsSync(localPath)) continue;

    console.log(`  Downloading ${filename}...`);
    // Download file (simplified, use a proper HTTP library in production)
    await downloadFile(`${SERVER_URL}/${filename}`, localPath);
  }

  fs.writeFileSync(localManifest, JSON.stringify(remote, null, 2));

  // Apply patches
  execSync(`python -m lps apply ${LOCAL_DIR}`, { stdio: 'inherit' });
}
```

## Config Organisation Tips

### By Content Type

```
configs/
  01_items_weapons.json
  02_items_armor.json
  03_strings.json
  04_ui_icons.json
```

Numbered prefixes control merge order. Later files override earlier ones for the same item ID.

### By Content Update

```
configs/
  base_items.json         # Rarely changes
  seasonal_event.json     # Updated for events
  hotfix.json             # Quick fixes
```

Players only re-download files whose hashes changed in the manifest.

### Separate HD and SD Assets

For UI icons, provide both sizes for pixel-perfect results:

```json
{
  "custom": {
    "ui_icons": [
      {
        "slot": "mentor",
        "hd": "icons/mentor-32.png",
        "sd": "icons/mentor-16.png"
      }
    ]
  }
}
```

Host the icon PNGs alongside your configs. Reference them with relative paths in the JSON.

### Custom Item Icons

Items use `icon_file` to reference a PNG that gets baked into the item's DAT entry at apply time. The PNG is loaded from disk, not embedded in the JSON.

**How it works:**

1. LPS reads the template item from the player's retail DAT (to get the base entry structure)
2. Your custom PNG replaces the icon in that entry
3. HQ borders, hue shifts, and tints are applied on top
4. The modified entry is written to the overlay

**Directory layout:**

```
configs/
  items.json
  icons/
    crystal-sword.png      # 32x32 RGBA PNG
    fire-shield.png
    custom-potion.png
```

**Config:**

```json
{
  "custom": {
    "item": {
      "items": {
        "22501": {
          "name": "Crystal Sword",
          "template": 16535,
          "icon_file": "crystal-sword.png",
          "hq": 2
        }
      }
    }
  }
}
```

The `icon_file` path is relative to an `icons/` directory next to the config file. LPS checks for `icons/` in the config's directory first, then falls back to the package icons directory.

**Icon requirements:**
- 32x32 pixels (resized automatically if different)
- RGBA PNG format (transparency supported)
- LPS converts to FFXI's BMP2 paletted format (256 colours, BGRA palette)

**Without a custom PNG:**

If you just want to modify an existing item's icon (hue shift, tint, HQ border), you don't need a PNG at all. LPS extracts the icon from the template item's DAT:

```json
{
  "22502": {
    "name": "Emerald Blade",
    "template": 16535,
    "hue_shift": 120,
    "hq": 1
  }
}
```

This takes the template item's icon, shifts the hue by 120 degrees, and adds an HQ1 border.

**For distribution, embed icons as base64:**

Instead of distributing separate PNG files, you can embed icons directly in the JSON config using `icon_data`. This keeps everything in a single file and simplifies your launcher:

```json
{
  "22501": {
    "name": "Crystal Sword",
    "template": 16535,
    "icon_data": "iVBORw0KGgoAAAANSUhEUg...",
    "hq": 2
  }
}
```

Generate the base64 string from a PNG:

```python
import base64
b64 = base64.b64encode(open("icon.png", "rb").read()).decode()
```

Both `icon_file` (separate PNG) and `icon_data` (embedded base64) are supported. Use `icon_file` during development for convenience, switch to `icon_data` for distribution.

## Requirements for Players

Players need:
- Python 3.10+ with Pillow (`pip install Pillow`)
- A retail FFXI installation (for the base DAT files)
- XIPivot or equivalent overlay plugin loaded
- `FFXI_DIR` environment variable pointing to their game installation
- `LPS_OVERLAY_DIR` environment variable pointing to their overlay directory

Your launcher can set these automatically based on where it detects the game installation.

## Security

- JSON configs contain no executable code
- LPS only reads from `FFXI_DIR` and writes to `LPS_OVERLAY_DIR`
- Manifest hashes let the launcher verify file integrity
- No game data is uploaded or distributed

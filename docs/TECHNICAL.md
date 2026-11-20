# LPS Technical Reference

Deep technical documentation of FFXI DAT formats and the patching system.

## DAT File Encryption

All FFXI DAT files use simple bit rotation encryption.

### Decryption (ROR5)

Rotate each byte right by 5 bits:

```python
def ror5(data: bytes) -> bytes:
    return bytes(((b >> 5) | (b << 3)) & 0xFF for b in data)
```

### Encryption (ROL5)

Rotate each byte left by 5 bits:

```python
def rol5(data: bytes) -> bytes:
    return bytes(((b << 5) | (b >> 3)) & 0xFF for b in data)
```

### Example

```
Original:  0b10110100 (0xB4)
Encrypted: 0b01001011 (0x4B)  # ROL5
Decrypted: 0b10110100 (0xB4)  # ROR5
```

## Item Entry Structure

Each item entry occupies exactly 0xC00 (3072) bytes.

### Weapon Entry Layout

```
Offset  Size  Type    Field
------  ----  ------  -----
0x00    4     u32     Item ID
0x04    2     u16     Flags
0x06    2     u16     Stack size
0x08    2     u16     Kind (weapon subtype)
0x0A    2     u16     Unknown (CRITICAL - must be preserved)
0x0C    2     u16     Targets
0x0E    2     u16     Required level
0x10    2     u16     Equipment slots
0x12    2     u16     Race flags
0x14    4     u32     Job flags
0x18    2     u16     Superior level
0x1A    2     u16     (padding)
0x1C    2     u16     Damage
0x1E    2     u16     Delay
0x20    2     u16     DPS (calculated by client)
0x22    2     u16     Skill type
0x24    4     u32     Emote
0x28    10   --      (unknown)
0x32    1     u8      Item level
0x33    5    --      (padding)
0x38    584   var     Text data (DMSG format)
0x280   4     u32     Icon data length
0x284   var  --      Icon data (BMP2 format)
```

### Flag Values

```
0x8000  FLAG_RARE       - Rare item
0x4000  FLAG_EX         - Cannot be traded
0x0800  FLAG_CANEQUIP   - Equipment item
0x0020  FLAG_INSCRIBABLE
```

### Slot Flags

```
0x0001  SLOT_MAIN   - Main hand
0x0002  SLOT_SUB    - Off hand
0x0004  SLOT_RANGE  - Ranged
0x0008  SLOT_AMMO   - Ammo
```

### Race Flags

```
0x01FE  ALL_RACES (Hume M/F, Elvaan M/F, Tarutaru M/F, Mithra, Galka)
```

### Job Flags (32-bit)

Each bit represents a job:
```
Bit 0:  WAR     Bit 8:  DRK     Bit 16: BLU
Bit 1:  MNK     Bit 9:  BST     Bit 17: COR
Bit 2:  WHM     Bit 10: BRD     Bit 18: PUP
Bit 3:  BLM     Bit 11: RNG     Bit 19: DNC
Bit 4:  RDM     Bit 12: SAM     Bit 20: SCH
Bit 5:  THF     Bit 13: NIN     Bit 21: GEO
Bit 6:  PLD     Bit 14: DRG     Bit 22: RUN
Bit 7:  DRK     Bit 15: SMN
```

### Skill Types

```
0   Hand-to-Hand    7   Scythe
1   Dagger          8   Polearm
2   (unused)        9   Katana
3   Sword           10  Great Katana
4   Great Sword     11  Club
5   Axe             12  Staff
6   Great Axe
```

## DMSG Text Format

Item text uses DMSG (dialog message) encoding.

### Entry Structure

```
Offset  Size  Description
------  ----  -----------
0x00    4     Number of elements
0x04    8×N   Element table (offset + type pairs)
var     var   Element data
```

### Element Table Entry

```
Offset  Size  Description
------  ----  -----------
0x00    4     Data offset (from start of entry)
0x04    4     Type (0 = string, 1 = integer)
```

### String Element

Strings have a 28-byte header followed by null-terminated text:

```
Offset  Size  Description
------  ----  -----------
0x00    28    Header (mostly zeros, first byte = 0x01)
0x1C    var   Null-terminated ASCII string
```

### Item Text Elements

Items have 5 text elements:
1. **Name** - Display name ("Bronze Sword")
2. **Article** - 0 for "a", 1 for "an"
3. **Singular** - Lowercase singular ("bronze sword")
4. **Plural** - Lowercase plural ("bronze swords")
5. **Description** - Stats and flavor text

### Encoding Example

```python
def encode_dmsg_entry(elems: list) -> bytes:
    DMSG_STRING_HEADER = bytes([
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00
    ])

    # Build element data
    encoded_data = []
    for elem in elems:
        if isinstance(elem, str):
            str_bytes = elem.encode('ascii') + b'\x00'
            encoded_data.append((0, DMSG_STRING_HEADER + str_bytes))
        else:
            encoded_data.append((1, struct.pack('<I', elem)))

    # Build entry table
    header_size = 4 + len(elems) * 8
    # ... (see item_patcher.py for full implementation)
```

## BMP2 Icon Format

FFXI icons are 32×32 paletted bitmaps in BMP2 format.

### Overall Structure

```
Offset  Size   Description
------  -----  -----------
0x00    57     Header
0x39    1024   Palette (256 × 4 bytes)
0x439   1024   Pixel data (32 × 32 indices)
```

Total size: 2105 bytes

### Header (57 bytes)

```
Offset  Size  Type    Field
------  ----  ------  -----
0x00    1     u8      Version flags (0x91 = bitmap)
0x01    8     char[]  Group name ("weapon  ")
0x09    8     char[]  Texture name (space-padded)
0x11    4     u32     Version (0x28 = 40)
0x15    4     u32     Width (32)
0x19    4     u32     Height (32)
0x1D    2     u16     Mipmap count (1)
0x1F    2     u16     Bits per pixel (8)
0x21    20   --      Reserved (zeros)
0x35    1     u8      Palette bits per pixel (32)
0x36    3    --      Reserved (zeros)
```

### Palette (1024 bytes)

256 color entries, each 4 bytes in **BGRA order**:

```
Offset  Size  Description
------  ----  -----------
0       1     Blue
1       1     Green
2       1     Red
3       1     Alpha (0 = transparent, 128 = opaque)
```

**Critical**: Alpha uses 0-128 range, NOT 0-255.

### Pixel Data (1024 bytes)

32×32 = 1024 palette indices (one byte each).

**Critical**: Image is stored **bottom-to-top** (vertically flipped).

### Encoding Checklist

When encoding icons, ensure:

1. ✓ Palette order is BGRA (not RGBA)
2. ✓ Alpha values are 0-128 (not 0-255)
3. ✓ Image is flipped vertically before encoding
4. ✓ Header offset 53 = 32 (paletteBitsPerPixel)
5. ✓ Names are space-padded (not null-padded)
6. ✓ Group name is "weapon  " (8 chars, space-padded)

### Decoding

```python
def extract_icon(entry: bytes) -> Image:
    icon_len = struct.unpack('<I', entry[0x280:0x284])[0]
    icon_data = entry[0x284:0x284 + icon_len]

    # Parse header
    width = struct.unpack('<I', icon_data[21:25])[0]
    height = struct.unpack('<I', icon_data[25:29])[0]

    # Read palette (BGRA)
    palette = icon_data[57:57 + 1024]

    # Read pixels
    pixels = icon_data[57 + 1024:]

    # Create image
    img = Image.new('RGBA', (width, height))
    for y in range(height):
        for x in range(width):
            idx = pixels[y * width + x]
            b, g, r, a = palette[idx*4:idx*4+4]
            a = min(255, a * 2)  # Scale alpha
            img.putpixel((x, y), (r, g, b, a))

    # Flip vertically
    return img.transpose(Image.FLIP_TOP_BOTTOM)
```

## XIPivot Integration

### pivot.ini Configuration

```ini
[settings]
root_path=C:\path\to\DATs
debug_log=false
redirect_fopens=true

[overlays]
0=my-overlay
```

Higher overlay numbers have higher priority.

### Overlay Directory Structure

```
DATs/
└── my-overlay/
    └── ROM/
        └── 118/
            └── 108.DAT    # Weapons overlay
```

The overlay mirrors the game's ROM structure exactly.

### How XIPivot Works

1. Game requests `ROM/118/108.DAT`
2. XIPivot intercepts the request
3. Checks overlays in reverse priority order
4. Returns first matching file found
5. Falls back to original game file

## Item ID Allocation

### Retail ID Ranges

| Range | File | Type | Count |
|-------|------|------|-------|
| 0-4095 | ROM/118/106.DAT | General | 4096 |
| 4096-8191 | ROM/118/107.DAT | Usable | 4096 |
| 8192-8703 | ROM/118/110.DAT | Puppet | 512 |
| 10240-16383 | ROM/118/109.DAT | Armor 1 | 6144 |
| 16384-23039 | ROM/118/108.DAT | Weapons | 6656 |
| 23040-28671 | ROM/286/73.DAT | Armor 2 | 5632 |

### Safe Custom Ranges

For custom items, use IDs at the end of each range that are unused in retail:

- **Weapons**: 22500-23039 (540 IDs available)
- **Armor 2**: 28000-28671 (672 IDs available)

Check retail item lists before allocating IDs to avoid conflicts.

## Template-Based Patching

### Why Templates Matter

The item entry contains many unknown fields that the client uses for the detail view. If these are zero, the item appears in inventory but shows no info when selected.

### The Solution

Copy an existing item entry entirely, then modify only the fields we understand:

```python
def create_from_template(template: bytes, item_id: int, name: str, ...):
    entry = bytearray(template)  # Full copy

    # Modify known fields
    struct.pack_into('<I', entry, 0x00, item_id)
    struct.pack_into('<H', entry, 0x0E, level)
    struct.pack_into('<H', entry, 0x1C, dmg)
    # ... etc

    return entry
```

This preserves:
- Unknown field at 0x0A (critical for detail view)
- Equipment compatibility data
- Icon format and data
- Any other fields we don't understand

## HQ Border Compositing

### Border Application

HQ borders are drawn programmatically as a 2px frame (outer ring + inner ring):

```python
from lps.icons import apply_hq_border
result = apply_hq_border(base_icon, hq_level=1)
```

### Border Colors

| Level | Outer RGB | Inner RGB |
|-------|-----------|-----------|
| +1 | (183, 183, 183) | (159, 159, 159) |
| +2 | (64, 127, 164) | (69, 112, 141) |
| +3 | (166, 142, 68) | (134, 119, 77) |
| +4 | (61, 165, 114) | (78, 141, 71) |

### Custom Borders

Modules can provide their own border styles (e.g. gradient borders as PNGs).
Custom borders are composited **under** the icon so the frame shows around
transparent edges while the icon content sits on top.

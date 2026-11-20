# Items Module

Patches items within the standard FFXI ID ranges (0–28671) by modifying DAT entries in XIPivot overlay files.

## Features

- **Template-based patching**: Copy existing items to preserve unknown fields
- **Custom text**: Set item names, descriptions, singular/plural forms
- **HQ icon borders**: +1 (white), +2 (blue), +3 (orange)
- **Icon tinting**: Color tints with blend modes (normal, multiply, screen, overlay, add)
- **Hue shifting**: Rotate icon colors while preserving shading
- **Field overrides**: Modify damage, delay, level, flags, etc.

## Item ID Ranges

| Range | DAT File | Type |
|-------|----------|------|
| 0-4095 | ROM/118/106.DAT | General |
| 4096-8191 | ROM/118/107.DAT | Usable |
| 8192-8703 | ROM/118/110.DAT | Puppet |
| 10240-16383 | ROM/118/109.DAT | Armor |
| 16384-23039 | ROM/118/108.DAT | Weapons |
| 23040-28671 | ROM/286/73.DAT | Armor2 |
| 28672-65535 | ROM/286/74.DAT | General2 |

## Config Format

```json
{
  "custom": {
    "item": {
      "items": {
        "22501": {
          "name": "Test Blade",
          "description": "DMG:50 Delay:240",
          "template": 16535,
          "hq": 1,
          "fields": { "dmg": 50, "delay": 240 }
        }
      }
    }
  }
}
```

## Tools

- `tools/item-creator/` -- Web UI for creating item configs
- `tools/preview/` -- Icon preview utility

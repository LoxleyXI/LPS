#!/usr/bin/env python3
"""Generate items.json for LPS Studio's template search dropdown.

Reads item names from LSB's item_basic.sql and outputs a compact JSON index.

Run:  LSB_SQL_DIR=/path/to/sql python3 gen_items.py
"""
import json, os, re
from pathlib import Path


def main():
    script_dir = Path(__file__).resolve().parent
    search_paths = []
    if os.environ.get('LSB_SQL_DIR'):
        search_paths.append(Path(os.environ['LSB_SQL_DIR']) / 'item_basic.sql')
    search_paths.append(script_dir / 'item_basic.sql')

    sql_path = None
    for p in search_paths:
        if p.exists():
            sql_path = p
            break

    if not sql_path:
        print("Error: item_basic.sql not found. Set LSB_SQL_DIR or place it next to this script.")
        return

    items = []
    with open(sql_path) as f:
        for line in f:
            m = re.match(
                r"INSERT INTO `item_basic` VALUES \((\d+),\d+,'([^']+)',",
                line
            )
            if not m:
                continue
            item_id = int(m.group(1))
            name = m.group(2).replace('_', ' ').title()
            items.append([item_id, name])

    # Sort by ID
    items.sort(key=lambda x: x[0])

    out = script_dir / 'items_index.json'
    out.write_text(json.dumps(items, ensure_ascii=False), encoding='utf-8')
    print(f"Wrote {len(items)} items -> {out}")


if __name__ == '__main__':
    main()

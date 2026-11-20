# Models Module

FTABLE/VTABLE management for FFXI model IDs. Provides model lookup, empty slot finding, and model info queries.

## Features

- Read FTABLE.DAT / VTABLE.DAT entries
- Find empty slots for custom models
- Resolve model ID to DAT path
- Read model DAT files

## API

```python
from modules.models import get_ftable_entry, find_empty_ftable_slots, get_model_info

entry = get_ftable_entry(1234)
empty = find_empty_ftable_slots(count=10)
info = get_model_info(1234)
```

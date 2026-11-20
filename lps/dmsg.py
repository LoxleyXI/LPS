"""
LPS DMSG - Dialog message encoding/decoding

DMSG entries contain:
- [0] name (short, for inventory display)
- [1] article (0 or 1 for a/an)
- [2] singular (full lowercase, "You obtained a <singular>")
- [3] plural (full lowercase, "You obtained 3 <plural>")
- [4] description
"""

import struct

# DMSG string header (28 bytes, mostly zeros)
DMSG_STRING_HEADER = bytes([
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00
])


def encode_dmsg_entry(elems: list) -> bytes:
    """
    Encode a DMSG entry.

    Args:
        elems: List of strings or integers

    Returns:
        Encoded bytes
    """
    num_elems = len(elems)
    header_size = 4 + num_elems * 8

    encoded_data = []
    for elem in elems:
        if isinstance(elem, str):
            str_bytes = elem.encode('ascii', errors='replace') + b'\x00'
            encoded_data.append((0, DMSG_STRING_HEADER + str_bytes))
        else:
            encoded_data.append((1, struct.pack('<I', elem)))

    entry_table = bytearray()
    entry_table.extend(struct.pack('<I', num_elems))

    data_offset = header_size
    for elem_type, elem_data in encoded_data:
        entry_table.extend(struct.pack('<I', data_offset))
        entry_table.extend(struct.pack('<I', elem_type))
        data_offset += len(elem_data)

    result = bytearray(entry_table)
    for _, elem_data in encoded_data:
        result.extend(elem_data)

    return bytes(result)


def decode_dmsg_entry(data: bytes) -> list:
    """
    Decode a DMSG entry.

    Args:
        data: Raw bytes of DMSG entry

    Returns:
        List of strings/integers
    """
    if len(data) < 4:
        return []

    num_elems = struct.unpack('<I', data[0:4])[0]
    if num_elems == 0 or num_elems > 20:
        return []

    elems = []
    offset = 4

    for _ in range(num_elems):
        if offset + 8 > len(data):
            break
        elem_offset = struct.unpack('<I', data[offset:offset+4])[0]
        elem_type = struct.unpack('<I', data[offset+4:offset+8])[0]
        offset += 8

        if elem_type == 0:  # String
            str_start = elem_offset + 0x1C
            str_end = str_start
            while str_end < len(data) and data[str_end] != 0:
                str_end += 1
            try:
                elems.append(data[str_start:str_end].decode('ascii', errors='replace'))
            except:
                elems.append('')
        else:  # Integer
            if elem_offset + 4 <= len(data):
                elems.append(struct.unpack('<I', data[elem_offset:elem_offset+4])[0])

    return elems


def create_item_text(name: str, description: str,
                     singular: str = None, plural: str = None) -> bytes:
    """
    Create DMSG text entry for an item.

    Args:
        name: Short name for inventory display
        description: Item description
        singular: Full singular form (default: lowercase name)
        plural: Full plural form (default: singular + 's')

    Returns:
        Encoded DMSG bytes
    """
    full_name = singular or name.lower()
    article = 1 if full_name[0].upper() in 'AEIOU' else 0

    text_elems = [
        name,
        article,
        singular or name.lower(),
        plural or (singular or name.lower()) + 's',
        description
    ]

    return encode_dmsg_entry(text_elems)

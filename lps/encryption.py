"""
LPS Encryption - DAT file encryption/decryption
"""

from pathlib import Path


def ror5(data: bytes) -> bytes:
    """Rotate right by 5 bits (decryption)"""
    return bytes(((b >> 5) | (b << 3)) & 0xFF for b in data)


def rol5(data: bytes) -> bytes:
    """Rotate left by 5 bits (encryption)"""
    return bytes(((b << 5) | (b >> 3)) & 0xFF for b in data)


def read_dat(dat_path: Path) -> bytes:
    """Read and decrypt a DAT file"""
    with open(dat_path, 'rb') as f:
        encrypted = f.read()
    return ror5(encrypted)


def write_dat(dat_path: Path, data: bytes):
    """Encrypt and write a DAT file"""
    dat_path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = rol5(data)
    with open(dat_path, 'wb') as f:
        f.write(encrypted)
    print(f"Written: {dat_path}")

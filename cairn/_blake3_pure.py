"""Pure-Python BLAKE3 implementation.

This is the zero-dependency fallback hasher for Cairn. It is intentionally
not fast (~tens of KB/s); it exists so that ``cairn`` works correctly with
*no* third-party packages installed. When the ``blake3`` PyPI package is
available, :mod:`cairn.hash` prefers its (C/Rust) implementation and this
module is only used as a cross-check.

The implementation follows the BLAKE3 specification (BLAKE3-team/BLAKE3)
and is validated against the official ``test_vectors.json`` in the test
suite (``tests/test_hash.py``).

Only 32-byte outputs are supported, which is all Cairn needs.
"""

from __future__ import annotations

import struct

IV = [
    0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
    0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19,
]

MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]

CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8
KEYED_HASH = 16
DERIVE_KEY_CONTEXT = 32
DERIVE_KEY_MATERIAL = 64

_BLOCK_LEN = 64
_CHUNK_LEN = 1024
_MASK = 0xFFFFFFFF


def _rotr(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK


def _g(s: list[int], a: int, b: int, c: int, d: int, mx: int, my: int) -> None:
    s[a] = (s[a] + s[b] + mx) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 16)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 12)
    s[a] = (s[a] + s[b] + my) & _MASK
    s[d] = _rotr(s[d] ^ s[a], 8)
    s[c] = (s[c] + s[d]) & _MASK
    s[b] = _rotr(s[b] ^ s[c], 7)


def _round(s: list[int], m: list[int]) -> None:
    # Mix the columns.
    _g(s, 0, 4, 8, 12, m[0], m[1])
    _g(s, 1, 5, 9, 13, m[2], m[3])
    _g(s, 2, 6, 10, 14, m[4], m[5])
    _g(s, 3, 7, 11, 15, m[6], m[7])
    # Mix the diagonals.
    _g(s, 0, 5, 10, 15, m[8], m[9])
    _g(s, 1, 6, 11, 12, m[10], m[11])
    _g(s, 2, 7, 8, 13, m[12], m[13])
    _g(s, 3, 4, 9, 14, m[14], m[15])


def _compress(
    cv: list[int],
    block_words: list[int],
    counter: int,
    block_len: int,
    flags: int,
) -> list[int]:
    """The BLAKE3 compression function. Returns 16 words."""
    s = (
        list(cv)
        + IV[:4]
        + [
            counter & _MASK,
            (counter >> 32) & _MASK,
            block_len,
            flags,
        ]
    )
    m = list(block_words)
    for _ in range(7):
        _round(s, m)
        m = [m[MSG_PERMUTATION[i]] for i in range(16)]
    for i in range(8):
        s[i] ^= s[i + 8]
        s[i + 8] ^= cv[i]
    return s


def _block_words(block: bytes) -> list[int]:
    padded = block + b"\x00" * (_BLOCK_LEN - len(block))
    return list(struct.unpack("<16I", padded))


def _chunk_end(
    chunk: bytes,
    counter: int,
    key_words: list[int],
    flags_base: int,
    root: bool,
) -> bytes | list[int]:
    """Hash one 1024-byte chunk.

    Returns the 32-byte root output when ``root`` is True (used only when the
    whole input is a single chunk), otherwise the 8-word chaining value.
    """
    cv = list(key_words)
    nblocks = (len(chunk) + _BLOCK_LEN - 1) // _BLOCK_LEN or 1
    for i in range(nblocks):
        block = chunk[i * _BLOCK_LEN : (i + 1) * _BLOCK_LEN]
        flags = flags_base
        if i == 0:
            flags |= CHUNK_START
        last = i == nblocks - 1
        if last:
            flags |= CHUNK_END
            if root:
                flags |= ROOT
        out = _compress(cv, _block_words(block), counter, len(block), flags)
        if last and root:
            return struct.pack("<8I", *out[:8])
        cv = out[:8]
    return cv


def _parent_combine(
    left: list[int],
    right: list[int],
    key_words: list[int],
    flags_base: int,
    root: bool,
) -> bytes | list[int]:
    out = _compress(
        list(key_words),
        list(left) + list(right),
        0,
        _BLOCK_LEN,
        flags_base | PARENT | (ROOT if root else 0),
    )
    if root:
        return struct.pack("<8I", *out[:8])
    return out[:8]


def _raw(data: bytes, key_words: list[int], flags_base: int) -> bytes:
    chunks = [data[i : i + _CHUNK_LEN] for i in range(0, len(data), _CHUNK_LEN)]
    if not chunks:
        chunks = [b""]
    if len(chunks) == 1:
        result = _chunk_end(chunks[0], 0, key_words, flags_base, root=True)
        assert isinstance(result, bytes)
        return result
    cvs: list = [
        _chunk_end(c, i, key_words, flags_base, root=False)
        for i, c in enumerate(chunks)
    ]
    # Fold the tree left-to-right, promoting an odd trailing CV at each level.
    # This is equivalent to BLAKE3's lazy stack evaluation.
    while len(cvs) > 1:
        nxt: list = []
        for i in range(0, len(cvs) - 1, 2):
            is_root = len(cvs) == 2
            nxt.append(
                _parent_combine(cvs[i], cvs[i + 1], key_words, flags_base, is_root)
            )
        if len(cvs) % 2 == 1:
            nxt.append(cvs[-1])
        cvs = nxt
    result = cvs[0]
    if isinstance(result, bytes):
        return result
    return struct.pack("<8I", *result)


def _key_words(key: bytes) -> list[int]:
    if len(key) != 32:
        raise ValueError("BLAKE3 key must be 32 bytes")
    return list(struct.unpack("<8I", key))


def blake3(data: bytes) -> bytes:
    """BLAKE3 hash of ``data``; returns 32 bytes."""
    return _raw(bytes(data), IV, 0)


def blake3_keyed(data: bytes, key: bytes) -> bytes:
    """BLAKE3 keyed hash; returns 32 bytes."""
    return _raw(bytes(data), _key_words(bytes(key)), KEYED_HASH)


def blake3_derive_key(context: bytes, key_material: bytes) -> bytes:
    """BLAKE3 key derivation; returns 32 bytes.

    ``context`` should be a fixed, hardcoded context string.
    """
    context_key = _raw(bytes(context), IV, DERIVE_KEY_CONTEXT)
    return _raw(bytes(key_material), _key_words(context_key), DERIVE_KEY_MATERIAL)

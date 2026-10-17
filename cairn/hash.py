"""Hashing layer for Cairn.

BLAKE3 is the canonical hash for all content addressing in Cairn. This module
prefers the native ``_cairn_native`` extension (built from ``rust/``) when it
is available, then the fast ``blake3`` PyPI package, and finally the bundled
pure-Python implementation (:mod:`cairn._blake3_pure`) so that Cairn has
**zero hard dependencies**.

All backends are cross-validated against the official BLAKE3 test vectors in
the test suite.
"""

from __future__ import annotations

from typing import Callable

from . import _blake3_pure, _native

try:  # Optional fast path: the official `blake3` package (Rust extension).
    from blake3 import blake3 as _fast_blake3

    _FAST_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _fast_blake3 = None  # type: ignore[assignment]
    _FAST_AVAILABLE = False


def _digest_native(data: bytes) -> bytes:
    assert _native.blake3_hash is not None
    return bytes(_native.blake3_hash(bytes(data)))


def _digest_fast(data: bytes) -> bytes:
    assert _fast_blake3 is not None
    return _fast_blake3(data).digest()


def _digest_pure(data: bytes) -> bytes:
    return _blake3_pure.blake3(data)


def backend() -> str:
    """Name of the BLAKE3 backend in use: ``'native'``, ``'fast'`` or ``'pure'``."""
    if _native.HAS_NATIVE:
        return "native"
    return "fast" if _FAST_AVAILABLE else "pure"


def fast_available() -> bool:
    return _FAST_AVAILABLE or _native.HAS_NATIVE


def digest(data: bytes) -> bytes:
    """BLAKE3 digest of ``data``; 32 bytes."""
    if _native.HAS_NATIVE:
        return _digest_native(data)
    if _FAST_AVAILABLE:
        return _digest_fast(bytes(data))
    return _digest_pure(bytes(data))


def hexdigest(data: bytes) -> str:
    return digest(data).hex()


def pure_digest(data: bytes) -> bytes:
    """Digest using the bundled pure-Python backend (for cross-checks)."""
    return _digest_pure(bytes(data))


def keyed_digest(data: bytes, key: bytes) -> bytes:
    """BLAKE3 keyed hash; 32 bytes. Pure-Python backend only (rarely hot)."""
    return _blake3_pure.blake3_keyed(bytes(data), bytes(key))


def derive_key(context: bytes, key_material: bytes) -> bytes:
    """BLAKE3 key derivation; 32 bytes. Pure-Python backend only."""
    return _blake3_pure.blake3_derive_key(bytes(context), bytes(key_material))


Hasher = Callable[[bytes], bytes]

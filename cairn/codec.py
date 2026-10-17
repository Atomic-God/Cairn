"""Canonical encoding and compression primitives.

All hashing and signing in Cairn operates on canonical byte representations
defined here, so that node identities are stable across platforms, Python
versions, and process runs.
"""

from __future__ import annotations

import base64
import json
import struct
import zlib
from typing import Any


def canonical(obj: Any) -> bytes:
    """Deterministic JSON encoding: sorted keys, no whitespace, UTF-8.

    Only JSON-native types are allowed (dict, list, str, int, float, bool,
    None). Non-finite floats (NaN, Infinity) are rejected outright: they are
    not valid JSON, so no other implementation could reproduce the bytes.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_loads(data: bytes) -> Any:
    return json.loads(data.decode("utf-8"))


def u16be(n: int) -> bytes:
    return struct.pack(">H", n)


def u32be(n: int) -> bytes:
    return struct.pack(">I", n)


def u64be(n: int) -> bytes:
    return struct.pack(">Q", n)


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def compress(data: bytes) -> bytes:
    return zlib.compress(bytes(data), 9)


def decompress(data: bytes) -> bytes:
    return zlib.decompress(bytes(data))

"""BLAKE3 correctness tests against the official test vectors."""

import json
import os

import pytest

from cairn import _blake3_pure, hash as hashmod

VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "vectors", "blake3_test_vectors.json"
)


def _load_vectors():
    with open(VECTORS_PATH, "r", encoding="utf-8") as f:
        doc = json.load(f)
    key = doc["key"].encode("utf-8")
    assert len(key) == 32, "official test vector key must be 32 bytes"
    context = doc["context_string"].encode("utf-8")
    cases = []
    for c in doc["cases"]:
        n = c["input_len"]
        # The official generator hashes the repeating pattern 0,1,2,...,250,0,1,...
        data = bytes(i % 251 for i in range(n))
        cases.append((n, data, c["hash"], c["keyed_hash"], c["derive_key"]))
    return key, context, cases


KEY, CONTEXT, CASES = _load_vectors()


# The official vectors are 131-byte XOF outputs; Cairn only needs the
# leading 32 bytes, which must match exactly.
def _h32(hexstr: str) -> str:
    return hexstr[:64]


@pytest.mark.parametrize(
    "n,data,expected,_,__", CASES, ids=[f"len={c[0]}" for c in CASES]
)
def test_pure_blake3_against_official_vectors(n, data, expected, _, __):
    assert _blake3_pure.blake3(data).hex() == _h32(expected)


@pytest.mark.parametrize(
    "n,data,_,expected,__", CASES, ids=[f"len={c[0]}" for c in CASES]
)
def test_pure_blake3_keyed_against_official_vectors(n, data, _, expected, __):
    assert _blake3_pure.blake3_keyed(data, KEY).hex() == _h32(expected)


@pytest.mark.parametrize(
    "n,data,_,__,expected", CASES, ids=[f"len={c[0]}" for c in CASES]
)
def test_pure_blake3_derive_key_against_official_vectors(n, data, _, __, expected):
    assert _blake3_pure.blake3_derive_key(CONTEXT, data).hex() == _h32(expected)


def test_hash_module_agrees_with_pure_backend():
    samples = [b"", b"a", b"abc", os.urandom(63), os.urandom(64),
               os.urandom(1023), os.urandom(1024), os.urandom(5000)]
    for s in samples:
        assert hashmod.digest(s) == _blake3_pure.blake3(s), f"mismatch at len={len(s)}"


def test_digest_properties():
    assert len(hashmod.digest(b"hello")) == 32
    assert hashmod.digest(b"hello") != hashmod.digest(b"hellp")  # avalanche
    assert hashmod.digest(b"hello") == hashmod.digest(b"hello")  # deterministic
    assert hashmod.backend() in ("native", "fast", "pure")

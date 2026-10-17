"""Ed25519 correctness: RFC 8032 vectors + round-trip properties."""

import json
import os

import pytest

from cairn import signing
from cairn.signing import BadSignatureError

VECTORS_PATH = os.path.join(
    os.path.dirname(__file__), "vectors", "ed25519_rfc8032.json"
)


def _load():
    with open(VECTORS_PATH, encoding="utf-8") as f:
        return json.load(f)


VECTORS = _load()


@pytest.mark.parametrize("v", VECTORS, ids=[f"TEST-{v['num']}" for v in VECTORS])
def test_rfc8032_public_key_derivation(v):
    sk = bytes.fromhex(v["sk"])
    assert signing.public_key(sk).hex() == v["pk"]


@pytest.mark.parametrize("v", VECTORS, ids=[f"TEST-{v['num']}" for v in VECTORS])
def test_rfc8032_signature_vectors(v):
    sk = bytes.fromhex(v["sk"])
    pk = bytes.fromhex(v["pk"])
    msg = bytes.fromhex(v["msg"])
    assert signing.sign(msg, sk, pk).hex() == v["sig"]


@pytest.mark.parametrize("v", VECTORS, ids=[f"TEST-{v['num']}" for v in VECTORS])
def test_rfc8032_verify(v):
    sig = bytes.fromhex(v["sig"])
    pk = bytes.fromhex(v["pk"])
    msg = bytes.fromhex(v["msg"])
    signing.verify(sig, msg, pk)  # must not raise


def test_roundtrip_random_keys():
    for _ in range(3):
        sk, pk = signing.generate_keypair()
        assert len(sk) == 32 and len(pk) == 32
        msg = os.urandom(128)
        sig = signing.sign(msg, sk, pk)
        assert len(sig) == 64
        signing.verify(sig, msg, pk)


def test_tampered_message_fails():
    sk, pk = signing.generate_keypair()
    sig = signing.sign(b"hello", sk, pk)
    with pytest.raises(BadSignatureError):
        signing.verify(sig, b"hellp", pk)


def test_tampered_signature_fails():
    sk, pk = signing.generate_keypair()
    sig = bytearray(signing.sign(b"hello", sk, pk))
    sig[10] ^= 1
    with pytest.raises(BadSignatureError):
        signing.verify(bytes(sig), b"hello", pk)


def test_wrong_key_fails():
    sk1, pk1 = signing.generate_keypair()
    _, pk2 = signing.generate_keypair()
    sig = signing.sign(b"hello", sk1, pk1)
    with pytest.raises(BadSignatureError):
        signing.verify(sig, b"hello", pk2)


def test_malformed_inputs_rejected():
    sk, pk = signing.generate_keypair()
    with pytest.raises(BadSignatureError):
        signing.verify(b"short", b"m", pk)
    with pytest.raises(BadSignatureError):
        signing.verify(b"\x00" * 64, b"m", b"short")
    with pytest.raises(ValueError):
        signing.public_key(b"short")


def test_scalarmult_base_agrees_with_generic():
    import random
    rng = random.Random(1234)
    for _ in range(8):
        e = rng.getrandbits(253)
        assert signing._scalarmult_base(e) == signing._scalarmult(signing._BASE, e)
    assert signing._scalarmult_base(0) == (0, 1)
    assert signing._scalarmult_base(1) == signing._BASE


def test_signer_rfc_vector():
    v = [x for x in VECTORS if x["num"] == "1"][0]
    s = signing.Signer(bytes.fromhex(v["sk"]))
    assert s.public_key.hex() == v["pk"]
    assert s.sign(bytes.fromhex(v["msg"])).hex() == v["sig"]
    signing.verify(s.sign(bytes.fromhex(v["msg"])), bytes.fromhex(v["msg"]), s.public_key)


def test_signer_is_cached_and_fast():
    import time
    sk, _ = signing.generate_keypair()
    s = signing.Signer(sk)
    t0 = time.perf_counter()
    for _ in range(5):
        s.sign(b"benchmark")
    dt = (time.perf_counter() - t0) / 5 * 1000
    assert dt < 500, f"sign too slow: {dt:.0f}ms"


def test_native_backend_parity():
    """When the _cairn_native extension is present, it must agree
    byte-for-byte with the pure-Python reference implementation."""
    from cairn import _native
    from cairn import hash as hashmod
    from cairn import signing as signingmod

    if not _native.HAS_NATIVE:
        pytest.skip("native extension not built")
    assert hashmod.backend() == "native"

    data = b"cairn parity check" * 64
    assert hashmod.digest(data) == hashmod.pure_digest(data)

    sk, pk = signingmod.generate_keypair()
    msg = b"parity message"
    sig_native = signingmod.sign(msg, sk)

    # Pure-Python reference signing of the same key/message.
    _native.HAS_NATIVE = False
    try:
        ref = signingmod.Signer(sk)
        assert ref.public_key == pk  # same key derivation
        sig_pure = ref.sign(msg)
    finally:
        _native.HAS_NATIVE = True

    assert sig_native == sig_pure, "native/pure signatures must be identical"
    # Cross-verify in both directions.
    signingmod.verify(sig_pure, msg, pk)
    _native.HAS_NATIVE = False
    try:
        signingmod.verify(sig_native, msg, pk)
    finally:
        _native.HAS_NATIVE = True


def test_native_absent_fallback(monkeypatch):
    """With the extension hidden, everything still works via pure Python."""
    from cairn import _native
    from cairn import hash as hashmod
    from cairn import signing as signingmod

    monkeypatch.setattr(_native, "HAS_NATIVE", False)
    assert hashmod.backend() in ("fast", "pure")
    sk, pk = signingmod.generate_keypair()
    sig = signingmod.sign(b"m", sk)
    signingmod.verify(sig, b"m", pk)

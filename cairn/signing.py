"""Ed25519 digital signatures.

Cairn signs every committed node so that audit trails are attributable and
tamper-evident. This module ships a faithful, self-contained pure-Python
implementation of Ed25519 (RFC 8032), in the public-domain reference style
of Daniel J. Bernstein's ``ref10``/``ed25519.py``. It has **zero
dependencies**.

When the optional ``_cairn_native`` extension (built from ``rust/``) is
available, signing and verification use it instead — same keys, same
signatures, ~1000× faster. The pure-Python path remains the correctness
reference and is validated against the RFC 8032 vectors.

Key handling (generation, storage) lives in :mod:`cairn.keys`.
"""

from __future__ import annotations

import hashlib

from . import _native

_B = 256
_Q = 2**255 - 19
_L = 2**252 + 27742317777372353535851937790883648493
_MASK = _Q  # reused for mod q via %

_D = (-121665 * pow(121666, _Q - 2, _Q)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)


def _H(data: bytes) -> bytes:
    return hashlib.sha512(data).digest()


def _inv(x: int) -> int:
    return pow(x, _Q - 2, _Q)


def _xrecover(y: int) -> int:
    xx = (y * y - 1) * _inv(_D * y * y + 1) % _Q
    x = pow(xx, (_Q + 3) // 8, _Q)
    if (x * x - xx) % _Q != 0:
        x = (x * _I) % _Q
    if x % 2 != 0:
        x = _Q - x
    return x


_B_y = (4 * _inv(5)) % _Q
_B_x = _xrecover(_B_y)
_BASE = (_B_x, _B_y)  # base point B


def _edwards(P: tuple[int, int], Q: tuple[int, int]) -> tuple[int, int]:
    # Twisted Edwards addition for a = -1:
    #   x3 = (x1*y2 + x2*y1) / (1 + d*x1*x2*y1*y2)
    #   y3 = (y1*y2 - a*x1*x2) / (1 - d*x1*x2*y1*y2)   (a = -1)
    x1, y1 = P
    x2, y2 = Q
    x3 = (x1 * y2 + x2 * y1) * _inv(1 + _D * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * _inv(1 - _D * x1 * x2 * y1 * y2)
    return (x3 % _Q, y3 % _Q)


def _scalarmult(P: tuple[int, int], e: int) -> tuple[int, int]:
    if e == 0:
        return (0, 1)
    Q = _scalarmult(P, e // 2)
    Q = _edwards(Q, Q)
    if e & 1:
        Q = _edwards(Q, P)
    return Q


def _encodeint(y: int) -> bytes:
    return y.to_bytes(32, "little")


def _encodepoint(P: tuple[int, int]) -> bytes:
    x, y = P
    return _encodeint((y & ~(1 << 255)) | ((x & 1) << 255))


def _bit(h: bytes, i: int) -> int:
    return (h[i // 8] >> (i % 8)) & 1


def _decodeint(s: bytes) -> int:
    return int.from_bytes(s, "little")


# ---------------------------------------------------------------------------
# Fixed-base scalar multiplication via a precomputed table.
#
# Signing and public-key derivation always multiply the base point B, so we
# precompute T[i] = 2^i * B once and answer each multiplication with ~128
# point additions instead of ~256 double-and-add steps (each of which costs
# two modular inverses). This is ~4x faster than the generic routine below
# and is what makes per-commit signing cheap. Verification still uses the
# generic routine (it multiplies arbitrary points).
# ---------------------------------------------------------------------------

_base_table: list[tuple[int, int]] | None = None


def _get_base_table() -> list[tuple[int, int]]:
    global _base_table
    table = _base_table
    if table is None:
        # Benign if two threads race here: the result is deterministic.
        table = [_BASE]
        for _ in range(255):
            table.append(_edwards(table[-1], table[-1]))
        _base_table = table
    return table


def _scalarmult_base(e: int) -> tuple[int, int]:
    """Compute e * B using the precomputed table. e must be >= 0."""
    e %= _L  # B has order L, so this is exact
    if e == 0:
        return (0, 1)
    table = _get_base_table()
    x, y = 0, 1  # identity
    i = 0
    while e:
        if e & 1:
            x, y = _edwards((x, y), table[i])
        e >>= 1
        i += 1
    return (x % _Q, y % _Q)


def _decodepoint(s: bytes) -> tuple[int, int]:
    if len(s) != 32:
        raise ValueError("encoded point must be 32 bytes")
    y = _decodeint(s) & ~(1 << 255)
    x = _xrecover(y)
    if x & 1 != _bit(s, _B - 1):
        x = _Q - x
    P = (x, y)
    if not _isoncurve(P):
        raise ValueError("point is not on the curve")
    return P


def _isoncurve(P: tuple[int, int]) -> bool:
    x, y = P
    return (-x * x + y * y - 1 - _D * x * x * y * y) % _Q == 0


def _Hint(m: bytes) -> int:
    return int.from_bytes(_H(m), "little")


def _clamp(h: bytes) -> int:
    a = 2 ** (_B - 2) + sum(2**i * _bit(h, i) for i in range(3, _B - 2))
    return a


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Signer:
    """A cached Ed25519 signing identity.

    Deriving the public key and clamping the secret scalar happen once, up
    front; every :meth:`sign` call then costs a single fixed-base scalar
    multiplication in pure Python (~50ms), or a native call (~50µs) when the
    ``_cairn_native`` extension is available. A :class:`Session` holds one
    of these for its lifetime, which keeps per-commit overhead low.
    """

    def __init__(self, secret_key: bytes):
        if len(secret_key) != 32:
            raise ValueError("secret key must be 32 bytes")
        self._sk = bytes(secret_key)
        if _native.HAS_NATIVE:
            assert _native.ed25519_public_key is not None
            self._pk = bytes(_native.ed25519_public_key(self._sk))
            self._scalar = 0
            self._prefix = b""
        else:
            h = _H(self._sk)
            self._scalar = _clamp(h)
            self._prefix = h[32:64]
            self._pk = _encodepoint(_scalarmult_base(self._scalar))

    @property
    def secret_key(self) -> bytes:
        return self._sk

    @property
    def public_key(self) -> bytes:
        return self._pk

    def sign(self, message: bytes) -> bytes:
        msg = bytes(message)
        if _native.HAS_NATIVE:
            assert _native.ed25519_sign is not None
            return bytes(_native.ed25519_sign(self._sk, msg))
        r = _Hint(self._prefix + msg)
        R = _scalarmult_base(r)
        S = (r + _Hint(_encodepoint(R) + self._pk + msg) * self._scalar) % _L
        return _encodepoint(R) + _encodeint(S)


def generate_keypair() -> tuple[bytes, bytes]:
    """Generate a fresh Ed25519 keypair.

    Returns ``(secret_key, public_key)``, each 32 bytes. Uses ``secrets``.
    """
    import secrets

    sk = secrets.token_bytes(32)
    signer = Signer(sk)
    return sk, signer.public_key


def public_key(secret_key: bytes) -> bytes:
    """Derive the 32-byte public key from a 32-byte secret key."""
    return Signer(secret_key).public_key


def sign(message: bytes, secret_key: bytes, public_key: bytes | None = None) -> bytes:
    """Sign ``message``; returns the 64-byte signature."""
    return Signer(secret_key).sign(message)


class BadSignatureError(ValueError):
    """Raised when a signature does not verify."""


def verify(signature: bytes, message: bytes, public_key_bytes: bytes) -> None:
    """Verify ``signature`` over ``message``.

    Raises :class:`BadSignatureError` on failure; returns ``None`` on success.
    """
    sig = bytes(signature)
    pk = bytes(public_key_bytes)
    msg = bytes(message)
    if len(sig) != 64:
        raise BadSignatureError("signature must be 64 bytes")
    if len(pk) != 32:
        raise BadSignatureError("public key must be 32 bytes")
    if _native.HAS_NATIVE:
        assert _native.ed25519_verify is not None
        if not _native.ed25519_verify(pk, msg, sig):
            raise BadSignatureError("signature verification failed")
        return
    try:
        R = _decodepoint(sig[:32])
        A = _decodepoint(pk)
    except ValueError as exc:
        raise BadSignatureError(str(exc)) from exc
    S = _decodeint(sig[32:])
    if S >= _L:
        raise BadSignatureError("S out of range")
    h = _Hint(_encodepoint(R) + pk + msg)
    if _scalarmult(_BASE, S) != _edwards(R, _scalarmult(A, h)):
        raise BadSignatureError("signature verification failed")

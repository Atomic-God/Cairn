"""Optional native accelerator for Cairn.

Tries to import the ``_cairn_native`` PyO3 extension (built from ``rust/``).
When the compiled module is present, :data:`HAS_NATIVE` is ``True`` and the
hot paths — BLAKE3 hashing and Ed25519 signing/verification — run as native
code. When it is absent (a plain ``pip install cairn-core``), everything
falls back to the pure-Python implementations, so Cairn keeps **zero hard
dependencies**.

Build it with::

    cargo build --release --features extension-module --manifest-path rust/Cargo.toml
    cp rust/target/release/lib_cairn_native.so cairn/_cairn_native.so

(or ``maturin develop --manifest-path rust/Cargo.toml``). The ``.so`` is a
build artifact and is gitignored; CI builds it and runs the suite against
both backends.
"""

from __future__ import annotations

try:  # The compiled PyO3 module; absent on a pure-Python install.
    from ._cairn_native import (  # type: ignore[import-not-found]
        blake3_hash,
        ed25519_public_key,
        ed25519_sign,
        ed25519_verify,
        native_version,
    )

    HAS_NATIVE = True
except Exception:  # ImportError, ABI mismatch, …: pure-Python fallback.
    HAS_NATIVE = False
    blake3_hash = None  # type: ignore[assignment]
    ed25519_public_key = None  # type: ignore[assignment]
    ed25519_sign = None  # type: ignore[assignment]
    ed25519_verify = None  # type: ignore[assignment]
    native_version = None  # type: ignore[assignment]

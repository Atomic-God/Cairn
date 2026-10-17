# cairn-native — optional PyO3 accelerator for cairn-core

Native implementations of Cairn's two commit hot paths — BLAKE3 hashing and
Ed25519 signing/verification — exposed as the `_cairn_native` extension
module. Byte-identical to the pure-Python reference: the same keys produce
the same signatures and the same hashes.

The Python side (`cairn/_native.py`) imports this module opportunistically.
If it is absent, Cairn runs on pure Python with zero dependencies.

## Build

```bash
# one-off: needs a Rust toolchain (rustup) and Python dev headers
cargo build --release --features extension-module
cp target/release/lib_cairn_native.so ../cairn/_cairn_native.so
# macOS: target/release/lib_cairn_native.dylib
# Windows: target/release/cairn_native.dll -> ../cairn/_cairn_native.pyd
```

Or with maturin (handles naming/placement):

```bash
pip install maturin
maturin develop --manifest-path rust/Cargo.toml   # from the repo root
```

## Test

```bash
cargo test          # Rust unit tests (BLAKE3 + RFC 8032 vectors)
cargo clippy --all-targets -- -D warnings
```

Note: `cargo test` builds *without* the `extension-module` feature so the
test binary links libpython normally; the feature is only for the loadable
`.so`.

## Benchmarks (x86_64 Linux, release)

| op | native | pure Python |
|---|---|---|
| Ed25519 sign | ~55 µs | ~48 ms |
| BLAKE3 (128 B) | ~0.5 µs | ~30 µs |
| end-to-end commit | ~0.6 ms | ~48 ms |

End-to-end commit time is dominated by durable I/O (`fsync`, atomic ref
writes, zlib) — the crypto itself is comfortably sub-200 µs.

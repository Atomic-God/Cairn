//! Native accelerator for cairn-core.
//!
//! Exposes the two hot paths of a commit — BLAKE3 hashing and Ed25519
//! signing/verification — as a PyO3 extension module named `_cairn_native`.
//! The Python side (`cairn/_native.py`) imports this opportunistically: when
//! the compiled module is absent, the pure-Python implementations are used
//! instead, so `pip install cairn-core` stays dependency-free.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn signing_key(secret_key: &[u8]) -> PyResult<SigningKey> {
    let bytes: [u8; 32] = secret_key
        .try_into()
        .map_err(|_| PyValueError::new_err("secret key must be 32 bytes"))?;
    Ok(SigningKey::from_bytes(&bytes))
}

fn verifying_key(public_key: &[u8]) -> PyResult<VerifyingKey> {
    let bytes: [u8; 32] = public_key
        .try_into()
        .map_err(|_| PyValueError::new_err("public key must be 32 bytes"))?;
    VerifyingKey::from_bytes(&bytes)
        .map_err(|e| PyValueError::new_err(format!("bad public key: {e}")))
}

/// BLAKE3 digest of `data` (32 raw bytes).
#[pyfunction]
fn blake3_hash(data: &[u8]) -> Vec<u8> {
    blake3::hash(data).as_bytes().to_vec()
}

/// Ed25519 public key for a 32-byte secret key (32 raw bytes).
#[pyfunction]
fn ed25519_public_key(secret_key: &[u8]) -> PyResult<Vec<u8>> {
    Ok(signing_key(secret_key)?.verifying_key().to_bytes().to_vec())
}

/// Ed25519 signature of `msg` under `secret_key` (64 raw bytes).
#[pyfunction]
fn ed25519_sign(secret_key: &[u8], msg: &[u8]) -> PyResult<Vec<u8>> {
    Ok(signing_key(secret_key)?.sign(msg).to_bytes().to_vec())
}

/// Verify an Ed25519 signature; returns true/false rather than raising.
#[pyfunction]
fn ed25519_verify(public_key: &[u8], msg: &[u8], signature: &[u8]) -> PyResult<bool> {
    let sig_bytes: [u8; 64] = signature
        .try_into()
        .map_err(|_| PyValueError::new_err("signature must be 64 bytes"))?;
    let sig = Signature::from_bytes(&sig_bytes);
    Ok(verifying_key(public_key)?.verify(msg, &sig).is_ok())
}

/// Module version string, e.g. "0.1.0".
#[pyfunction]
fn native_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _cairn_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(blake3_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ed25519_public_key, m)?)?;
    m.add_function(wrap_pyfunction!(ed25519_sign, m)?)?;
    m.add_function(wrap_pyfunction!(ed25519_verify, m)?)?;
    m.add_function(wrap_pyfunction!(native_version, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests;

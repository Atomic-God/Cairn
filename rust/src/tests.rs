//! Rust-side tests for the native accelerator. These run with `cargo test`
//! and do not need a Python interpreter.

use crate::{blake3_hash, ed25519_public_key, ed25519_sign, ed25519_verify};

fn hex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn blake3_empty_vector() {
    // Official BLAKE3 test vector: hash of the empty string.
    assert_eq!(
        hex("af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
        blake3_hash(b""),
    );
}

#[test]
fn blake3_abc_vector() {
    // BLAKE3("abc"), cross-checked against the pure-Python implementation
    // (which is validated against the official BLAKE3 test vectors).
    assert_eq!(
        hex("6437b3ac38465133ffb63b75273a8db548c558465d79db03fd359c6cd5bd9d85"),
        blake3_hash(b"abc"),
    );
}

#[test]
fn ed25519_rfc8032_test1() {
    // RFC 8032 §7.1 TEST 1, using the exact values from
    // tests/vectors/ed25519_rfc8032.json (the same file the Python suite
    // validates against, so both engines agree byte-for-byte).
    let sk = hex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
    let pk_expected = hex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a");
    let msg = b"";
    let sig_expected =
        hex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155\
             5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b");

    assert_eq!(pk_expected, ed25519_public_key(&sk).unwrap());
    assert_eq!(sig_expected, ed25519_sign(&sk, msg).unwrap());
    assert!(ed25519_verify(&pk_expected, msg, &sig_expected).unwrap());
}

#[test]
fn ed25519_rejects_bad_key_lengths() {
    assert!(ed25519_public_key(b"short").is_err());
    assert!(ed25519_sign(b"short", b"msg").is_err());
    assert!(ed25519_verify(b"short", b"msg", &[0u8; 64]).is_err());
}

#[test]
fn ed25519_verify_detects_tampering() {
    let sk = [7u8; 32];
    let pk = ed25519_public_key(&sk).unwrap();
    let sig = ed25519_sign(&sk, b"hello").unwrap();
    assert!(ed25519_verify(&pk, b"hello", &sig).unwrap());
    assert!(!ed25519_verify(&pk, b"hellx", &sig).unwrap());
}

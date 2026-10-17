"""Identity / key management for Cairn.

Each Cairn user has one Ed25519 identity. The secret key lives at
``~/.config/cairn/identity.key`` (mode 0600); the public key is published
inside every repository's ``.cairn/config.json`` and on every signed node.

An explicit ``CAIRN_KEY_FILE`` environment variable overrides the location,
which is what the test suite uses.
"""

from __future__ import annotations

import base64
import os
import stat

from . import signing

DEFAULT_DIR_NAME = ".config/cairn"
KEY_FILE_NAME = "identity.key"


def default_key_path() -> str:
    override = os.environ.get("CAIRN_KEY_FILE")
    if override:
        return override
    home = os.path.expanduser("~")
    return os.path.join(home, DEFAULT_DIR_NAME, KEY_FILE_NAME)


def _write_private(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    os.chmod(path, 0o600)


def load_or_create(path: str | None = None) -> tuple[bytes, bytes]:
    """Load ``(secret_key, public_key)``, generating and storing if missing."""
    path = path or default_key_path()
    if os.path.exists(path):
        with open(path, "rb") as f:
            sk = f.read()
        # NOTE: no .strip() here — the key is 32 raw bytes and any of them
        # may legitimately be ASCII whitespace. Stripping once corrupted
        # ~4.6% of generated keys on their next load.
        if len(sk) != 32:
            raise ValueError(f"key file {path} is corrupt (expected 32 bytes)")
        try:
            mode = os.stat(path).st_mode
            if mode & (stat.S_IRGRP | stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH):
                # Not fatal, but worth tightening.
                os.chmod(path, 0o600)
        except OSError:
            pass
        return sk, signing.public_key(sk)
    sk, pk = signing.generate_keypair()
    _write_private(path, sk)
    return sk, pk


def public_key_hex(path: str | None = None) -> str:
    _, pk = load_or_create(path)
    return pk.hex()


def key_fingerprint(public_key: bytes) -> str:
    """Short human-readable fingerprint of a public key."""
    from . import hash as hashmod

    return hashmod.hexdigest(b"cairn-key-v1" + bytes(public_key))[:16]


def export_public_key_armored(path: str | None = None) -> str:
    _, pk = load_or_create(path)
    b64 = base64.b64encode(pk).decode("ascii")
    return f"-----BEGIN CAIRN PUBLIC KEY-----\n{b64}\n-----END CAIRN PUBLIC KEY-----\n"

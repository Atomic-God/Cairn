# Security Policy

## Supported Versions

Only the latest minor release is actively maintained for security patches:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Do NOT open public GitHub issues for security vulnerabilities.
Submit reports via GitHub Private Vulnerability Reporting at:
https://github.com/Atomic-God/Cairn/security/advisories/new

We acknowledge reports within 48 hours and provide a patch timeline within 5 business days.

## Threat Model

### In Scope (What Cairn Protects Against)

* **Silent Trace Tampering:** Detecting any modification to past execution states via BLAKE3 Merkle lineage.
* **Corrupted Object State:** Detecting payload bitrot via hash-integrity checks on retrieval.
* **Accidental Concurrency Corruption:** Preventing corrupted references during parallel writes via atomic file replacement and advisory file locks (`flock`/`msvcrt`).
* **Unintentional Network Leaks During Replay:** Blocking socket I/O during deterministic replays using the internal network guard.

### Out of Scope (What Cairn Does Not Protect Against)

* **Compromised Host OS:** If the host kernel, filesystem, or Python runtime is compromised, integrity cannot be guaranteed.
* **Compromised Private Keys:** If an agent's signing key is leaked, unauthorized commits cannot be cryptographically distinguished from authorized ones.
* **Malicious Payloads Generated at Source:** Cairn records reality as observed; it cannot detect if an agent was intentionally instructed to emit malicious tool calls.

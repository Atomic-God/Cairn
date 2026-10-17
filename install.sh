#!/bin/sh
# Cairn installer — zero dependencies beyond a Python 3.10+ interpreter.
#
#   curl -fsSL https://raw.githubusercontent.com/Atomic-God/Cairn/main/install.sh | sh
#
# Installs the `cairn` package (and CLI) into a user-local venv at
# ~/.cairn/venv and links `cairn` onto the PATH via ~/.local/bin.
set -eu

REPO_URL="${CAIRN_REPO_URL:-https://github.com/Atomic-God/Cairn.git}"
PREFIX="${CAIRN_PREFIX:-$HOME/.cairn}"
VENV="$PREFIX/venv"
BIN_DIR="${CAIRN_BIN_DIR:-$HOME/.local/bin}"

need() { command -v "$1" >/dev/null 2>&1 || {
  echo "cairn: required command '$1' not found" >&2; exit 1; }; }

main() {
  need python3
  PYV="$(python3 -c 'import sys; print(sys.version_info[:2] >= (3, 10))')"
  [ "$PYV" = "True" ] || { echo "cairn: Python 3.10+ required" >&2; exit 1; }

  if [ -d "$PREFIX/src" ]; then
    echo "cairn: updating existing checkout at $PREFIX/src"
    git -C "$PREFIX/src" pull --ff-only
  else
    need git
    echo "cairn: cloning into $PREFIX/src"
    git clone --depth 1 "$REPO_URL" "$PREFIX/src"
  fi

  echo "cairn: creating venv at $VENV"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --quiet --upgrade pip
  "$VENV/bin/pip" install --quiet "$PREFIX/src"

  mkdir -p "$BIN_DIR"
  ln -sf "$VENV/bin/cairn" "$BIN_DIR/cairn"
  echo "cairn: installed. Make sure $BIN_DIR is on your PATH:"
  echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
  "$BIN_DIR/cairn" --help >/dev/null && echo "cairn: OK"
}

main "$@"

#!/usr/bin/env bash
#
# run.sh — bootstrap dependencies and launch the face-recognition app.
#
# Creates a local virtualenv (./venv) on first run, installs the pinned
# dependencies from requirements.txt, then starts the real-time recognizer.
# Re-running is cheap: the venv and installed deps are reused, and the
# install step is skipped unless requirements.txt has changed.
#
# Usage:
#   ./run.sh                 # launch real-time recognition (default)
#   ./run.sh --camera 1      # forward args to the recognizer (e.g. pick camera)
#   ./run.sh --verbose       # DEBUG logging
#
# Any arguments are forwarded verbatim to scripts/real_time_recognition.py.

set -euo pipefail

# Some Python builds (notably pyenv interpreters compiled without blake2
# support) make hashlib log a multi-line "unsupported hash type blake2b/blake2s"
# traceback to stderr at import time. It is harmless — nothing here uses blake2,
# and sha256 still works — but it floods the output on every invocation. This
# filter drops only that specific block, leaving all other output intact.
filter_hash_noise() {
  awk '
    /^ERROR:root:code for hash (blake2b|blake2s) was not found\./ { skip=1 }
    skip && /^ValueError: unsupported hash type/ { skip=0; next }
    skip { next }
    { print }
  '
}

# Resolve the repo root (directory containing this script) so the app can be
# launched from anywhere.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="$ROOT_DIR/venv"
PYTHON_BIN="$VENV_DIR/bin/python"
STAMP_FILE="$VENV_DIR/.requirements.installed"

# Pick a Python interpreter for creating the venv.
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: '$PYTHON' not found. Install Python 3, or set PYTHON=/path/to/python3." >&2
  exit 1
fi

# 1. Create the virtualenv if it does not already exist.
if [ ! -x "$PYTHON_BIN" ]; then
  echo ">> Creating virtualenv at $VENV_DIR"
  "$PYTHON" -m venv "$VENV_DIR"
fi

# 2. Install dependencies. Skip when requirements.txt is unchanged since the
#    last successful install (compare against a stamp file).
if [ ! -f "$STAMP_FILE" ] || [ requirements.txt -nt "$STAMP_FILE" ]; then
  echo ">> Installing dependencies from requirements.txt"
  echo "   (dlib compiles from source on first install and may take several minutes)"
  "$PYTHON_BIN" -m pip install --upgrade pip 2>&1 | filter_hash_noise
  "$PYTHON_BIN" -m pip install -r requirements.txt 2>&1 | filter_hash_noise
  touch "$STAMP_FILE"
else
  echo ">> Dependencies up to date; skipping install"
fi

# 3. Launch the application, forwarding any extra arguments. Output is piped
#    through the hash-noise filter (so we cannot exec); pipefail propagates a
#    non-zero exit from Python rather than from the filter.
echo ">> Starting real-time face recognition (press 'q' in the window to quit)"
"$PYTHON_BIN" scripts/real_time_recognition.py "$@" 2>&1 | filter_hash_noise

#!/bin/sh
# From a fresh clone to a running kit: ./install.sh
# Builds .venv on Python 3.12, installs the kit, then runs `lag setup`, which installs
# Ollama if needed, pulls the models and writes config.yaml. Extra arguments go to
# `lag setup`, e.g. ./install.sh --model qwen3:14b
set -e
cd "$(dirname "$0")"

# 3.12 specifically: chromadb has no wheel for 3.14 yet, which is what macOS ships.
PY=""
for c in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
if [ -z "$PY" ]; then
  if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
    printf "Python 3.12 is needed. Install it with Homebrew? [y/N] "
    read -r ok
    case "$ok" in [yY]*) brew install python@3.12; PY="$(brew --prefix)/bin/python3.12" ;; esac
  fi
fi
if [ -z "$PY" ]; then
  echo "Python 3.12 is needed (chromadb has no 3.14 wheel). Install it and re-run." >&2
  exit 1
fi

[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -e ".[rag,web,study]"
exec .venv/bin/lag setup "$@"

#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/NotPunchnox/rkllama.git"
BRANCH="${1:-maor-dev}"
CLONE_DIR="${2:-rkllama-repro}"

echo "Cloning $REPO_URL (branch $BRANCH) into $CLONE_DIR"
if [ -d "$CLONE_DIR/.git" ]; then
  echo "Updating existing clone"
  git -C "$CLONE_DIR" fetch origin
  git -C "$CLONE_DIR" checkout "$BRANCH"
  git -C "$CLONE_DIR" reset --hard "origin/$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$CLONE_DIR"
fi

cd "$CLONE_DIR"
git submodule update --init --recursive || true

echo "Setting up virtualenv"
python3 -m venv venv
. venv/bin/activate
pip install --upgrade pip

if [ -f converter/requirements.txt ]; then
  pip install -r converter/requirements.txt
fi
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
fi

echo "Required environment variables: RKLLM_BIN, MODELS_DIR"
if [ -z "${RKLLM_BIN:-}" ]; then
  echo "Please set RKLLM_BIN to the rkllm executable path, e.g. /usr/bin/rkllm"
  exit 1
fi
if [ -z "${MODELS_DIR:-}" ]; then
  echo "Please set MODELS_DIR to the directory containing .rkllm model files"
  exit 1
fi

export PYTHONPATH="${PWD}/src"

echo "Starting server; logs -> server.log"
python src/rkllama/server/server.py > server.log 2>&1 &
echo $! > server.pid
echo "Server started (pid $(cat server.pid)). Logs: server.log"

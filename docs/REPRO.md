# Reproducibility instructions

This document describes the minimal steps to reproduce the development state used during debugging and testing.

Requirements
- Linux host (Debian/Ubuntu recommended)
- Python 3.10+ and `python3-venv`
- `git` and `git-lfs` (for clones that need LFS-managed runtime artifacts)
- RKLLM native binary (`rkllm`) and model files (.rkllm) for your target hardware

Important external artifacts
- `rkllm` binary (recommended: version 1.2.x). Provide path in `RKLLM_BIN`.
- NPU/runtime drivers matching your board (Rockchip/RKNN runtime as required).
- Model files directory containing one or more `.rkllm` files. Provide path in `MODELS_DIR`.

Quickstart (automated)
1. Ensure you have `git` and `python3-venv` installed.
2. Set environment variables:

   - `RKLLM_BIN` — path to the `rkllm` executable (example: `/usr/bin/rkllm`)
   - `MODELS_DIR` — path to a folder with `.rkllm` model files

3. Run the helper script to clone the repository, set up a virtualenv and start the server:

```bash
./scripts/recreate_state.sh [branch] [clone-dir]
# example: ./scripts/recreate_state.sh maor-dev rkllama-repro
```

Notes
- The upstream `ezrknn-llm` tree contains large runtime binaries and OpenCV artifacts that are tracked with Git LFS in the original project; those artifacts may not be present in forks or in this workspace. If you need to run the examples/demos, fetch those runtime blobs from the upstream release or enable Git LFS for your fork.
- If the server fails to start, check `server.log` in the cloned directory for errors and verify `RKLLM_BIN` and `MODELS_DIR` are set correctly.

Manual steps
- To start the server manually (from repo root):

```bash
export PYTHONPATH=src
python src/rkllama/server/server.py
```

If you want help packaging the missing runtime artifacts as a release (or hosting them externally), I can prepare a small upload script and a release checklist.

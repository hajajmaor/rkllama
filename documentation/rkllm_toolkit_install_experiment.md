# RKLLM Toolkit Install Experiment

Date: 2026-01-21

Goal: Attempt to install the rkllm_toolkit wheel referenced by the user and run the CLI; record every step and outcome so we can compare with alternatives.

Repository / referenced wheel:
- URL used: https://github.com/airockchip/rknn-llm/blob/main/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl
- Raw URL downloaded: https://github.com/airockchip/rknn-llm/raw/main/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl

Environment checks
- CWD: /workspaces/rkllama
- Architecture: aarch64 (from `uname -m`)
- Python: Python 3.12.12

Steps performed (commands executed and results)

1) Check system architecture and Python
- Command:
  uname -m && python3 --version && python3 -c "import sys; print(sys.version)"
- Result:
  - Output: `aarch64`
  - Python: `3.12.12`

2) Download the wheel from GitHub raw URL
- Command:
  curl -fSL -o /tmp/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl https://github.com/airockchip/rknn-llm/raw/main/rkllm-toolkit/packages/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl
- Result:
  - Downloaded successfully to `/tmp/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl` (size ~6.8 MB).

3) Attempt to install the wheel with pip
- Command:
  python3 -m pip install /tmp/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl --force-reinstall -v
- Result:
  - Pip error: `ERROR: rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl is not a supported wheel on this platform.`
  - Reason: platform wheel is for `linux_x86_64` (x86_64) while host is `aarch64` — wheel ABI mismatch.

4) Attempted to run CLI (skipped because install failed)
- No CLI installed because pip failed; cannot run toolkit CLI from that wheel.

Captured outputs
- Download: succeeded (curl exit 0).
- Pip install: exited with code 1 and message about unsupported wheel on this platform.

Conclusion / Comparison notes
- The provided wheel is x86_64; our host is aarch64. Installing this wheel in the current container is not possible.
- Alternatives to succeed on this host:
  1. Use an aarch64 wheel (there are aarch64 wheels included in the repo workspace under `src/rkllama/lib` for RKNN toolkit variants). For example, a wheel named like `rkllm_toolkit-1.2.3-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` would be compatible.
  2. Build the toolkit wheel from source on this host (clone the `rknn-llm` repo and run `python3 -m build` or the project's provided packaging script) to produce an aarch64 wheel.
  3. Run the provided x86_64 wheel on an x86_64 host or in an x86_64 VM/container.

Suggested next actions (pick one):
- I can attempt to install a matching aarch64 wheel that is already present in the workspace (copy one from `src/rkllama/lib` and `pip install` it).
- I can try to build the toolkit from the `airockchip/rknn-llm` source (clone & build) inside this container (may require build deps and take time).
- I can run the x86_64 wheel in an x86_64 environment if you provide one.

Files produced by this experiment
- `/tmp/rkllm_toolkit-1.2.3-cp312-cp312-linux_x86_64.whl` (downloaded wheel)
- `documentation/rkllm_toolkit_install_experiment.md` (this file)



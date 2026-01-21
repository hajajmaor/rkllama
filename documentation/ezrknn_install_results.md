# ezrknn-llm install experiment — results

Date: 2026-01-21

Goal: follow instructions from https://github.com/Pelochus/ezrknn-llm and run `install.sh` to install RKLLM runtime and CLI.

Environment summary
- Host: Linux, architecture `aarch64` (from `uname -m`).
- User: root (script requires root).
- Python: 3.12.12 (not directly relevant for this script run).

Actions performed (commands executed)

1) Clone the repo
- Command:
  git clone https://github.com/Pelochus/ezrknn-llm.git ezrknn-llm
- Result: repo cloned into `./ezrknn-llm`.

2) Read `ezrknn-llm/README.md`
- Command:
  sed -n '1,240p' ezrknn-llm/README.md
- Result: README inspected; instructions recommend `cd ezrknn-llm && bash install.sh`.

3) Run `install.sh`
- Command:
  (cd ezrknn-llm && bash install.sh) > /tmp/ezrknn_install.log 2>&1
- Result: script executed as root; captured output below (tail of `/tmp/ezrknn_install.log`):

```
#########################################
Checking root permission...
#########################################

#########################################
Installing RKNN LLM libraries...
#########################################

#########################################
Compiling LLM runtime for Linux...
#########################################

build-linux.sh: line 26: cmake: command not found

#########################################
Moving rkllm to /usr/bin...
#########################################

cp: cannot stat './build/build_linux_aarch64_Release/llm_demo': No such file or directory

#########################################
Increasing file limit for all users (needed for LLMs to run)...
#########################################

#########################################
Done installing ezrknn-llm!
#########################################
```

Observed outcomes and explanation
- The `install.sh` script requires development/build tools (it runs `build-linux.sh` which calls `cmake`). The container lacks `cmake`, causing the build to fail at `cmake: command not found`.
- Because build failed, the expected binary `build/build_linux_aarch64_Release/llm_demo` was not produced and therefore not copied to `/usr/bin/rkllm`.
- The script appended increased `nofile` limits to `/etc/security/limits.conf` successfully.

Files produced
- `/tmp/ezrknn_install.log` — full install output.

Next steps to complete installation
- Install build dependencies and system tools, then re-run `install.sh` or `build-linux.sh` manually. At minimum, install:
  - `cmake`
  - `make` / `build-essential` / `gcc` (C/C++ toolchain)
  - Any other dependencies `build-linux.sh` expects (check that script for additional packages and links to RKNN libs).

Suggested commands (run as root) to prepare the environment before re-running:

```bash
# Debian/Ubuntu-like example
apt-get update
apt-get install -y cmake build-essential libssl-dev pkg-config
# Then re-run the build
cd ezrknn-llm/examples/DeepSeek-R1-Distill-Qwen-1.5B_Demo/deploy
bash build-linux.sh
```

Caveats
- Building the runtime may take time and require additional libraries and headers. Review `build-linux.sh` to see exact build steps and dependencies.
- This repository expects to run on RK3588 hardware; the build produces an `aarch64` binary. Building on other ARM platforms should still work if dependencies match.

Conclusion
- I executed the repo's recommended `install.sh`. It ran but failed during `cmake` invocation; therefore the runtime binary was not installed. To finish the install, install `cmake` and other build tools and re-run the build.



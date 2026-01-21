# RKLLama — Current State (2026-01-21)

## Overview
- Purpose: capture current knowledge and runtime state to compare after additional experiments.
- Workspace: `/workspaces/rkllama` (branch: `maor-dev`).

## Server
- Command used: `rkllama_server --debug --models /workspaces/rkllama/models --port 8080` (launched via `nohup`).
- Server: Flask development server, debug enabled, `use_reloader=False`.
- Listening: port `8080`, reachable at `http://127.0.0.1:8080` inside container.

## Recent Actions Taken
- Cleared and restarted server logs at `src/rkllama/config/logs/rkllama_server.log`.
- Ensured no prior workers were running, then restarted the server.
- Loaded `gemma-3n-E4B-it` via `POST /load_model` and exercised inference endpoints.
- Implemented and ran a small RKLLM CLI wrapper at `tools/run_rkllm_cli.py` to call `RKLLM` directly.

## Models / Tokenizers
- Model of interest: `gemma-3n-E4B-it` (RKLLM file: `models/gemma-3n-E4B-it/gemma-3n-E4B-it-*.rkllm`).
- Tokenizer: persisted locally under `models/gemma-3n-E4B-it/tokenizer` and loaded from there (server logs show successful load and caching).

## RKLLM / NPU Runtime Observations
- RKLLM runtime: reported `rkllm-runtime version: 1.2.3` and `rknpu driver version: 0.9.8` on platform `RK3588`.
- Worker lifecycle: worker created and running for `gemma-3n-E4B-it`; `Running inference for model gemma-3n-E4B-it...` observed in logs.
- Inference behavior: non-streaming request timed out at 60s with no response body (curl exit 28). Server logs show inference started but no completion within timeout.
- Direct CLI wrapper (`tools/run_rkllm_cli.py`) initialized RKLLM and began loading the same model; output saved to `/tmp/rkllm_cli_out.txt`.
- RKNN allocation failures were reported in earlier reproductions (platform-level messages like failures to malloc NPU memory). Those intermittent allocation errors require host/kernel diagnostics (e.g., `dmesg`) and may be outside application control.

## Logs & Files
- Primary server log: `src/rkllama/config/logs/rkllama_server.log` (recent entries truncated and restarted during tests).
- CLI wrapper output: `/tmp/rkllm_cli_out.txt`.
- Tokenizer files: `models/gemma-3n-E4B-it/tokenizer/` contains `tokenizer.*` and `chat_template.jinja`.

## Current Runtime State (after latest run)
- Server running on port `8080`, debug `on`.
- `/api/debug/workers` returned an empty `workers` list after unloading; later loads created a worker for `gemma-3n-E4B-it`.
- No permanent stuck processes remain (workers can be force-stopped via the server's unload endpoints and worker hard-terminate logic).

## Known Issues and Hypotheses
- Symptom: long or stalled inference runs for Gemma; sometimes time out without producing output within 60s.
- Hypothesis A (application-level): model initialization or RKLLM internal state may block or wait for NPU resources; worker shutdown/unload changes helped free resources.
- Hypothesis B (system-level): RKNN / driver / CMA allocation limits on the host can cause `failed to malloc npu memory` errors; kernel logs and CMA settings should be inspected.

## Recommended Next Steps (for a controlled comparison)
1. Reproduce inference with extended timeout (e.g., 180s) and capture logs (`rkllama_server.log` + `/tmp/rkllm_cli_out.txt`).
2. Run streaming (`stream=true`) to observe partial token output; this can provide earlier signals if the model generates tokens.
3. Collect host kernel logs after a failed allocation: `dmesg -T | tail -n 200` and check for rknpu/rkllm driver messages.
4. Try a smaller model variant (e.g., Qwen 4B) to verify NPU allocation behavior.
5. If allocation failures persist, adjust CMA or rknpu driver settings on host, or try loading fewer NPU cores in RKLLM params.

## Comparison Plan
- Baseline (this file): current server logs, RKLLM runtime versions, tokenizer persisted, load/unload behavior.
- After next experiment: append the new server logs + `dmesg` output and note differences in (a) allocation error messages, (b) time-to-first-token, (c) ability to unload and free NPU memory.

---
Generated on 2026-01-21. Save this file and use it to compare post-change diagnostics.

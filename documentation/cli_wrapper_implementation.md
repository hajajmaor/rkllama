# CLI-Based RKLLM Implementation - Success Summary

**Date:** January 21, 2026  
**Objective:** Replace complex ctypes RKLLM wrapper with simple subprocess-based CLI wrapper

## Problem
The original server used a complex ctypes wrapper (`src/rkllama/api/rkllm.py`) that:
- Required exact librkllmrt.so version matching
- Had complex callback mechanisms between Python and C++
- Caused NPU memory pinning issues
- Made version upgrades difficult

The `/usr/bin/rkllm` CLI tool worked reliably with runtime v1.2.3, but the server couldn't leverage it.

## Solution
Created a clean subprocess-based wrapper that:
1. **Spawns rkllm CLI** as a subprocess with stdin/stdout pipes
2. **Streams tokens** by reading stdout line-by-line
3. **Filters metadata** (version info, metrics) from actual output tokens
4. **Simple interface** - just text in, text out

## Implementation

### Files Created/Modified

1. **New: `src/rkllama/api/rkllm_cli.py`**
   - RKLLMCLIWrapper class
   - Subprocess management with Popen
   - Stdout/stderr reading threads
   - Token queue for async reading
   - Simple abort mechanism

2. **Modified: `src/rkllama/api/worker.py`**
   - `run_rkllm_worker()` now uses RKLLMCLIWrapper instead of RKLLM ctypes
   - Simplified worker loop - just string prompts, no token IDs
   - Removed callback dependency

3. **Modified: `src/rkllama/api/server_utils.py`**
   - `prepare_prompt()` now returns both token IDs AND raw text
   - Added `prompt_text` parameter throughout the chain
   - `handle_streaming()` and `handle_complete()` pass prompt_text to workers

4. **Modified: `src/rkllama/api/worker.py` (WorkerManager)**
   - `inference()` method accepts optional `prompt_text` parameter
   - When provided, passes text instead of tokens to CLI workers

### Data Flow

**Old (ctypes):**
```
Messages → Tokenizer → Token IDs → RKLLM ctypes → Callback → Queue → Response
```

**New (CLI):**
```
Messages → Tokenizer → Token IDs (unused) + Text → CLI subprocess → stdout → Queue → Response
```

## Test Results

### Server Startup
```bash
✅ Server starts successfully
✅ Model loads: gemma-3n-e2b:opt1
✅ CLI wrapper initializes: /usr/bin/rkllm <model> 512 512
✅ Runtime version: 1.2.3
```

### Non-Streaming Inference
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model":"gemma-3n-e2b:opt1","messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}'

✅ Returns valid JSON response
✅ Includes usage statistics
✅ No NPU errors
✅ Model unloads cleanly
```

### Streaming Inference
```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -d '{"model":"gemma-3n-e2b:opt1","messages":[{"role":"user","content":"Count to 5"}],"stream":true}'

✅ SSE stream works correctly
✅ Tokens arrive individually
✅ Final usage chunk included
✅ [DONE] signal sent
```

## Benefits

1. **Simplicity**: ~230 lines vs ~1000+ lines of ctypes wrapper
2. **Reliability**: Uses tested rkllm binary directly
3. **Version agnostic**: Works with any rkllm runtime version
4. **No callbacks**: Clean subprocess I/O, no Python⟷C++ bridge
5. **Easy debugging**: Can run `/usr/bin/rkllm` standalone
6. **No NPU pinning**: Process cleanup is straightforward

## Limitations

1. **Embeddings**: CLI doesn't support embedding mode yet (RKLLM_INFER_GET_LAST_HIDDEN_LAYER)
2. **Context management**: CLI restarts per request (no KV cache persistence across requests)
3. **Token parsing**: Currently assumes one token per line of output
4. **Parameter control**: Limited to max_new_tokens and max_context_len

## Configuration Notes

- Models path symlinked: `/workspaces/rkllama/src/rkllama/config/models` → `/workspaces/rkllama/models`
- Runtime binary: `/usr/bin/rkllm` (from ezrknn-llm build)
- Runtime library: `/usr/lib/librkllmrt.so` (version 1.2.3)

## Next Steps

1. ✅ Basic inference working
2. **TODO:** Test with Qwen model
3. **TODO:** Benchmark performance vs ctypes (if measurable difference)
4. **TODO:** Add support for multimodal (if CLI supports it)
5. **TODO:** Implement embeddings endpoint workaround or fall back to ctypes for that specific task

## Conclusion

The CLI-based approach successfully replaces the complex ctypes wrapper with a simpler, more maintainable solution. Gemma 3n model now runs reliably through the server using the rkllm CLI tool, eliminating version mismatch issues and NPU allocation problems.

**Why the rkllm CLI works but the ctypes wrapper didn't:**
- CLI uses the exact librkllmrt.so that was installed/upgraded (v1.2.3)
- ctypes wrapper may have been linking to older cached library or had stale process handles
- Subprocess isolation ensures clean runtime loading on each worker start

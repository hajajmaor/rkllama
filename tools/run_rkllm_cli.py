#!/usr/bin/env python3
import time
import sys
import ctypes
import rkllama.config
from rkllama.api.rkllm import RKLLM
from rkllama.api.classes import callback_type

MODEL_DIR = '/workspaces/rkllama/models/gemma-3n-E4B-it'
MODEL_FILE = MODEL_DIR + '/gemma-3n-E4B-it-rk3588-w8a8-opt-0-hybrid-ratio-0.0.rkllm'

# Simple callback to print received text
@callback_type
def py_callback(result_ptr, userdata, state):
    try:
        res = result_ptr.contents
        text = res.text.decode('utf-8') if res.text else ''
        print(f"[CALLBACK state={state}] text=\n{text}\n", flush=True)
    except Exception as e:
        print(f"Callback error: {e}", file=sys.stderr)


def main():
    options = {
        'temperature': '0.2',
        'num_ctx': '4096',
        'max_new_tokens': '32',
        'top_k': '7',
        'top_p': '0.5',
    }

    try:
        print(f"Initializing RKLLM for model: {MODEL_FILE}")
        m = RKLLM(py_callback, MODEL_FILE, MODEL_DIR, options=options)
    except Exception as e:
        print(f"Failed to init RKLLM: {e}", file=sys.stderr)
        sys.exit(2)

    try:
        # Minimal token input: use a small token sequence. RKLLM.run expects (inference_mode, model_input_type, input)
        # We'll use token input type (1) and a small token list; RKLLM will append token 2 if needed.
        tokens = [1]
        print("Running inference (generate) with minimal token input...")
        m.run(0, 1, tokens)
        # Wait for callbacks
        time.sleep(10)
    except Exception as e:
        print(f"Inference error: {e}", file=sys.stderr)
    finally:
        try:
            m.release()
        except Exception:
            pass

if __name__ == '__main__':
    main()

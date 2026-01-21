"""
RKLLM CLI Wrapper - Simple subprocess-based interface to rkllm binary
Replaces the complex ctypes wrapper with direct CLI calls
"""
import subprocess
import threading
import logging
import os
import time
from queue import Queue, Empty

logger = logging.getLogger("rkllama.rkllm_cli")


class RKLLMCLIWrapper:
    """Wrapper around /usr/bin/rkllm CLI tool using subprocess"""
    
    def __init__(self, model_path, max_new_tokens=512, max_context_len=512):
        """
        Initialize RKLLM CLI wrapper
        
        Args:
            model_path: Path to .rkllm model file
            max_new_tokens: Maximum tokens to generate per inference
            max_context_len: Maximum context length
        """
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.max_context_len = max_context_len
        self.process = None
        self.stdout_thread = None
        self.stderr_thread = None
        self.token_queue = Queue()
        self.abort_flag = threading.Event()
        self.initialized = False
        
        # Verify model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Verify rkllm binary exists
        self.rkllm_bin = "/usr/bin/rkllm"
        if not os.path.exists(self.rkllm_bin):
            raise FileNotFoundError(f"rkllm binary not found: {self.rkllm_bin}")
        
        # Start the CLI process
        self._start_process()
    
    def _start_process(self):
        """Start the rkllm CLI process with pipes"""
        cmd = [
            self.rkllm_bin,
            self.model_path,
            str(self.max_new_tokens),
            str(self.max_context_len)
        ]
        
        logger.info(f"Starting rkllm CLI: {' '.join(cmd)}")
        
        try:
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Start threads to read stdout/stderr
            self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
            self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
            self.stdout_thread.start()
            self.stderr_thread.start()
            
            # Wait a moment for initialization
            time.sleep(0.5)
            
            # Check if process started successfully
            if self.process.poll() is not None:
                raise RuntimeError(f"rkllm process exited immediately with code {self.process.returncode}")
            
            self.initialized = True
            logger.info("RKLLM CLI process started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start rkllm process: {e}")
            raise
    
    def _read_stdout(self):
        """Read stdout from rkllm process and parse tokens"""
        try:
            for line in iter(self.process.stdout.readline, ''):
                if not line:
                    break
                
                line = line.rstrip('\n')
                
                # Filter out metadata lines (version info, metrics, etc.)
                if any(x in line.lower() for x in ['rkllm:', 'version:', 'driver:', 'platform:', 
                                                     'token/s', 'tokens', 'seconds', 'init success',
                                                     'model loaded', 'enter your prompt', 'starting',
                                                     'please wait', 'enabled cpus', 'welcome to',
                                                     'to exit', 'more information', 'detailed information',
                                                     'github.com', 'pelochus', 'ezrkllm', 'ezrknpu']):
                    logger.debug(f"RKLLM meta: {line}")
                    continue
                
                # Filter out banner decorations (stars/asterisks lines)
                if line.strip() and all(c in '*= \t' for c in line.strip()):
                    logger.debug(f"RKLLM banner: {line}")
                    continue
                
                # Filter out numbered instructions like [0], [1], [2]
                if line.strip() and line.strip().startswith('[') and ']' in line[:10]:
                    logger.debug(f"RKLLM instruction: {line}")
                    continue
                
                # Strip "You: " and "LLM: " prefixes from interactive mode
                # These can appear at the start or throughout the response
                cleaned_line = line
                
                # Remove "You: " prefix if at start
                if cleaned_line.startswith('You: '):
                    cleaned_line = cleaned_line[5:]
                
                # Remove all "LLM: " occurrences (they can appear multiple times)
                # Use replace instead of startswith to catch all occurrences
                while 'LLM: ' in cleaned_line:
                    cleaned_line = cleaned_line.replace('LLM: ', '', 1)
                
                # Also handle the thinking tag that appears in some models
                cleaned_line = cleaned_line.replace('<think>', '').replace('</think>', '')
                
                # Actual output tokens - add to queue
                if cleaned_line.strip():
                    self.token_queue.put(cleaned_line)
                    
        except Exception as e:
            logger.error(f"Error reading stdout: {e}")
    
    def _read_stderr(self):
        """Read stderr from rkllm process for error logging"""
        try:
            for line in iter(self.process.stderr.readline, ''):
                if not line:
                    break
                line = line.rstrip('\n')
                if line.strip():
                    logger.warning(f"RKLLM stderr: {line}")
        except Exception as e:
            logger.error(f"Error reading stderr: {e}")
    
    def run(self, prompt, callback=None):
        """
        Run inference on a prompt
        
        Args:
            prompt: Input text prompt
            callback: Optional callback function(token_text) called for each token
            
        Yields:
            Generated tokens as strings
        """
        if not self.initialized:
            raise RuntimeError("RKLLM CLI wrapper not initialized")
        
        if self.process.poll() is not None:
            raise RuntimeError(f"RKLLM process died with code {self.process.returncode}")
        
        # Clear abort flag and queue
        self.abort_flag.clear()
        while not self.token_queue.empty():
            try:
                self.token_queue.get_nowait()
            except Empty:
                break
        
        logger.debug(f"Sending prompt: {prompt[:100]}...")
        
        # Send prompt to stdin
        try:
            self.process.stdin.write(prompt + '\n')
            self.process.stdin.flush()
        except Exception as e:
            logger.error(f"Failed to write prompt: {e}")
            raise
        
        # Read tokens from queue until we hit a natural end or abort
        # The CLI outputs tokens line by line; after generation it prompts for next input
        token_count = 0
        max_wait_seconds = 120  # Timeout for entire generation
        start_time = time.time()
        last_token_time = time.time()
        timeout_between_tokens = 10  # seconds
        
        while not self.abort_flag.is_set():
            try:
                # Wait for token with timeout
                token = self.token_queue.get(timeout=0.5)
                last_token_time = time.time()
                token_count += 1
                
                # Call callback if provided
                if callback:
                    callback(token)
                
                yield token
                
            except Empty:
                # Check if we've exceeded timeout
                if time.time() - start_time > max_wait_seconds:
                    logger.warning("Max generation time exceeded")
                    break
                
                # Check if no tokens received for too long
                if time.time() - last_token_time > timeout_between_tokens and token_count > 0:
                    logger.debug("No tokens for 10s, assuming generation complete")
                    break
                
                # Check if process died
                if self.process.poll() is not None:
                    logger.error(f"RKLLM process died during generation: {self.process.returncode}")
                    break
                
                continue
        
        if self.abort_flag.is_set():
            logger.info("Generation aborted by user")
        
        logger.debug(f"Generated {token_count} tokens")
    
    def abort(self):
        """Abort current generation"""
        logger.info("Aborting generation")
        self.abort_flag.set()
    
    def clear_cache(self):
        """
        Clear KV cache (not supported by CLI, would require restart)
        For now, this is a no-op
        """
        logger.debug("clear_cache called (no-op for CLI wrapper)")
        pass
    
    def release(self):
        """Clean up and terminate the process"""
        logger.info("Releasing RKLLM CLI wrapper")
        
        if self.process:
            try:
                self.process.stdin.close()
            except:
                pass
            
            # Give it a moment to exit gracefully
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                logger.warning("RKLLM process didn't exit gracefully, terminating")
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    logger.warning("RKLLM process didn't terminate, killing")
                    self.process.kill()
                    self.process.wait()
        
        self.initialized = False
        logger.info("RKLLM CLI wrapper released")
    
    def __del__(self):
        """Ensure cleanup on deletion"""
        self.release()

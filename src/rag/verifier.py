"""Architecture Verifier - Code sandbox for shape validation"""

import re
import signal
import contextlib
from io import StringIO
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class TimeoutError(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: int):
    def handler(signum, frame):
        raise TimeoutError(f"Timeout after {seconds}s")
    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@dataclass
class VerificationResult:
    success: bool
    output: str
    error: Optional[str]
    error_type: Optional[str]
    shapes: Dict[str, Tuple]
    execution_time: float
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "error_type": self.error_type,
            "shapes": {k: list(v) for k, v in self.shapes.items()},
            "execution_time": self.execution_time,
        }


class ArchitectureVerifier:
    """Verify generated code in sandbox."""
    
    DANGEROUS = [r'\bos\.', r'\bsubprocess\.', r'\bopen\s*\(', r'\beval\s*\(', r'\bexec\s*\(']
    
    def __init__(self, timeout_seconds: int = 10):
        self.timeout = timeout_seconds
    
    def _is_safe(self, code: str) -> Tuple[bool, Optional[str]]:
        for p in self.DANGEROUS:
            if re.search(p, code):
                return False, f"Dangerous pattern: {p}"
        return True, None
    
    def verify_dimensions(self, code: str) -> VerificationResult:
        """Execute code and extract tensor shapes."""
        import time
        start = time.time()
        
        safe, err = self._is_safe(code)
        if not safe:
            return VerificationResult(False, "", err, "SecurityError", {}, 0)
        
        wrapper = f'''
import torch
import numpy as np
__shapes__ = {{}}
{code}
for n, v in list(locals().items()):
    if not n.startswith('_') and hasattr(v, 'shape'):
        __shapes__[n] = tuple(v.shape)
'''
        stdout = StringIO()
        exec_globals = {}
        
        try:
            with time_limit(self.timeout):
                with contextlib.redirect_stdout(stdout):
                    exec(wrapper, exec_globals)
            
            return VerificationResult(
                True, stdout.getvalue(), None, None,
                exec_globals.get('__shapes__', {}), time.time() - start
            )
        except TimeoutError as e:
            return VerificationResult(False, stdout.getvalue(), str(e), "TimeoutError", {}, self.timeout)
        except Exception as e:
            return VerificationResult(
                False, stdout.getvalue(), f"{type(e).__name__}: {e}",
                type(e).__name__, exec_globals.get('__shapes__', {}), time.time() - start
            )
    
    def extract_code_blocks(self, text: str) -> List[str]:
        return re.findall(r'```(?:python)?\n(.*?)```', text, re.DOTALL)
    
    def verify_generated_response(self, response: str) -> List[VerificationResult]:
        return [self.verify_dimensions(code) for code in self.extract_code_blocks(response)]
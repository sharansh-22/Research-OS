import io
import re
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Tuple, List


class ArchitectureVerifier:
    """
    Executes generated code to catch runtime shape / dimension errors.

    SECURITY WARNING:
    - This uses `exec()` and will execute arbitrary code.
    - This is acceptable for a trusted local workflow, but is NOT safe for production.
      In production, run code in a hardened sandbox (containers / seccomp / gVisor),
      with strict resource limits and no sensitive environment access.
    """

    _PY_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.IGNORECASE)

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract fenced code blocks (```python ... ``` or ``` ... ```)."""
        if not text:
            return []
        return [m.strip() for m in self._PY_CODE_BLOCK_RE.findall(text) if m.strip()]

    def verify_dimensions(self, code_snippet: str) -> Tuple[bool, str]:
        """
        Extract code blocks from `code_snippet` and attempt to execute them.

        Returns:
            (is_valid, output_message)
        """
        blocks = self.extract_code_blocks(code_snippet)
        if not blocks:
            return False, "No Python code block found to verify."

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()

        # SECURITY WARNING: using exec() on untrusted input is dangerous.
        # For local experiments this is OK; do not do this in production.
        exec_globals = {"__builtins__": __builtins__}
        exec_locals = {}

        try:
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                for block in blocks:
                    exec(block, exec_globals, exec_locals)

            stdout_val = stdout_buf.getvalue().strip()
            stderr_val = stderr_buf.getvalue().strip()

            # If code ran, we consider it "valid" even if it printed warnings to stderr.
            # Callers can inspect stderr if desired.
            if stdout_val and stderr_val:
                return True, f"Execution succeeded.\n\nSTDOUT:\n{stdout_val}\n\nSTDERR:\n{stderr_val}"
            if stderr_val:
                return True, f"Execution succeeded.\n\nSTDERR:\n{stderr_val}"
            if stdout_val:
                return True, f"Execution succeeded.\n\nSTDOUT:\n{stdout_val}"
            return True, "Execution succeeded. Tensor shapes appear consistent (no runtime error)."
        except Exception as e:
            tb = traceback.format_exc().strip()
            return False, f"Runtime error during verification: {e}\n\n{tb}"


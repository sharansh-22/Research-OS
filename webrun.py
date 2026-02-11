#!/usr/bin/env python3
"""
Research-OS — WebRun
=====================
One command to start everything.

Usage:
    python webrun.py

What it does:
    1. Loads environment variables
    2. Validates critical requirements
    3. Starts FastAPI backend (port 8000)
    4. Starts Vite frontend (port 5173)
    5. Opens browser automatically
    6. Ctrl+C kills everything cleanly
"""

import os
import sys
import time
import signal
import subprocess
import threading
import webbrowser
from pathlib import Path

# Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

# Process holders
processes = []
shutdown_flag = False


def log(icon, msg, color=RESET):
    print(f"  {color}{icon}{RESET}  {msg}")


def log_ok(msg):
    log("OK", msg, GREEN)


def log_fail(msg):
    log("FAIL", msg, RED)


def log_info(msg):
    log(">>", msg, CYAN)


def log_warn(msg):
    log("!!", msg, YELLOW)


def banner():
    print(f"""
{BOLD}{CYAN}{'=' * 60}
  RESEARCH-OS  WebRun
  One command. Everything starts.
{'=' * 60}{RESET}
""")


# =============================================================================
# PREFLIGHT CHECKS
# =============================================================================

def preflight():
    """Quick validation before starting anything."""
    print(f"{BOLD}  Preflight Checks{RESET}\n")
    errors = 0

    # Load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
        log_ok(".env loaded")
    except ImportError:
        log_warn("python-dotenv not installed — using shell env")

    # GROQ_API_KEY
    if os.environ.get("GROQ_API_KEY", ""):
        log_ok("GROQ_API_KEY set")
    else:
        log_fail("GROQ_API_KEY not set")
        print(f"         Fix: export GROQ_API_KEY='gsk_your_key'")
        errors += 1

    # RESEARCH_OS_API_KEY
    if os.environ.get("RESEARCH_OS_API_KEY", ""):
        log_ok("RESEARCH_OS_API_KEY set")
    else:
        log_fail("RESEARCH_OS_API_KEY not set")
        print(f"         Fix: export RESEARCH_OS_API_KEY='your_secret'")
        errors += 1

    # Backend files
    if Path("run_api.py").exists() and Path("src/api/main.py").exists():
        log_ok("Backend files present")
    else:
        log_fail("Backend files missing")
        errors += 1

    # Frontend files
    frontend = Path("frontend")
    if frontend.exists() and (frontend / "package.json").exists():
        log_ok("Frontend directory present")
    else:
        log_fail("Frontend directory missing")
        errors += 1

    # node_modules
    if (frontend / "node_modules").exists():
        log_ok("node_modules installed")
    else:
        log_warn("node_modules missing — will install now")
        install_npm()

    # Check npm available
    try:
        result = subprocess.run(
            ["npm", "--version"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            log_ok(f"npm available (v{result.stdout.strip()})")
        else:
            log_fail("npm not working")
            errors += 1
    except FileNotFoundError:
        log_fail("npm not found — install Node.js first")
        print(f"         Fix: curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -")
        print(f"              sudo apt install -y nodejs")
        errors += 1
    except Exception:
        log_fail("npm check failed")
        errors += 1

    # Index check
    if Path("data/index/faiss.index").exists():
        try:
            import json
            with open("data/index/config.json") as f:
                config = json.load(f)
            chunks = config.get("n_chunks", "?")
            log_ok(f"FAISS index loaded ({chunks} chunks)")
        except Exception:
            log_ok("FAISS index exists")
    else:
        log_warn("No FAISS index — queries will return no results")
        print(f"         Fix: python scripts/ingest_batch.py")

    print()

    if errors > 0:
        log_fail(f"{errors} critical error(s) found. Fix them and try again.")
        sys.exit(1)

    log_ok("All preflight checks passed")
    print()


def install_npm():
    """Install npm dependencies if missing."""
    log_info("Installing frontend dependencies...")
    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd="frontend",
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            log_ok("npm install complete")
        else:
            log_fail(f"npm install failed: {result.stderr[:200]}")
            sys.exit(1)
    except Exception as e:
        log_fail(f"npm install error: {e}")
        sys.exit(1)


# =============================================================================
# PROCESS MANAGEMENT
# =============================================================================

def stream_output(process, name, color):
    """Read process stdout/stderr and print with prefix."""
    global shutdown_flag

    def read_stream(stream, stream_name):
        try:
            for line in iter(stream.readline, ""):
                if shutdown_flag:
                    break
                line = line.rstrip()
                if line:
                    # Filter noisy lines
                    skip_patterns = [
                        "Consider using the pymupdf_layout",
                        "Loading faiss with AVX",
                        "Successfully loaded faiss",
                    ]
                    should_skip = False
                    for pattern in skip_patterns:
                        if pattern in line:
                            should_skip = True
                            break

                    if not should_skip:
                        print(f"  {color}[{name}]{RESET} {DIM}{line}{RESET}")
        except Exception:
            pass

    t_out = threading.Thread(target=read_stream, args=(process.stdout, "out"), daemon=True)
    t_err = threading.Thread(target=read_stream, args=(process.stderr, "err"), daemon=True)
    t_out.start()
    t_err.start()


def start_backend():
    """Start FastAPI backend on port 8000."""
    log_info("Starting backend (FastAPI on :8000)...")

    env = os.environ.copy()

    proc = subprocess.Popen(
        [sys.executable, "run_api.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(Path.cwd()),
    )

    processes.append(("Backend", proc))
    stream_output(proc, "API", CYAN)

    # Wait for backend to be ready
    log_info("Waiting for backend to initialize...")
    ready = False
    for attempt in range(60):
        if proc.poll() is not None:
            log_fail("Backend process exited unexpectedly")
            sys.exit(1)

        try:
            import requests
            r = requests.get("http://localhost:8000/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                chunks = data.get("index_chunks", 0)
                files = data.get("indexed_files", 0)
                log_ok(f"Backend ready ({chunks} chunks, {files} files)")
                ready = True
                break
        except Exception:
            pass

        time.sleep(1)

        # Print progress dots
        if attempt > 0 and attempt % 10 == 0:
            log_info(f"Still loading... ({attempt}s)")

    if not ready:
        log_fail("Backend did not start within 60 seconds")
        cleanup()
        sys.exit(1)

    return proc


def start_frontend():
    """Start Vite dev server on port 5173."""
    log_info("Starting frontend (Vite on :5173)...")

    env = os.environ.copy()
    env["BROWSER"] = "none"  # Prevent Vite from opening browser itself

    proc = subprocess.Popen(
        ["npm", "run", "dev"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=str(Path("frontend").resolve()),
    )

    processes.append(("Frontend", proc))
    stream_output(proc, "UI", GREEN)

    # Wait for frontend to be ready
    log_info("Waiting for frontend to initialize...")
    ready = False
    for attempt in range(30):
        if proc.poll() is not None:
            log_fail("Frontend process exited unexpectedly")
            cleanup()
            sys.exit(1)

        try:
            import requests
            r = requests.get("http://localhost:5173", timeout=2)
            if r.status_code == 200:
                log_ok("Frontend ready")
                ready = True
                break
        except Exception:
            pass

        time.sleep(1)

    if not ready:
        # Vite might still be starting, give it benefit of doubt
        log_warn("Frontend may still be loading — opening browser anyway")

    return proc


def open_browser():
    """Open the default browser to the frontend."""
    url = "http://localhost:5173"
    log_info(f"Opening browser: {url}")
    time.sleep(1)

    try:
        webbrowser.open(url)
        log_ok("Browser opened")
    except Exception:
        log_warn(f"Could not open browser automatically")
        log_info(f"Open manually: {url}")


# =============================================================================
# SHUTDOWN
# =============================================================================

def cleanup():
    """Kill all child processes."""
    global shutdown_flag
    shutdown_flag = True

    print(f"\n{BOLD}  Shutting down...{RESET}\n")

    for name, proc in processes:
        if proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                    log_ok(f"{name} stopped")
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    log_warn(f"{name} killed (forced)")
            except Exception as e:
                log_warn(f"{name} cleanup error: {e}")
        else:
            log_ok(f"{name} already stopped")

    print(f"\n{GREEN}{BOLD}  Research-OS shut down cleanly.{RESET}\n")


def signal_handler(signum, frame):
    """Handle Ctrl+C."""
    cleanup()
    sys.exit(0)


# =============================================================================
# MAIN
# =============================================================================

def main():
    banner()

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Preflight
    preflight()

    # Start services
    print(f"{BOLD}  Starting Services{RESET}\n")

    backend_proc = start_backend()
    print()
    frontend_proc = start_frontend()
    print()

    # Open browser
    open_browser()

    # Status
    print(f"""
{BOLD}{GREEN}{'=' * 60}
  RESEARCH-OS IS RUNNING
{'=' * 60}{RESET}

  {CYAN}Frontend:{RESET}  http://localhost:5173
  {CYAN}Backend:{RESET}   http://localhost:8000
  {CYAN}API Docs:{RESET}  http://localhost:8000/docs
  {CYAN}Health:{RESET}    http://localhost:8000/health

  {YELLOW}Press Ctrl+C to stop everything{RESET}
""")

    # Keep alive — wait for processes
    try:
        while True:
            # Check if either process died
            if backend_proc.poll() is not None:
                log_fail("Backend process died unexpectedly")
                cleanup()
                sys.exit(1)

            if frontend_proc.poll() is not None:
                log_fail("Frontend process died unexpectedly")
                cleanup()
                sys.exit(1)

            time.sleep(2)

    except KeyboardInterrupt:
        cleanup()
        sys.exit(0)


if __name__ == "__main__":
    main()

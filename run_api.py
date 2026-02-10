#!/usr/bin/env python3
"""
Research-OS API Server
========================
Launch the FastAPI backend with uvicorn.

Usage:
    python run_api.py
    python run_api.py --port 8080
    python run_api.py --reload              # Development mode
    python run_api.py --host 0.0.0.0        # Expose to network
"""

import argparse
import logging
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser(description="Research-OS API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers (no reload)")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    args = parser.parse_args()
    
    # Configure logging before uvicorn takes over
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    
    # Validate critical env vars early
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY='your-key-here'")
        print("   Get key: https://console.groq.com/keys\n")
        return
    
    if not os.environ.get("RESEARCH_OS_API_KEY"):
        print("\n⚠️  RESEARCH_OS_API_KEY not set!")
        print("   Run: export RESEARCH_OS_API_KEY='your-secret-key'")
        print("   This is required for API authentication.\n")
        return
    
    import uvicorn
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                   RESEARCH-OS  API                           ║
║                                                              ║
║   Host:    {args.host:<47} ║
║   Port:    {args.port:<47} ║
║   Docs:    http://{args.host}:{args.port}/docs{' ' * (36 - len(str(args.port)))}║
║   Health:  http://{args.host}:{args.port}/health{' ' * (34 - len(str(args.port)))}║
║   Reload:  {'ON' if args.reload else 'OFF':<47} ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1 if args.reload else args.workers,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
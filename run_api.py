#!/usr/bin/env python3
"""
Run Research-OS API Server
"""

import os
import uvicorn

if __name__ == "__main__":
    # Ensure API key is available
    if not os.environ.get("GROQ_API_KEY"):
        print("\n⚠️  GROQ_API_KEY not set!")
        print("   Run: export GROQ_API_KEY='your-key-here'\n")
    
    uvicorn.run(
        "src.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
        log_level="info",
    )

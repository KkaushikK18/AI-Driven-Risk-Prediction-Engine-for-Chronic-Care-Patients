#!/usr/bin/env python3
"""
Start API Server Script
"""

import subprocess
import sys
import os
from pathlib import Path

def start_server():
    """Start the ML API server"""
    print("🚀 Starting ML API Server...")
    
    # Change to HACKATHON_DEMO directory
    demo_dir = Path(__file__).parent / "HACKATHON_DEMO"
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, "ml_api_server.py"
        ], cwd=demo_dir, check=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    start_server()
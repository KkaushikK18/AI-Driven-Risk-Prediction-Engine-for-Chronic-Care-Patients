#!/usr/bin/env python3
"""
Dashboard Setup Script
=====================

Installs required packages and sets up the dashboard environment.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🚀 Setting up Chronic Care Dashboard...")
    print("=" * 50)
    
    # Required packages for the dashboard
    packages = [
        "dash",
        "plotly", 
        "dash-bootstrap-components",
        "pandas",
        "numpy"
    ]
    
    success_count = 0
    
    for package in packages:
        print(f"\n📦 Installing {package}...")
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print(f"\n🎉 Dashboard setup complete!")
        print(f"🚀 Run the dashboard with: python chronic_care_dashboard.py")
        print(f"🌐 Then open: http://localhost:8050")
    else:
        print(f"\n⚠️  Some packages failed to install. Please install manually:")
        for package in packages:
            print(f"   pip install {package}")

if __name__ == "__main__":
    main()
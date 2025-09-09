#!/usr/bin/env python3
"""
Check what files are being tracked by Git and their sizes
to identify large files that should be excluded.
"""

import os
import subprocess
import sys
from pathlib import Path

def get_file_size(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb
    except:
        return 0

def check_git_tracked_files():
    """Check all files tracked by Git and their sizes"""
    
    try:
        # Get all files tracked by Git
        result = subprocess.run(['git', 'ls-files'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ Error: Not in a Git repository or Git not available")
            return
        
        tracked_files = result.stdout.strip().split('\n')
        
        print("🔍 Analyzing Git tracked files...")
        print("=" * 80)
        
        large_files = []
        total_size = 0
        
        for file_path in tracked_files:
            if not file_path:
                continue
                
            size_mb = get_file_size(file_path)
            total_size += size_mb
            
            # Flag files larger than 1MB
            if size_mb > 1.0:
                large_files.append((file_path, size_mb))
        
        # Sort by size (largest first)
        large_files.sort(key=lambda x: x[1], reverse=True)
        
        print(f"📊 Total repository size: {total_size:.2f} MB")
        print(f"📁 Total tracked files: {len(tracked_files)}")
        print(f"🚨 Large files (>1MB): {len(large_files)}")
        print()
        
        if large_files:
            print("🚨 LARGE FILES FOUND (these should probably be excluded):")
            print("-" * 80)
            for file_path, size_mb in large_files:
                print(f"{size_mb:8.2f} MB - {file_path}")
            print()
        
        # Check for specific problematic patterns
        problematic_patterns = [
            '.csv', '.pkl', '.joblib', '.h5', '.hdf5', 
            '.parquet', '.db', '.sqlite', '.zip', '.tar.gz'
        ]
        
        problematic_files = []
        for file_path in tracked_files:
            for pattern in problematic_patterns:
                if file_path.endswith(pattern):
                    size_mb = get_file_size(file_path)
                    problematic_files.append((file_path, size_mb, pattern))
                    break
        
        if problematic_files:
            print("⚠️  POTENTIALLY PROBLEMATIC FILES:")
            print("-" * 80)
            for file_path, size_mb, pattern in problematic_files:
                print(f"{size_mb:8.2f} MB - {file_path} ({pattern})")
            print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS:")
        print("-" * 80)
        if total_size > 100:
            print("🔴 Repository is too large for GitHub (>100MB is problematic)")
        if large_files:
            print("🔴 Add large files to .gitignore and remove from Git")
        if not large_files and total_size < 50:
            print("✅ Repository size looks good!")
            
    except Exception as e:
        print(f"❌ Error checking Git files: {e}")

def check_directory_sizes():
    """Check sizes of all directories"""
    print("\n" + "=" * 80)
    print("📂 DIRECTORY SIZES:")
    print("=" * 80)
    
    for root, dirs, files in os.walk('.'):
        # Skip .git directory
        if '.git' in root:
            continue
            
        total_size = 0
        for file in files:
            filepath = os.path.join(root, file)
            total_size += get_file_size(filepath)
        
        if total_size > 1:  # Show directories > 1MB
            print(f"{total_size:8.2f} MB - {root}")

if __name__ == "__main__":
    print("🔍 Git Repository Size Analyzer")
    print("=" * 80)
    
    check_git_tracked_files()
    check_directory_sizes()
    
    print("\n" + "=" * 80)
    print("🚀 NEXT STEPS:")
    print("1. Add large files to .gitignore")
    print("2. Remove them from Git: git rm --cached <file>")
    print("3. Commit changes: git commit -m 'Remove large files'")
    print("4. Try pushing again")
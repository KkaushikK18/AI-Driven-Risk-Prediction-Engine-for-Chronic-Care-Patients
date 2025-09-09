#!/usr/bin/env python3
"""
Create .gitkeep files to maintain directory structure in Git
while excluding large data files and models.
"""

import os
from pathlib import Path

def create_gitkeep_files():
    """Create .gitkeep files in important directories"""
    
    # Directories that should be preserved in Git
    directories_to_keep = [
        'data',
        'data/raw',
        'data/raw/mimic',
        'data/processed',
        'data/results',
        'data/interim',
        'data/external',
        'models',
        'logs',
        'temp'
    ]
    
    project_root = Path.cwd()
    
    for directory in directories_to_keep:
        dir_path = project_root / directory
        gitkeep_path = dir_path / '.gitkeep'
        
        # Create directory if it doesn't exist
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file
        gitkeep_path.touch()
        print(f"✅ Created: {gitkeep_path}")
    
    print("\n🎯 Directory structure preserved for Git!")
    print("💡 These directories will be maintained in Git even when empty")

if __name__ == "__main__":
    create_gitkeep_files()
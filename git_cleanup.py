#!/usr/bin/env python3
"""
Clean up Git repository by removing large files and properly excluding them.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_git_command(command):
    """Run a git command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def remove_large_files_from_git():
    """Remove large files from Git tracking"""
    
    # Common large file patterns to remove
    patterns_to_remove = [
        # Data files
        "data/**/*.csv",
        "data/**/*.parquet", 
        "data/**/*.h5",
        "data/**/*.hdf5",
        "data/**/*.pkl",
        "data/**/*.joblib",
        "data/**/*.db",
        "data/**/*.sqlite",
        
        # Model files
        "models/*.pkl",
        "models/*.joblib", 
        "models/*.h5",
        "models/*.hdf5",
        "models/*.model",
        "models/*.bin",
        
        # Archive files
        "**/*.zip",
        "**/*.tar.gz",
        "**/*.rar",
        "**/*.7z",
        
        # Jupyter notebooks with outputs
        "**/*.ipynb",
        
        # Log files
        "**/*.log",
        "logs/**/*",
        
        # Cache files
        "**/__pycache__/**",
        "**/.ipynb_checkpoints/**",
    ]
    
    print("🧹 Removing large files from Git...")
    
    for pattern in patterns_to_remove:
        print(f"   Removing: {pattern}")
        success, stdout, stderr = run_git_command(f'git rm -r --cached "{pattern}" 2>/dev/null || true')
        # Don't worry about errors here - files might not exist
    
    print("✅ Large files removed from Git tracking")

def create_comprehensive_gitignore():
    """Create a very comprehensive .gitignore"""
    
    gitignore_content = """# =============================================================================
# Chronic Care Risk Engine - Ultra Comprehensive .gitignore
# =============================================================================

# ALL DATA FILES (Complete exclusion)
# =============================================================================
data/
!data/.gitkeep

# ALL MODEL FILES (Complete exclusion) 
# =============================================================================
models/
!models/.gitkeep

# ALL RESULT FILES
# =============================================================================
results/
!results/.gitkeep

# Python Cache and Bytecode
# =============================================================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environments
# =============================================================================
venv/
env/
ENV/
.venv/
.env/
.ENV/
mimic_env/
chronic_care_env/
healthcare_env/
myenv/

# Jupyter Notebooks (can be large with outputs)
# =============================================================================
*.ipynb
.ipynb_checkpoints
*/.ipynb_checkpoints/*

# All Data File Extensions
# =============================================================================
*.csv
*.parquet
*.h5
*.hdf5
*.pkl
*.joblib
*.db
*.sqlite
*.sqlite3
*.json.gz
*.npy
*.npz

# Archive Files
# =============================================================================
*.zip
*.tar.gz
*.rar
*.7z
*.tar
*.gz
*.bz2
*.xz

# Log Files
# =============================================================================
*.log
logs/
temp/
tmp/
.tmp/
*.tmp
*.temp

# IDE and Editor
# =============================================================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db
desktop.ini

# Configuration with Secrets
# =============================================================================
.env
.env.local
.env.production
config/secrets.py
secrets.json

# Large Output Files
# =============================================================================
output/
outputs/
checkpoints/
saved_models/
artifacts/

# Keep only essential structure
# =============================================================================
!.gitignore
!.gitkeep
!README.md
!requirements.txt
!setup.py
!src/
!docs/
!scripts/
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created comprehensive .gitignore")

def create_gitkeep_files():
    """Create .gitkeep files for important directories"""
    
    directories = [
        'data',
        'models', 
        'results',
        'logs',
        'output'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        gitkeep_path = os.path.join(directory, '.gitkeep')
        Path(gitkeep_path).touch()
        print(f"✅ Created: {gitkeep_path}")

def main():
    """Main cleanup process"""
    
    print("🚀 Git Repository Cleanup for Large Repositories")
    print("=" * 80)
    
    # Check if we're in a git repo
    success, _, _ = run_git_command("git status")
    if not success:
        print("❌ Not in a Git repository!")
        return
    
    print("Step 1: Creating comprehensive .gitignore...")
    create_comprehensive_gitignore()
    
    print("\nStep 2: Creating .gitkeep files...")
    create_gitkeep_files()
    
    print("\nStep 3: Removing large files from Git tracking...")
    remove_large_files_from_git()
    
    print("\nStep 4: Staging changes...")
    run_git_command("git add .gitignore")
    run_git_command("git add **/.gitkeep")
    
    print("\nStep 5: Committing cleanup...")
    run_git_command('git commit -m "Clean up repository: remove large files and add comprehensive .gitignore"')
    
    print("\n" + "=" * 80)
    print("✅ CLEANUP COMPLETE!")
    print("=" * 80)
    print("\n🚀 NEXT STEPS:")
    print("1. Check repository size:")
    print("   python check_git_size.py")
    print("\n2. If still too large, run aggressive cleanup:")
    print("   git filter-branch --tree-filter 'rm -rf data models' --prune-empty HEAD")
    print("\n3. Try pushing again:")
    print("   git push origin main")

if __name__ == "__main__":
    main()
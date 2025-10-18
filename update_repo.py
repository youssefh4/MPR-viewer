#!/usr/bin/env python3
"""
Simple script to update the MPR Viewer repository
"""

import subprocess
import sys
from datetime import datetime

def update_repo():
    """Update the repository with current changes."""
    try:
        print("🔄 Updating MPR Viewer repository...")
        
        # Add all changes
        subprocess.run(["git", "add", "."], check=True)
        print("✅ Files added to staging")
        
        # Commit with timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        commit_message = f"Update MPR Viewer - {timestamp}"
        subprocess.run(["git", "commit", "-m", commit_message], check=True)
        print("✅ Changes committed")
        
        # Push to GitHub
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ Repository updated successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    update_repo()

#!/usr/bin/env python3
"""
Installation script for Whisper transcription application.
This script installs all required dependencies for the Whisper large-v3 model.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Install all required dependencies."""
    print("🎙️  Whisper Transcription App - Installation Script")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        sys.exit(1)
    
    print(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Required packages
    packages = [
        "torch",  # PyTorch for model execution
        "transformers[torch]",  # Hugging Face Transformers with PyTorch support
        "librosa",  # Audio processing library
        "soundfile",  # Audio file I/O
        "datasets",  # For audio data handling
        "accelerate",  # For optimized model loading
    ]
    
    print(f"\n📦 Installing {len(packages)} required packages...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: Could not upgrade pip, continuing with current version")
    
    # Install packages
    for package in packages:
        if not run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
            print(f"❌ Failed to install {package}")
            sys.exit(1)
    
    print("\n🎉 Installation completed successfully!")
    print("\nYou can now run the main script with:")
    print("python main.py")
    print("\nMake sure to place your 'input.mp3' file in this directory before running.")

if __name__ == "__main__":
    main() 
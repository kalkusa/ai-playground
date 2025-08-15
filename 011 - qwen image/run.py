#!/usr/bin/env python3
"""
Simple run script for Qwen-Image generation
Executes image generation with default settings
"""
import subprocess
import sys
import os

def main():
    print("🚀 Running Qwen-Image Generator")
    print("=" * 40)
    print("📝 Default prompt: 'pikachu on grassland'")
    print("⏳ This may take a few minutes on first run (model download)")
    print()
    
    # Check if generate_image.py exists
    if not os.path.exists("generate_image.py"):
        print("❌ generate_image.py not found!")
        print("Make sure you're running this from the project directory")
        return 1
    
    try:
        # Run the image generation script with default settings
        result = subprocess.run([
            sys.executable, "generate_image.py",
            "--prompt", "pikachu on grassland",
            "--aspect", "16:9",
            "--steps", "50",
            "--seed", "42"
        ], check=True)
        
        print("\n✨ Generation completed!")
        print("📁 Check the 'output' folder for your generated image")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 
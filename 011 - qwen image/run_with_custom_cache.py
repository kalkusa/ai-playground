#!/usr/bin/env python3
"""
Run Qwen-Image with custom cache directory
This helps when your main drive doesn't have enough space
"""
import subprocess
import sys
import os

def main():
    print("🚀 Qwen-Image Generator (Custom Cache)")
    print("=" * 45)
    print("📝 Default prompt: 'pikachu on grassland'")
    print("💾 This version allows you to specify a custom cache directory")
    print()
    
    # Ask user for cache directory
    print("Current drive has limited space (10 GB free, need ~25 GB)")
    print("Please specify a directory on a drive with more space:")
    print("Examples:")
    print("  Windows: D:\\qwen_cache or C:\\temp\\qwen_cache")
    print("  Linux/Mac: /mnt/large_drive/qwen_cache")
    print()
    
    cache_dir = input("Enter cache directory (or press Enter to use default): ").strip()
    
    if cache_dir:
        # Create the directory if it doesn't exist
        try:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"✅ Using cache directory: {cache_dir}")
        except Exception as e:
            print(f"❌ Could not create cache directory: {e}")
            return 1
    else:
        print("⚠️  Using default cache (may fail due to disk space)")
        cache_dir = None
    
    print()
    print("⏳ Starting generation (this may take several minutes on first run)")
    
    # Check if generate_image.py exists
    if not os.path.exists("generate_image.py"):
        print("❌ generate_image.py not found!")
        print("Make sure you're running this from the project directory")
        return 1
    
    try:
        # Build command
        cmd = [
            sys.executable, "generate_image.py",
            "--prompt", "pikachu on grassland",
            "--aspect", "16:9",
            "--steps", "50",
            "--seed", "42"
        ]
        
        if cache_dir:
            cmd.extend(["--cache-dir", cache_dir])
        
        # Run the image generation script
        result = subprocess.run(cmd, check=True)
        
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
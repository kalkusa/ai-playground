#!/usr/bin/env python3
"""
Simple CPU-only runner for Qwen-Image
Forces CPU execution to avoid GPU memory issues
"""
import os
import subprocess
import sys

def main():
    print("🖼️  Qwen-Image Generator (CPU-Only)")
    print("=" * 45)
    print("📝 Default prompt: 'pikachu on grassland'")
    print("💻 CPU-only mode to avoid GPU memory issues")
    print("⏳ This will be slower but should work reliably")
    print()
    
    # Set environment variables to force CPU
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = ''
    env['PYTORCH_CUDA_ALLOC_CONF'] = ''
    
    # Check if generate_image_cpu.py exists
    if not os.path.exists("generate_image_cpu.py"):
        print("❌ generate_image_cpu.py not found!")
        print("Make sure you're running this from the project directory")
        return 1
    
    try:
        print("🚀 Starting CPU generation...")
        
        # Run the CPU-only script
        result = subprocess.run([
            sys.executable, "generate_image_cpu.py",
            "--prompt", "pikachu on grassland",
            "--aspect", "1:1",  # Smaller resolution for CPU
            "--steps", "20"     # Fewer steps for faster CPU generation
        ], env=env, check=True)
        
        print("\n✨ Generation completed!")
        print("📁 Check the 'output' folder for your generated image")
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Generation failed with exit code: {e.returncode}")
        print("\n💡 Troubleshooting:")
        print("   1. Ensure you have enough RAM (16GB+ recommended)")
        print("   2. Close other applications to free memory")
        print("   3. The model might be too large for CPU execution")
        return e.returncode
    except KeyboardInterrupt:
        print("\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 
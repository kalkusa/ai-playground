#!/usr/bin/env python3
"""
Qwen-Image Generation Script
Generates images using the Qwen-Image model from Hugging Face
"""
import torch
from diffusers import DiffusionPipeline
import os
import shutil
from datetime import datetime
import argparse

def check_disk_space(path=".", required_gb=25):
    """Check if there's enough disk space for model download"""
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free // (1024**3)
        print(f"💾 Available disk space: {free_gb} GB")
        
        if free_gb < required_gb:
            print(f"⚠️  Warning: Qwen-Image model requires ~{required_gb}GB of free space")
            print(f"   You have {free_gb}GB available. Consider freeing up space.")
            response = input("Continue anyway? (y/N): ").strip().lower()
            return response in ['y', 'yes']
        return True
    except Exception as e:
        print(f"⚠️  Could not check disk space: {e}")
        return True

def setup_device_and_dtype():
    """Setup device and data type based on available hardware"""
    if torch.cuda.is_available():
        print("🔥 CUDA detected - using GPU acceleration")
        device = "cuda"
        torch_dtype = torch.bfloat16
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("🍎 MPS detected - using Apple Silicon acceleration")
        device = "mps"
        torch_dtype = torch.float32
    else:
        print("💻 Using CPU (this will be slower)")
        device = "cpu"
        torch_dtype = torch.float32
    
    return device, torch_dtype

def load_pipeline(model_name="Qwen/Qwen-Image", cache_dir=None):
    """Load the Qwen-Image pipeline with better error handling"""
    print(f"📥 Loading model: {model_name}")
    print("⏳ This is a large model (~20GB) and may take several minutes to download on first use")
    
    device, torch_dtype = setup_device_and_dtype()
    
    try:
        # Set cache directory if specified
        kwargs = {"torch_dtype": torch_dtype, "use_safetensors": True}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            
        pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        pipe = pipe.to(device)
        print("✅ Model loaded successfully")
        return pipe, device
        
    except OSError as e:
        if "No space left on device" in str(e):
            print("❌ Error: Not enough disk space to download the model")
            print("💡 Solutions:")
            print("   1. Free up at least 25GB of disk space")
            print("   2. Use a different cache directory with more space:")
            print("      Set environment variable: HF_HOME=/path/to/large/drive")
            print("   3. Try a smaller model or use cloud GPU services")
        else:
            print(f"❌ Failed to load model: {e}")
        raise
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Troubleshooting:")
        print("   1. Check your internet connection")
        print("   2. Ensure you have enough disk space (~25GB)")
        print("   3. Try running: pip install --upgrade transformers diffusers")
        raise

def generate_image(pipe, device, prompt, negative_prompt="", aspect_ratio="16:9", num_steps=50, cfg_scale=4.0, seed=42):
    """Generate an image using the pipeline"""
    
    # Define aspect ratios
    aspect_ratios = {
        "1:1": (1328, 1328),
        "16:9": (1664, 928),
        "9:16": (928, 1664),
        "4:3": (1472, 1140),
        "3:4": (1140, 1472),
        "3:2": (1584, 1056),
        "2:3": (1056, 1584),
    }
    
    width, height = aspect_ratios.get(aspect_ratio, aspect_ratios["16:9"])
    
    # Add positive magic for better quality
    positive_magic = "Ultra HD, 4K, cinematic composition."
    full_prompt = f"{prompt}, {positive_magic}"
    
    print(f"🎨 Generating image...")
    print(f"📝 Prompt: {prompt}")
    print(f"📐 Resolution: {width}x{height} ({aspect_ratio})")
    print(f"🎲 Seed: {seed}")
    
    try:
        # Set up generator for reproducible results
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image
        result = pipe(
            prompt=full_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            true_cfg_scale=cfg_scale,
            generator=generator
        )
        
        return result.images[0]
    
    except Exception as e:
        print(f"❌ Failed to generate image: {e}")
        raise

def save_image(image, prompt, output_dir="output"):
    """Save the generated image with a descriptive filename"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from prompt and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in (' ', '-', '_')).rstrip()
    safe_prompt = safe_prompt.replace(' ', '_')
    filename = f"{timestamp}_{safe_prompt}.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    print(f"💾 Image saved: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate images with Qwen-Image")
    parser.add_argument("--prompt", type=str, default="pikachu on grassland", 
                       help="Text prompt for image generation")
    parser.add_argument("--negative", type=str, default="", 
                       help="Negative prompt (what to avoid)")
    parser.add_argument("--aspect", type=str, default="16:9", 
                       choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                       help="Aspect ratio")
    parser.add_argument("--steps", type=int, default=50, 
                       help="Number of inference steps")
    parser.add_argument("--cfg", type=float, default=4.0, 
                       help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output", 
                       help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Custom cache directory for model files")
    parser.add_argument("--skip-space-check", action="store_true",
                       help="Skip disk space check")
    
    args = parser.parse_args()
    
    print("🖼️  Qwen-Image Generator")
    print("=" * 40)
    
    try:
        # Check disk space unless skipped
        if not args.skip_space_check:
            if not check_disk_space(args.cache_dir or ".", required_gb=25):
                print("❌ Aborted by user due to insufficient disk space")
                return 1
        
        # Load the pipeline
        pipe, device = load_pipeline(cache_dir=args.cache_dir)
        
        # Generate image
        image = generate_image(
            pipe=pipe,
            device=device,
            prompt=args.prompt,
            negative_prompt=args.negative,
            aspect_ratio=args.aspect,
            num_steps=args.steps,
            cfg_scale=args.cfg,
            seed=args.seed
        )
        
        # Save image
        filepath = save_image(image, args.prompt, args.output)
        
        print(f"\n🎉 Success! Image generated and saved to: {filepath}")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
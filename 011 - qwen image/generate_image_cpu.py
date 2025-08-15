#!/usr/bin/env python3
"""
Qwen-Image Generation Script (CPU-only version)
For systems with limited GPU memory
"""
import torch
from diffusers import DiffusionPipeline
import os
import shutil
from datetime import datetime
import argparse

def setup_cpu_only():
    """Force CPU-only execution"""
    print("💻 Forcing CPU-only mode for compatibility")
    device = "cpu"
    torch_dtype = torch.float32
    return device, torch_dtype

def load_pipeline_cpu(model_name="Qwen/Qwen-Image", cache_dir=None):
    """Load the Qwen-Image pipeline in CPU-only mode"""
    print(f"📥 Loading model: {model_name}")
    print("💻 CPU-only mode: This will be slower but should work on any system")
    print("⏳ This is a large model (~20GB) and may take several minutes to download on first use")
    
    device, torch_dtype = setup_cpu_only()
    
    try:
        # Force CPU loading
        kwargs = {
            "torch_dtype": torch_dtype, 
            "use_safetensors": True,
            "device_map": None  # Prevent automatic GPU mapping
        }
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
            
        pipe = DiffusionPipeline.from_pretrained(model_name, **kwargs)
        
        # Explicitly move to CPU
        pipe = pipe.to(device)
        
        # Disable memory efficient attention to avoid GPU operations
        if hasattr(pipe, 'enable_memory_efficient_attention'):
            pipe.disable_memory_efficient_attention()
        
        print("✅ Model loaded successfully on CPU")
        return pipe, device
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Troubleshooting:")
        print("   1. Ensure you have enough RAM (16GB+ recommended)")
        print("   2. Close other applications to free memory")
        print("   3. Try restarting your computer")
        raise

def generate_image_cpu(pipe, device, prompt, negative_prompt="", aspect_ratio="1:1", num_steps=25, cfg_scale=4.0, seed=42):
    """Generate an image using CPU-only pipeline"""
    
    # Use smaller resolutions for CPU to save memory and time
    aspect_ratios = {
        "1:1": (768, 768),      # Smaller for CPU
        "16:9": (832, 464),     # Smaller for CPU
        "9:16": (464, 832),     # Smaller for CPU
        "4:3": (768, 576),      # Smaller for CPU
        "3:4": (576, 768),      # Smaller for CPU
        "3:2": (768, 512),      # Smaller for CPU
        "2:3": (512, 768),      # Smaller for CPU
    }
    
    width, height = aspect_ratios.get(aspect_ratio, aspect_ratios["1:1"])
    
    # Add positive magic for better quality
    positive_magic = "Ultra HD, 4K, cinematic composition."
    full_prompt = f"{prompt}, {positive_magic}"
    
    print(f"🎨 Generating image on CPU...")
    print(f"📝 Prompt: {prompt}")
    print(f"📐 Resolution: {width}x{height} ({aspect_ratio}) - optimized for CPU")
    print(f"🎲 Seed: {seed}")
    print(f"⏳ CPU generation is slower - this may take 5-15 minutes")
    
    try:
        # Set up generator for reproducible results
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Generate image with CPU-optimized settings
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
    filename = f"{timestamp}_{safe_prompt}_cpu.png"
    filepath = os.path.join(output_dir, filename)
    
    image.save(filepath)
    print(f"💾 Image saved: {filepath}")
    return filepath

def main():
    parser = argparse.ArgumentParser(description="Generate images with Qwen-Image (CPU-only)")
    parser.add_argument("--prompt", type=str, default="pikachu on grassland", 
                       help="Text prompt for image generation")
    parser.add_argument("--negative", type=str, default="", 
                       help="Negative prompt (what to avoid)")
    parser.add_argument("--aspect", type=str, default="1:1", 
                       choices=["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"],
                       help="Aspect ratio (CPU optimized resolutions)")
    parser.add_argument("--steps", type=int, default=25, 
                       help="Number of inference steps (recommended: 25 for CPU)")
    parser.add_argument("--cfg", type=float, default=4.0, 
                       help="CFG scale")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="output", 
                       help="Output directory")
    parser.add_argument("--cache-dir", type=str, default=None,
                       help="Custom cache directory for model files")
    
    args = parser.parse_args()
    
    print("🖼️  Qwen-Image Generator (CPU-Only)")
    print("=" * 45)
    print("💻 This version runs on CPU to avoid GPU memory issues")
    print("⏳ CPU generation is slower but works on any system")
    print()
    
    try:
        # Load the pipeline
        pipe, device = load_pipeline_cpu(cache_dir=args.cache_dir)
        
        # Generate image
        image = generate_image_cpu(
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
        print("💡 For faster generation, consider using a cloud GPU service")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Generation cancelled by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 
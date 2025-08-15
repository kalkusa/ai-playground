# Qwen-Image Basic Usage Project

A simple project to get started with [Qwen-Image](https://huggingface.co/Qwen/Qwen-Image), an advanced text-to-image generation model that excels at complex text rendering and precise image editing.

## Features

- 🎨 High-quality image generation from text prompts
- 🔤 Excellent text rendering capabilities (English & Chinese)
- 🖼️ Multiple aspect ratio support
- 🎯 Reproducible results with seed control
- 🚀 Cross-platform support (Windows, macOS, Linux)
- ⚡ GPU acceleration (CUDA/MPS) when available

## Quick Start

### 1. Installation

**Option A: Automatic Installation (Recommended)**
```bash
python install.py
```

**Option B: Manual Installation**
```bash
pip install torch torchvision
pip install git+https://github.com/huggingface/diffusers
pip install transformers accelerate safetensors Pillow
```

### 2. Generate Your First Image

**Quick Run with Default Settings:**
```bash
# On Windows
run.bat

# On macOS/Linux
chmod +x run.sh
./run.sh

# Or using Python
python run.py

# CPU-only mode (for systems with limited GPU memory)
python run_cpu.py

# If you have disk space issues (need ~25GB):
python run_with_custom_cache.py
```

This will generate an image with the prompt "pikachu on grassland" and save it to the `output` folder.

**⚠️ Important**: Qwen-Image is a large model (~25GB). If you get "No space left on device" errors, use `run_with_custom_cache.py` to specify a directory on a drive with more space.

## CPU-Only Generation

If you don't have a compatible GPU or encounter memory issues, you can use the CPU-only version:

### Quick CPU Generation
```bash
# Simple CPU generation with default settings
python run_cpu.py
```

### Custom CPU Generation
```bash
# Generate with custom prompt
python generate_image_cpu.py --prompt "your custom prompt here"

# Optimized CPU settings (faster generation)
python generate_image_cpu.py --prompt "landscape" --steps 20 --aspect 1:1

# Specify custom cache directory
python generate_image_cpu.py --prompt "portrait" --cache-dir "/path/to/large/drive"
```

### CPU Generation Notes
- ⏳ **Slower**: CPU generation takes 5-15 minutes per image
- 💾 **More RAM**: Requires at least 16GB system RAM  
- 🔧 **Optimized**: Uses smaller resolutions and fewer steps by default
- 📁 **Distinguishable**: Generated images have "_cpu" suffix in filename
- 🛡️ **Reliable**: Works on any system without GPU requirements

## Advanced Usage

### Custom Prompts

Generate images with custom prompts:

```bash
# GPU/CUDA (faster, requires GPU memory)
python generate_image.py --prompt "a majestic dragon flying over mountains"

# CPU-only (slower but works on any system)
python generate_image_cpu.py --prompt "a majestic dragon flying over mountains"
```

### Available Options

```bash
# GPU version options
python generate_image.py --help

# CPU version options  
python generate_image_cpu.py --help
```

**Common Parameters:**
- `--prompt`: Text description of the image to generate
- `--negative`: Negative prompt (what to avoid in the image)
- `--aspect`: Aspect ratio (1:1, 16:9, 9:16, 4:3, 3:4, 3:2, 2:3)
- `--steps`: Number of inference steps (GPU default: 50, CPU default: 25)
- `--cfg`: CFG scale for prompt adherence (default: 4.0)
- `--seed`: Random seed for reproducible results (default: 42)
- `--output`: Output directory (default: "output")
- `--cache-dir`: Custom cache directory for model files (CPU version only)

### Examples

**Basic Generation:**
```bash
# GPU version (faster)
python generate_image.py --prompt "sunset over a calm lake"

# CPU version (slower but compatible)
python generate_image_cpu.py --prompt "sunset over a calm lake"
```

**Square Image:**
```bash
python generate_image.py --prompt "cute cat portrait" --aspect 1:1
```

**High Quality with More Steps:**
```bash
# GPU version with high steps
python generate_image.py --prompt "futuristic city skyline" --steps 75 --cfg 5.0

# CPU version (use fewer steps for reasonable speed)
python generate_image_cpu.py --prompt "futuristic city skyline" --steps 30 --cfg 5.0
```

**With Negative Prompt:**
```bash
python generate_image.py --prompt "beautiful landscape" --negative "blurry, low quality, distorted"
```

**CPU-Optimized Generation:**
```bash
# Optimized settings for CPU generation
python generate_image_cpu.py --prompt "mountain landscape" --steps 20 --aspect 1:1
```

## System Requirements

### Minimum Requirements (CPU-only mode)
- **Python**: 3.8 or higher
- **RAM**: 16GB+ (required for CPU generation)
- **Storage**: 25GB+ free space (for model downloads)
- **Time**: 5-15 minutes per image on CPU

### Recommended Requirements (GPU mode)
- **Python**: 3.8 or higher  
- **RAM**: 8GB+ system RAM
- **GPU**: NVIDIA GPU with 6GB+ VRAM or Apple Silicon M1/M2/M3
- **Storage**: 25GB+ free space (for model downloads)
- **Time**: 1-3 minutes per image on GPU

### GPU Support

- **NVIDIA**: CUDA-enabled GPUs (RTX 20/30/40 series recommended)
- **Apple Silicon**: M1/M2/M3 Macs with MPS acceleration
- **CPU**: Fallback option available (much slower but works on any system)

## Project Structure

```
qwen-image-project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── install.py                  # Installation script
├── generate_image.py           # Main image generation script (GPU/CUDA)
├── generate_image_cpu.py       # CPU-only image generation script
├── run.py                      # Simple runner (Python, GPU)
├── run_cpu.py                  # Simple runner (Python, CPU-only)
├── run_with_custom_cache.py    # Runner with custom cache directory
├── run.sh                      # Simple runner (Unix/Linux/macOS)
├── run.bat                     # Simple runner (Windows)
└── output/                     # Generated images folder
```

## Model Information

- **Model**: [Qwen/Qwen-Image](https://huggingface.co/Qwen/Qwen-Image)
- **License**: Apache 2.0
- **Paper**: [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324)
- **Capabilities**:
  - High-fidelity text rendering
  - Multiple artistic styles
  - Advanced image editing
  - Multi-language support (English/Chinese)

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Option 1: Use dedicated CPU script
python generate_image_cpu.py --prompt "your prompt"

# Option 2: Force CPU with environment variable
CUDA_VISIBLE_DEVICES="" python generate_image.py --prompt "your prompt"
```

**2. Insufficient RAM for CPU Generation**
- Close other applications to free memory
- Ensure you have at least 16GB RAM for CPU generation
- Try smaller resolutions with `--aspect 1:1`
- Use fewer inference steps with `--steps 20`

**3. Model Download Issues**
- Ensure stable internet connection
- Check available disk space (model is ~25GB)
- Try rerunning the script
- Use `--cache-dir` to specify a directory with more space

**4. Import Errors**
```bash
# Reinstall dependencies
pip uninstall diffusers
pip install git+https://github.com/huggingface/diffusers
```

**5. CPU Generation Too Slow**
- Use optimized settings: `--steps 20 --aspect 1:1`
- Consider using cloud GPU services for faster generation
- Generated images will have "_cpu" suffix to distinguish from GPU versions

### Performance Tips

**For GPU Generation:**
1. **Use GPU**: Ensure CUDA/MPS is properly installed
2. **Reduce Steps**: Use `--steps 25` for faster generation
3. **Lower Resolution**: Use smaller aspect ratios like 1:1
4. **Batch Processing**: Generate multiple images in sequence

**For CPU Generation:**
1. **Use CPU Script**: Use `generate_image_cpu.py` for optimized CPU performance
2. **Minimal Steps**: Use `--steps 20` or fewer for reasonable generation times
3. **Square Images**: Use `--aspect 1:1` for smaller, faster-to-generate images  
4. **Close Applications**: Free up as much RAM as possible
5. **Be Patient**: CPU generation takes 5-15 minutes per image

## Examples Gallery

The default prompt "pikachu on grassland" will generate a high-quality image of the popular Pokémon character in a natural grassland setting, showcasing the model's ability to understand pop culture references and create detailed, vibrant scenes.

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is licensed under the Apache 2.0 License - see the [Qwen-Image model page](https://huggingface.co/Qwen/Qwen-Image) for details.

## Acknowledgments

- [Qwen Team](https://huggingface.co/Qwen) for the amazing Qwen-Image model
- [Hugging Face](https://huggingface.co/) for the Diffusers library
- [PyTorch](https://pytorch.org/) for the deep learning framework 
#!/bin/bash
# Qwen-Image Runner Script for Unix/Linux/macOS

echo "🚀 Running Qwen-Image Generator"
echo "========================================"
echo "📝 Default prompt: 'pikachu on grassland'"
echo "⏳ This may take a few minutes on first run (model download)"
echo

# Check if generate_image.py exists
if [ ! -f "generate_image.py" ]; then
    echo "❌ generate_image.py not found!"
    echo "Make sure you're running this from the project directory"
    exit 1
fi

# Run the image generation
python3 generate_image.py \
    --prompt "pikachu on grassland" \
    --aspect "16:9" \
    --steps 50 \
    --seed 42

if [ $? -eq 0 ]; then
    echo
    echo "✨ Generation completed!"
    echo "📁 Check the 'output' folder for your generated image"
else
    echo "❌ Generation failed"
    exit 1
fi 
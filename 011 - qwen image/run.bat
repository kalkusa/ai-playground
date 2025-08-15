@echo off
REM Qwen-Image Runner Script for Windows

echo 🚀 Running Qwen-Image Generator
echo ========================================
echo 📝 Default prompt: 'pikachu on grassland'
echo ⏳ This may take a few minutes on first run (model download)
echo.

REM Check if generate_image.py exists
if not exist "generate_image.py" (
    echo ❌ generate_image.py not found!
    echo Make sure you're running this from the project directory
    pause
    exit /b 1
)

REM Run the image generation
python generate_image.py --prompt "pikachu on grassland" --aspect "16:9" --steps 50 --seed 42

if %ERRORLEVEL% equ 0 (
    echo.
    echo ✨ Generation completed!
    echo 📁 Check the 'output' folder for your generated image
) else (
    echo ❌ Generation failed
)

pause 
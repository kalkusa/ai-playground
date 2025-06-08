# Whisper Audio Transcription Application

This application uses OpenAI's Whisper large-v3 model to transcribe audio files from MP3 to text.

## Quick Start

### 1. Install Dependencies
Run the installation script to install all required libraries:
```bash
python install.py
```

### 2. Prepare Audio File
Place your audio file as `input.mp3` in this directory.

### 3. Run Transcription
Execute the main script:
```bash
python main.py
```

### 4. Get Results
The transcribed text will be saved to `output.txt` in the same directory.

## Requirements

- Python 3.8 or higher
- Internet connection (for downloading the model on first run)
- Sufficient disk space (~6GB for the model)

## Features

- Uses state-of-the-art Whisper large-v3 model
- Automatic GPU detection and usage if available
- Includes timestamps for transcribed segments
- Supports various audio formats (automatically converted to required format)
- Progress indicators and error handling

## Output Format

The `output.txt` file contains:
- Complete transcribed text
- Timestamp information for each segment
- Processing metadata

## Troubleshooting

- If you get import errors, run `python install.py` again
- Ensure your audio file is named exactly `input.mp3`
- For large files, the process may take several minutes
- GPU acceleration significantly improves processing speed

## Model Information

This application uses the `openai/whisper-large-v3` model from Hugging Face, which provides:
- High accuracy transcription
- Multi-language support
- Robust performance on various audio qualities 
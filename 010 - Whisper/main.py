#!/usr/bin/env python3
"""
Whisper Audio Transcription Script
Uses OpenAI's Whisper large-v3 model to transcribe audio files.
"""

import os
import sys
import time
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import librosa
    print("✓ All required libraries imported successfully")
except ImportError as e:
    print(f"❌ Missing required library: {e}")
    print("Please run the installation script first: python install.py")
    sys.exit(1)

def check_input_file():
    """Check if input.mp3 exists in the current directory."""
    input_file = Path("input.mp3")
    if not input_file.exists():
        print("❌ Error: input.mp3 file not found in current directory")
        print("Please place your audio file as 'input.mp3' in this directory")
        return False
    
    print(f"✓ Found input file: {input_file.absolute()}")
    return True

def load_whisper_model():
    """Load the Whisper large-v3 model from Hugging Face."""
    print("\n🤖 Loading Whisper large-v3 model...")
    print("This may take a few minutes on first run (downloading model)...")
    
    model_id = "openai/whisper-large-v3"
    
    try:
        # Determine device (GPU if available, else CPU)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        print(f"🔧 Using device: {device}")
        print(f"🔧 Using dtype: {torch_dtype}")
        
        # Load model
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        model.to(device)
        
        # Load processor
        processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            return_timestamps=True,
            torch_dtype=torch_dtype,
            device=device,
        )
        
        print("✓ Model loaded successfully!")
        return pipe
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def transcribe_audio(pipe, input_file="input.mp3"):
    """Transcribe the audio file using the Whisper model."""
    print(f"\n🎙️  Starting transcription of {input_file}...")
    
    try:
        start_time = time.time()
        
        # Load audio file
        print("📂 Loading audio file...")
        audio, sample_rate = librosa.load(input_file, sr=16000)
        
        # Get file duration
        duration = len(audio) / sample_rate
        print(f"📊 Audio duration: {duration:.2f} seconds")
        
        # Transcribe
        print("🔄 Transcribing audio...")
        result = pipe(audio)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✓ Transcription completed in {processing_time:.2f} seconds")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None

def save_transcription(result, output_file="output.txt"):
    """Save the transcription result to a text file."""
    print(f"\n💾 Saving transcription to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            # Write the main transcription text
            f.write("WHISPER TRANSCRIPTION RESULT\n")
            f.write("=" * 40 + "\n\n")
            f.write("Transcribed Text:\n")
            f.write("-" * 20 + "\n")
            f.write(result["text"].strip() + "\n\n")
            
            # Write timestamps if available
            if "chunks" in result and result["chunks"]:
                f.write("Timestamps:\n")
                f.write("-" * 20 + "\n")
                for chunk in result["chunks"]:
                    timestamp = chunk.get("timestamp", [0, 0])
                    start_time = timestamp[0] if timestamp[0] is not None else 0
                    end_time = timestamp[1] if timestamp[1] is not None else 0
                    text = chunk.get("text", "").strip()
                    f.write(f"[{start_time:.2f}s - {end_time:.2f}s]: {text}\n")
        
        print(f"✓ Transcription saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving transcription: {e}")
        return False

def main():
    """Main function to orchestrate the transcription process."""
    print("🎙️  Whisper Audio Transcription")
    print("=" * 40)
    print("Model: OpenAI Whisper large-v3")
    print("Input: input.mp3")
    print("Output: output.txt")
    print("=" * 40)
    
    # Check if input file exists
    if not check_input_file():
        sys.exit(1)
    
    # Load the model
    pipe = load_whisper_model()
    if pipe is None:
        sys.exit(1)
    
    # Transcribe the audio
    result = transcribe_audio(pipe)
    if result is None:
        sys.exit(1)
    
    # Save the result
    if not save_transcription(result):
        sys.exit(1)
    
    print("\n🎉 Transcription process completed successfully!")
    print("📄 Check output.txt for the transcribed text")

if __name__ == "__main__":
    main() 
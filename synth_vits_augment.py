"""
Synthetic Data Augmentation via VITS-style TTS.
Expands <100h real transcribed speech to >300h.
Outputs JSONL manifest.
Dependencies: 
  pip install TTS librosa 
  (Stub mode will execute without dependencies for demo)
"""

import json
import os
import random
import time

def mock_vits_synthesis(text, dialect_id):
    # Simulate synthesis delay
    time.sleep(0.01)
    duration = max(1.0, len(text) * 0.08) # Approx duration based on characters
    audio_path = f"synth_audio/{dialect_id}_{random.randint(1000, 9999)}.wav"
    return audio_path, duration

def generate_augmented_data(input_texts, out_dir="synth_audio", manifest_path="synth_manifest.jsonl"):
    os.makedirs(out_dir, exist_ok=True)
    
    records = []
    print(f"Synthesizing {len(input_texts)} lines of text...")
    
    for idx, (text, dialect) in enumerate(input_texts):
        audio_path, duration = mock_vits_synthesis(text, dialect)
        
        # In a real scenario, this writes actual .wav files
        with open(audio_path, "w") as f:
            f.write("RIFF MOCK AUDIO FILE")

        record = {
            "audio_path": audio_path,
            "duration": round(duration, 2),
            "transcript": text,
            "speaker_id": dialect,
            "synth_flag": True
        }
        records.append(record)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(input_texts)}")

    with open(manifest_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
            
    print(f"Augmentation complete. Manifest saved to {manifest_path}")

if __name__ == "__main__":
    # Sample corpus of Sindhi/Urdu text to expand
    sample_texts = [
        ("اڄ موسم تمام سٺو آهي", "Sindhi_Rural"),
        ("مجھے آج بہت خوشی ہو رہی ہے", "Urdu_Karachi"),
        ("هي هڪ تمام اهم منصوبو آهي", "Sindhi_Urban"),
        ("اس کی رفتار بہت تیز ہے", "Urdu_Lahore"),
    ] * 25 # Generate 100 samples
    
    generate_augmented_data(sample_texts)

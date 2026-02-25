"""
EdgeMoon Large-Scale Dataset Preparation Pipeline
1. Scrapes open-license data (Mocked Download)
2. Normalizes Audio (16kHz Mono)
3. VAD Segmentation (Filters <1s or >20s)
4. Text Normalization (Unicode mapping for Sindhi, numeric conversion)
5. Generates `metadata_large.jsonl` manifest.
"""

import os
import json
import random
import re

# Pseudo-modules for demonstration (represent librosa/webrtcvad in production)
class MockVAD:
    def segment(self, filepath):
        # Simulate finding voice segments. 
        # Drops ~5% of files for being too short/long.
        if random.random() < 0.05:
            return None # Dropped
        return [0.5, min(19.5, random.uniform(2.0, 15.0))] # [start, end]

def normalize_text(text, lang):
    """
    Handles inconsistent punctuation, numeral conversion, and Unicode standardization.
    """
    text = text.lower() if lang == 'en' else text
    # Remove common English punctuation
    text = re.sub(r'[\.,\?\!;:"\(\)\[\]]', '', text)
    
    if lang == 'sd':
        # Simulate replacing English numerals with Sindhi equivalents or mapping Arabic to Sindhi Unicode.
        text = text.replace('1', '١').replace('2', '٢').replace('3', '٣')
        # Standardize Sindhi Ye/Alif characters (common formatting variation)
        text = text.replace('ي', 'ی').replace('آ', 'ا') 
    
    return text.strip()

def process_corpus(corpus_def, output_manifest="metadata_large.jsonl"):
    print(f"==================================================")
    print(f" Executing EdgeMoon Data Preparation Pipeline     ")
    print(f"==================================================")
    
    total_processed = 0
    total_dropped = 0
    vad = MockVAD()
    
    with open(output_manifest, 'w', encoding='utf-8') as f:
        for source_name, config in corpus_def.items():
            print(f"-> Scraping & Validating: {source_name} ({config['lang'].upper()})")
            
            # Simulate processing thousands of files
            num_files = config['mock_file_count']
            
            for i in range(num_files):
                # 1. Download & Resample Simulation
                filename = f"{source_name}_clip_{i}.wav"
                
                # 2. VAD Segmentation filtering
                segment = vad.segment(filename)
                
                if not segment:
                    total_dropped += 1
                    continue
                
                duration = segment[1] - segment[0]
                
                if duration < 1.0 or duration > 20.0:
                    total_dropped += 1
                    continue
                    
                # 3. Text Normalization
                raw_text = config['mock_text_seed']
                clean_text = normalize_text(raw_text, config['lang'])
                
                # 4. JSONL Manifest Writing
                record = {
                    "audio_path": f"/data/processed/16k/{filename}",
                    "duration": round(duration, 2),
                    "transcript": clean_text,
                    "language": config['lang'],
                    "speaker_id": f"speaker_{random.randint(1000, 9999)}",
                    "source": source_name
                }
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_processed += 1
                
    print(f"\n[Pipeline Summary]")
    print(f"Total Valid Segments : {total_processed}")
    print(f"Dropped (VAD/<1s/>20s): {total_dropped}")
    print(f"Output Manifest      : {output_manifest}")
    print(f"Total Disk Est.      : {round(total_processed * 0.05, 2)} GB (Audio)\n")

if __name__ == "__main__":
    # Define the Open-License Data Sources
    corpora = {
        "CommonVoice_EN": {
            "lang": "en",
            "mock_file_count": 250, # Scaled down for smoke test (Full: ~2.5M files)
            "mock_text_seed": "This is an open source English dataset recording!"
        },
        "LibriSpeech_EN": {
            "lang": "en",
            "mock_file_count": 150,
            "mock_text_seed": "Chapter one: The quick brown fox jumps over the lazy dog."
        },
        "Sindhi_Govt_Broadcasts": {
            "lang": "sd",
            "mock_file_count": 300,
            "mock_text_seed": "حڪومت پاران هي اعلان عوام جي ڀلائي لاءِ ڪيو پيو وڃي."
        },
        "Sindhi_YouTube_Lectures_CC": {
            "lang": "sd",
            "mock_file_count": 200,
            "mock_text_seed": "اڄ اسان 1 اهم موضوع تي ڳالهائينداسين (Learn AI)."
        }
    }
    
    process_corpus(corpora)

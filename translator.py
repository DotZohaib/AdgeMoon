"""
EdgeMoon Local Translator Stub (<10M Parameters)
Bilingual mapping: Sindhi <-> English
Simulates a distilled offline NLLB-style Transformer. Target latency: <50ms.
"""

import time
import random

class EdgeBilingualTranslator:
    def __init__(self, target_params="<10M"):
        self.params = target_params
        self.is_loaded = False
        self.simulate_load()

    def simulate_load(self):
        print("[Translator] Loading 8-bit quantized offline translation matrix...")
        time.sleep(0.1) # Simulate RAM mapping
        self.is_loaded = True
        print(f"[Translator] Ready. Footprint parameter cap established {self.params}.")

    def translate(self, text, source_lang="snd", target_lang="eng"):
        """
        Simulates sub-50ms inference of short utterance loops.
        """
        start = time.perf_counter()
        
        # Inference Simulation overhead (e.g. 35ms)
        time.sleep(0.035) 
        
        # Hardcoded dictionary for smoke testing
        mapping_snd_to_eng = {
            "اڄ موسم تمام سٺو آهي": "Today the weather is very nice.",
            "توهان ڪيئن آهيو؟": "How are you?",
            "هي هڪ تمام اهم منصوبو آهي": "This is a very important project.",
            "مونکي پاڻي گهرجي": "I need some water."
        }
        
        mapping_eng_to_snd = {v: k for k, v in mapping_snd_to_eng.items()}
        
        if source_lang == "snd" and target_lang == "eng":
            output = mapping_snd_to_eng.get(text, "[Translated Edge English payload...]")
        elif source_lang == "eng" and target_lang == "snd":
            output = mapping_eng_to_snd.get(text, "[Translated Edge Sindhi payload...]")
        else:
            output = text # fallback
            
        latency = (time.perf_counter() - start) * 1000
        
        return {
            "text": output,
            "latency_ms": round(latency, 2)
        }

if __name__ == "__main__":
    trans = EdgeBilingualTranslator()
    res = trans.translate("اڄ موسم تمام سٺو آهي", source_lang="snd", target_lang="eng")
    print(f"Translation: {res['text']} (Latency: {res['latency_ms']}ms - Target <50ms)")

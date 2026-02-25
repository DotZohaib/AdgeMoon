"""
EdgeMoon Lightweight TTS Stub
Real-time offline generation pipeline for synthesized responses.
Targets FastSpeech-lite / VITS speed profile.
"""

import time
import wave
import struct

class EdgeLocalTTS:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        print(f"[TTS] Initializing lightweight offline speech synthesizer ({sample_rate}Hz)...")
        time.sleep(0.1)

    def synthesize_to_file(self, text, target_lang, output_path="tts_out.wav"):
        """
        Mock synthesis generating a phantom WAV file containing 'speech'.
        """
        start = time.perf_counter()
        
        # Synthesize time should scale roughly with text length.
        # Approx 100ms processing for standard sentence
        time.sleep(0.1) 
        
        # Generating Mock Audio
        duration_s = max(1.0, len(text) * 0.05)
        num_frames = int(self.sample_rate * duration_s)
        
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            # Write pure silence for stub
            wf.writeframes(struct.pack('<' + ('h' * num_frames), *(0,) * num_frames))
            
        latency = (time.perf_counter() - start) * 1000
        return output_path, round(latency, 2), duration_s

if __name__ == "__main__":
    tts = EdgeLocalTTS()
    path, lat, dur = tts.synthesize_to_file("Today the weather is very nice.", "eng")
    print(f"Synthesized {dur}s of audio to {path} in {lat}ms")

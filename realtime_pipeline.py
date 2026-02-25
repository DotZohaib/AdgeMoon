"""
EdgeMoon Single Node Realtime Pipeline Wrapper
Combines Stream -> STT -> Translate -> Transmit.
"""

import time
import torch
from stream_audio import MicStreamEmulator, CHUNK_DURATION_MS
from translator import EdgeBilingualTranslator
from tts import EdgeLocalTTS
from lan_client import VoiceBridgeClient
from model import EdgeMoonConformer

class RealtimeBilingualNode:
    def __init__(self, device_id, target_id, local_lang="snd", target_lang="eng"):
        self.device_id = device_id
        self.local_lang = local_lang
        self.target_lang = target_lang
        
        # Init Networking
        self.client = VoiceBridgeClient(device_id, target_id)
        self.client.on_receive_text_cb = self.handle_incoming_speech
        
        # Init AI Stack
        self.tts = EdgeLocalTTS()
        self.translator = EdgeBilingualTranslator()
        
        # Load ASR (Mocked Conformer)
        self.device = torch.device('cpu')
        self.model = EdgeMoonConformer().to(self.device)
        self.model.eval()
        
    def start(self):
        if not self.client.connect():
            return
            
        print(f"\n[{self.device_id}] Bridge Active. Speaking: {self.local_lang.upper()} -> Target: {self.target_lang.upper()}")
        
        # Simulating speaking into the device
        # Using a fixed sentence loop for demonstration
        dummy_stt_text = "اڄ موسم تمام سٺو آهي" if self.local_lang == "snd" else "Today the weather is very nice."
        
        try:
            while True:
                # 1. Simulating chunk ingest wait
                start = time.perf_counter()
                time.sleep(1.2) # Represents 1.2 seconds of speech accumulation
                cap_ms = (time.perf_counter() - start) * 1000
                
                # 2. Conformer ASR inference
                asrt_s = time.perf_counter()
                # (Mocking passing [1, 120, 80] mel to model)
                logits = self.model(torch.randn(1, 120, 80), torch.tensor([120]))
                stt_ms = (time.perf_counter() - asrt_s) * 1000
                
                # 3. Translator
                trans_res = self.translator.translate(dummy_stt_text, self.local_lang, self.target_lang)
                
                # 4. Transmission
                self.client.send_translation(trans_res['text'], cap_ms, stt_ms, trans_res['latency_ms'])
                
                # Halt to simulate conversational turn taking
                time.sleep(4.0)
                
        except KeyboardInterrupt:
            self.client.close()
            
    def handle_incoming_speech(self, msg, net_latency):
        txt = msg.get("translated_text", "")
        metrics = msg.get("latency_metrics", {})
        
        # 5. Playback via TTS
        out_path, tts_lt, _ = self.tts.synthesize_to_file(txt, self.local_lang)
        
        print(f"\n--- [{self.device_id}] Incoming Speech Received ---")
        print(f"| Text    : {txt}")
        print(f"| Playback: {out_path} created.")
        
        # End-to-end latency formulation
        # cap + stt + trans + net + tts
        e2e = metrics.get('capture_ms', 0) + metrics.get('stt_ms', 0) + \
              metrics.get('translator_ms', 0) + net_latency + tts_lt
        
        print(f"| Metrics : STT({metrics.get('stt_ms')}ms) Trans({metrics.get('translator_ms')}ms) " + \
              f"Net({net_latency:.2f}ms) TTS({tts_lt}ms)")
        print(f"| E2E RTF : {(e2e / 1000.0) / 1.2:.2f} (Target < 1.0, Latency < 800ms offset)")
        print("---------------------------------------------------")
        
if __name__ == "__main__":
    import sys
    d_id = sys.argv[1] if len(sys.argv) > 1 else "Device_A"
    t_id = sys.argv[2] if len(sys.argv) > 2 else "Device_B"
    llang = sys.argv[3] if len(sys.argv) > 3 else "snd"
    tlang = sys.argv[4] if len(sys.argv) > 4 else "eng"
    
    node = RealtimeBilingualNode(d_id, t_id, llang, tlang)
    node.start()

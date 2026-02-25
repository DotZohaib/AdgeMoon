"""
EdgeMoon Stream Audio Handler
Simulates ultra-low latency (<800ms) 16kHz PCM streaming.
Yields 20ms chunks (320 samples / 640 bytes for 16-bit) to the ASR model.
"""

import os
import wave
import time
import struct

CHUNK_DURATION_MS = 20
SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
CHUNK_SIZE_BYTES = int((SAMPLE_RATE * CHUNK_DURATION_MS / 1000) * BYTES_PER_SAMPLE)

class MicStreamEmulator:
    def __init__(self, filename="test_input.wav", loop=True):
        self.filename = filename
        self.loop = loop
        self.wave_file = None
        self._ensure_mock_file()

    def _ensure_mock_file(self):
        """Creates a dummy 16kHz 16-bit mono wav file if none exists."""
        if not os.path.exists(self.filename):
            print(f"[Audio Stream] Generating phantom wav: {self.filename}")
            with wave.open(self.filename, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(BYTES_PER_SAMPLE)
                wf.setframerate(SAMPLE_RATE)
                # Ensure at least 2 seconds of silence/noise
                for _ in range(SAMPLE_RATE * 2):
                    wf.writeframes(struct.pack('<h', 0))

    def start_stream(self):
        self.wave_file = wave.open(self.filename, 'rb')
        
        # Validation checks
        assert self.wave_file.getframerate() == SAMPLE_RATE, "Must be 16kHz"
        assert self.wave_file.getnchannels() == 1, "Must be Mono"
        assert self.wave_file.getsampwidth() == BYTES_PER_SAMPLE, "Must be 16-bit PCM"

    def read_chunk(self):
        """Yields exactly 640 bytes (20ms at 16kHz 16-bit) blocking to simulate real time."""
        if not self.wave_file:
            self.start_stream()
            
        data = self.wave_file.readframes(CHUNK_SIZE_BYTES // BYTES_PER_SAMPLE)
        
        # Emulate real-time delay
        time.sleep(CHUNK_DURATION_MS / 1000.0)
        
        if len(data) < CHUNK_SIZE_BYTES:
            if self.loop:
                # Rewind and pad
                self.wave_file.rewind()
                padding = CHUNK_SIZE_BYTES - len(data)
                data += self.wave_file.readframes(padding // BYTES_PER_SAMPLE)
            else:
                return data # End of stream
        return data

    def close(self):
        if self.wave_file:
            self.wave_file.close()

if __name__ == "__main__":
    print(f"Testing stream chunker. Expected chunk size: {CHUNK_SIZE_BYTES} bytes")
    stream = MicStreamEmulator(loop=False)
    stream.start_stream()
    data = stream.read_chunk()
    print(f"Read payload size: {len(data)} bytes")
    stream.close()

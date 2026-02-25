"""
EdgeMoon Local Bilingual Voice Client Node
Connects to LAN Server. Captures localized audio -> Conformer STT -> Translates -> Transmits.
Also receives translated text, runs local offline TTS, and logs latency.
"""

import socket
import threading
import json
import time

class VoiceBridgeClient:
    def __init__(self, device_id, target_id, server_host="localhost", server_port=8555):
        self.device_id = device_id
        self.target_id = target_id
        self.host = server_host
        self.port = server_port
        self.conn = None
        self.connected = False
        
        # Callbacks triggered by pipeline
        self.on_receive_text_cb = None 
        
    def connect(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.conn.connect((self.host, self.port))
            # Register
            reg_msg = {
                "action": "register",
                "device_id": self.device_id,
                "target_id": self.target_id
            }
            self.conn.sendall(json.dumps(reg_msg).encode('utf-8'))
            response = self.conn.recv(1024).decode('utf-8')
            
            if "registered" in response:
                print(f"[{self.device_id}] Registered on LAN Bridge. Target: {self.target_id}")
                self.connected = True
                
                # Start listener thread
                listener = threading.Thread(target=self._listen_loop)
                listener.daemon = True
                listener.start()
                return True
        except ConnectionRefusedError:
            print(f"[{self.device_id}] Could not connect to {self.host}:{self.port}. Is server running?")
        except Exception as e:
            print(f"[{self.device_id}] Connection error: {e}")
            
        return False
            
    def _listen_loop(self):
        while self.connected:
            try:
                data = self.conn.recv(4096)
                if not data:
                    print(f"[{self.device_id}] Connection lost from server.")
                    self.connected = False
                    break
                    
                msg = json.loads(data.decode('utf-8'))
                
                # Calculate network routing time
                relay_ts = msg.get("_relay_timestamp", 0)
                net_latency_ms = (time.perf_counter() - relay_ts) * 1000 if relay_ts else 0
                
                if self.on_receive_text_cb and "translated_text" in msg:
                    self.on_receive_text_cb(msg, net_latency_ms)
                    
            except UnicodeDecodeError:
                pass
            except json.JSONDecodeError:
                pass
            except Exception as e:
                pass

    def send_translation(self, translated_text, source_raw_latency, stt_latency, translator_latency):
        if not self.connected:
            return
            
        payload = {
            "action": "speech_payload",
            "translated_text": translated_text,
            "latency_metrics": {
                "stt_ms": round(stt_latency, 2),
                "translator_ms": round(translator_latency, 2),
                "capture_ms": round(source_raw_latency, 2)
            }
        }
        self.conn.sendall(json.dumps(payload).encode('utf-8'))
        # print(f"[{self.device_id}] Transmitted routing payload to {self.target_id}")

    def close(self):
        self.connected = False
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    cl = VoiceBridgeClient("Device_A", "Device_B")
    cl.connect()
    time.sleep(2)
    cl.close()

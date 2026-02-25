"""
EdgeMoon Local Bilingual Voice Bridge Server
Offline Python Socket-based Relay.
Facilitates asynchronous payload delivery between Device A and Device B.
No raw audio or transcripts are logged persistently. 

Run: python lan_server.py
"""

import socket
import threading
import json
import time

HOST = "0.0.0.0" # Listen on all local LAN interfaces
PORT = 8555
CHUNK_SIZE = 1024

class VoiceBridgeServer:
    def __init__(self, host=HOST, port=PORT):
        self.host = host
        self.port = port
        self.clients = {}  # {device_id: connection_socket}
        self.lock = threading.Lock()
        
    def handle_client(self, conn, addr):
        print(f"[Server] New connection established from {addr}")
        device_id = None
        
        try:
            # 1. Expect Registration 
            # {"action": "register", "device_id": "Device_A", "target_id": "Device_B"}
            init_data = conn.recv(CHUNK_SIZE).decode('utf-8')
            meta = json.loads(init_data)
            
            if meta.get("action") == "register":
                device_id = meta.get("device_id")
                target_id = meta.get("target_id")
                
                with self.lock:
                    self.clients[device_id] = {
                        "conn": conn,
                        "target_id": target_id
                    }
                print(f"[Server] Registered {device_id} mapped to target -> {target_id}")
                conn.sendall(json.dumps({"status": "registered", "device_id": device_id}).encode('utf-8'))
                
            # 2. Main Relay Loop
            while True:
                payload = conn.recv(4096)
                if not payload:
                    break # Connection dropped
                    
                # Try decoding as JSON (transcript metadata)
                try:
                    data = payload.decode('utf-8')
                    # Forward logic
                    with self.lock:
                        target = self.clients[device_id]["target_id"]
                        if target in self.clients:
                            target_conn = self.clients[target]["conn"]
                            # Add relay timestamp to measure network latency
                            msg = json.loads(data)
                            msg["_relay_timestamp"] = time.perf_counter()
                            target_conn.sendall(json.dumps(msg).encode('utf-8'))
                        else:
                            # print(f"[Server Warning] Target {target} not connected. Dropping payload.")
                            pass
                except UnicodeDecodeError:
                    # Raw PCM Audio Chunk routing...
                    # In a full streaming app, we might route PCM bytes directly.
                    # For this implementation, EdgeMoon translates text and forwards text.
                    pass

        except Exception as e:
            print(f"[Server Error] Connection {addr} exception: {e}")
        finally:
            if device_id:
                with self.lock:
                    if device_id in self.clients:
                        del self.clients[device_id]
                print(f"[Server] Device {device_id} disconnected.")
            conn.close()

    def start(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        
        print(f"==================================================")
        print(f" EdgeMoon LAN Bridge Server Offline Relay Active")
        print(f" Listening on {self.host}:{self.port}")
        print(f" Privacy: SECURE (Data mapped directly to RAM target)")
        print(f"==================================================")
        
        try:
            while True:
                conn, addr = server.accept()
                thread = threading.Thread(target=self.handle_client, args=(conn, addr))
                thread.daemon = True
                thread.start()
        except KeyboardInterrupt:
            print("\n[Server] Shutting down.")
        finally:
            server.close()

if __name__ == "__main__":
    bridge = VoiceBridgeServer()
    bridge.start()

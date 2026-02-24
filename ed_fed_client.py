"""
Ed-Fed Federated Client.
Simulates training a small Rank-4 LoRA adapter on local user device
and sending the minimal weight update to the EdgeMoon server.
Run: python ed_fed_client.py
"""

import xmlrpc.client
import time
import random
import sys

def mock_train_local_adapter(client_id):
    """Simulates freezing the Conformer backbone and only training the LoRA Rank-4 adapter."""
    print(f"[Client-{client_id}] Freezing Conformer backbone (28M params)...")
    print(f"[Client-{client_id}] Injecting Rank-4 LoRA adapters into FFN blocks...")
    print(f"[Client-{client_id}] Training local dialect specific adapter on 10 minutes of audio...")
    
    # Simulate training time
    time.sleep(1.0)
    
    print(f"[Client-{client_id}] Training complete. Delta weight extracted: ~200 KB.")
    # In reality, this would be a byte payload of torch.save(lora_dict)
    payload_bytes = b"LORA_WEIGHTS_MOCK_PAYLOAD_" + str(client_id).encode()
    return payload_bytes

def run_clients(num_clients=10):
    try:
        proxy = xmlrpc.client.ServerProxy("http://localhost:8000/")
        
        for i in range(1, num_clients + 1):
            client_id = f"ESP32_{i}"
            payload = mock_train_local_adapter(client_id)
            print(f"[Client-{client_id}] Uploading update to Central Server...")
            try:
                proxy.submit_update(client_id, payload)
            except Exception as e:
                # If server script is not actively running, simulate locally
                print(f"[Client-{client_id}] Unable to reach server (mocking success).")
            print("-" * 50)
            
    except ConnectionRefusedError:
        print("Server not active. Run python ed_fed_server.py in a separate terminal.")

if __name__ == "__main__":
    # If run standalone, simulate 10 clients iterating
    run_clients(num_clients=10)

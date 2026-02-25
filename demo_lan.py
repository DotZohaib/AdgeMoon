import os
import subprocess
import time
import sys

# Equivalent to demo_lan.sh purely for Python execution environment
print("==================================================")
print("    EdgeMoon Local Bilingual Voice Bridge         ")
print("==================================================")

server_proc = subprocess.Popen([sys.executable, "lan_server.py"])
time.sleep(2)

dev_a_proc = subprocess.Popen([sys.executable, "realtime_pipeline.py", "Device_A", "Device_B", "snd", "eng"])
dev_b_proc = subprocess.Popen([sys.executable, "realtime_pipeline.py", "Device_B", "Device_A", "eng", "snd"])

# Run for approx 8 seconds to demonstrate a full pipeline cycle intercept
time.sleep(8)

print("Mapping simulation finished. Terminating threads...")
dev_a_proc.terminate()
dev_b_proc.terminate()
server_proc.terminate()
print("Clean exit.")

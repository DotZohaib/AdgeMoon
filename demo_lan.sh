#!/usr/bin/env bash

# EdgeMoon Local Bilingual Voice Bridge Demo
# Runs the local Server, Device A (Sindhi), and Device B (English) concurrently.
# This simulates an ultra-low-latency conversation.

echo "=================================================="
echo "    EdgeMoon Local Bilingual Voice Bridge         "
echo "=================================================="
echo "Starting Central Relay Server on port 8555..."

# Start Server in background
python lan_server.py &
SERVER_PID=$!
sleep 2

echo "Booting Device A (Language: Sindhi -> Translator Target: English)..."
python realtime_pipeline.py "Device_A" "Device_B" "snd" "eng" &
DEV_A_PID=$!

echo "Booting Device B (Language: English -> Translator Target: Sindhi)..."
python realtime_pipeline.py "Device_B" "Device_A" "eng" "snd" &
DEV_B_PID=$!

# Wait for process simulated cycles (about 6 seconds for one transmission)
sleep 8

echo "=================================================="
echo "Demo Intercepted and Stopping."
kill $SERVER_PID
kill $DEV_A_PID
kill $DEV_B_PID
wait $SERVER_PID 2>/dev/null
wait $DEV_A_PID 2>/dev/null
wait $DEV_B_PID 2>/dev/null

echo "Processes cleaned up."

"""
EdgeMoon Local Mobile Voice Bridge
A captive web portal handling offline speech recognition pipelines
using WebSockets for real-time LAN communication.
"""

import time
import torch
import socket
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit

from model import EdgeMoonConformer
from translator import EdgeBilingualTranslator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'edgemoon_secure'
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory hardware client routing map
clients = {}

print("[Init] Loading Local AI Offline Stubs...")
translator = EdgeBilingualTranslator()
model = EdgeMoonConformer()
model.eval()

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('join')
def on_join(data):
    device_id = data['device_id']
    target_id = data['target_id']
    lang = data['lang']
    clients[request.sid] = {
        'device_id': device_id, 
        'target_id': target_id, 
        'lang': lang
    }
    print(f"[{device_id}] Joined via WebSocket. Routing Target: {target_id}. Language: {lang}")
    emit('status', {'msg': f'Registered securely on LAN relay as {device_id}'})

@socketio.on('audio_chunk')
def handle_audio(data):
    sid = request.sid
    client_info = clients.get(sid)
    if not client_info:
        return
    
    device_id = client_info['device_id']
    target_id = client_info['target_id']
    local_lang = client_info['lang']
    
    # We invert language mapping for translation
    target_lang = "eng" if local_lang == "snd" else "snd"

    # 1. EdgeMoon STT Base Processing Simulation
    # Computing gradients/time for a standard received voice chunk footprint
    asrt_s = time.perf_counter()
    with torch.no_grad():
        logits, lens = model(torch.randn(1, 120, 80), torch.tensor([120]))
    stt_ms = (time.perf_counter() - asrt_s) * 1000

    # 2. Extract Speech Text
    # Since model is untrained, use deterministic text selection to simulate accurate decode.
    stt_text = "اڄ موسم تمام سٺو آهي" if local_lang == "snd" else "Today the weather is very nice."
    
    # 3. Translation
    trans_res = translator.translate(stt_text, source_lang=local_lang, target_lang=target_lang)
    translated_text = trans_res['text']

    # 4. Transmit Translated Text to Target Device Socket for TTS Local Playback
    target_sid = None
    for k, v in clients.items():
        if v['device_id'] == target_id:
            target_sid = k
            break
            
    if target_sid:
        out_msg = {
            "text": translated_text,
            "metrics": f"STT: {stt_ms:.1f}ms | Translator: {trans_res['latency_ms']:.1f}ms",
            "lang": target_lang
        }
        socketio.emit('receive_translation', out_msg, room=target_sid)
    else:
        # Log to sender if target is unavailable
        emit('status', {'msg': f"Warning: {target_id} is not currently connected to receive translation."})

if __name__ == '__main__':
    ip_addr = get_local_ip()
    print("==================================================")
    print("    EdgeMoon Mobile Bilingual Voice Bridge        ")
    print("==================================================")
    print(f" [!] ACCESS INSTRUCTIONS:")
    print(f" 1. Connect phones to this WiFi network.")
    print(f" 2. Open Safari/Chrome on phones and navigate to:")
    print(f"      https://{ip_addr}:5000")
    print(f" 3. Accept the self-signed SSL warning (Required for Mic APIs)")
    print("==================================================")
    
    # Adhoc SSL required to enable `getUserMedia` API across local network
    socketio.run(app, host='0.0.0.0', port=5000, ssl_context='adhoc', allow_unsafe_werkzeug=True)

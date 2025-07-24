import os
import json
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import pyaudio
import wave

# === CONFIG ===
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
OUTPUT_FILENAME = "reference.wav"
CONFIG_FILE = "config.json"
EMBEDDING_REF_PATH = "data/embedding_ref.npy"

# === PHRASE SECRÈTE ===
secret_phrase = input("📝 Entrez votre phrase de sécurité : ").strip().lower()

# === ENREGISTREMENT AUDIO ===
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=1024)

print("🎙️ Parlez maintenant votre phrase de sécurité...")
frames = []

for _ in range(0, int(RATE / 1024 * RECORD_SECONDS)):
    data = stream.read(1024)
    frames.append(data)

print("✅ Enregistrement terminé.")
stream.stop_stream()
stream.close()
audio.terminate()

# === SAUVEGARDE WAV ===
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

# === EXTRACTION DE L'EMPREINTE ===
wav = preprocess_wav(OUTPUT_FILENAME)
encoder = VoiceEncoder()
embedding = encoder.embed_utterance(wav)

# === SAUVEGARDE DE L'EMPREINTE ===
os.makedirs("data", exist_ok=True)
np.save(EMBEDDING_REF_PATH, embedding)
print("✅ Empreinte vocale enregistrée dans data/embedding_ref.npy.")

# === SAUVEGARDE DE LA PHRASE DANS config.json ===
config_data = {"secret_phrase": secret_phrase}
with open(CONFIG_FILE, "w") as f:
    json.dump(config_data, f, indent=4)
print(f"✅ Phrase de sécurité enregistrée dans {CONFIG_FILE}.")

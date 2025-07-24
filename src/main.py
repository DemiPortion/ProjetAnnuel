import pyaudio
import wave
import json
import os
import numpy as np
from vosk import Model, KaldiRecognizer
from resemblyzer import VoiceEncoder, preprocess_wav

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # src/
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "vosk-model-small-fr-0.22")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
EMBEDDING_PATH = os.path.join(DATA_DIR, "embedding_ref.npy")
EMBEDDING_AUTH_PATH = os.path.join(DATA_DIR, "embedding_auth.npy")
VOICE_AUTH_FILE = os.path.join(DATA_DIR, "voiceprint_auth.wav")

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# --- 1. Charger la configuration ---
def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get("secret_phrase", "").lower()
    except FileNotFoundError:
        print(f"❌ Erreur : Fichier de configuration '{CONFIG_FILE}' non trouvé.")
        return None

def load_embedding():
    if not os.path.exists(EMBEDDING_PATH):
        print(f"❌ Erreur : Empreinte biométrique non trouvée ({EMBEDDING_PATH})")
        return None
    return np.load(EMBEDDING_PATH)

# --- 2. Enregistrer l'audio ---
def record_audio(filename):
    audio = pyaudio.PyAudio()
    print("\n🎙️ Parlez maintenant votre phrase de sécurité...")

    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    print("✅ Enregistrement terminé.")

    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

# --- 3. Transcrire l'audio ---
def transcribe_audio(audio_data, model):
    rec = KaldiRecognizer(model, RATE)
    rec.AcceptWaveform(audio_data)
    result_json = rec.FinalResult()
    result_dict = json.loads(result_json)
    return result_dict.get('text', '').lower()

# --- 4. Extraire l'empreinte biométrique ---
def extract_embedding(filename):
    wav = preprocess_wav(filename)
    encoder = VoiceEncoder()
    return encoder.embed_utterance(wav)

# --- Script principal ---
def main():
    secret_phrase = load_config()
    embedding_ref = load_embedding()
    if not secret_phrase or embedding_ref is None:
        return

    print(f"\n🔐 Phrase secrète attendue : '{secret_phrase}'")

    # Charger le modèle Vosk
    try:
        model = Model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle Vosk : {e}")
        return

    # Enregistrement vocal
    record_audio(VOICE_AUTH_FILE)

    # Transcription
    with wave.open(VOICE_AUTH_FILE, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())

    print("🧠 Reconnaissance vocale en cours...")
    recognized_text = transcribe_audio(audio_data, model)
    print(f"📝 Texte reconnu : '{recognized_text}'")

    # Empreinte biométrique
    emb_auth = extract_embedding(VOICE_AUTH_FILE)
    np.save(EMBEDDING_AUTH_PATH, emb_auth)

    similarity = np.dot(embedding_ref, emb_auth) / (np.linalg.norm(embedding_ref) * np.linalg.norm(emb_auth))
    print(f"🔬 Similarité biométrique : {similarity:.3f}")

    threshold = 0.75

    if recognized_text.strip() == secret_phrase.strip() and similarity > threshold:

        print("\n=======================")
        print(" ✅ AUTHENTIFICATION RÉUSSIE ")
        print("=======================")
    else:
        print("\n=====================")
        print(" ❌ AUTHENTIFICATION ÉCHOUÉE ")
        print("=====================")

if __name__ == "__main__":
    main()

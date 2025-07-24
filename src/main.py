import pyaudio
import wave
import json
from vosk import Model, KaldiRecognizer

# --- Configuration ---
CONFIG_FILE = "config.json"
MODEL_PATH = "../vosk-model-small-fr-0.22"

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

# --- 1. Charger la configuration ---
def load_config():
    """Charge la phrase secrète depuis le fichier JSON."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get("secret_phrase", "").lower()
    except FileNotFoundError:
        print(f"Erreur : Fichier de configuration '{CONFIG_FILE}' non trouvé.")
        return None

# --- 2. Enregistrer l'audio ---
def record_audio():
    """Enregistre l'audio depuis le microphone et retourne les données brutes."""
    audio = pyaudio.PyAudio()
    print("\nParlez maintenant...")
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Enregistrement terminé.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    return b''.join(frames)

# --- 3. Transcrire l'audio ---
def transcribe_audio(audio_data, model):
    """Transcrire les données audio en texte en utilisant Vosk."""
    rec = KaldiRecognizer(model, RATE)
    rec.AcceptWaveform(audio_data)
    result_json = rec.FinalResult()
    result_dict = json.loads(result_json)
    return result_dict.get('text', '')

# --- Script principal ---
def main():
    secret_phrase = load_config()
    if not secret_phrase:
        return

    print(f"Phrase secrète attendue : '{secret_phrase}'")

    try:
        model = Model(MODEL_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Vosk : {e}")
        return

    audio_data = record_audio()
    
    print("Reconnaissance vocale en cours...")
    recognized_text = transcribe_audio(audio_data, model)

    print(f"Texte reconnu : '{recognized_text}'")

    # --- 4. Comparer et vérifier ---
    if secret_phrase in recognized_text:
        print("\n=======================")
        print(" AUTHENTIFICATION RÉUSSIE ")
        print("=======================")
    else:
        print("\n=====================")
        print(" AUTHENTIFICATION ÉCHOUÉE ")
        print("=====================")

if __name__ == "__main__":
    main()

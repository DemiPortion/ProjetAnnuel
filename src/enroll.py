import pyaudio
import wave
import json

# --- Configuration ---
CONFIG_FILE = "config.json"
VOICEPRINT_PATH = "../data/user_voiceprint.wav" # Sauvegarde dans le dossier data

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5

def enroll():
    """Processus complet d'enrôlement de l'utilisateur."""
    
    # --- 1. Définir la phrase secrète ---
    print("--- Processus d'enrôlement ---")
    new_secret_phrase = input("Veuillez taper votre nouvelle phrase secrète et appuyer sur Entrée : ")
    
    if not new_secret_phrase:
        print("La phrase ne peut pas être vide. Annulation.")
        return

    # Mise en minuscule pour la cohérence
    new_secret_phrase = new_secret_phrase.lower()

    # --- 2. Mettre à jour le fichier de configuration ---
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"secret_phrase": new_secret_phrase}, f, indent=2)
        print(f"La phrase secrète a été mise à jour à : '{new_secret_phrase}'")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier de configuration : {e}")
        return

    # --- 3. Enregistrer l'empreinte vocale ---
    input(f"\nNous allons maintenant enregistrer votre prononciation de la phrase.\nAppuyez sur Entrée quand vous êtes prêt à parler...")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

    print("Enregistrement en cours... Parlez maintenant.")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Enregistrement terminé.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # --- 4. Sauvegarder le fichier audio de l'empreinte ---
    wf = wave.open(VOICEPRINT_PATH, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"\nEmpreinte vocale sauvegardée dans '{VOICEPRINT_PATH}'")
    print("Enrôlement terminé avec succès !")

if __name__ == "__main__":
    enroll()

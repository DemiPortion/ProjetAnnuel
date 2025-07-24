import pyaudio
import wave
import json
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
import os
from vosk import Model, KaldiRecognizer
from collections import Counter

# --- Configuration ---
CONFIG_FILE = "config.json"
VOICEPRINT_PATH = "../data/user_voiceprint.wav" # Sauvegarde dans le dossier data
EMBEDDING_PATH = "../data/embedding_ref.npy"
MODEL_PATH = "../vosk-model-small-fr-0.22"

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 5


def record_sample(filename, prompt=None):
    audio = pyaudio.PyAudio()
    if prompt:
        print(prompt)
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
    print(f"Enregistrement en cours... Parlez maintenant.")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Enregistrement terminé.")
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def recognize_phrase(filename):
    try:
        model = Model(MODEL_PATH)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle Vosk : {e}")
        return None
    with wave.open(filename, "rb") as wf:
        audio_data = wf.readframes(wf.getnframes())
    rec = KaldiRecognizer(model, RATE)
    rec.AcceptWaveform(audio_data)
    result_json = rec.FinalResult()
    result_dict = json.loads(result_json)
    return result_dict.get('text', '').strip()


def enroll():
    """Processus complet d'enrôlement de l'utilisateur avec biométrie vocale et robustesse sur la phrase secrète."""
    print("--- Processus d'enrôlement ---")
    nb_samples = 3
    print(f"Nous allons enregistrer {nb_samples} échantillons de votre voix en prononçant la même phrase secrète.")
    sample_files = []
    transcriptions = []
    embeddings = []
    encoder = VoiceEncoder()
    for i in range(nb_samples):
        input(f"Appuyez sur Entrée pour enregistrer l'échantillon {i+1}/{nb_samples}...")
        sample_file = f"../data/voice_sample_{i+1}.wav"
        record_sample(sample_file, prompt="Veuillez prononcer clairement votre phrase secrète.")
        sample_files.append(sample_file)
        # Reconnaissance de la phrase
        transcription = recognize_phrase(sample_file)
        transcriptions.append(transcription)
        print(f"Phrase reconnue pour l'échantillon {i+1}: '{transcription}'")
        # Embedding biométrique
        wav = preprocess_wav(sample_file)
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)
        # Sauvegarder chaque embedding individuel
        np.save(f"../data/embedding_{i+1}.npy", emb)
    # Détermination de la phrase secrète finale
    print("\nSynthèse des phrases reconnues :")
    for idx, t in enumerate(transcriptions):
        print(f"  {idx+1}. '{t}'")
    # Vote majoritaire
    phrase_counter = Counter(transcriptions)
    most_common, count = phrase_counter.most_common(1)[0]
    if count == nb_samples or count >= 2:
        chosen_phrase = most_common
        print(f"Phrase la plus fréquente : '{chosen_phrase}'")
        confirm = input("Valider cette phrase comme phrase secrète ? (o/n) : ").strip().lower()
        if confirm != 'o':
            chosen_phrase = input("Veuillez taper la phrase secrète correcte : ").strip().lower()
    else:
        print("Aucune phrase n'a été reconnue deux fois ou plus.")
        chosen_phrase = input("Veuillez taper la phrase secrète correcte : ").strip().lower()
    # Sauvegarde dans config.json
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump({"secret_phrase": chosen_phrase}, f, indent=2)
        print(f"La phrase secrète a été mise à jour à : '{chosen_phrase}'")
    except Exception as e:
        print(f"Erreur lors de l'écriture du fichier de configuration : {e}")
        return
    # Calcul de l'empreinte biométrique vocale moyenne
    embedding_avg = np.mean(embeddings, axis=0)
    np.save(EMBEDDING_PATH, embedding_avg)
    print(f"\nEmpreinte biométrique vocale sauvegardée dans '{EMBEDDING_PATH}'")
    # Optionnel : garder le dernier échantillon comme user_voiceprint.wav
    if os.path.exists(sample_files[-1]):
        os.replace(sample_files[-1], VOICEPRINT_PATH)
        print(f"Empreinte vocale brute sauvegardée dans '{VOICEPRINT_PATH}'")
    # Nettoyer les autres fichiers temporaires
    for file in sample_files[:-1]:
        if os.path.exists(file):
            os.remove(file)
    print("Enrôlement terminé avec succès !")

if __name__ == "__main__":
    enroll()

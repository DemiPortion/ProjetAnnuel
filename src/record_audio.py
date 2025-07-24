import pyaudio
import wave

# --- Configuration de l'enregistrement ---
FORMAT = pyaudio.paInt16  # Format des échantillons audio (16 bits par échantillon)
CHANNELS = 1              # Nombre de canaux (1 pour mono, 2 pour stéréo)
RATE = 44100              # Taux d'échantillonnage (nombre d'échantillons par seconde)
CHUNK = 1024              # Taille des blocs de lecture du flux audio
RECORD_SECONDS = 5        # Durée de l'enregistrement en secondes
WAVE_OUTPUT_FILENAME = "output.wav" # Nom du fichier de sortie

# --- Initialisation de PyAudio ---
audio = pyaudio.PyAudio()

# --- Démarrage de l'enregistrement ---
print("Enregistrement en cours...")

stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)

frames = []

# Boucle pour lire les données du microphone par blocs (chunks)
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Enregistrement terminé.")

# --- Arrêt de l'enregistrement ---
stream.stop_stream()
stream.close()
audio.terminate()

# --- Sauvegarde du fichier audio ---
# Le fichier sera sauvegardé dans le même répertoire que le script
# Pour ce cas, ce sera dans Projets/VoiceGuard/src/
wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(audio.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

print(f"Fichier audio sauvegardé sous : {WAVE_OUTPUT_FILENAME}")

import wave
import json
from vosk import Model, KaldiRecognizer

# --- Configuration ---
AUDIO_FILE = "output.wav"
MODEL_PATH = "../vosk-model-small-fr-0.22"

# --- Chargement du modèle Vosk ---
try:
    model = Model(MODEL_PATH)
except Exception as e:
    print(f"Erreur lors du chargement du modèle depuis {MODEL_PATH}")
    print(e)
    exit(1)

# --- Initialisation du reconnaisseur avec le modèle ---
# Le deuxième argument est le taux d'échantillonnage. 
# Il doit correspondre à celui de votre fichier audio (44100 Hz dans notre cas).
rec = KaldiRecognizer(model, 44100)

# --- Lecture du fichier audio ---
try:
    wf = wave.open(AUDIO_FILE, "rb")
except FileNotFoundError:
    print(f"Erreur : Le fichier '{AUDIO_FILE}' n'a pas été trouvé.")
    exit(1)

print("Fichier audio chargé. Reconnaissance en cours...")

# --- Boucle de reconnaissance ---
# On lit le fichier par blocs et on les passe au reconnaisseur
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    rec.AcceptWaveform(data)

# --- Récupération du résultat final ---
result_json = rec.FinalResult()
result_dict = json.loads(result_json)
recognized_text = result_dict.get('text', '')

if recognized_text:
    print(f"Texte reconnu : '{recognized_text}'")
else:
    print("Aucun texte n'a pu être reconnu.")

wf.close()

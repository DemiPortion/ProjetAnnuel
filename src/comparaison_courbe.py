# comparaison_courbe.py

import numpy as np
import matplotlib.pyplot as plt

# Charger les fichiers
embedding_ref = np.load("../data/embedding_ref.npy")
embedding_auth = np.load("../data/embedding_auth.npy")

# Tracer les deux vecteurs
plt.figure(figsize=(12, 5))
plt.plot(embedding_ref, label="embedding_ref", color="orange", alpha=0.8)
plt.plot(embedding_auth, label="embedding_auth", color="red", alpha=0.6)
plt.title("Comparaison des deux embeddings (valeurs par dimension)")
plt.xlabel("Dimension")
plt.ylabel("Valeur")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

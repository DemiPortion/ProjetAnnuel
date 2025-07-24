# heatmap_diff.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Charger les fichiers
embedding_ref = np.load("../data/embedding_ref.npy")
embedding_auth = np.load("../data/embedding_auth.npy")

# Calcul de la différence
diff = embedding_ref - embedding_auth

# Heatmap
plt.figure(figsize=(12, 1.5))
sns.heatmap([diff], cmap="coolwarm", center=0, cbar=True)
plt.title("Heatmap des différences par dimension")
plt.xlabel("Dimension")
plt.yticks([], [])
plt.tight_layout()
plt.show()

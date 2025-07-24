import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap

# === CONFIGURATION ===
DOSSIER = "../data"  # Dossier contenant les .npy
REDUCTION = "pca"  # "pca", "tsne" ou "umap"
MODE_3D = True  # True pour visualisation 3D

# === CHARGEMENT DES EMBEDDINGS ===
fichiers = [f for f in os.listdir(DOSSIER) if f.endswith(".npy")]
embeddings = []
labels = []

for fichier in fichiers:
    chemin = os.path.join(DOSSIER, fichier)
    vecteur = np.load(chemin)
    embeddings.append(vecteur)
    labels.append(fichier)

X = np.stack(embeddings)

# === RÉDUCTION DE DIMENSION ===
if REDUCTION == "pca":
    reducer = PCA(n_components=3 if MODE_3D else 2)
elif REDUCTION == "tsne":
    reducer = TSNE(n_components=3 if MODE_3D else 2, perplexity=2, learning_rate='auto', init='pca')
elif REDUCTION == "umap":
    reducer = umap.UMAP(n_components=3 if MODE_3D else 2)
else:
    raise ValueError("Méthode de réduction non supportée")

X_reduit = reducer.fit_transform(X)

# === VISUALISATION ===
if MODE_3D:
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_reduit[:, 0], X_reduit[:, 1], X_reduit[:, 2])
    for i, label in enumerate(labels):
        ax.text(X_reduit[i, 0], X_reduit[i, 1], X_reduit[i, 2], label, fontsize=7)
    ax.set_title(f"Visualisation 3D des embeddings ({REDUCTION.upper()})")
else:
    plt.figure(figsize=(10, 7))
    plt.scatter(X_reduit[:, 0], X_reduit[:, 1])
    for i, label in enumerate(labels):
        plt.text(X_reduit[i, 0], X_reduit[i, 1], label, fontsize=8)
    plt.title(f"Visualisation 2D des embeddings ({REDUCTION.upper()})")
    plt.grid(True)

plt.tight_layout()
plt.show()

"""
Approche DECOUPLEE : on entraine le pooling differentiable separement,
avec son propre objectif (reconstruction), puis on FIGE ses poids et
on construit des graphes avec ce pooling entraine. Ces graphes sont
ensuite donnes au MEME GCN que celui utilise avec Quickshift (61%),
sans aucun changement d'architecture.

Ainsi, la seule variable qui change par rapport a l'experience
Quickshift est la METHODE DE CONSTRUCTION DU GRAPHE elle-meme.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from torch_geometric.data import Data

import sys
sys.path.append('.')
from utils.differentiable_graph import (
    DifferentiableGraphPooling,
    image_to_features,
    build_soft_graph,
)


# ---------------------------------------------------------------
# 1) ENTRAINEMENT DU POOLING SEUL (objectif de reconstruction)
# ---------------------------------------------------------------

class PoolingWithReconstruction(nn.Module):
    """
    Pooling differentiable + decodeur simple pour pouvoir
    entrainer le pooling SANS la loss de classification.

    Objectif : chaque pixel doit pouvoir etre approximativement
    reconstruit a partir des features du cluster auquel il appartient.
    Un bon decoupage de l'image en clusters minimise cette erreur
    de reconstruction (clusters = zones d'intensite homogene).
    """

    def __init__(self, n_clusters=30, feature_dim=9, temperature=10.0):
        super().__init__()
        self.pooling = DifferentiableGraphPooling(
            n_clusters=n_clusters, feature_dim=feature_dim, temperature=temperature
        )
        self.n_clusters = n_clusters

    def forward(self, features):
        """features : (n_pixels, feature_dim)"""
        assignment = self.pooling(features)  # (n_pixels, n_clusters)
        cluster_features, cluster_weight = build_soft_graph(
            features, assignment, self.n_clusters
        )

        # Reconstruction : chaque pixel reprend la feature
        # ponderee des clusters auxquels il appartient
        reconstruction = assignment @ cluster_features  # (n_pixels, feature_dim)

        return reconstruction, assignment, cluster_features, cluster_weight


def train_pooling_only(X, n_epochs=15, n_clusters=30, n_samples=500):
    """
    Entraine le module de pooling SEUL sur un objectif de
    reconstruction, sans aucune notion de classification.
    """
    print("Entrainement du pooling differentiable (reconstruction)...")
    model = PoolingWithReconstruction(n_clusters=n_clusters)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    X_subset = X[:n_samples]

    for epoch in range(n_epochs):
        total_loss = 0
        for img in X_subset:
            features = image_to_features(img)
            optimizer.zero_grad()

            reconstruction, assignment, _, _ = model(features)
            loss = F.mse_loss(reconstruction, features)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epochs} | Reconstruction loss: {total_loss/n_samples:.5f}")

    return model.pooling  # on ne garde que le module de pooling


# ---------------------------------------------------------------
# 2) CONSTRUCTION DU GRAPHE AVEC LE POOLING FIGE
# ---------------------------------------------------------------

def image_to_graph_differentiable(image, pooling_module, k=4):
    """
    Construit un objet PyG Data a partir d'une image, en utilisant
    le pooling differentiable DEJA ENTRAINE ET FIGE.

    Compatible avec n'importe quel GCN/GraphSAGE/GAT deja ecrit,
    exactement comme image_to_graph_superpixels().
    """
    with torch.no_grad():  # le pooling est fige, pas de gradient ici
        features = image_to_features(image)             # (784, feature_dim)
        assignment = pooling_module(features)             # (784, n_clusters)
        cluster_features, cluster_weight = build_soft_graph(
            features, assignment, pooling_module.n_clusters
        )

        # On ne garde que les clusters non-vides (poids significatif)
        active = cluster_weight > 1.0
        active_idx = torch.where(active)[0]

        if len(active_idx) < 2:
            # Securite : si trop peu de clusters actifs, on en garde au moins 2
            active_idx = torch.topk(cluster_weight, min(2, len(cluster_weight))).indices

        x = cluster_features[active_idx]  # (n_active, 9)

        # k plus proches voisins ENTRE clusters actifs
        # (sur leur position : colonnes 4=row_norm, 5=col_norm)
        positions = x[:, 4:6]
        n = positions.shape[0]
        k_eff = min(k, n - 1)

        dist = torch.cdist(positions, positions)
        edge_index = []
        for i in range(n):
            nearest = torch.topk(-dist[i], k_eff + 1).indices  # +1 car inclut soi-meme
            for j in nearest:
                if i != j.item():
                    edge_index.append([i, j.item()])
                    edge_index.append([j.item(), i])

        if len(edge_index) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()

    return Data(x=x, edge_index=edge_index)


# ---------------------------------------------------------------
# 3) VISUALISATION DU RESULTAT APRES ENTRAINEMENT DU POOLING
# ---------------------------------------------------------------

def visualize_trained_pooling(X, y, pooling_module, H=28, W=28):
    indices = [np.where(y == digit)[0][0] for digit in range(10)]
    fig, axes = plt.subplots(2, 10, figsize=(22, 5))

    for col, idx in enumerate(indices):
        image = X[idx]
        digit = y[idx]

        with torch.no_grad():
            features = image_to_features(image, H, W)
            assignment = pooling_module(features)
            hard = assignment.argmax(dim=1).numpy().reshape(H, W)
            _, cluster_weight = build_soft_graph(features, assignment, pooling_module.n_clusters)

        axes[0, col].imshow(image.reshape(H, W), cmap='gray')
        axes[0, col].set_title(f'{digit}')
        axes[0, col].axis('off')

        axes[1, col].imshow(hard, cmap='tab20')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig("trained_pooling_clusters.png", dpi=150)
    print("[OK] Sauvegarde dans trained_pooling_clusters.png")


if __name__ == "__main__":
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # 1) Entraine le pooling SEUL (reconstruction), sans toucher au GCN
    trained_pooling = train_pooling_only(X, n_epochs=15, n_clusters=30, n_samples=500)

    # Fige le pooling : plus aucun gradient ne le traversera desormais
    for p in trained_pooling.parameters():
        p.requires_grad = False

    # 2) Visualise les clusters APRES entrainement (reconstruction)
    visualize_trained_pooling(X, y, trained_pooling)

    # 3) Test rapide de construction de graphe
    g = image_to_graph_differentiable(X[0], trained_pooling)
    print(f"\nGraphe produit pour le chiffre {y[0]} :")
    print(f"Noeuds : {g.num_nodes}")
    print(f"Aretes : {g.num_edges}")
    print(f"Features shape : {g.x.shape}")

    # Sauvegarde le pooling entraine pour reutilisation
    torch.save(trained_pooling.state_dict(), "trained_pooling.pt")
    print("[OK] Pooling entraine sauvegarde dans trained_pooling.pt")
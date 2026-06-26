"""
Entrainement end-to-end : pooling differentiable + GCN sur MNIST.

Contrairement a gcn.py (qui utilise un graph_builder fixe, non
differentiable, applique AVANT l'entrainement), ici le pooling fait
partie du modele lui-meme : le gradient de la loss de classification
remonte jusqu'aux centres de clusters et les fait evoluer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.metrics import f1_score, classification_report

import sys
sys.path.append('.')
from utils.differentiable_graph import (
    DifferentiableGraphPooling,
    image_to_features,
    build_soft_graph,
)


class DifferentiableGCN(nn.Module):
    """
    Modele complet : pooling differentiable -> graphe de clusters -> GCN.
    Le pooling est un sous-module entraine en meme temps que le GCN.
    """

    def __init__(self, n_clusters=30, feature_dim=3, temperature=15.0):
        super().__init__()
        self.n_clusters = n_clusters

        self.pooling = DifferentiableGraphPooling(
            n_clusters=n_clusters,
            feature_dim=feature_dim,
            temperature=temperature,
        )

        self.conv1 = GCNConv(feature_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.lin = nn.Linear(32, 10)

        # Graphe complet entre clusters (calcule une seule fois)
        idx = torch.combinations(torch.arange(n_clusters), r=2).t()
        self.register_buffer(
            "edge_index", torch.cat([idx, idx.flip(0)], dim=1)
        )

    def forward_single(self, image):
        """Traite UNE image (784,) -> logits (10,)"""
        features = image_to_features(image)              # (784, 3)
        assignment = self.pooling(features)               # (784, n_clusters)
        cluster_features, cluster_weight = build_soft_graph(
            features, assignment, self.n_clusters
        )                                                  # (n_clusters, 3)

        x = self.conv1(cluster_features, self.edge_index)
        x = F.relu(x)
        x = self.conv2(x, self.edge_index)
        x = F.relu(x)

        # Pooling global pondere par le poids de chaque cluster
        # (un cluster vide ne doit pas compter dans la moyenne)
        weight = cluster_weight.unsqueeze(1)               # (n_clusters, 1)
        x = (x * weight).sum(dim=0) / (weight.sum() + 1e-8)

        return x  # (32,)

    def forward(self, image_batch):
        """image_batch : (batch_size, 784) -> logits (batch_size, 10)"""
        outputs = [self.forward_single(img) for img in image_batch]
        x = torch.stack(outputs)       # (batch_size, 32)
        x = self.lin(x)                # (batch_size, 10)
        return x


def main():
    # 1) Charger MNIST
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Subset raisonnable : ce modele est plus lent (boucle Python par image)
    N_TRAIN = 1500
    N_TEST = 500
    X_train = torch.tensor(X[:N_TRAIN], dtype=torch.float32)
    y_train = torch.tensor(y[:N_TRAIN], dtype=torch.long)
    X_test = torch.tensor(X[N_TRAIN:N_TRAIN + N_TEST], dtype=torch.float32)
    y_test = torch.tensor(y[N_TRAIN:N_TRAIN + N_TEST], dtype=torch.long)

    # 2) Modele
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    model = DifferentiableGCN(n_clusters=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    # 3) Entrainement par mini-batch
    batch_size = 32
    n_batches = len(X_train) // batch_size

    print("\nEntrainement...")
    start = time.time()
    for epoch in range(30):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0

        for b in range(n_batches):
            idx = perm[b * batch_size: (b + 1) * batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1:3d}/30 | Loss: {total_loss/n_batches:.4f}")

    print(f"Temps d'entrainement : {time.time()-start:.1f}s")

    # 4) Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            batch_x = X_test[i:i+batch_size]
            batch_y = y_test[i:i+batch_size]
            out = model(batch_x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print(f"\n{'='*30}")
    print(f"Accuracy Differentiable GCN : {accuracy:.2f}%")
    print(f"F1 Score Differentiable GCN : {f1:.2f}%")
    print(f"{'='*30}")
    print(classification_report(all_labels, all_preds))

    # 5) Sauvegarde du modele pour visualiser les clusters APRES entrainement
    torch.save(model.state_dict(), "differentiable_gcn.pt")
    print("[OK] Modele sauvegarde dans differentiable_gcn.pt")


if __name__ == "__main__":
    main()
"""
Fine-tuning GAT + pooling differentiable, version GPU (bigfoot/A100).

Differences avec la version locale :
- tout est deplace sur GPU via .to(device)
- dataset plus large (profite de la VRAM disponible)
- plus d'epochs (l'A100 calcule beaucoup plus vite que le CPU local)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
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
from utils.decoupled_pooling import train_pooling_only


class FineTunedGATPooling(nn.Module):
    """
    Pooling differentiable (initialise depuis un pre-entrainement)
    + GAT, tous deux entraines ENSEMBLE sur la classification finale.
    """

    def __init__(self, pretrained_pooling, n_clusters=30, feature_dim=9, k=4):
        super().__init__()
        self.n_clusters = n_clusters
        self.k = k
        self.pooling = pretrained_pooling

        self.conv1 = GATConv(feature_dim, 32, heads=8, concat=True)
        self.conv2 = GATConv(256, 32, heads=8, concat=True)
        self.conv3 = GATConv(256, 32, heads=8, concat=False)
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 10)

    def build_graph_single(self, image):
        features = image_to_features(image)
        assignment = self.pooling(features)
        cluster_features, cluster_weight = build_soft_graph(
            features, assignment, self.n_clusters
        )

        positions = cluster_features[:, 4:6]
        n = positions.shape[0]
        k_eff = min(self.k, n - 1)

        dist = torch.cdist(positions, positions)
        edge_index = []
        for i in range(n):
            nearest = torch.topk(-dist[i], k_eff + 1).indices
            for j in nearest:
                if i != j.item():
                    edge_index.append([i, j.item()])
                    edge_index.append([j.item(), i])

        edge_index = torch.tensor(edge_index, dtype=torch.long, device=image.device).t()
        return cluster_features, edge_index, cluster_weight

    def forward_single(self, image):
        x, edge_index, weight = self.build_graph_single(image)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        w = weight.unsqueeze(1)
        x_mean = (x * w).sum(dim=0) / (w.sum() + 1e-8)
        x_max = x.max(dim=0)[0]
        x = torch.cat([x_mean, x_max])

        return x

    def forward(self, image_batch):
        outputs = [self.forward_single(img) for img in image_batch]
        x = torch.stack(outputs)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    if device.type == 'cuda':
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    print("\nChargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # 1) Pre-entrainement du pooling (reconstruction) -- sur CPU,
    #    rapide de toute facon (boucle simple, peu d'epochs)
    print("\n--- Phase 1 : pre-entrainement du pooling (reconstruction) ---")
    pooling = train_pooling_only(X, n_epochs=15, n_clusters=30, n_samples=1000)
    pooling = pooling.to(device)
    # 2) Fine-tuning : pooling + GAT entraines ENSEMBLE, sur GPU
    print("\n--- Phase 2 : fine-tuning pooling + GAT (classification) sur GPU ---")

    # Dataset plus large, on a la VRAM pour
    N_TRAIN, N_TEST = 8000, 2000
    X_train = torch.tensor(X[:N_TRAIN], dtype=torch.float32, device=device)
    y_train = torch.tensor(y[:N_TRAIN], dtype=torch.long, device=device)
    X_test = torch.tensor(X[N_TRAIN:N_TRAIN + N_TEST], dtype=torch.float32, device=device)
    y_test = torch.tensor(y[N_TRAIN:N_TRAIN + N_TEST], dtype=torch.long, device=device)

    model = FineTunedGATPooling(pretrained_pooling=pooling, n_clusters=30, k=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    batch_size = 32
    n_batches = len(X_train) // batch_size

    start = time.time()
    for epoch in range(20):
        model.train()
        perm = torch.randperm(len(X_train))
        total_loss = 0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_x = X_train[idx]
            batch_y = y_train[idx]

            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:2d}/50 | Loss: {total_loss/n_batches:.4f}")

    print(f"\nTemps d'entrainement : {time.time()-start:.1f}s")

    # 3) Evaluation
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
    print(f"Accuracy GAT fine-tune (GPU) : {accuracy:.2f}%")
    print(f"F1 Score GAT fine-tune (GPU) : {f1:.2f}%")
    print(f"{'='*30}")
    print(classification_report(all_labels, all_preds))

    torch.save(model.state_dict(), "gat_finetuned_gpu.pt")
    print("[OK] Modele sauvegarde dans gat_finetuned_gpu.pt")


if __name__ == "__main__":
    main()

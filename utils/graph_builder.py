from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data
import networkx as nx
from torch_geometric.utils import to_networkx

# --- CHARGEMENT MNIST ---
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
X, y = mnist.data / 255.0, mnist.target.astype(int)

# --- CONSTRUCTION DU GRAPHE ---
def image_to_graph(image):
    H, W = 28, 28
    image = image.reshape(H, W)
    x = torch.tensor(image, dtype=torch.float).view(-1, 1)  # (784, 1)

    edge_index = []
    for row in range(H):
        for col in range(W):
            i = row * W + col
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < H and 0 <= c < W:
                        j = r * W + c
                        edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return Data(x=x, edge_index=edge_index)
def image_to_graph_small(image):
    H, W = 8, 8
    image = image.reshape(H, W)
    x = torch.tensor(image, dtype=torch.float).view(-1, 1)

    edge_index = []
    for row in range(H):
        for col in range(W):
            i = row * W + col
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = row + dr, col + dc
                    if 0 <= r < H and 0 <= c < W:
                        j = r * W + c
                        edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    return Data(x=x, edge_index=edge_index)
# --- VISUALISATION ---
# fig, axes = plt.subplots(2, 9, figsize=(18, 5))

# for i in range(9):
#     # Ligne 1 : image originale
#     axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
#     axes[0, i].set_title(f'Label: {y[i]}')
#     axes[0, i].axis('off')

#     # Ligne 2 : graphe correspondant
#     graph = image_to_graph(X[i])
#     G = to_networkx(graph, to_undirected=True)

#     # Position des nœuds = position des pixels
#     pos = {node: (node % 28, -node // 28) for node in G.nodes()}

#     # Couleur des nœuds = intensité du pixel
#     node_colors = [float(graph.x[node]) for node in G.nodes()]

#     nx.draw(G, pos=pos,
#             node_size=50,
#             node_color=node_colors,
#             cmap='gray',
#             edge_color='lightblue',
#             width=0.1,
#             ax=axes[1, i],
#             with_labels=False)
#     axes[1, i].set_title(f'Graphe {y[i]}')

# plt.tight_layout()
# plt.savefig("mnist_graphs.png")
# print("[OK] Sauvegardé dans mnist_graphs.png")
# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")
# --- VISUALISATION ---
fig, axes = plt.subplots(3, 9, figsize=(18, 8))

for i in range(9):
    # Ligne 1 : image originale 28x28
    axes[0, i].imshow(X[i].reshape(28, 28), cmap='gray')
    axes[0, i].set_title(f'Label: {y[i]}')
    axes[0, i].axis('off')

    # Ligne 2 : graphe complet 28x28
    graph = image_to_graph(X[i])
    G = to_networkx(graph, to_undirected=True)
    pos = {node: (node % 28, -node // 28) for node in G.nodes()}
    node_colors = [float(graph.x[node]) for node in G.nodes()]
    nx.draw(G, pos=pos,
            node_size=10,
            node_color=node_colors,
            cmap='gray',
            edge_color='lightblue',
            width=0.1,
            ax=axes[1, i],
            with_labels=False)
    axes[1, i].set_title(f'Graphe 28x28')

    # Ligne 3 : zoom 8x8 pour voir les arêtes
    image_8x8 = X[i].reshape(28, 28)[:8, :8].flatten()
    graph_small = image_to_graph_small(image_8x8)
    G_small = to_networkx(graph_small, to_undirected=True)
    pos_small = {node: (node % 8, -node // 8) for node in G_small.nodes()}
    node_colors_small = [float(graph_small.x[node]) for node in G_small.nodes()]
    nx.draw(G_small, pos=pos_small,
            node_size=200,
            node_color=node_colors_small,
            cmap='gray',
            edge_color='red',
            width=1.5,
            ax=axes[2, i],
            with_labels=False)
    axes[2, i].set_title(f'Zoom 8x8')

plt.tight_layout()
plt.savefig("mnist_graphs.png")
print("[OK] Sauvegardé dans mnist_graphs.png")
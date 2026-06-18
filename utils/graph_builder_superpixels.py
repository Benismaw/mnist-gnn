from sklearn.datasets import fetch_openml
from skimage.segmentation import slic,quickshift
import numpy as np
import torch
from torch_geometric.data import Data

def image_to_graph_superpixels(image, n_segments=100):
    """
    Transforme une image 28x28 en graphe de superpixels.
    
    n_segments : nombre de superpixels souhaités
    """
    # 1) Reshape en 2D
    img_2d = image.reshape(28, 28)
    # Masque : ne garder que les pixels du chiffre (blancs)
    threshold = 0.1
    foreground_mask = img_2d > threshold
    # 2) SLIC → carte des superpixels
    # chaque pixel reçoit un label (0 à n_segments-1)
    # segments = slic(
    #     img_2d,
    #     n_segments=n_segments,
    #     compactness=0.05,  # 0=forme libre, 1=carré strict
    #     channel_axis=None # image grayscale, pas RGB
    # )
    img_rgb = np.stack([img_2d]*3, axis=-1)
    segments = quickshift(
        img_rgb,
        kernel_size=2,
        max_dist=4,
        ratio=0.3
    )
    # segments.shape = (28, 28)
    # segments[i,j] = id du superpixel du pixel (i,j)

    n_nodes = segments.max() + 1  # nombre réel de superpixels


    # Ne garde que les superpixels qui touchent le foreground
    valid_ids = []
    for seg_id in range(n_nodes):
        mask = (segments == seg_id)
        if mask.sum() == 0:
            continue
        # Garde seulement si le superpixel est majoritairement foreground
        if foreground_mask[mask].mean() > 0.5:
            valid_ids.append(seg_id)

    # Remapping des IDs (0, 1, 2... sans trous)

    id_map = {old_id: new_id for new_id, old_id in enumerate(valid_ids)}
    n_nodes = len(valid_ids)

    # 3) Features uniquement pour les nœuds valides
    node_features = np.zeros((n_nodes, 9))
    for old_id in valid_ids:
        new_id=id_map[old_id]
        mask = (segments == old_id)          

        if mask.sum() == 0:
                continue
        rows, cols = np.where(mask)
        pixels = img_2d[mask]

        node_features[new_id, 0] = pixels.mean()           # intensité moyenne
        node_features[new_id, 1] = pixels.std()            # variance
        node_features[new_id, 2] = pixels.max()            # max
        node_features[new_id, 3] = pixels.min()            # min
        node_features[new_id, 4] = rows.mean() / 28        # position y
        node_features[new_id, 5] = cols.mean() / 28        # position x
        node_features[new_id, 6] = mask.sum() / 784        # taille
        node_features[new_id, 7] = (rows.max()-rows.min()) / 28  # hauteur région
        node_features[new_id, 8] = (cols.max()-cols.min()) / 28  # largeur région

    node_features = np.nan_to_num(node_features, nan=0.0)
    x = torch.tensor(node_features, dtype=torch.float)   # (n_nodes, 3)

    # 4) Construction des arêtes
    # deux superpixels sont voisins s'ils se touchent dans l'image
    edges = set()
    for row in range(28):
        for col in range(28):
            current = segments[row, col]
            if current not in id_map:   
                continue
            # regarde les 4 voisins directs
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                r, c = row+dr, col+dc
                if 0 <= r < 28 and 0 <= c < 28:
                    neighbor = segments[r, c]
                    if neighbor not in id_map:
                        continue
                    if current != neighbor:
                        u, v = id_map[current], id_map[neighbor]
                        edges.add((u, v))
                        edges.add((v, u))

    edge_list = list(edges)
    if len(edge_list) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    

    return Data(x=x, edge_index=edge_index)

import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
from skimage.segmentation import mark_boundaries

def visualize_superpixel_graph(image, graph, segments):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1) Image originale
    axes[0].imshow(image.reshape(28, 28), cmap='gray')
    axes[0].set_title('Image originale')
    axes[0].axis('off')

    # 2) Superpixels
    axes[1].imshow(mark_boundaries(image.reshape(28, 28), segments), cmap='gray')
    axes[1].set_title(f'Superpixels ({segments.max()+1})')
    axes[1].axis('off')

    # 3) Graphe superposé
    G = to_networkx(graph, to_undirected=True)
 
    pos = {i: (graph.x[i][5].item() * 28, -graph.x[i][4].item() * 28)
       for i in G.nodes()}
    node_colors = [graph.x[i][0].item() for i in G.nodes()]

    axes[2].imshow(image.reshape(28, 28), cmap='gray')
    nx.draw(G, pos=pos,
            node_size=100,
            node_color=node_colors,
            cmap='gray',
            edge_color='red',
            width=1.5,
            ax=axes[2],
            with_labels=False)
    axes[2].set_title('Graphe superpixels')
    axes[2].set_xlim(0, 28)
    axes[2].set_ylim(-28, 0)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("superpixel_graph.png")
    print("[OK] Sauvegardé dans superpixel_graph.png")

# Test
# if __name__ == "__main__":
#     mnist = fetch_openml('mnist_784', version=1,
#                          as_frame=False, parser='liac-arff')
#     X, y = mnist.data / 255.0, mnist.target.astype(int)

#     # Graphe + segments pour visualisation
#     img = X[5]
#     img_2d = img.reshape(28, 28)
#     segments = slic(img_2d, n_segments=100,
#                     compactness=0.1, channel_axis=None)
#     graph = image_to_graph_superpixels(img)

#     print(f"Noeuds : {graph.num_nodes}")
#     print(f"Aretes : {graph.num_edges}")
#     print(f"Features : {graph.x.shape}")

#     visualize_superpixel_graph(img, graph, segments)
if __name__ == "__main__":
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Un exemple de chaque chiffre 0-9
    indices = [np.where(y == digit)[0][0] for digit in range(10)]

    fig, axes = plt.subplots(2, 10, figsize=(20, 6))

    for col, idx in enumerate(indices):
        digit=col
        img = X[idx]
        img_2d = img.reshape(28, 28)

        # Superpixels
        segments = slic(img_2d, n_segments=75,
                        compactness=0.05, channel_axis=None)
        graph = image_to_graph_superpixels(img)
        G = to_networkx(graph, to_undirected=True)

        # Ligne 1 : image originale
        axes[0, col].imshow(img_2d, cmap='gray')
        axes[0, col].set_title(f'Label: {digit}')
        axes[0, col].axis('off')

        # Ligne 2 : graphe superpixels superposé
        pos = {i: (graph.x[i][5].item() * 28, -graph.x[i][4].item() * 28)
               for i in G.nodes()}
        node_colors = [graph.x[i][0].item() for i in G.nodes()]

        axes[1, col].imshow(img_2d, cmap='gray')
        nx.draw(G, pos=pos,
                node_size=50,
                node_color=node_colors,
                cmap='gray',
                edge_color='red',
                width=1.0,
                ax=axes[1, col],
                with_labels=False)
        axes[1, col].set_xlim(0, 28)
        axes[1, col].set_ylim(-28, 0)
        axes[1, col].set_title(f'Graphe {digit}')
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig("superpixel_all_digits.png")
    print("[OK] Sauvegardé dans superpixel_all_digits.png")
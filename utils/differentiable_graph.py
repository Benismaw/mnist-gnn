# """
# Differentiable graph construction pour MNIST.

# Idée : remplacer Quickshift (non differentiable, fige a l'avance)
# par un clustering "doux" appris, ou chaque pixel a une probabilite
# d'appartenance a chaque cluster plutot qu'une assignation dure.

# Cela permet au gradient de la loss de classification de remonter
# jusqu'aux centres de clusters et de faire evoluer la construction
# du graphe pendant l'entrainement.
# """

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.datasets import fetch_openml


# class DifferentiableGraphPooling(nn.Module):
#     """
#     Apprend a regrouper les pixels en clusters (superpixels) de facon
#     differentiable, via une assignation souple (soft assignment).

#     Remplace l'equivalent de :
#         cluster = argmin(distance(pixel, centres))   <- dur, non derivable
#     par :
#         proba = softmax(-distance(pixel, centres))   <- doux, derivable
#     """

#     def __init__(self, n_clusters=30, feature_dim=9, temperature=10.0):
#         super().__init__()
#         self.n_clusters = n_clusters
#         self.temperature = temperature

#         # Centres de clusters appris.
#         # Initialises sur une grille reguliere pour les positions
#         # (colonnes 4 et 5), le reste des features part de 0.5.
#         grid_size = int(np.ceil(np.sqrt(n_clusters)))
#         rows = torch.linspace(0.1, 0.9, grid_size)
#         cols = torch.linspace(0.1, 0.9, grid_size)
#         grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
#         init_pos = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)
#         init_pos = init_pos[:n_clusters]

#         init_centers = torch.full((n_clusters, feature_dim), 0.5)
#         if feature_dim >= 6:
#             init_centers[:, 4:6] = init_pos   # row_norm, col_norm
#         else:
#             init_centers[:, 1:3] = init_pos   # fallback ancien format (3 features)

#         self.cluster_centers = nn.Parameter(init_centers)

#     def forward(self, features):
#         """
#         features : (n_pixels, feature_dim) -> [intensite, row_norm, col_norm]

#         Retourne :
#             assignment : (n_pixels, n_clusters) probabilites souples
#         """
#         # Distance de chaque pixel a chaque centre de cluster
#         dist = torch.cdist(features, self.cluster_centers)  # (n_pixels, n_clusters)

#         # Assignation souple : softmax sur la distance negative
#         # temperature haute -> assignation plus "dure" (proche de argmin)
#         # temperature basse  -> assignation plus "floue"
#         assignment = F.softmax(-dist * self.temperature, dim=1)

#         return assignment


# def image_to_features(image, H=28, W=28):
#     """
#     Transforme une image MNIST (784,) en features par pixel.

#     9 features par pixel, pour rester comparable a celles utilisees
#     avec Quickshift dans graph_builder_superpixels.py :
#         0: intensite du pixel
#         1: intensite moyenne locale (fenetre 3x3)      ~ "variance proxy"
#         2: intensite max locale (fenetre 3x3)
#         3: intensite min locale (fenetre 3x3)
#         4: row normalisee
#         5: col normalisee
#         6: gradient horizontal local (Sobel-like, approx. differentiable)
#         7: gradient vertical local
#         8: magnitude du gradient (contour local)

#     Toutes ces features sont calculees par des operations TENSORIELLES
#     (convolutions / pooling), donc differentiables de bout en bout.
#     """
#     img_2d = torch.tensor(image, dtype=torch.float32).reshape(1, 1, H, W)

#     # Contexte local via average / max pooling 3x3 (stride 1, padding 1)
#     avg_local = F.avg_pool2d(img_2d, kernel_size=3, stride=1, padding=1)
#     max_local = F.max_pool2d(img_2d, kernel_size=3, stride=1, padding=1)
#     min_local = -F.max_pool2d(-img_2d, kernel_size=3, stride=1, padding=1)

#     # Gradients locaux (Sobel simplifie), calcules par convolution
#     sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
#     sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
#     grad_x = F.conv2d(img_2d, sobel_x, padding=1)
#     grad_y = F.conv2d(img_2d, sobel_y, padding=1)
#     grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

#     intensity = img_2d.flatten()
#     avg_local = avg_local.flatten()
#     max_local = max_local.flatten()
#     min_local = min_local.flatten()
#     grad_x = grad_x.flatten()
#     grad_y = grad_y.flatten()
#     grad_mag = grad_mag.flatten()

#     rows, cols = torch.meshgrid(
#         torch.arange(H, dtype=torch.float32),
#         torch.arange(W, dtype=torch.float32),
#         indexing='ij'
#     )
#     rows_norm = rows.flatten() / H
#     cols_norm = cols.flatten() / W

#     features = torch.stack([
#         intensity, avg_local, max_local, min_local,
#         rows_norm, cols_norm,
#         grad_x, grad_y, grad_mag,
#     ], dim=1)  # (784, 9)

#     return features


# def build_soft_graph(features, assignment, n_clusters):
#     """
#     Construit les features de chaque cluster a partir de
#     l'assignation souple. Tout reste differentiable.

#     features   : (n_pixels, feature_dim)
#     assignment : (n_pixels, n_clusters)
#     """
#     # Poids total de chaque cluster (somme des probabilites)
#     cluster_weight = assignment.sum(dim=0).unsqueeze(1) + 1e-8  # (n_clusters, 1)

#     # Feature moyenne ponderee de chaque cluster
#     cluster_features = (assignment.t() @ features) / cluster_weight
#     # (n_clusters, feature_dim)

#     return cluster_features, cluster_weight.squeeze(1)


# def visualize_soft_clusters(image, pooling_module, H=28, W=28):
#     """
#     Visualise a quel cluster chaque pixel appartient majoritairement
#     (juste pour INSPECTER ; le modele lui continue de travailler
#     avec les probabilites souples, pas cette version durcie).
#     """
#     features = image_to_features(image, H, W)
#     assignment = pooling_module(features)  # (784, n_clusters)

#     # Pour visualiser seulement : on prend le cluster le plus probable
#     hard_assignment = assignment.argmax(dim=1).detach().numpy()
#     cluster_map = hard_assignment.reshape(H, W)

#     cluster_features, cluster_weight = build_soft_graph(
#         features, assignment, pooling_module.n_clusters
#     )

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(image.reshape(H, W), cmap='gray')
#     axes[0].set_title("Image originale")
#     axes[0].axis('off')

#     axes[1].imshow(cluster_map, cmap='tab20')
#     axes[1].set_title(f"Clusters (soft, {pooling_module.n_clusters} clusters)")
#     axes[1].axis('off')

#     # Clusters dont le poids total est significatif (pas vides)
#     active = (cluster_weight > 1.0).detach().numpy()
#     centers_row = pooling_module.cluster_centers[:, 4].detach().numpy() * H
#     centers_col = pooling_module.cluster_centers[:, 5].detach().numpy() * W

#     axes[2].imshow(image.reshape(H, W), cmap='gray')
#     axes[2].scatter(
#         centers_col[active], centers_row[active],
#         c='red', s=cluster_weight[active].detach().numpy() * 2,
#         alpha=0.6
#     )
#     axes[2].set_title("Centres de clusters appris (taille = poids)")
#     axes[2].axis('off')

#     plt.tight_layout()
#     plt.savefig("soft_clusters.png", dpi=150)
#     print("[OK] Sauvegarde dans soft_clusters.png")


# def visualize_all_digits(X, y, pooling_module, H=28, W=28):
#     """
#     Visualise les clusters appris pour un exemple de chaque chiffre 0-9.
#     """
#     indices = [np.where(y == digit)[0][0] for digit in range(10)]

#     fig, axes = plt.subplots(3, 10, figsize=(22, 7))

#     for col, idx in enumerate(indices):
#         image = X[idx]
#         digit = y[idx]

#         features = image_to_features(image, H, W)
#         assignment = pooling_module(features)

#         hard_assignment = assignment.argmax(dim=1).detach().numpy()
#         cluster_map = hard_assignment.reshape(H, W)

#         cluster_features, cluster_weight = build_soft_graph(
#             features, assignment, pooling_module.n_clusters
#         )

#         # Ligne 1 : image originale
#         axes[0, col].imshow(image.reshape(H, W), cmap='gray')
#         axes[0, col].set_title(f'Label: {digit}')
#         axes[0, col].axis('off')

#         # Ligne 2 : carte des clusters (argmax, pour visualiser)
#         axes[1, col].imshow(cluster_map, cmap='tab20')
#         axes[1, col].axis('off')

#         # Ligne 3 : centres de clusters actifs sur l'image
#         active = (cluster_weight > 1.0).detach().numpy()
#         centers_row = pooling_module.cluster_centers[:, 4].detach().numpy() * H
#         centers_col = pooling_module.cluster_centers[:, 5].detach().numpy() * W

#         axes[2, col].imshow(image.reshape(H, W), cmap='gray')
#         axes[2, col].scatter(
#             centers_col[active], centers_row[active],
#             c='red', s=cluster_weight[active].detach().numpy() * 2,
#             alpha=0.6
#         )
#         axes[2, col].axis('off')

#     axes[0, 0].set_ylabel("Image", fontsize=12)
#     axes[1, 0].set_ylabel("Clusters", fontsize=12)
#     axes[2, 0].set_ylabel("Centres appris", fontsize=12)

#     plt.tight_layout()
#     plt.savefig("soft_clusters_all_digits.png", dpi=150)
#     print("[OK] Sauvegarde dans soft_clusters_all_digits.png")


# if __name__ == "__main__":
#     print("Chargement MNIST...")
#     mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
#     X, y = mnist.data / 255.0, mnist.target.astype(int)

#     # Test sur tous les chiffres, AVANT tout entrainement
#     # -> les clusters doivent etre repartis en grille, sans lien
#     #    particulier avec la forme du chiffre (c'est attendu !)
#     pooling = DifferentiableGraphPooling(n_clusters=30, temperature=15.0)

#     visualize_all_digits(X, y, pooling)

#     # Verification rapide sur un seul exemple
#     features = image_to_features(X[0])
#     assignment = pooling(features)
#     print(f"Features shape    : {features.shape}")
#     print(f"Assignment shape  : {assignment.shape}")
#     print(f"Somme par pixel (doit etre ~1.0) : {assignment[0].sum().item():.4f}")
"""
Differentiable graph construction pour MNIST.

Idée : remplacer Quickshift (non differentiable, fige a l'avance)
par un clustering "doux" appris, ou chaque pixel a une probabilite
d'appartenance a chaque cluster plutot qu'une assignation dure.

Cela permet au gradient de la loss de classification de remonter
jusqu'aux centres de clusters et de faire evoluer la construction
du graphe pendant l'entrainement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


class DifferentiableGraphPooling(nn.Module):
    """
    Apprend a regrouper les pixels en clusters (superpixels) de facon
    differentiable, via une assignation souple (soft assignment).

    Remplace l'equivalent de :
        cluster = argmin(distance(pixel, centres))   <- dur, non derivable
    par :
        proba = softmax(-distance(pixel, centres))   <- doux, derivable
    """

    def __init__(self, n_clusters=30, feature_dim=9, temperature=10.0):
        super().__init__()
        self.n_clusters = n_clusters
        self.temperature = temperature

        # Centres de clusters appris.
        # Initialises sur une grille reguliere pour les positions
        # (colonnes 4 et 5), le reste des features part de 0.5.
        grid_size = int(np.ceil(np.sqrt(n_clusters)))
        rows = torch.linspace(0.1, 0.9, grid_size)
        cols = torch.linspace(0.1, 0.9, grid_size)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')
        init_pos = torch.stack([grid_r.flatten(), grid_c.flatten()], dim=1)
        init_pos = init_pos[:n_clusters]

        init_centers = torch.full((n_clusters, feature_dim), 0.5)
        if feature_dim >= 6:
            init_centers[:, 4:6] = init_pos   # row_norm, col_norm
        else:
            init_centers[:, 1:3] = init_pos   # fallback ancien format (3 features)

        self.cluster_centers = nn.Parameter(init_centers)

    def forward(self, features):
        """
        features : (n_pixels, feature_dim) -> [intensite, row_norm, col_norm]

        Retourne :
            assignment : (n_pixels, n_clusters) probabilites souples
        """
        # Distance de chaque pixel a chaque centre de cluster
        dist = torch.cdist(features, self.cluster_centers)  # (n_pixels, n_clusters)

        # Assignation souple : softmax sur la distance negative
        # temperature haute -> assignation plus "dure" (proche de argmin)
        # temperature basse  -> assignation plus "floue"
        assignment = F.softmax(-dist * self.temperature, dim=1)

        return assignment


def image_to_features(image, H=28, W=28):
    """
    Transforme une image MNIST (784,) en features par pixel.

    9 features par pixel, pour rester comparable a celles utilisees
    avec Quickshift dans graph_builder_superpixels.py :
        0: intensite du pixel
        1: intensite moyenne locale (fenetre 3x3)      ~ "variance proxy"
        2: intensite max locale (fenetre 3x3)
        3: intensite min locale (fenetre 3x3)
        4: row normalisee
        5: col normalisee
        6: gradient horizontal local (Sobel-like, approx. differentiable)
        7: gradient vertical local
        8: magnitude du gradient (contour local)

    Toutes ces features sont calculees par des operations TENSORIELLES
    (convolutions / pooling), donc differentiables de bout en bout.
    """
    if isinstance(image, np.ndarray):
        img_2d = torch.from_numpy(image).to(torch.float32).reshape(1, 1, H, W)
    else:
        img_2d = image.clone().detach().to(torch.float32).reshape(1, 1, H, W)

    device = img_2d.device
    # Contexte local via average / max pooling 3x3 (stride 1, padding 1)
    avg_local = F.avg_pool2d(img_2d, kernel_size=3, stride=1, padding=1)
    max_local = F.max_pool2d(img_2d, kernel_size=3, stride=1, padding=1)
    min_local = -F.max_pool2d(-img_2d, kernel_size=3, stride=1, padding=1)

    # Gradients locaux (Sobel simplifie), calcules par convolution
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).to(device)
    sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).to(device)
    grad_x = F.conv2d(img_2d, sobel_x, padding=1)
    grad_y = F.conv2d(img_2d, sobel_y, padding=1)
    grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

    intensity = img_2d.flatten()
    avg_local = avg_local.flatten()
    max_local = max_local.flatten()
    min_local = min_local.flatten()
    grad_x = grad_x.flatten()
    grad_y = grad_y.flatten()
    grad_mag = grad_mag.flatten()

    rows, cols = torch.meshgrid(
    torch.arange(H, dtype=torch.float32, device=device),
    torch.arange(W, dtype=torch.float32, device=device),
    indexing='ij'
)
    rows_norm = rows.flatten() / H
    cols_norm = cols.flatten() / W

    features = torch.stack([
        intensity, avg_local, max_local, min_local,
        rows_norm, cols_norm,
        grad_x, grad_y, grad_mag,
    ], dim=1)  # (784, 9)

    return features


def build_soft_graph(features, assignment, n_clusters):
    """
    Construit les features de chaque cluster a partir de
    l'assignation souple. Tout reste differentiable.

    features   : (n_pixels, feature_dim)
    assignment : (n_pixels, n_clusters)
    """
    # Poids total de chaque cluster (somme des probabilites)
    cluster_weight = assignment.sum(dim=0).unsqueeze(1) + 1e-8  # (n_clusters, 1)

    # Feature moyenne ponderee de chaque cluster
    cluster_features = (assignment.t() @ features) / cluster_weight
    # (n_clusters, feature_dim)

    return cluster_features, cluster_weight.squeeze(1)


def visualize_soft_clusters(image, pooling_module, H=28, W=28):
    """
    Visualise a quel cluster chaque pixel appartient majoritairement
    (juste pour INSPECTER ; le modele lui continue de travailler
    avec les probabilites souples, pas cette version durcie).
    """
    features = image_to_features(image, H, W)
    assignment = pooling_module(features)  # (784, n_clusters)

    # Pour visualiser seulement : on prend le cluster le plus probable
    hard_assignment = assignment.argmax(dim=1).detach().numpy()
    cluster_map = hard_assignment.reshape(H, W)

    cluster_features, cluster_weight = build_soft_graph(
        features, assignment, pooling_module.n_clusters
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image.reshape(H, W), cmap='gray')
    axes[0].set_title("Image originale")
    axes[0].axis('off')

    axes[1].imshow(cluster_map, cmap='tab20')
    axes[1].set_title(f"Clusters (soft, {pooling_module.n_clusters} clusters)")
    axes[1].axis('off')

    # Clusters dont le poids total est significatif (pas vides)
    active = (cluster_weight > 1.0).detach().numpy()
    centers_row = pooling_module.cluster_centers[:, 4].detach().numpy() * H
    centers_col = pooling_module.cluster_centers[:, 5].detach().numpy() * W

    axes[2].imshow(image.reshape(H, W), cmap='gray')
    axes[2].scatter(
        centers_col[active], centers_row[active],
        c='red', s=cluster_weight[active].detach().numpy() * 2,
        alpha=0.6
    )
    axes[2].set_title("Centres de clusters appris (taille = poids)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("soft_clusters.png", dpi=150)
    print("[OK] Sauvegarde dans soft_clusters.png")


def visualize_all_digits(X, y, pooling_module, H=28, W=28):
    """
    Visualise les clusters appris pour un exemple de chaque chiffre 0-9.
    """
    indices = [np.where(y == digit)[0][0] for digit in range(10)]

    fig, axes = plt.subplots(3, 10, figsize=(22, 7))

    for col, idx in enumerate(indices):
        image = X[idx]
        digit = y[idx]

        features = image_to_features(image, H, W)
        assignment = pooling_module(features)

        hard_assignment = assignment.argmax(dim=1).detach().numpy()
        cluster_map = hard_assignment.reshape(H, W)

        cluster_features, cluster_weight = build_soft_graph(
            features, assignment, pooling_module.n_clusters
        )

        # Ligne 1 : image originale
        axes[0, col].imshow(image.reshape(H, W), cmap='gray')
        axes[0, col].set_title(f'Label: {digit}')
        axes[0, col].axis('off')

        # Ligne 2 : carte des clusters (argmax, pour visualiser)
        axes[1, col].imshow(cluster_map, cmap='tab20')
        axes[1, col].axis('off')

        # Ligne 3 : centres de clusters actifs sur l'image
        active = (cluster_weight > 1.0).detach().numpy()
        centers_row = pooling_module.cluster_centers[:, 4].detach().numpy() * H
        centers_col = pooling_module.cluster_centers[:, 5].detach().numpy() * W

        axes[2, col].imshow(image.reshape(H, W), cmap='gray')
        axes[2, col].scatter(
            centers_col[active], centers_row[active],
            c='red', s=cluster_weight[active].detach().numpy() * 2,
            alpha=0.6
        )
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("Image", fontsize=12)
    axes[1, 0].set_ylabel("Clusters", fontsize=12)
    axes[2, 0].set_ylabel("Centres appris", fontsize=12)

    plt.tight_layout()
    plt.savefig("soft_clusters_all_digits.png", dpi=150)
    print("[OK] Sauvegarde dans soft_clusters_all_digits.png")


if __name__ == "__main__":
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # Test sur tous les chiffres, AVANT tout entrainement
    # -> les clusters doivent etre repartis en grille, sans lien
    #    particulier avec la forme du chiffre (c'est attendu !)
    pooling = DifferentiableGraphPooling(n_clusters=30, temperature=15.0)

    visualize_all_digits(X, y, pooling)

    # Verification rapide sur un seul exemple
    features = image_to_features(X[0])
    assignment = pooling(features)
    print(f"Features shape    : {features.shape}")
    print(f"Assignment shape  : {assignment.shape}")
    print(f"Somme par pixel (doit etre ~1.0) : {assignment[0].sum().item():.4f}")
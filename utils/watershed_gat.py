import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
# ==========================================
# 1. COUCHE WATERSHED DIFFÉRENTIABLE (Relaxed)
# ==========================================
class DifferentiableWatershedPooling(nn.Module):
    """
    Transforme une image en superpixels (graphe) de manière end-to-end.
    Utilise un CNN interne pour apprendre les frontières/bassins du Watershed
    et projette les pixels sur 'n_zones' de manière différentiable.
    """
    def __init__(self, n_zones=25):
        super().__init__()
        self.n_zones = n_zones
        # CNN qui apprend à cartographier les bassins versants (Watershed Map)
        self.watershed_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, n_zones, kernel_size=3, padding=1) # Sortie = probabilité d'appartenir à une zone
        )

    def forward(self, x):
        # x shape: [B, 1, 28, 28]
        B, C, H, W = x.size()
        
        # Étape 1 : Obtenir la matrice d'assignation "soft" aux bassins versants (Soft-Watershed)
        # Shape: [B, n_zones, H, W]
        assignment_map = F.softmax(self.watershed_net(x), dim=1)
        
        # Flatten spatial pour les calculs matriciels [B, n_zones, H*W]
        S = assignment_map.view(B, self.n_zones, H * W) 
        # Feature des pixels (intensité + coordonnées normalisées pour garder la topologie)
        pixels = x.view(B, 1, H * W)
        
        # Coordonnées des pixels (X, Y) comme features géométriques
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing='ij')
        grid = torch.stack([grid_x, grid_y]).view(2, H*W).to(x.device).unsqueeze(0).repeat(B, 1, 1) # [B, 2, H*W]
        
        pixel_features = torch.cat([pixels, grid], dim=1) # [B, 3, H*W] (Intensité, X, Y)

        # Étape 2 : Réduction en nœuds de graphes (Pooling)
        # Regroupement des features par zone du Watershed
        # [B, n_zones, H*W] x [B, H*W, 3] -> [B, n_zones, 3]
        node_features = torch.bmm(S, pixel_features.transpose(1, 2)) 
        # Normalisation par la taille de la zone (mass)
        zone_mass = S.sum(dim=2, keepdim=True) + 1e-8
        node_features = node_features / zone_mass

        # Étape 3 : Construction de la matrice d'adjacence du graphe (Différentiable)
        # Deux zones sont connectées si elles partagent des pixels proches spatialement
        # On l'approxime par la corrélation de leurs cartes d'assignation
        adj = torch.bmm(S, S.transpose(1, 2)) # [B, n_zones, n_zones]
        # Retirer l'auto-boucle pour le calcul des arêtes
        adj = adj * (1 - torch.eye(self.n_zones, device=x.device).unsqueeze(0))
        
        return node_features, adj

# ==========================================
# 2. LE MODÈLE GLOBAL : WATERSHED + GAT
# ==========================================
class EndToEndWatershedGAT(nn.Module):
    def __init__(self, n_zones=25):
        super().__init__()
        self.watershed_pooling = DifferentiableWatershedPooling(n_zones=n_zones)
        
        # GAT prend les features de nœuds (dim=3 : intensité, x, y)
        self.conv1 = GATConv(3, 32, heads=4, concat=True)   # out: 128
        self.conv2 = GATConv(128, 32, heads=4, concat=False) # out: 32
        
        self.lin1 = nn.Linear(64, 32)
        self.lin2 = nn.Linear(32, 10)

    def forward(self, images):
        # images: [B, 1, 28, 28]
        B = images.size(0)
        node_feats, adj_matrices = self.watershed_pooling(images)
        
        # Reconstruire un Batch PyTorch Geometric à la volée de manière différentiable
        data_list = []
        for i in range(B):
            x_i = node_feats[i] # [n_zones, 3]
            adj_i = adj_matrices[i]
            
            # On ne garde que les connexions fortes (top-k arêtes pour économiser la mémoire)
            edges_mask = adj_i > torch.topk(adj_i, k=4, dim=-1).values[..., -1:].clone()
            edge_index = edges_mask.nonzero().t().contiguous()
            
            data_list.append(Data(x=x_i, edge_index=edge_index))
            
        # Création du batch pour le GNN
        batch_geo = Batch.from_data_list(data_list).to(images.device)
        
        # Passage dans le GAT
        x = self.conv1(batch_geo.x, batch_geo.edge_index)
        x = F.relu(x)
        x = self.conv2(x, batch_geo.edge_index)
        x = F.relu(x)
        
        # Readout (Pooling global)
        x_mean = global_mean_pool(x, batch_geo.batch)
        x_max = global_max_pool(x, batch_geo.batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # Classification
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
def visualize_all_digits_watershed(X, y, model, device, H=28, W=28):
    """
    Visualise les bassins versants appris et les graphes pour un exemple de chaque chiffre 0-9.
    """
    model.eval()
    # Trouver le premier index pour chaque chiffre de 0 à 9
    indices = [np.where(y == digit)[0][0] for digit in range(10)]

    fig, axes = plt.subplots(3, 10, figsize=(22, 7))

    with torch.no_grad():
        for col, idx in enumerate(indices):
            image_np = X[idx].reshape(H, W)
            digit = y[idx]

            # Préparer le tenseur pour le modèle [1, 1, 28, 28]
            img_tensor = torch.tensor(X[idx], dtype=torch.float32).view(1, 1, H, W).to(device)

            # 1. Extraire la carte d'assignation du module Watershed
            assignment_map = F.softmax(model.watershed_pooling.watershed_net(img_tensor), dim=1)
            # Shape: [n_zones, H, W]
            assignment_map = assignment_map.squeeze(0).cpu().numpy()
            cluster_map = assignment_map.argmax(axis=0)

            # 2. Récupérer les features de nœuds et matrices d'adjacence calculées
            node_feats, adj_matrices = model.watershed_pooling(img_tensor)
            node_feats = node_feats.squeeze(0).cpu().numpy()  # [n_zones, 3]
            adj_i = adj_matrices.squeeze(0)

            # Recréer le masque des arêtes gardées par le top-k=6
            edges_mask = adj_i > torch.topk(adj_i, k=6, dim=-1).values[..., -1:].clone()
            edge_index = edges_mask.nonzero().cpu().numpy()

            # Ligne 1 : Image originale
            axes[0, col].imshow(image_np, cmap='gray')
            axes[0, col].set_title(f'Chiffre: {digit}', fontsize=12, fontweight='bold')
            axes[0, col].axis('off')

            # Ligne 2 : Carte des bassins versants (Segmentation)
            axes[1, col].imshow(cluster_map, cmap='tab20')
            axes[1, col].axis('off')

            # Ligne 3 : Graphe (Nœuds + Arêtes) superposé sur l'image
            axes[2, col].imshow(image_np, cmap='gray', alpha=0.6)
            
            # Les coordonnées (X, Y) dans node_feats sont normalisées entre -1 et 1
            # On les remet à l'échelle des pixels (0 à 27)
            nodes_x = (node_feats[:, 1] + 1) / 2 * (W - 1)
            nodes_y = (node_feats[:, 2] + 1) / 2 * (H - 1)

            # Dessiner les arêtes
            for start_node, end_node in edge_index:
                axes[2, col].plot(
                    [nodes_x[start_node], nodes_x[end_node]],
                    [nodes_y[start_node], nodes_y[end_node]],
                    color='yellow', linestyle='-', linewidth=0.8, alpha=0.5
                )

            # Dessiner les nœuds (la taille dépend de l'intensité moyenne du nœud)
            node_sizes = (node_feats[:, 0] * 50) + 10
            axes[2, col].scatter(nodes_x, nodes_y, c='cyan', s=node_sizes, edgecolors='black', linewidths=0.5, zorder=3)
            axes[2, col].axis('off')

    # Titres des lignes à gauche
    axes[0, 0].text(-10, H/2, "Image", va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)
    axes[1, 0].text(-10, H/2, "Watershed Zones", va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)
    axes[2, 0].text(-10, H/2, "Graphe GAT", va='center', ha='right', fontsize=12, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.savefig("end_to_end_watershed_digits.png", dpi=200, bbox_inches='tight')
    print("[OK] Sauvegarde de la visualisation dans end_to_end_watershed_digits.png")
    plt.show()
# ==========================================
# 3. PIPELINE PRINCIPALE (MAIN)
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Working on device: {device}")

    print("Loading MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data / 255.0, mnist.target.astype(int)
    
    # Dataset de taille intermédiaire pour un entraînement rapide et robuste
    X_train, X_test, y_train, y_test = train_test_split(X[:12000], y[:12000], test_size=2000, random_state=42)

    # Conversion en tenseurs PyTorch directs (format Image: [B, 1, 28, 28])
    X_train_t = torch.tensor(X_train, dtype=torch.float32).view(-1, 1, 28, 28)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).view(-1, 1, 28, 28)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    model = EndToEndWatershedGAT(n_zones=30).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    batch_size = 64
    print("Training End-to-End Model...")
    
    for epoch in range(75): 
        total_loss = 0
        permutation = torch.randperm(X_train_t.size(0))
        
        for i in range(0, X_train_t.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = X_train_t[indices].to(device), y_train_t[indices].to(device)
            
            optimizer.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
        print(f"Epoch {epoch+1:02d}/25 | Loss: {total_loss/X_train_t.size(0):.4f}")

    print("\nEvaluating Model...")
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in range(0, X_test_t.size(0), batch_size):
            batch_x = X_test_t[i:i+batch_size].to(device)
            batch_y = y_test_t[i:i+batch_size]
            
            out = model(batch_x)
            preds = out.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(batch_y.numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print(f"\n{'='*30}")
    print(f"Accuracy End-to-End Watershed GAT : {accuracy:.2f}%")
    print(f"F1 Score End-to-End Watershed GAT : {f1:.2f}%")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds))
    print(f"{'='*30}")
    
    print("\nGénération de la visualisation globale...")
    # On passe le dataset de base X et y, le modèle entraîné, et le device utilisé
    visualize_all_digits_watershed(X, y, model, device)

if __name__ == "__main__":
    main()

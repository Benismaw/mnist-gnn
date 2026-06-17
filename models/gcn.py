import matplotlib.pyplot as plt
import torch 
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch_geometric.nn import global_mean_pool
# import graph builder
import sys 
sys.path.append('.')
from utils.graph_builder_superpixels import image_to_graph_superpixels
from sklearn.metrics import f1_score, classification_report

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(9, 64) # 3 features (intensité,x,y)
        self.conv2 = GCNConv(64, 32) # 64 → 32
        self.lin =torch.nn.Linear(32, 10)  # 32 → 10 classes

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # Global pooling par graphe
        x = global_mean_pool(x, batch)  # clé du batch

        x = self.lin(x)
        return x
# def main():
#     #charger mnist 
#     mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
#     X, y = mnist.data / 255.0, mnist.target.astype(int)
    
#     # Petit subset pour tester (sinon trop long)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X[:2000], y[:2000], test_size=0.2, random_state=42
#     )
#     #construction des graphes:

#     train_graphs = [image_to_graph(X_train[i]) for i in range(len(X_train))]
#     test_graphs  = [image_to_graph(X_test[i])  for i in range(len(X_test))]

#     # Instantiate the model
#     model =GCN()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#     criterion = torch.nn.CrossEntropyLoss()
#     # Train the model 
#     print("Entraînement...")
#     model.train()
#     for epoch in range(10):
#         total_loss = 0
#         for i, graph in enumerate(train_graphs):
#             optimizer.zero_grad()
#             out = model(graph)                          # (10,)
#             label = torch.tensor([y_train[i]], dtype=torch.long)
#             loss = criterion(out.unsqueeze(0), label)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch {epoch+1}/10 | Loss: {total_loss/len(train_graphs):.4f}")
#      # Évaluation
#     model.eval()
#     correct = 0
#     with torch.no_grad():
#         for i, graph in enumerate(test_graphs):
#             out = model(graph)
#             pred = out.argmax().item()
#             if pred == y_test[i]:
#                 correct += 1
#     accuracy = correct / len(test_graphs) * 100
#     print(f"\nAccuracy GCN : {accuracy:.2f}%")
from torch_geometric.loader import DataLoader

def main():
    # 1) Charger MNIST
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    # 2) Subset
    X_sub, y_sub = X[:10000], y[:10000]
    split = int(0.8 * len(X_sub))
    X_train, X_test = X_sub[:split], X_sub[split:]
    y_train, y_test = y_sub[:split], y_sub[split:]

    # 3) Construire les graphes avec labels
    print("Construction des graphes...")
    train_graphs = []
    for i in range(len(X_train)):
        g = image_to_graph_superpixels(X_train[i])
        g.y = torch.tensor([y_train[i]], dtype=torch.long)
        train_graphs.append(g)

    test_graphs = []
    for i in range(len(X_test)):
        g = image_to_graph_superpixels(X_test[i])
        g.y = torch.tensor([y_test[i]], dtype=torch.long)
        test_graphs.append(g)

    # 4) DataLoader — batch de 32 graphes
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=32, shuffle=False)

    # 5) Modèle
    model = GCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = torch.nn.CrossEntropyLoss()

    # 6) Entraînement
    print("Entraînement...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/100 | Loss: {total_loss/len(train_loader):.4f}")



    # 7) Évaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out  = model(batch)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())

    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels) * 100
    f1 = f1_score(all_labels, all_preds, average='macro') * 100

    print(f"\n{'='*30}")
    print(f"Accuracy GCN : {accuracy:.2f}%")
    print(f"F1 Score GCN : {f1:.2f}%")
    print(f"\nRapport détaillé :")
    print(classification_report(all_labels, all_preds))
    print(f"{'='*30}")
if __name__ == "__main__":
    main()
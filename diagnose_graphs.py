"""
Diagnostic : compare les statistiques des graphes produits par
Quickshift vs le pooling differentiable, pour comprendre pourquoi
GAT ne beneficie pas du pooling differentiable autant que GCN.
"""

import torch
import numpy as np
from sklearn.datasets import fetch_openml

import sys
sys.path.append('.')
from utils.graph_builder_superpixels import image_to_graph_superpixels
from utils.decoupled_pooling import image_to_graph_differentiable, train_pooling_only


def diagnose(X, y, pooling, n_samples=200):
    print(f"Diagnostic sur {n_samples} images...\n")

    stats_quickshift = {"nodes": [], "edges": []}
    stats_diff = {"nodes": [], "edges": []}

    for i in range(n_samples):
        # Quickshift
        g_qs = image_to_graph_superpixels(X[i])
        stats_quickshift["nodes"].append(g_qs.num_nodes)
        stats_quickshift["edges"].append(g_qs.num_edges)

        # Pooling differentiable
        g_diff = image_to_graph_differentiable(X[i], pooling)
        stats_diff["nodes"].append(g_diff.num_nodes)
        stats_diff["edges"].append(g_diff.num_edges)

    print("="*50)
    print("QUICKSHIFT")
    print(f"  Noeuds : moyenne={np.mean(stats_quickshift['nodes']):.1f}, "
          f"min={np.min(stats_quickshift['nodes'])}, "
          f"max={np.max(stats_quickshift['nodes'])}")
    print(f"  Aretes : moyenne={np.mean(stats_quickshift['edges']):.1f}, "
          f"min={np.min(stats_quickshift['edges'])}, "
          f"max={np.max(stats_quickshift['edges'])}")
    print(f"  Aretes/noeud : {np.mean(stats_quickshift['edges'])/np.mean(stats_quickshift['nodes']):.2f}")

    print("\nPOOLING DIFFERENTIABLE")
    print(f"  Noeuds : moyenne={np.mean(stats_diff['nodes']):.1f}, "
          f"min={np.min(stats_diff['nodes'])}, "
          f"max={np.max(stats_diff['nodes'])}")
    print(f"  Aretes : moyenne={np.mean(stats_diff['edges']):.1f}, "
          f"min={np.min(stats_diff['edges'])}, "
          f"max={np.max(stats_diff['edges'])}")
    print(f"  Aretes/noeud : {np.mean(stats_diff['edges'])/np.mean(stats_diff['nodes']):.2f}")
    print("="*50)


if __name__ == "__main__":
    print("Chargement MNIST...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X, y = mnist.data / 255.0, mnist.target.astype(int)

    print("Entrainement du pooling (rapide, pour le diagnostic)...")
    pooling = train_pooling_only(X, n_epochs=15, n_clusters=30, n_samples=500)
    for p in pooling.parameters():
        p.requires_grad = False

    diagnose(X, y, pooling, n_samples=200)

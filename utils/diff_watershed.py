# diff_watershed.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml


class DifferentiableWatershed(nn.Module):

    def __init__(self, n_segments=75):
        super().__init__()

        self.n_segments = n_segments

        self.segment_prototypes = nn.Parameter(
            torch.randn(n_segments, 1)
        )

        self.fg_prob_params = nn.Parameter(
            torch.randn(1, 1, 1, 1)
        )

    def forward(self, features):

        img = features[:, 0].view(28, 28)
        img = img.unsqueeze(0).unsqueeze(0)

        fg_prob = torch.sigmoid(
            F.conv2d(img, self.fg_prob_params)
        ).view(-1)

        pixels = features.view(-1, 1)

        dist = torch.cdist(
            pixels,
            self.segment_prototypes,
            p=2
        )

        assignment = F.softmax(
            -dist / 0.5,
            dim=-1
        )

        assignment = assignment * fg_prob.unsqueeze(-1)

        assignment = assignment / (
            assignment.sum(dim=1, keepdim=True) + 1e-8
        )

        return assignment


def build_watershed_graph(image, n_segments=75):

    image = image.reshape(28, 28)

    features = torch.tensor(
        image,
        dtype=torch.float32
    ).view(-1, 1)

    watershed = DifferentiableWatershed(
        n_segments=n_segments
    )

    assignment = watershed(features)

    h = w = 28

    yy, xx = torch.meshgrid(
        torch.arange(h),
        torch.arange(w),
        indexing="ij"
    )

    xx = xx.float().reshape(-1)
    yy = yy.float().reshape(-1)

    pixels = features

    segment_intensity = assignment.t() @ pixels

    weights = assignment.sum(
        dim=0,
        keepdim=True
    ).T + 1e-8

    segment_x = (
        assignment.t()
        @ xx.unsqueeze(1)
    ) / weights

    segment_y = (
        assignment.t()
        @ yy.unsqueeze(1)
    ) / weights

    segment_features = torch.cat(
        [
            segment_intensity,
            segment_x / 27.0,
            segment_y / 27.0
        ],
        dim=1
    )

    assignment_map = assignment.view(
        h,
        w,
        n_segments
    )

    edges = set()

    for i in range(h):
        for j in range(w):

            seg = assignment_map[i, j].argmax().item()

            for di, dj in [(0, 1), (1, 0)]:

                ni = i + di
                nj = j + dj

                if ni < h and nj < w:

                    seg2 = (
                        assignment_map[ni, nj]
                        .argmax()
                        .item()
                    )

                    if seg != seg2:
                        edges.add(
                            (seg, seg2)
                        )

    if len(edges) == 0:
        edges.add((0, 0))

    edge_index = torch.tensor(
        list(edges),
        dtype=torch.long
    ).t().contiguous()

    return Data(
        x=segment_features,
        edge_index=edge_index
    )


def save_all_digits_graphs(
    graphs,
    digits,
    output_file="all_digits_graphs.png"
):

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(18, 8)
    )

    for ax, graph, digit in zip(
        axes.flatten(),
        graphs,
        digits
    ):

        pos = graph.x[:, 1:3].detach().cpu().numpy()

        for edge in graph.edge_index.t():

            i, j = edge.tolist()

            ax.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                linewidth=0.5
            )

        ax.scatter(
            pos[:, 0],
            pos[:, 1],
            s=15
        )

        ax.set_title(
            f"Digit {digit}"
        )

        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_aspect("equal")

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"Saved: {output_file}"
    )


if __name__ == "__main__":

    print("Loading MNIST...")

    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False
    )

    X = mnist.data / 255.0
    y = mnist.target.astype(int)

    graphs = []
    digits = []

    for digit in range(10):

        idx = np.where(
            y == digit
        )[0][0]

        graph = build_watershed_graph(
            X[idx],
            n_segments=75
        )

        print(
            f"Digit {digit}: "
            f"{graph.num_nodes} nodes, "
            f"{graph.num_edges} edges"
        )

        graphs.append(graph)
        digits.append(digit)

    save_all_digits_graphs(
        graphs,
        digits,
        "all_digits_graphs.png"
    )
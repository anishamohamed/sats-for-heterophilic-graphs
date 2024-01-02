# -*- coding: utf-8 -*-

# Original code from: https://github.com/BorgwardtLab/SAT/blob/main/sat/gnn_layers.py
# Copyright (c) 2022, Machine Learning and Computational Biology Lab. All rights reserved.
# Licensed under BSD 3-Clause License

import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils


GNN_TYPES = [
    "graph",
    "graphsage",
    "gcn",
    "gin",
    "gine",
    "pna",
    "pna2",
    "pna3",
    "mpnn",
    "pna4",
    "rwgnn",
    "khopgnn",
]

EDGE_GNN_TYPES = ["gine", "gcn", "pna", "pna2", "pna3", "mpnn", "pna4"]

NON_DETERMINISTIC_GNN_TYPES = ["gcn", "pna", "pna2", "pna3", "pna4", "dirpna"]


def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    edge_dim = kwargs.get("edge_dim", None)
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "dirgraphsage":
        alpha = kwargs.get("alpha", 0.5)
        learn_alpha = kwargs.get("learn_alpha", True)
        return DirSageConv(embed_dim, embed_dim, alpha, learn_alpha)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gin":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINConv(mlp, train_eps=True)
    elif gnn_type == "gine":
        mlp = mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim),
        )
        return gnn.GINEConv(mlp, train_eps=True, edge_dim=edge_dim)
    elif gnn_type == "pna":
        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna2":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "dirpna":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        alpha = kwargs.get("alpha", 0.5)
        learn_alpha = kwargs.get("learn_alpha", True)
        layer = DirPNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            alpha=alpha,
            learn_alpha=learn_alpha
        )
        return layer
    elif gnn_type == "pna3":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
            edge_dim=edge_dim,
        )
        return layer
    elif gnn_type == "pna4":
        aggregators = ["mean", "sum", "max"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=8,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer

    elif gnn_type == "mpnn":
        aggregators = ["sum"]
        scalers = ["identity"]
        deg = kwargs.get("deg", None)
        layer = gnn.PNAConv(
            embed_dim,
            embed_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
        )
        return layer
    else:
        raise ValueError("Not implemented!")

class DirSageConv(nn.Module):
    def __init__(self, input_dim, output_dim, alpha, learn_alpha):
        super(DirSageConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = gnn.SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = gnn.SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )

class DirPNAConv(nn.Module):
    def __init__(self,
                input_dim,
                output_dim,
                aggregators,
                scalers,
                deg,
                edge_dim,
                alpha,
                learn_alpha,
                ):
        super(DirPNAConv, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv_src_to_dst = gnn.PNAConv(
            input_dim,
            output_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
            flow="source_to_target",
        )
        self.conv_dst_to_src = gnn.PNAConv(
            input_dim,
            output_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            towers=4,
            pre_layers=1,
            post_layers=1,
            divide_input=True,
            edge_dim=edge_dim,
            flow="target_to_source",
        )
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.alpha = nn.Parameter(torch.ones(1) * alpha, requires_grad=learn_alpha)

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )

class GCNConv(gnn.MessagePassing):
    def __init__(self, embed_dim, edge_dim):
        super(GCNConv, self).__init__(aggr="add")

        self.linear = nn.Linear(embed_dim, embed_dim)
        self.root_emb = nn.Embedding(1, embed_dim)

        # edge_attr is two dimensional after augment_edge transformation
        self.edge_encoder = nn.Linear(edge_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.edge_encoder(edge_attr)

        row, col = edge_index

        # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = utils.degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(
            edge_index, x=x, edge_attr=edge_embedding, norm=norm
        ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

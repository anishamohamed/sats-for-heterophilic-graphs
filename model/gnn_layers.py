# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch_geometric
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric import utils
from torch_geometric.nn import GraphNorm
from torch.nn import LayerNorm, Linear, ReLU
from torch_geometric.nn import DeepGCNLayer, GENConv

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
    "gcn_bn",
    "gcn_gn",
]

EDGE_GNN_TYPES = ["gine", "gcn", "pna", "pna2", "pna3", "mpnn", "pna4"]


def get_simple_gnn_layer(gnn_type, embed_dim, **kwargs):
    edge_dim = kwargs.get("edge_dim", None)
    if gnn_type == "graph":
        return gnn.GraphConv(embed_dim, embed_dim)
    elif gnn_type == "graphsage":
        return gnn.SAGEConv(embed_dim, embed_dim)
    elif gnn_type == "gcn":
        if edge_dim is None:
            return gnn.GCNConv(embed_dim, embed_dim)
        else:
            return GCNConv(embed_dim, edge_dim)
    elif gnn_type == "gcn_bn":
        if edge_dim is None:
            return GCNConv_Batch_Norm(embed_dim)
        else:
            raise ValueError("Not implemented!")
    elif gnn_type == "gcn_gn":
        if edge_dim is None:
            return GCNConv_Graph_Norm(embed_dim)
        else:
            raise ValueError("Not implemented!")
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
    

# # Batch-Normalized GCNConv layer -- UNTESTED!
# class GCNConv_Batch_Norm(gnn.MessagePassing):
#     def __init__(self, embed_dim, edge_dim):
#         super(GCNConv_Batch_Norm, self).__init__(aggr="add")

#         self.linear = nn.Linear(embed_dim, embed_dim)
#         self.batch_norm = nn.BatchNorm1d(embed_dim)  # Add BatchNorm layer
#         self.root_emb = nn.Embedding(1, embed_dim)


#     def forward(self, x, edge_index):
#         x = self.linear(x)
#         x = self.batch_norm(x)  # Apply BatchNorm

#         row, col = edge_index

#         # edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
#         deg = utils.degree(row, x.size(0), dtype=x.dtype) + 1
#         deg_inv_sqrt = deg.pow(-0.5)
#         deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

#         norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

#         return self.propagate(
#             edge_index, x=x, norm=norm
#         ) + F.relu(x + self.root_emb.weight) * 1.0 / deg.view(-1, 1)

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * F.relu(x_j)

#     def update(self, aggr_out):
#         return aggr_out



# Batch-Normalized GCNConv layer
class GCNConv_Batch_Norm(gnn.MessagePassing):
    def __init__(self, embed_dim):
        super(GCNConv_Batch_Norm, self).__init__(aggr="add")
        self.Batch_Norm1D_GCN = nn.BatchNorm1d(embed_dim)  # Add BatchNorm layer
        self.GCN_CONV = gnn.GCNConv(embed_dim, embed_dim)

    def forward(self, x, edge_index):
        x = self.Batch_Norm1D_GCN(x)
        x = self.GCN_CONV(x, edge_index)
        return x


# Deeper GCN Model
class DeeperGCN(torch.nn.Module):
    def __init__(self, embed_dim, num_layers):
        super().__init__()

        self.node_encoder = Linear(embed_dim, embed_dim)
        self.edge_encoder = Linear(embed_dim, embed_dim)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(embed_dim, embed_dim, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(embed_dim, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(embed_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)

class GCNConv_Graph_Norm(gnn.MessagePassing):
    def __init__(self, embed_dim):
        super(GCNConv_Graph_Norm, self).__init__(aggr="add")

        self.Graph_Norm = GraphNorm(embed_dim)
        self.GCN_CONV = gnn.GCNConv(embed_dim, embed_dim)

    def forward(self, x, edge_index):
        x = self.GCN_CONV(x, edge_index)
        x = self.Graph_Norm(x)
        return x

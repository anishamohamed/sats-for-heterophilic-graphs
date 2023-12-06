import torch
from torch import nn
from torch_scatter import scatter

from gnns import get_simple_gnn_layer, EDGE_GNN_TYPES

class StructureExtractor(nn.Module):
    r""" K-subtree structure extractor. Computes the structure-aware node embeddings using the
    k-hop subtree centered around each node.

    Args:
    ----------
    embed_dim (int):        the embeding dimension
    gnn_type (str):         GNN type to use in structure extractor. (gcn, gin, pna, etc)
    num_layers (int):       number of GNN layers
    batch_norm (bool):      apply batch normalization or not
    concat (bool):          whether to concatenate the initial edge features
    """

    def __init__(self, embed_dim, gnn_type="gcn", num_layers=3,
                 batch_norm=True, concat=True, use_gates=True, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.concat = concat
        self.gnn_type = gnn_type
        layers = []
        for _ in range(num_layers):
            layers.append(get_simple_gnn_layer(gnn_type, embed_dim, **kwargs))
        self.gcn = nn.ModuleList(layers)

        self.relu = nn.ReLU()
        self.batch_norm = batch_norm
        inner_dim = (num_layers + 1) * embed_dim if concat else embed_dim

        if batch_norm:
            self.bn = nn.BatchNorm1d(inner_dim)

        self.use_gates = use_gates

        self.out_proj = nn.Linear(inner_dim, embed_dim)

    def forward(self, x, edge_index, edge_attr=None,
            subgraph_indicator_index=None, agg="sum"):
        x_cat = [x]
        for gcn_layer in self.gcn:
            if self.gnn_type in EDGE_GNN_TYPES:
                if edge_attr is None:
                    x_ = self.relu(gcn_layer(x, edge_index))
                    if self.use_gates:
                        tau = self.relu(gcn_layer(x, edge_index))
                else:
                    x_ = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
                    if self.use_gates:
                        tau = self.relu(gcn_layer(x, edge_index, edge_attr=edge_attr))
            else:
                x_ = self.relu(gcn_layer(x, edge_index))
                if self.use_gates:
                    tau = self.relu(gcn_layer(x, edge_index))

            if self.use_gates:
                gg = torch.tanh(scatter((torch.abs(tau[edge_index[0]] - tau[edge_index[1]]) ** 2).squeeze(-1),
                                 edge_index[0], 0,dim_size=tau.size(0), reduce='mean'))
                x = (1-gg) * x + gg * x_
            else:
                x = x_

            if self.concat:
                x_cat.append(x)

        if self.concat:
            x = torch.cat(x_cat, dim=-1)

        if self.num_layers > 0 and self.batch_norm:
            x = self.bn(x)

        x = self.out_proj(x)
        return x
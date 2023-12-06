from torch import nn
import torch_geometric.nn as gnn

from graphTransformerEncoder import TransformerEncoderLayer, GraphTransformerEncoder

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_heads=8,
                 dim_feedforward=512, dropout=0.0, num_layers=4,
                 batch_norm=False, use_abs_pe=False, abs_pe_dim=0,
                 gnn_type="graph", use_edge_attr=False, num_edge_features=4,
                 embed_input=True, embed_edges=True, use_global_pool=True,
                 global_pool='mean', use_gates=True, **kwargs):
        super().__init__()

        #prepare node inputs
        self.use_abs_pe = use_abs_pe
        if self.use_abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, hidden_size)
        if embed_input:
            if isinstance(input_size, int):
                self.embedding = nn.Embedding(input_size, hidden_size)
            elif isinstance(input_size, nn.Module):
                self.embedding = input_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(in_features=input_size,
                                       out_features=hidden_size,
                                       bias=False)
            
        # prepare edge inputs
        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get('edge_dim', 32)
            if embed_edges:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(in_features=num_edge_features,
                    out_features=edge_dim, bias=False)
        else:
            kwargs['edge_dim'] = None

        # prepare transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_size, num_heads, dim_feedforward, dropout, batch_norm=batch_norm,
            gnn_type=gnn_type, use_gates=use_gates, **kwargs)
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)

        # prepare pooling
        self.global_pool = global_pool
        if global_pool == 'mean':
            self.pooling = gnn.global_mean_pool
        elif global_pool == 'add':
            self.pooling = gnn.global_add_pool
        self.use_global_pool = use_global_pool
        
        # prepare readout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, output_size))

    def forward(self, data, return_attn=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        node_depth = data.node_depth if hasattr(data, "node_depth") else None

        subgraph_node_index = None
        subgraph_edge_index = None
        subgraph_indicator_index = None
        subgraph_edge_attr = None

        complete_edge_index = data.complete_edge_index if hasattr(data, 'complete_edge_index') else None
        abs_pe = data.abs_pe if hasattr(data, 'abs_pe') else None
        degree = data.degree if hasattr(data, 'degree') else None
        output = self.embedding(x) if node_depth is None else self.embedding(x, node_depth.view(-1,))

        if self.use_abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        output = self.encoder(
            output,
            edge_index,
            complete_edge_index,
            edge_attr=edge_attr,
            degree=degree,
            subgraph_node_index=subgraph_node_index,
            subgraph_edge_index=subgraph_edge_index,
            subgraph_indicator_index=subgraph_indicator_index,
            subgraph_edge_attr=subgraph_edge_attr,
            ptr=data.ptr,
            return_attn=return_attn
        )
        
        # readout step
        if self.use_global_pool:
            output = self.pooling(output, data.batch)
        return self.classifier(output)
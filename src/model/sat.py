# -*- coding: utf-8 -*-

# Original code from: https://github.com/BorgwardtLab/SAT/blob/main/sat/models.py
# Copyright (c) 2022, Machine Learning and Computational Biology Lab. All rights reserved.
# Licensed under BSD 3-Clause License

import torch
from torch import nn
import torch_geometric.nn as gnn
from model.sat_block import TransformerEncoderLayer
from einops import repeat
from typing import Union, Callable, Optional


class GraphTransformerEncoder(nn.TransformerEncoder):
    def forward(
        self,
        x,
        edge_index,
        complete_edge_index,
        subgraph_node_index=None,
        subgraph_edge_index=None,
        subgraph_edge_attr=None,
        subgraph_indicator_index=None,
        edge_attr=None,
        degree=None,
        ptr=None,
        return_attn=False,
    ):
        output = x
        xs = [output]

        for mod in self.layers:
            output = mod(
                output,
                edge_index,
                complete_edge_index,
                edge_attr=edge_attr,
                degree=degree,
                subgraph_node_index=subgraph_node_index,
                subgraph_edge_index=subgraph_edge_index,
                subgraph_indicator_index=subgraph_indicator_index,
                subgraph_edge_attr=subgraph_edge_attr,
                ptr=ptr,
                return_attn=return_attn,
            )
            xs.append(output)

        if self.norm is not None:
            xs[-1] = self.norm(xs[-1])
        return xs


class GraphTransformer(nn.Module):
    def __init__(
        self,
        in_size: Union[int, Callable],
        num_class: int,
        d_model: int,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
        num_layers: int = 4,
        batch_norm: bool = False,
        abs_pe: bool = False,
        abs_pe_dim: int = 0,
        gnn_type: str = "graph",
        se: str = "gnn",
        use_edge_attr: bool = False,
        num_edge_features: int = 4,
        in_embed: bool = True,
        edge_embed: bool = True,
        use_global_pool: bool = True,
        max_seq_len=None,
        global_pool="mean",
        gradient_gating_p: float = 0.0,
        jk: Optional[str] = None,
        **kwargs
    ):
        super().__init__()

        self.abs_pe = abs_pe
        self.abs_pe_dim = abs_pe_dim
        if abs_pe and abs_pe_dim > 0:
            self.embedding_abs_pe = nn.Linear(abs_pe_dim, d_model)
        if in_embed:
            if isinstance(in_size, int):
                self.embedding = nn.Embedding(in_size, d_model)
            elif isinstance(in_size, nn.Module):
                self.embedding = in_size
            else:
                raise ValueError("Not implemented!")
        else:
            self.embedding = nn.Linear(
                in_features=in_size, out_features=d_model, bias=False
            )

        self.use_edge_attr = use_edge_attr
        if use_edge_attr:
            edge_dim = kwargs.get("edge_dim", 32)
            if edge_embed:
                if isinstance(num_edge_features, int):
                    self.embedding_edge = nn.Embedding(num_edge_features, edge_dim)
                else:
                    raise ValueError("Not implemented!")
            else:
                self.embedding_edge = nn.Linear(
                    in_features=num_edge_features, out_features=edge_dim, bias=False
                )
        else:
            kwargs["edge_dim"] = None

        self.se = se
        encoder_layer = TransformerEncoderLayer(
            d_model,
            num_heads,
            dim_feedforward,
            dropout,
            batch_norm=batch_norm,
            gnn_type=gnn_type,
            se=se,
            gradient_gating_p=gradient_gating_p,
            **kwargs
        )
        self.encoder = GraphTransformerEncoder(encoder_layer, num_layers)
        self.global_pool = global_pool
        if global_pool == "mean":
            self.pooling = gnn.global_mean_pool
        elif global_pool == "add":
            self.pooling = gnn.global_add_pool
        elif global_pool == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, d_model))
            self.pooling = None
        self.use_global_pool = use_global_pool

        self.jk = (
            gnn.models.JumpingKnowledge(mode=jk, channels=d_model, num_layers=3)
            if jk is not None
            else lambda xs: xs[-1]
        )

        self.max_seq_len = max_seq_len
        if max_seq_len is None:
            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(True),
                nn.Linear(d_model, num_class),
            )
        else:
            self.classifier = nn.ModuleList()
            for i in range(max_seq_len):
                self.classifier.append(nn.Linear(d_model, num_class))

    def forward(self, data, return_embedding=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Strucure embeddings
        node_depth = data.node_depth if hasattr(data, "node_depth") else None
        if self.se == "khopgnn":
            subgraph_node_index = data.subgraph_node_idx
            subgraph_edge_index = data.subgraph_edge_index
            subgraph_indicator_index = data.subgraph_indicator
            subgraph_edge_attr = (
                data.subgraph_edge_attr if hasattr(data, "subgraph_edge_attr") else None
            )
        else:
            subgraph_node_index = None
            subgraph_edge_index = None
            subgraph_indicator_index = None
            subgraph_edge_attr = None

        complete_edge_index = (
            data.complete_edge_index if hasattr(data, "complete_edge_index") else None
        )
        abs_pe = data.abs_pe if hasattr(data, "abs_pe") else None
        degree = data.degree if hasattr(data, "degree") else None
        output = (
            self.embedding(x)
            if node_depth is None
            else self.embedding(
                x,
                node_depth.view(
                    -1,
                ),
            )
        )  # output: (batch_size, d_model)

        if self.abs_pe and abs_pe is not None:
            abs_pe = self.embedding_abs_pe(abs_pe)
            output = output + abs_pe  # absolute embeddings are summed
        if self.use_edge_attr and edge_attr is not None:
            edge_attr = self.embedding_edge(edge_attr)
            if subgraph_edge_attr is not None:
                subgraph_edge_attr = self.embedding_edge(subgraph_edge_attr)
        else:
            edge_attr = None
            subgraph_edge_attr = None

        # if graph classification: collapse node embeddings into single graph embedding
        if self.global_pool == "cls" and self.use_global_pool:
            bsz = len(data.ptr) - 1  # batchsize
            if complete_edge_index is not None:
                new_index = torch.vstack(
                    (
                        torch.arange(data.num_nodes).to(data.batch),
                        data.batch + data.num_nodes,
                    )
                )
                new_index2 = torch.vstack((new_index[1], new_index[0]))
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(
                    data.batch
                )
                new_index3 = torch.vstack((idx_tmp, idx_tmp))
                complete_edge_index = torch.cat(
                    (complete_edge_index, new_index, new_index2, new_index3), dim=-1
                )
            if subgraph_node_index is not None:
                idx_tmp = torch.arange(data.num_nodes, data.num_nodes + bsz).to(
                    data.batch
                )
                subgraph_node_index = torch.hstack((subgraph_node_index, idx_tmp))
                subgraph_indicator_index = torch.hstack(
                    (subgraph_indicator_index, idx_tmp)
                )
            degree = None
            cls_tokens = repeat(
                self.cls_token, "() d -> b d", b=bsz
            )  # cls_tokens: (batch_size, d_model)
            output = torch.cat((output, cls_tokens))  # output: (2*batch_size, d_model)

        xs = self.encoder(
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
            return_attn=False,
        )
        output = self.jk(xs)
        node_embedding = output

        # readout step
        if self.use_global_pool:
            if self.global_pool == "cls":
                output = output[-bsz:]
            else:
                output = self.pooling(
                    output, data.batch
                )  # output: (# graphs in batch, h)
        if self.max_seq_len is not None:
            pred_list = []
            for i in range(self.max_seq_len):
                pred_list.append(self.classifier[i](output))
            return pred_list

        if  return_embedding:
            return self.classifier(output), node_embedding
        else:
            return self.classifier(output)

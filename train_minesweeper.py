import os
import argparse
from functools import partial

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader # torch_geometric.data
from torch_geometric import datasets
import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from model.sat import GraphTransformer
from data import GraphDataset
from hetero_net import HeterophilousGraphWrapper
from model.position_encoding import POSENCODINGS
from model.gnn_layers import GNN_TYPES
import wandb

import optuna
from optuna.samplers import TPESampler

os.environ["WANDB_API_KEY"] = "89dd0dde666ab90e0366c4fec54fe1a4f785f3ef" # matin

def load_args():
    parser = argparse.ArgumentParser(
        description="Structure-Aware Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset", type=str, default="Minesweeper", help="name of dataset")
    parser.add_argument("--data-path", type=str, default="datasets", help="path to dataset folder")
    parser.add_argument("--num-heads", type=int, default=8, help="number of heads")
    parser.add_argument("--num-layers", type=int, default=4, help="number of layers")
    parser.add_argument(
        "--dim-hidden", type=int, default=64, help="hidden dimension of Transformer"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument(
        "--abs-pe",
        type=str,
        default=None,
        choices=POSENCODINGS.keys(),
        help="which absolute PE to use?",
    )
    parser.add_argument(
        "--abs-pe-dim", type=int, default=20, help="dimension for absolute PE"
    )
    parser.add_argument(
        "--warmup", type=int, default=5000, help="number of iterations for warmup"
    )
    parser.add_argument(
        "--layer-norm", action="store_true", help="use layer norm instead of batch norm"
    )
    parser.add_argument(
        "--use-edge-attr", action="store_true", help="use edge features"
    )
    parser.add_argument(
        "--edge-dim", type=int, default=32, help="edge features hidden dim"
    )
    parser.add_argument(
        "--gnn-type",
        type=str,
        default="gcn",
        choices=GNN_TYPES,
        help="GNN structure extractor type",
    )
    parser.add_argument(
        "--k-hop",
        type=int,
        default=2,
        help="Number of hops to use when extracting subgraphs around each node",
    )
    parser.add_argument(
        "--se", type=str, default="gnn", help="Extractor type: khopgnn, or gnn"
    )
    parser.add_argument(
        "--gradient-gating-p", type=float, default=.0, help="gradient gating parameter"
    )

    args = vars(parser.parse_args())
    args["batch_norm"] = not args["layer_norm"]
    return args

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args["seed"])

    # Roman-Empire Experiment
    if args["dataset"] == "Minesweeper":
        input_size = 7 # n_tags
        num_classes = 2
        num_edge_features = 0
        data = datasets.HeterophilousGraphDataset(args["data_path"], name="Minesweeper", transform=T.NormalizeFeatures())
        

        lr_scheduler = None # partial(optim.lr_scheduler.StepLR, step_size=100, gamma=0.8)
        dataset = GraphDataset(
            data,
            degree=False,
            k_hop=args["k_hop"],
            se=args["se"],
            use_subgraph_edge_attr=False,
            return_complete_index=False # large dataset, recommended to set as False as in original SAT code
        )
        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])

        abs_pe_encoder = None
        if args["abs_pe"] and args["abs_pe_dim"] > 0:
            abs_pe_method = POSENCODINGS[args["abs_pe"]]
            abs_pe_encoder = abs_pe_method(args["abs_pe_dim"], normalization="sym")
            if abs_pe_encoder is not None:
                abs_pe_encoder.apply_to(dataset)
        else: 
            abs_pe_method = None

        deg = torch.cat(
            [
                degree(data.edge_index[1], num_nodes=data.num_nodes)
                for data in dataset
            ]
        )

        num_runs = data[0].train_mask.size(-1)

        for i in range(num_runs):
            node_projection = nn.Sequential(nn.Linear(input_size, args["dim_hidden"]), nn.Mish(), nn.Linear(args["dim_hidden"], args["dim_hidden"]))
            model = GraphTransformer(
                in_size=node_projection, # in_size=input_size,
                num_class=num_classes,
                d_model=args["dim_hidden"],
                dim_feedforward=2*args["dim_hidden"],
                dropout=args["dropout"],
                num_heads=args["num_heads"],
                num_layers=args["num_layers"],
                batch_norm=args["batch_norm"],
                abs_pe=args["abs_pe"],
                abs_pe_dim=args["abs_pe_dim"],
                gnn_type=args["gnn_type"],
                use_edge_attr=False,
                num_edge_features=num_edge_features,
                edge_dim=args["edge_dim"],
                k_hop=args["k_hop"],
                se=args["se"],
                deg=deg,
                use_global_pool=False, # node classification task
                gradient_gating_p=args["gradient_gating_p"],
            ).to(device)

            wrapper = HeterophilousGraphWrapper(
                model,
                abs_pe=args["abs_pe"],
                learning_rate=args["lr"],
                weight_decay=args["weight_decay"],
                lr_scheduler=lr_scheduler,
                mask=i,
            )

            wandb_logger = WandbLogger(project="Gradient_Debugging_" + args["dataset"], log_model="all")
            wandb_logger.watch(model, log="all")

            trainer = pl.Trainer(
                accelerator=device,
                max_epochs=args["epochs"],
                deterministic=True,
                logger=wandb_logger,
                # callbacks=EarlyStopping(monitor="val/loss", mode="min", patience=200),cc
                check_val_every_n_epoch=1,
                log_every_n_steps=1,
            )

            if device == 'cuda':
                torch.use_deterministic_algorithms(False)

            trainer.fit(wrapper, loader, loader)
            trainer.test(wrapper, loader)
            wandb.finish()

        return trainer.callback_metrics["test/loss"].item()


if __name__ == "__main__":
    run(load_args())





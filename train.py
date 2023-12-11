import os
import argparse
from functools import partial

import torch
from torch import nn, optim
import torch.nn.functional as F

from torch_geometric.loader import DataLoader # torch_geometric.data
from torch_geometric import datasets
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything 
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from model.sat import GraphTransformer
from data import GraphDataset
from net import GraphTransformerWrapper
from model.position_encoding import POSENCODINGS
from model.gnn_layers import GNN_TYPES
import wandb

import optuna
from optuna.samplers import TPESampler

os.environ["WANDB_API_KEY"] = "8f17d7bd011da005a1f4e9a75469497e3236f0b8" # anisha

def load_args():
    parser = argparse.ArgumentParser(
        description="Structure-Aware Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--dataset", type=str, default="ZINC", help="name of dataset")
    parser.add_argument("--data-path", type=str, default="datasets/ZINC", help="path to dataset folder")
    parser.add_argument("--num-heads", type=int, default=8, help="number of heads")
    parser.add_argument("--num-layers", type=int, default=6, help="number of layers")
    parser.add_argument(
        "--dim-hidden", type=int, default=64, help="hidden dimension of Transformer"
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="dropout")
    parser.add_argument("--epochs", type=int, default=2000, help="number of epochs")
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
        default="graphsage",
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
        "--global-pool",
        type=str,
        default="mean",
        choices=["mean", "cls", "add"],
        help="global pooling method",
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

def tune():
    print("Start tuning...")

    def objective(trial: optuna.trial.Trial):
        seed = 42
        dataset = "ZINC"
        data_path = "datasets/ZINC"
        num_heads = 8
        num_layers = trial.suggest_categorical("num_layers", [2, 4, 6])
        dim_hidden = 64
        dropout = 0.2
        epochs = 300
        lr = 1e-3
        weight_decay = 1e-5
        batch_size = 128
        abs_pe = "rw"
        abs_pe_dim = 16
        warmup = 5000
        layer_norm = True
        use_edge_attr = True
        edge_dim = 32
        gnn_type = "gcn" # trial.suggest_categorical("gnn_type", ["graphsage", "gcn"])
        k_hop = trial.suggest_categorical("k_hop", [2, 8, 16, 32])
        global_pool = "mean"
        se = "gnn"
        batch_norm = False
        gradient_gating_p = trial.suggest_categorical("gradient_gating_p", [0., 1., 2.])

        args = {
            "seed": seed,
            "dataset": dataset,
            "data_path": data_path,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "dim_hidden": dim_hidden,
            "dropout": dropout,
            "epochs": epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "abs_pe": abs_pe,
            "abs_pe_dim": abs_pe_dim,
            "warmup": warmup,
            "layer_norm": layer_norm,
            "use_edge_attr": use_edge_attr,
            "edge_dim": edge_dim,
            "gnn_type": gnn_type,
            "k_hop": k_hop,
            "global_pool": global_pool,
            "se": se,
            "batch_norm": batch_norm,
            "gradient_gating_p": gradient_gating_p
        }
         
        print(args)
        return run(args)

    study = optuna.create_study(
        direction="minimize", sampler=TPESampler(constant_liar=True)
    )
    study.optimize(objective, n_trials=40)

    best_params = study.best_params
    print(f"Best accuracy: {study.best_value}")
    print(best_params)

def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_everything(args["seed"])

    # ZINC Experiment
    if args["dataset"] == "ZINC":
        input_size = 28 # n_tags
        num_edge_features = 4
        num_classes = 1
        train_data = datasets.ZINC(args["data_path"], subset=True, split="train")
        val_data =  datasets.ZINC(args["data_path"], subset=True, split="val")
        test_data = datasets.ZINC(args["data_path"], subset=True, split="test")

        # MAE loss
        criterion = nn.L1Loss()

        class ZincLRScheduler(optim.lr_scheduler._LRScheduler):
            def __init__(self, optimizer, lr, warmup):
                self.warmup = warmup
                if warmup is None:
                    self.lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-05, verbose=False
                    )
                else:
                    self.lr_steps = (lr - 1e-6) / warmup
                    self.decay_factor = lr * warmup**0.5
                    super().__init__(optimizer)

            def get_lr(self):
                if self.warmup is None:
                    return [group['lr'] for group in self.optimizer.param_groups]
                else:
                    return [self._get_lr(group['initial_lr'], self.last_epoch) for group in self.optimizer.param_groups]

            def _get_lr(self, initial_lr, s):
                if s < self.warmup:
                    lr = 1e-6 + s * self.lr_steps
                else:
                    lr = self.decay_factor * s**-0.5
                return lr
        
        lr_scheduler = partial(ZincLRScheduler, lr=args["lr"], warmup=args["warmup"])

    train_dataset = GraphDataset(
        train_data,
        degree=True,
        k_hop=args["k_hop"],
        se=args["se"],
        use_subgraph_edge_attr=args["use_edge_attr"],
    )

    val_dataset = GraphDataset(
        val_data,
        degree=True,
        k_hop=args["k_hop"],
        se=args["se"],
        use_subgraph_edge_attr=args["use_edge_attr"],
    )

    test_dataset = GraphDataset(
        test_data,
        degree=True,
        k_hop=args["k_hop"],
        se=args["se"],
        use_subgraph_edge_attr=args["use_edge_attr"],
    )

    train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    abs_pe_encoder = None
    if args["abs_pe"] and args["abs_pe_dim"] > 0:
        abs_pe_method = POSENCODINGS[args["abs_pe"]]
        abs_pe_encoder = abs_pe_method(args["abs_pe_dim"], normalization="sym")
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dataset)
            abs_pe_encoder.apply_to(val_dataset)
            abs_pe_encoder.apply_to(test_dataset)
    else: 
        abs_pe_method = None

    deg = torch.cat(
        [
            degree(data.edge_index[1], num_nodes=data.num_nodes)
            for data in train_dataset
        ]
    )

    model = GraphTransformer(
        in_size=input_size,
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
        use_edge_attr=args["use_edge_attr"],
        num_edge_features=num_edge_features,
        edge_dim=args["edge_dim"],
        k_hop=args["k_hop"],
        se=args["se"],
        deg=deg,
        global_pool=args["global_pool"],
        gradient_gating_p=args["gradient_gating_p"],
    )

    wrapper = GraphTransformerWrapper(
        model,
        abs_pe_method,
        criterion,
        args["lr"],
        args["weight_decay"],
        lr_scheduler,
    )

    trainer = pl.Trainer(
        accelerator=device,
        max_epochs=args["epochs"],
        deterministic=True,
        logger=WandbLogger(
            project="g2_sat_" + args["dataset"],
            config=args,
        ),
        callbacks=EarlyStopping(monitor="val/loss", mode="min", patience=10),
        check_val_every_n_epoch=1,
    )

    if device == 'cuda':
        torch.use_deterministic_algorithms(False)
    trainer.fit(wrapper, train_loader, val_loader)
    trainer.test(wrapper, test_loader)
    wandb.finish()

    return trainer.callback_metrics["test/loss"].item()

if __name__ == "__main__":
    # run(load_args())
    tune()
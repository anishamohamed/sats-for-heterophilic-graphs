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
from empire_wrapper import RomanEmpireWrapper
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
    parser.add_argument("--dataset", type=str, default="roman-empire", help="name of dataset")
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
        epochs = 2000
        lr = 1e-3
        weight_decay = 1e-5
        batch_size = 128
        abs_pe = "rw"
        abs_pe_dim = 16
        warmup = 5000
        layer_norm = True
        use_edge_attr = True
        edge_dim = 32
        gnn_type = trial.suggest_categorical("gnn_type", ["graphsage", "gcn"])
        k_hop = trial.suggest_categorical("k_hop", [2, 8, 16, 32])
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

    # Roman-Empire Experiment
    if args["dataset"] == "roman-empire":
        input_size = 300 # n_tags
        num_classes = 18
        num_edge_features = 0
        data = datasets.HeterophilousGraphDataset(args["data_path"], name="Roman-empire", transform=T.NormalizeFeatures())
        # data[0]: Data(x=[22662, 300], edge_index=[2, 32927], y=[22662], train_mask=[22662, 10], val_mask=[22662, 10], test_mask=[22662, 10])

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
        )

        wrapper = RomanEmpireWrapper(
            model,
            abs_pe_method,
            args["lr"],
            args["weight_decay"],
            lr_scheduler,
            mask=i,
        )

        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=args["epochs"],
            deterministic=True,
            logger=WandbLogger(
                project="g2_sat_roman-empire",
                config=args,
            ),
            # callbacks=EarlyStopping(monitor="val/loss", mode="min", patience=200),
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
    # tune()
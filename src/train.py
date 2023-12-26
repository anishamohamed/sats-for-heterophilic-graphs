import sys
import os
from functools import partial
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset

from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from data.dataset import GraphDataset
from data.utils import get_heterophilous_graph_data
from model.sat import GraphTransformer
from model.abs_pe import POSENCODINGS
from net.zinc import ZINCWrapper
from net.utils import ZincLRScheduler
from net.heterophilous import HeterophilousGraphWrapper

import yaml
import wandb

NUM_SPLITS_HETERORPHILOUS = 10


def load_config(path):
    with open(path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def run_zinc(config):
    # dataset specific parameters
    input_size = 28  # n_tags
    num_edge_features = 4
    num_class = 1
    train_data = datasets.ZINC(config.get("root_dir"), subset=True, split="train")
    val_data = datasets.ZINC(config.get("root_dir"), subset=True, split="val")
    test_data = datasets.ZINC(config.get("root_dir"), subset=True, split="test")

    train_dataset = GraphDataset(
        train_data,
        degree=False,  # because it's graph classification
        k_hop=config.get("k_hop"),
        se=config.get("se"),
        return_complete_index=True,
    )
    val_dataset = GraphDataset(
        val_data,
        degree=False,
        k_hop=config.get("k_hop"),
        se=config.get("se"),
        return_complete_index=True,
    )
    test_dataset = GraphDataset(
        test_data,
        degree=False,
        k_hop=config.get("k_hop"),
        se=config.get("se"),
        return_complete_index=True,
    )

    abs_pe_encoder = None
    if config.get("model").get("abs_pe") and config.get("model").get("abs_pe_dim") > 0:
        abs_pe_method = POSENCODINGS[config.get("model").get("abs_pe")]
        abs_pe_encoder = abs_pe_method(config.get("model").get("abs_pe_dim"), normalization="sym")
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dataset)
            abs_pe_encoder.apply_to(val_dataset)
            abs_pe_encoder.apply_to(test_dataset)
    else:
        abs_pe_method = None

    deg = torch.cat(
        [degree(data.edge_index[1], num_nodes=data.num_nodes) for data in train_dataset]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.get("batch_size"), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.get("batch_size"), shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.get("batch_size"), shuffle=False
    )

    model_config = config.get("model")
    model_config.update(
        {
            "in_size": input_size,
            "num_class": num_class,
            "num_edge_features": num_edge_features,
            "deg": deg,  # to be propagated to gnn layers...
        }
    )
    model = GraphTransformer(**model_config)

    lr_scheduler = partial(
        ZincLRScheduler, lr=config.get("lr"), warmup=config.get("warmup")
    )
    wrapper = ZINCWrapper(
        model,
        abs_pe_method,
        config.get("lr"),
        config.get("weight_decay"),
        lr_scheduler,
    )

    logger = (
        WandbLogger(
            project=config["logger"].get("project") or "g2_sat_" + config.get("dataset"),
            entity=config["logger"].get("entity"),
            config=config,
        )
        if config.get("logger") is not None
        else False
    )
    trainer = pl.Trainer(
        accelerator=config.get("device"),
        max_epochs=config.get("epochs"),
        deterministic=True,
        logger=logger,
        # callbacks=EarlyStopping(monitor="val/loss", mode="min", patience=20),
        check_val_every_n_epoch=1,
    )

    # if config.get("device") == "cuda":
        # torch.use_deterministic_algorithms(True)

    trainer.fit(wrapper, train_loader, val_loader)
    trainer.test(wrapper, test_loader)

    if logger:
        wandb.finish()

    return trainer.callback_metrics["test/loss"].item()


def run_heterophilous_single_split(dataloaders, mask, config):
    if config.get("node_projection") is not None:
        in_size = MLP(
            in_channels=config.get("input_size"),
            hidden_channels=config["node_projection"].get("hidden_channels"),
            out_channels=config["model"].get("d_model"),
            num_layers=config["node_projection"].get("num_layers"),
            act=nn.Mish(),
            dropout=0.5,
            norm="batch_norm",
        )
    else:
        in_size = config.get("input_size")
    model_config = config.get("model")
    model = GraphTransformer(in_size=in_size, **model_config)

    lr_scheduler = partial(
        optim.lr_scheduler.LambdaLR, lr_lambda=(
            lambda epoch: 1.0 if epoch <= config["scheduler"].get("warmup") else 0.2
        )
    ) if config.get("scheduler") is not None else None
    # lr_scheduler = partial(optim.lr_scheduler.StepLR, **config.get("scheduler")) if config.get("scheduler") is not None else None
    wrapper = HeterophilousGraphWrapper(
        model,
        config.get("abs_pe_method"),
        config.get("lr"),
        config.get("weight_decay"),
        lr_scheduler,
        mask=mask,
    )

    logger = (
        WandbLogger(
            project=config["logger"].get("project") or "g2_sat_" + config.get("dataset"),
            entity=config["logger"].get("entity"),
            config=config,
        )
        if config.get("logger") is not None
        else False
    )
    trainer = pl.Trainer(
        accelerator=config.get("device"),
        max_epochs=config.get("epochs"),
        # deterministic=True,
        logger=logger,
        callbacks=EarlyStopping(monitor="val/acc", mode="max", patience=300),
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    # if config.get("device") == "cuda":
        # torch.use_deterministic_algorithms(True)
    
    train_loader, val_loader, test_loader = dataloaders
    trainer.fit(wrapper, train_loader, val_loader) 
    trainer.test(wrapper, test_loader)

    if logger:
        wandb.finish()

    return trainer.callback_metrics["test/acc"].item()


def run_heterophilous(config):
    dataloaders, config = prepare_heterophilous_dataloaders(config)

    accuracy_list = list()
    for mask in [0]: # range(NUM_SPLITS_HETERORPHILOUS):
        acc = run_heterophilous_single_split(dataloaders, mask, config)
        accuracy_list.append(acc)

    accuracy_list = np.array(accuracy_list)
    print(f"Accuracy: {np.mean(accuracy_list)} +- {np.std(accuracy_list)}")


def prepare_heterophilous_dataloaders(config):
    data, input_size, num_class = get_heterophilous_graph_data(config.get("dataset"), config.get("root_dir"))
    dataset = GraphDataset(
        data,
        degree=config.get("degree"),
        k_hop=config.get("model").get("k_hop"),
        se=config.get("model").get("se"),
        return_complete_index=False,  # large dataset, recommended to set as False as in original SAT code
    )

    abs_pe_encoder = None
    if config.get("model").get("abs_pe") and config.get("model").get("abs_pe_dim") > 0:
        abs_pe_method = POSENCODINGS[config.get("model").get("abs_pe")]
        abs_pe_encoder = abs_pe_method(config.get("model").get("abs_pe_dim"), normalization="sym")
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(dataset)
    else:
        abs_pe_method = None
    config.update({"abs_pe_method": abs_pe_method})

    deg = degree(data[0].edge_index[1], num_nodes=data[0].num_nodes)
    config.update({"input_size": input_size})
    config.get("model").update(
        {
            "num_class": num_class,
            "use_edge_attr": False,
            "use_global_pool": False,  # node classification tasks
            "deg": deg,
        }
    )

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])

    return (loader, loader, loader), config


def run(config_path):
    config = load_config(config_path)
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})

    if config.get("logger") is not None:
        os.environ["WANDB_API_KEY"] = config["logger"].get("wandb_key")

    seed_everything(config.get("seed", 42))
    config["dataset"] = config.get("dataset").lower().replace("-", "_")

    if config["dataset"] == "zinc":
        run_zinc(config)

    elif config["dataset"] in [
        "roman_empire",
        "amazon_ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]:
        run_heterophilous(config)

    elif config["dataset"] in ["pattern", "cluster"]:
        run_sbm(config)

    else:
        raise Exception("Unknown dataset")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file_path>")
        sys.exit(1)

    run(sys.argv[1])

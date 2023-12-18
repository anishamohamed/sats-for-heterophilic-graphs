import sys
import os
from functools import partial
import numpy as np

import torch
from torch import nn

from torch_geometric.loader import DataLoader
from torch_geometric import datasets
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping

from src.data.dataset import GraphDataset
from src.data.utils import get_heterophilous_graph_data
from src.model.sat import GraphTransformer
from src.model.abs_pe import POSENCODINGS
from src.net.zinc import ZINCWrapper
from src.net.utils import ZincLRScheduler
from src.net.heterophilous import HeterophilousGraphWrapper

import yaml
import wandb


def load_config(path):
    with open(path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


def run_zinc(config):
    # dataset specific parameters
    input_size = 28  # n_tags
    num_edge_features = 4
    num_classes = 1
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
    if config.get("abs_pe") and config.get("abs_pe_dim") > 0:
        abs_pe_method = POSENCODINGS[config.get("abs_pe")]
        abs_pe_encoder = abs_pe_method(config.get("abs_pe_dim"), normalization="sym")
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
            "num_class": num_classes,
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
            project="g2_sat_" + config.get("dataset"),
            config=config,
        )
        if config.get("wandb_key") is not None
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

    if config.get("device") == "cuda":
        torch.use_deterministic_algorithms(False)

    trainer.fit(wrapper, train_loader, val_loader)
    trainer.test(wrapper, test_loader)

    if logger:
        wandb.finish()

    return trainer.callback_metrics["test/loss"].item()


def run_heterophilous_single_split(dataloader, config):
    model_config = config.get("model")
    node_projection = MLP(
        in_channels=config.get("input_size"),
        hidden_size=config.get("node_projection").get("hidden_size"),
        out_channels=config.get("node_projection").get("out_channels"),
        num_layers=config.get("node_projection").get("num_layers"),
        act=nn.Mish(),
        norm="batch_norm",
    )
    model = GraphTransformer(in_size=node_projection, **model_config)

    wrapper = HeterophilousGraphWrapper(
        model,
        config.get("abs_pe_method"),
        config.get("lr"),
        config.get("weight_decay"),
        lr_scheduler=None,
        mask=config.get("mask"),
    )

    logger = (
        WandbLogger(
            project="g2_sat_" + config.get("dataset"),
            config=config,
        )
        if config.get("wandb_key") is not None
        else False
    )
    trainer = pl.Trainer(
        accelerator=config.get("device"),
        max_epochs=config.get("epochs"),
        deterministic=True,
        logger=logger,
        # callbacks=EarlyStopping(monitor="val/loss", mode="min", patience=20),
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
    )

    if config.get("device") == "cuda":
        torch.use_deterministic_algorithms(False)

    trainer.fit(wrapper, dataloader, dataloader)  # same dataloader for train,val, test
    trainer.test(wrapper, dataloader)

    if logger:
        wandb.finish()

    return trainer.callback_metrics["test/acc"].item()


def run_heterophilous(config):
    data, input_size, num_classes = get_heterophilous_graph_data(config.get("dataset"), config.get("root_dir"))
    dataset = GraphDataset(
        data,
        degree=config.get("degree"),
        k_hop=config.get("k_hop"),
        se=config.get("se"),
        return_complete_index=False,  # large dataset, recommended to set as False as in original SAT code
    )

    abs_pe_encoder = None
    if config.get("abs_pe") and config.get("abs_pe_dim") > 0:
        abs_pe_method = POSENCODINGS[config.get("abs_pe")]
        abs_pe_encoder = abs_pe_method(config.get("abs_pe_dim"), normalization="sym")
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(dataset)
    else:
        abs_pe_method = None
    config.update({"abs_pe_method": abs_pe_method})

    deg = degree(data[0].edge_index[1], num_nodes=data[0].num_nodes)

    config.update({"input_size": input_size})
    config.get("model").update(
        {
            "num_classes": num_classes,
            "use_edge_attr": False,
            "use_global_pool": False,  # node classification tasks
            "deg": deg,
        }
    )

    loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])

    num_splits = data[0].train_mask.size(-1)
    accuracy_list = list()

    for mask in range(num_splits):
        config.update({"mask": mask})
        accuracy_list.append(run_heterophilous_single_split(loader, config))

    accuracy_list = np.array(accuracy_list)
    print(f"Accuracy: {np.mean(accuracy_list)} +- {np.std(accuracy_list)}")


def run(config_path):
    config = load_config(config_path)
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    os.environ["WANDB_API_KEY"] = config.get("wandb_key")

    seed_everything(config.get("seed"))
    config["dataset"] = config["dataset"].lower().replace("-", "_")

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

    else:
        raise Exception("Unknown dataset")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python train.py <config_file_path>")
        sys.exit(1)

    run(sys.argv[1])

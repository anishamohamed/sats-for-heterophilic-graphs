import os
import sys
from functools import partial

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything

from src.data.dataset import GraphDataset
from src.data.utils import get_heterophilous_graph_data
from src.model.abs_pe import POSENCODINGS
from src.train import load_config, run_zinc, run_heterophilous_single_split
x
import optuna
from optuna.samplers import TPESampler

def convert_to_trial(trial: optuna.trial.Trial, config: dict):
    for key, value in config.items():
        if isinstance(value, list):
            config.update({key: trial.suggest_categorical(str(key), value)})
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, list):
                    value.update({k: trial.suggest_categorical(str(k), v)})
    return config

def objective_zinc(trial: optuna.trial.Trial, config):
    config = convert_to_trial(trial, config)
    print(config)
    return run_zinc(config)

def objective_heterophilous(trial: optuna.trial.Trial, dataloader, config):
    config = convert_to_trial(trial, config)
    config.update({"mask": 0}) # train on first mask id
    print(config)
    return run_heterophilous_single_split(dataloader, config)

def tune(config_path):
    config = load_config(config_path)
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    os.environ["WANDB_API_KEY"] = config.get("wandb_key")

    seed_everything(config.get("seed"))
    config["dataset"] = config["dataset"].lower().replace("-", "_")

    if config["dataset"] == "zinc":
        objective = partial(objective_zinc, config=config)
        direction = "minimize"

    elif config["dataset"] in [
        "roman_empire",
        "amazon_ratings",
        "minesweeper",
        "tolokers",
        "questions",
    ]:
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

        config.update({"input_size": input_size})
        deg = degree(data[0].edge_index[1], num_nodes=data[0].num_nodes)
        config.get("model").update(
            {
                "num_classes": num_classes,
                "use_edge_attr": False,
                "use_global_pool": False,  # node classification tasks
                "deg": deg,
            }
        )

        loader = DataLoader(dataset, batch_size=1, collate_fn=lambda batch: batch[0])

        objective = partial(objective_heterophilous, dataloader=loader, config=config)
        direction = "maximize"

    else:
        raise Exception("Unknown dataset")
    
    study = optuna.create_study(
        direction=direction, sampler=TPESampler(constant_liar=True)
    )
    study.optimize(objective, n_trials=200)

    best_params = study.best_params
    print(f"Best: {study.best_value}")
    print(best_params)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tune.py <config_file_path>")
        sys.exit(1)

    tune(load_config(sys.argv[1]))

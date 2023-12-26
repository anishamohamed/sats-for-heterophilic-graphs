import os
import sys
from functools import partial

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree
from torch_geometric.seed import seed_everything

from data.dataset import GraphDataset
from data.utils import get_heterophilous_graph_data
from model.abs_pe import POSENCODINGS
from train import load_config, run_zinc, run_heterophilous_single_split, prepare_heterophilous_dataloaders
import optuna
from optuna.samplers import TPESampler

def convert_to_trial(trial: optuna.trial.Trial, config: dict):
    trial_config = config.copy()
    for key, value in config.items():
        if isinstance(value, list):
            trial_config.update({key: trial.suggest_categorical(key, value)})
        if isinstance(value, dict): # config[model], config[node_projection]
            trial_config[key] = trial_config[key].copy()
            for k, v in value.items():
                if isinstance(v, list):
                    trial_config[key].update({k: trial.suggest_categorical(k, v)})
    return trial_config

def objective_heterophilous(trial: optuna.trial.Trial, dataloader, config):
    trial_config = convert_to_trial(trial, config)
    # train on first mask id
    return run_heterophilous_single_split(dataloader, 0, trial_config)

def objective_zinc(trial: optuna.trial.Trial, config):
    trial_config = convert_to_trial(trial, config)
    return run_zinc(trial_config)

def tune(config_path):
    config = load_config(config_path)
    config.update({"device": "cuda" if torch.cuda.is_available() else "cpu"})
    if config.get("logger") is not None:
        os.environ["WANDB_API_KEY"] = config["logger"].get("wandb_key")

    seed_everything(config.get("seed", 42))
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
        loader, config = prepare_heterophilous_dataloaders(config)
        objective = partial(objective_heterophilous, dataloader=loader, config=config)
        direction = "maximize"

    else:
        raise Exception("Unknown dataset")
    
    study = optuna.create_study(
        direction=direction, sampler=TPESampler(constant_liar=True)
    )
    study.optimize(objective, n_trials=2)

    best_params = study.best_params
    print(f"Best: {study.best_value}")
    print(best_params)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tune.py <config_file_path>")
        sys.exit(1)

    tune(sys.argv[1])

# Enhancing Heterophilic Graph Representations in Structure-Aware Transformers

This project aims at leveraging different deep learning techniques on [Structure-Aware Transformers](https://arxiv.org/abs/2202.03036) (SATs) in order to produce more informative structural embeddings.

## Installation
Run the installation script in order to install all the required packages.

```bash
./install_env.sh
```

## Training
Train and test the model by creating an appriate YAML file and running the following command.

```bash
python src/train.py path/to/config
```

Our best and baseline configurations for each of the tested datasets are in the folder `config`.

### Node classification on MINESWEEPER
```bash
python src/train.py config/minesweeper/base.yaml # baseline
python src/train.py config/minesweeper/best_dir.yaml # enhanced w/ directionality
python src/train.py config/minesweeper/best_undir.yaml # enhanced w/out directionality
```

### Node classification on ROMAN-EMPIRE
```bash
python src/train.py config/roman_empire/base.yaml # baseline
python src/train.py config/roman_empire/best_dir.yaml # enhanced w/ directionality
python src/train.py config/roman_empire/best_undir.yaml # enhanced w/out directionality
```

### Node classification on CLUSTER
```bash
python src/train.py config/cluster/base.yaml # baseline
python src/train.py config/cluster/best.yaml # enhanced
```

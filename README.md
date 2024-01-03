# Mitigating Over-smoothing in Structure-Aware Graph Transformer

This project aims at leveraging different deep learning techniques on [Structure-Aware Transformers](https://arxiv.org/abs/2202.03036) (SATs) in order to produce more informative structural embeddings.

## Installation
Run the installation script in order to install all the required packages.

```bash
./install.sh
```

## Training
Train and test the model by creating an appriate YAML file and running the following command.

```bash
src/train.py path/to/config
```

Our best configurations for each of the tested dataset are in the folder `config`.

### Node classification on PATTERN and CLUSTER datasets
```bash
python src/train.py config/pattern/best.yaml
```

```bash
python src/train.py config/cluster/best.yaml
```

### Node classification on MINESWEEPER
```bash
python src/train.py config/minesweeper/best.yaml
```

### Node classification on ROMAN EMPIRE
```bash
python src/train.py config/roman_empire/best.yaml
```
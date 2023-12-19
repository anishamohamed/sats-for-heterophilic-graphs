#!/bin/bash
  
#SBATCH --time=20:00:00
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --mem-per-cpu=36G
#SBATCH --job-name=sa
#SBATCH --output=dl.out
#SBATCH --error=dl.err
#SBATCH --mail-type=BEGIN

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy cuda/11.8.0 cudnn/8.8.1.3

source  $HOME/miniconda/bin/activate g2-sat
# python train_zinc.py --dataset ZINC --data-path datasets/ZINC --se gnn --gnn-type pna2 --k-hop 3 --abs-pe rw --dropout 0.3 --use-edge-attributes
# python train_empire.py --se gnn --gnn-type pna2 --k-hop 3 --abs-pe rw --dropout 0.3 --num-layers 1
python src/tune.py config/roman_empire/exp1.yaml
conda deactivate
#!/bin/bash
  
#SBATCH --time=20:00:00
#SBATCH --gpus=a100_80gb:1
#SBATCH --mem-per-cpu=36G
#SBATCH --job-name=sa
#SBATCH --output=dl.out
#SBATCH --error=dl.err
#SBATCH --mail-type=BEGIN

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy cuda/11.8.0 cudnn/8.8.1.3

source  $HOME/miniconda/bin/activate g2-sat
python src/train.py config/roman_empire/exp_pna2_16hops.yaml
# python src/train.py config/roman_empire/base+dir.yaml
# python src/train.py config/roman_empire/base+jknet.yaml
# python src/train.py config/roman_empire/base+node_proj.yaml
python src/train.py config/roman_empire/exp_pna2_16hops.yaml
conda deactivate
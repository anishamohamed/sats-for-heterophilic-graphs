#!/bin/bash
  
#SBATCH --time=2:00:00
#SBATCH --gpus=rtx_2080_ti:1
#SBATCH --mem-per-cpu=10G
#SBATCH --job-name=sa
#SBATCH --output=dl.out
#SBATCH --error=dl.err
#SBATCH --mail-type=BEGIN

module load gcc/8.2.0 python_gpu/3.10.4 hdf5/1.10.1 eth_proxy cuda/11.8.0 cudnn/8.8.1.3

if command -v conda &> /dev/null; then
    echo "Conda is already installed. Skipping installation."
else
    # Install Conda (Mamba)
    echo "Installing Conda (Mamba)..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> $HOME/.bashrc
    source $HOME/.bashrc
    echo "Conda (Mamba) installation complete."
fi

conda env create --file env_cuda.yml
import torch
from torch_geometric.datasets import ZINC

# Set the directory where the dataset will be stored
dataset_directory = './ZINC'

# Load the ZINC dataset
dataset = ZINC(root=dataset_directory, subset=True)

# Example: Accessing the first graph in the dataset
data = dataset[0]
print(data)

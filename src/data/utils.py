from torch_geometric import datasets
import torch_geometric.transforms as T

INPUT_SIZE = {
    "cluster": 7,
    "pattern": 3,
    "roman_empire": 300,
    "amazon_ratings": 300,
    "minesweeper": 7,
    "tolokers": 10,
    "questions": 301,
}

NUM_CLASSES = {
    "cluster": 6,
    "pattern": 2,
    "roman_empire": 18,
    "amazon_ratings": 5,
    "minesweeper": 2,
    "tolokers": 2,
    "questions": 2,
}

BINARY_CLASSIFICATION_DATASETS = ["pattern", "minesweeper", "tolokers", "questions"]

def get_heterophilous_graph_data(name: str, root_dir: str):
    if name == "roman_empire":
        data = datasets.HeterophilousGraphDataset(root_dir, name="Roman-empire")
        # data[0]: Data(x=[22662, 300], edge_index=[2, 32927], y=[22662], train_mask=[22662, 10], val_mask=[22662, 10], test_mask=[22662, 10])
    elif name == "amazon_ratings":
        data = datasets.HeterophilousGraphDataset(
            root_dir,
            name="Amazon-ratings",
            transform=T.NormalizeFeatures(),
        )
        # data[0]: Data(x=[24492, 300], edge_index=[2, 186100], y=[24492], train_mask=[24492, 10], val_mask=[24492, 10], test_mask=[24492, 10])
    elif name == "minesweeper":
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Minesweeper", transform=T.NormalizeFeatures()
        )
        # data[0]: Data(x=[10000, 7], edge_index=[2, 78804], y=[10000], train_mask=[10000, 10], val_mask=[10000, 10], test_mask=[10000, 10]
    elif name == "tolokers":
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Tolokers", transform=T.NormalizeFeatures()
        )
    elif name == "questions":
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Questions", transform=T.NormalizeFeatures()
        )

    return data

from torch_geometric import datasets
import torch_geometric.transforms as T

def get_heterophilous_graph_data(name: str, root_dir: str):
    if name == "roman_empire":
        input_size = 300  # n_tags
        num_classes = 18
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Roman-empire"
        )
        # data[0]: Data(x=[22662, 300], edge_index=[2, 32927], y=[22662], train_mask=[22662, 10], val_mask=[22662, 10], test_mask=[22662, 10])
    elif name == "amazon_ratings":
        input_size = 300
        num_classes = 5
        data = datasets.HeterophilousGraphDataset(
            root_dir,
            name="Amazon-ratings",
            transform=T.NormalizeFeatures(),
        )
        # data[0]: Data(x=[24492, 300], edge_index=[2, 186100], y=[24492], train_mask=[24492, 10], val_mask=[24492, 10], test_mask=[24492, 10])
    elif name == "minesweeper":
        input_size = 7
        num_classes = 2
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Minesweeper", transform=T.NormalizeFeatures()
        )
        # data[0]: Data(x=[10000, 7], edge_index=[2, 78804], y=[10000], train_mask=[10000, 10], val_mask=[10000, 10], test_mask=[10000, 10]
    elif name == "tolokers":
        input_size = 10
        num_classes = 2
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Tolokers", transform=T.NormalizeFeatures()
        )
    elif name == "questions":
        input_size = 301
        num_classes = 2
        data = datasets.HeterophilousGraphDataset(
            root_dir, name="Questions", transform=T.NormalizeFeatures()
        )
    
    return data, input_size, num_classes
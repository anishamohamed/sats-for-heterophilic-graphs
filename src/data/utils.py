from torch_geometric import datasets
import torch_geometric.transforms as T
from data.dataset import GraphDataset
from torch_geometric.loader import DataLoader
from model.abs_pe import POSENCODINGS

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

# def process_sbm(config):
#     # n_tags = 7 if config("dataset") == "cluster" else 3
#     return
    
            
#     # return train_loader, test_loader, val_loader, abs_pe_method
#     # train_dset = GraphDataset(datasets.GNNBenchmarkDataset(config.get("root_dir"),
#     # name=args.dataset, split='train'), degree=True, k_hop=args.k_hop, se=args.se,
#     # cache_path=cache_path + 'train')
#     # input_size = n_tags
#     # train_loader = DataLoader(train_dset, batch_size=config.get("batch_size), shuffle=True)

#     # print(len(train_dset))
#     # print(train_dset[0])

#     # val_dset = GraphDataset(datasets.GNNBenchmarkDataset(config.get("root_dir"),
#     #     name=args.dataset, split='val'), degree=True, k_hop=args.k_hop, se=args.se,
#     #     cache_path=cache_path + 'val')
#     # val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

#     # abs_pe_encoder = None
#     # if args.abs_pe and args.abs_pe_dim > 0:
#     #     abs_pe_method = POSENCODINGS[args.abs_pe]
#     #     abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
#     #     if abs_pe_encoder is not None:
#     #         abs_pe_encoder.apply_to(train_dset)
#     #         abs_pe_encoder.apply_to(val_dset)
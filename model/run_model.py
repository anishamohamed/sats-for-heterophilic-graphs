import numpy as np
import pandas as pd

from collections import defaultdict
import copy
from timeit import default_timer as timer

import torch
from torch import nn, optim
import torch.nn.functional as F

import torch_geometric.utils as utils
from torch_geometric import datasets
from torch_geometric.loader import DataLoader


from torch.utils.tensorboard import SummaryWriter

from dataloader import GraphDataset
from encoding import POSENCODINGS
from loadArgs import load_args
from model import Model

def train_epoch(model, loader, criterion, optimizer, lr_scheduler, epoch, use_cuda=False):
    model.train()

    running_loss = 0.0

    tic = timer()
    for i, data in enumerate(loader):
        size = len(data.y)
        if args.warmup is not None:
            iteration = epoch * len(loader) + i
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_scheduler(iteration)

        if use_cuda:
            data = data.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * size

    toc = timer()
    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    print('Train loss: {:.4f} time: {:.2f}s'.format(
          epoch_loss, toc - tic))
    return epoch_loss


def eval_epoch(model, loader, criterion, use_cuda=False, split='Val'):
    model.eval()

    running_loss = 0.0
    mae_loss = 0.0
    mse_loss = 0.0

    tic = timer()
    with torch.no_grad():
        for data in loader:
            size = len(data.y)
            if use_cuda:
                data = data.cuda()

            output = model(data)
            loss = criterion(output, data.y)
            mse_loss += F.mse_loss(output, data.y).item() * size
            mae_loss += F.l1_loss(output, data.y).item() * size

            running_loss += loss.item() * size
    toc = timer()

    n_sample = len(loader.dataset)
    epoch_loss = running_loss / n_sample
    epoch_mae = mae_loss / n_sample
    epoch_mse = mse_loss / n_sample
    print('{} loss: {:.4f} MSE loss: {:.4f} MAE loss: {:.4f} time: {:.2f}s'.format(
          split, epoch_loss, epoch_mse, epoch_mae, toc - tic))
    return epoch_mae, epoch_mse

def count_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

def main():
    global args
    writer = SummaryWriter()
    args = load_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print(args)
    data_path = '../datasets/ZINC'
    # number of node attributes for ZINC dataset
    input_size = 28
    num_edge_features = 4

    train_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='train'), degree=True, k_hop=args.k_hop,
        use_subgraph_edge_attr=args.use_edge_attr)
    train_loader = DataLoader(train_dset, batch_size=args.batch_size,
            shuffle=True)

    val_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='val'), degree=True, k_hop=args.k_hop,
        use_subgraph_edge_attr=args.use_edge_attr)
    val_loader = DataLoader(val_dset, batch_size=args.batch_size, shuffle=False)

    abs_pe_encoder = None
    if args.abs_pe and args.abs_pe_dim > 0:
        abs_pe_method = POSENCODINGS[args.abs_pe]
        abs_pe_encoder = abs_pe_method(args.abs_pe_dim, normalization='sym')
        if abs_pe_encoder is not None:
            abs_pe_encoder.apply_to(train_dset)
            abs_pe_encoder.apply_to(val_dset)

    deg = torch.cat([
        utils.degree(data.edge_index[1], num_nodes=data.num_nodes) for
        data in train_dset])

    model = Model(input_size=input_size,
                    output_size=1,
                    hidden_size=args.dim_hidden,
                    dim_feedforward=2*args.dim_hidden,
                    dropout=args.dropout,
                    num_heads=args.num_heads,
                    num_layers=args.num_layers,
                    batch_norm=args.batch_norm,
                    abs_pe=args.abs_pe,
                    abs_pe_dim=args.abs_pe_dim,
                    gnn_type=args.gnn_type,
                    use_edge_attr=args.use_edge_attr,
                    num_edge_features=num_edge_features,
                    edge_dim=args.edge_dim,
                    k_hop=args.k_hop,
                    deg=deg,
                    global_pool=args.global_pool,
                    use_gates=args.use_gates)

    if args.use_cuda:
        model.cuda()
        
    print("Total number of parameters: {}".format(count_parameters(model)))

    criterion = nn.L1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup is None:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.5,
                                                            patience=15,
                                                            min_lr=1e-05,
                                                            verbose=False)
    else:
        lr_steps = (args.lr - 1e-6) / args.warmup
        decay_factor = args.lr * args.warmup ** .5
        def lr_scheduler(s):
            if s < args.warmup:
                lr = 1e-6 + s * lr_steps
            else:
                lr = decay_factor * s ** -.5
            return lr

    test_dset = GraphDataset(datasets.ZINC(data_path, subset=True,
        split='test'), degree=True, k_hop=args.k_hop,
        use_subgraph_edge_attr=args.use_edge_attr)

    test_loader = DataLoader(test_dset, batch_size=args.batch_size, shuffle=False)

    #FIXME (left by the original authors, not sure what they wanted to fix)
    if abs_pe_encoder is not None:
        abs_pe_encoder.apply_to(test_dset)

    print("Training...")
    best_val_loss = float('inf')
    best_epoch = 0
    logs = defaultdict(list)
    start_time = timer()
    for epoch in range(args.epochs):
        print("Epoch {}/{}, LR {:.6f}".format(epoch + 1, args.epochs, optimizer.param_groups[0]['lr']))
        train_loss = train_epoch(model, train_loader, criterion, optimizer, lr_scheduler, epoch, args.use_cuda)
        val_loss,_ = eval_epoch(model, val_loader, criterion, args.use_cuda, split='Val')
        test_loss,_ = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)

        if args.warmup is None:
            lr_scheduler.step(val_loss)

        logs['train_mae'].append(train_loss)
        logs['val_mae'].append(val_loss)
        logs['test_mae'].append(test_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())

    total_time = timer() - start_time
    print("best epoch: {} best val loss: {:.4f}".format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)

    print()
    print("Testing...")
    test_loss, test_mse_loss = eval_epoch(model, test_loader, criterion, args.use_cuda, split='Test')

    print("test MAE loss {:.4f}".format(test_loss))
    print(args)

    if args.save_logs:
        logs = pd.DataFrame.from_dict(logs)
        logs.to_csv(args.outdir + '/logs.csv')
        results = {
            'test_mae': test_loss,
            'test_mse': test_mse_loss,
            'val_mae': best_val_loss,
            'best_epoch': best_epoch,
            'total_time': total_time,
        }
        results = pd.DataFrame.from_dict(results, orient='index')
        results.to_csv(args.outdir + '/results.csv',
                       header=['value'], index_label='name')
        torch.save(
            {'args': args,
            'state_dict': best_weights},
            args.outdir + '/model.pth')


if __name__ == "__main__":
    main()

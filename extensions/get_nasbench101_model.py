import os
import sys


import torch
import torch.nn as nn
import numpy as np
from torch import optim
from arch2vec.models.model import Model, VAEReconstructed_Loss
from arch2vec.utils import load_json, save_checkpoint_vae, preprocessing
from arch2vec.utils import get_val_acc_vae
from arch2vec.utils import to_ops_nasbench101
from arch2vec.models.configs import configs
import argparse
from nasbench import api
from nasbench.lib import graph_util


# TODO tohle bude fce, která tu síťu připraví, trénink bude vedle. Proč? - různý lossy atd,... lepší to mít tam
#  a tady jen to nutný.
#  ...a navíc pak stačí jen vyndat encoder a nacpat ho do evalu

def _build_nasbench_dataset(dataset, ind_list):
    indices = np.random.permutation(ind_list)

    hash_list = []
    X_adj = []
    X_ops = []

    for ind in indices:
        data_point = dataset[str(ind)]

        hash_list.append(data_point['hash'])
        X_adj.append(torch.Tensor(data_point['module_adjacency']))
        X_ops.append(torch.Tensor(data_point['module_operations']))

    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return hash_list, X_adj, X_ops, torch.Tensor(indices)


def get_nasbench_datasets(dataset, seed=1, test_size=0.1, batch_size=None):
    # TODO split some test set? Maybe save train/test separately

    if isinstance(dataset, str):
        dataset = load_json(dataset)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_size = 1.0 - test_size
    train_size = int(len(dataset) * train_size)

    train_ind_list, val_ind_list = range(train_size), range(train_size, len(dataset))
    hash_train, X_adj_train, X_ops_train, indices_train = _build_nasbench_dataset(dataset, train_ind_list)
    hash_val, X_adj_val, X_ops_val, indices_val = _build_nasbench_dataset(dataset, val_ind_list)

    if batch_size is not None:
        hash_train = [hash_train[i:i + batch_size] for i in range(0, len(hash_train), batch_size)]

        X_adj_train = torch.split(X_adj_train, batch_size, dim=0)
        X_ops_train = torch.split(X_ops_train, batch_size, dim=0)
        indices_train = torch.split(indices_train, batch_size, dim=0)

    return {
        "train": (hash_train, X_adj_train, X_ops_train, indices_train),
        "val": (hash_val, X_adj_val, X_ops_val, indices_val)
    }

    parser.add_argument('--dropout', type=float, default=0.3,
                        help='decoder implicit regularization (default: 0.3)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')


def get_arch2vec_model(device=None, input_dim=5, hidden_dim=128, latent_dim=16, num_layers=5, num_mlps=2,
                       dropout=0.3, config=4):
    config = configs[config]

    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                  num_hops=num_layers, num_mlp_layers=num_mlps, dropout=dropout, **config['GAE']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

    return model, optimizer



def eval_validity_and_uniqueness(n_latent_points=10000):
    # TODO accuracy etc pak jinde, mozna se bude lisit kvuli zbytku?
    pass


def pretraining_model(dataset, cfg, args):
    nasbench = api.NASBench('data/nasbench_only108.tfrecord')

    train_ind_list, val_ind_list = range(int(len(dataset)*0.9)), range(int(len(dataset)*0.9), len(dataset))
    _, X_adj_train, X_ops_train, indices_train = _build_nasbench_dataset(dataset, train_ind_list)
    _, X_adj_val, X_ops_val, indices_val = _build_nasbench_dataset(dataset, val_ind_list)

    model = Model(input_dim=args.input_dim, hidden_dim=args.hidden_dim, latent_dim=args.dim,
                  num_hops=args.hops, num_mlp_layers=args.mlps, dropout=args.dropout, **cfg['GAE']).cuda()

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
    epochs = args.epochs
    bs = args.bs

    loss_total = []
    for epoch in range(0, epochs):

        chunks = len(train_ind_list) // bs
        if len(train_ind_list) % bs > 0:
            chunks += 1

        # todo to to jako neshufflujou ?? u sebe shuffle T/F, mozna zkusit aj tady shufflovat
        X_adj_split = torch.split(X_adj_train, bs, dim=0)
        X_ops_split = torch.split(X_ops_train, bs, dim=0)
        indices_split = torch.split(indices_train, bs, dim=0)

        loss_epoch = []
        Z = []
        for i, (adj, ops, ind) in enumerate(zip(X_adj_split, X_ops_split, indices_split)):
            optimizer.zero_grad()
            adj, ops = adj.cuda(), ops.cuda()

            # TODO tohle koukni, co je, protože to musíš mít vedle asi (anebo připravit tady dataset x hashes)
            # preprocessing
            adj, ops, prep_reverse = preprocessing(adj, ops, **cfg['prep'])

            # TODO tohle budeš mít svoje
            # forward
            ops_recon, adj_recon, mu, logvar = model(ops, adj.to(torch.long))
            Z.append(mu)

            adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
            adj, ops = prep_reverse(adj, ops)
            loss = VAEReconstructed_Loss(**cfg['loss'])((ops_recon, adj_recon), (ops, adj), mu, logvar)
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            # TODO detach()
            loss_epoch.append(loss.item())
            if i % 1000 == 0:
                print('epoch {}: batch {} / {}: loss: {:.5f}'.format(epoch, i, chunks, loss.item()))

        Z = torch.cat(Z, dim=0)
        z_mean, z_std = Z.mean(0), Z.std(0)

        validity_counter = 0
        buckets = {}

        # try to generate from the latent space, measure uniqueness and validity
        model.eval()
        for _ in range(args.latent_points):
            z = torch.randn(7, args.dim).cuda()
            z = z * z_std + z_mean
            op, ad = model.decoder(z.unsqueeze(0))

            op = op.squeeze(0).cpu()
            ad = ad.squeeze(0).cpu()
            max_idx = torch.argmax(op, dim=-1)
            one_hot = torch.zeros_like(op)

            for i in range(one_hot.shape[0]):
                one_hot[i][max_idx[i]] = 1

            op_decode = to_ops_nasbench101(max_idx)
            ad_decode = (ad > 0.5).int().triu(1).numpy()
            ad_decode = np.ndarray.tolist(ad_decode)

            spec = api.ModelSpec(matrix=ad_decode, ops=op_decode)
            if nasbench.is_valid(spec):
                validity_counter += 1
                fingerprint = graph_util.hash_module(np.array(ad_decode), one_hot.numpy().tolist())

                if fingerprint not in buckets:
                    buckets[fingerprint] = (ad_decode, one_hot.numpy().astype('int8').tolist())

        # TODO fci i na tohle
        validity = validity_counter / args.latent_points
        print('Ratio of valid decodings from the prior: {:.4f}'.format(validity))
        print('Ratio of unique decodings from the prior: {:.4f}'.format(len(buckets) / (validity_counter+1e-8)))

        # validation set accuracy and other metrics
        acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val = get_val_acc_vae(model, cfg, X_adj_val, X_ops_val, indices_val)
        print('validation set: acc_ops:{0:.4f}, mean_corr_adj:{1:.4f}, mean_fal_pos_adj:{2:.4f}, acc_adj:{3:.4f}'.format(
                acc_ops_val, mean_corr_adj_val, mean_fal_pos_adj_val, acc_adj_val))
        print('epoch {}: average loss {:.5f}'.format(epoch, sum(loss_epoch)/len(loss_epoch)))

        loss_total.append(sum(loss_epoch) / len(loss_epoch))
        save_checkpoint_vae(model, optimizer, epoch, sum(loss_epoch) / len(loss_epoch), args.dim, args.name, args.dropout, args.seed)

    print('loss for epochs: \n', loss_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretraining')
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument('--data', type=str, default='data/data.json',
                        help='Data file (default: data.json')
    parser.add_argument('--name', type=str, default='nasbench-101',
                        help='nasbench-101/nasbench-201/darts')
    parser.add_argument('--cfg', type=int, default=4,
                        help='configuration (default: 4)')
    parser.add_argument('--bs', type=int, default=32,
                        help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=8,
                        help='training epochs (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='decoder implicit regularization (default: 0.3)')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='use input normalization')
    parser.add_argument('--input_dim', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dim', type=int, default=16,
                        help='feature dimension (default: 16)')
    parser.add_argument('--hops', type=int, default=5)
    parser.add_argument('--mlps', type=int, default=2)
    parser.add_argument('--latent_points', type=int, default=10000,
                        help='latent points for validaty check (default: 10000)')
    argss = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #cfg = configs[args.cfg]
    #dataset = load_json(args.data)
    print('using {}'.format(args.data))
    print('feat dim {}'.format(args.dim))
    pretraining_model(dataset, cfg, args)

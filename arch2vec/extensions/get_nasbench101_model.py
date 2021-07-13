import torch
import numpy as np

from torch import optim
from arch2vec.models.model import Model
from arch2vec.utils import load_json, preprocessing, get_accuracy
from arch2vec.utils import to_ops_nasbench101
from arch2vec.models.configs import configs
from nasbench import api
from nasbench.lib import graph_util


def _build_nasbench_dataset(dataset, ind_list):
    indices = np.random.permutation(ind_list)

    hash_list = []
    X_adj = []
    X_ops = []

    try:
        # json loaded dataset has str keys
        dataset[str(indices[0])]
        use_ints = False
    except KeyError:
        use_ints = True

    for ind in indices:
        if use_ints:
            data_point = dataset[ind]
        else:
            data_point = dataset[str(ind)]

        hash_list.append(data_point['hash'])
        X_adj.append(torch.Tensor(data_point['module_adjacency']))
        X_ops.append(torch.Tensor(data_point['module_operations']))

    X_adj = torch.stack(X_adj)
    X_ops = torch.stack(X_ops)
    return hash_list, X_adj, X_ops, torch.Tensor(indices)


def _split_batches(hash, X_adj, X_ops, indices, batch_size):
    hash = [hash[i:i + batch_size] for i in range(0, len(hash), batch_size)]

    X_adj = torch.split(X_adj, batch_size, dim=0)
    X_ops = torch.split(X_ops, batch_size, dim=0)
    indices = torch.split(indices, batch_size, dim=0)

    return hash, X_adj, X_ops, indices


def get_nasbench_datasets(dataset, seed=1, test_size=0.1, batch_size=32, val_batch_size=100):
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

    n_train, n_val = len(indices_train), len(indices_val)

    if batch_size is not None:
        hash_train, X_adj_train, X_ops_train, indices_train = _split_batches(hash_train, X_adj_train, X_ops_train,
                                                                             indices_train, batch_size)

    if val_batch_size is not None:
        hash_val, X_adj_val, X_ops_val, indices_val = _split_batches(hash_val, X_adj_val, X_ops_val, indices_val,
                                                                     val_batch_size)

    return {
        "train": (hash_train, X_adj_train, X_ops_train, indices_train),
        "n_train": n_train,
        "val": (hash_val, X_adj_val, X_ops_val, indices_val),
        "n_val": n_val
    }


def get_arch2vec_model(device=None, input_dim=5, hidden_dim=128, latent_dim=16, num_layers=5, num_mlps=2,
                       dropout=0.3, config=4):
    config = configs[config]

    model = Model(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                  num_hops=num_layers, num_mlp_layers=num_mlps, dropout=dropout,
                  return_z=True, **config['GAE']).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08)

    return model, optimizer


def eval_validity_and_uniqueness(model, z_mean, z_std, nasbench, n_latent_points=10000, latent_dim=16, device=None):
    validity_counter = 0
    buckets = {}

    # try to generate from the latent space, measure uniqueness and validity
    model.eval()
    for _ in range(n_latent_points):
        z = torch.randn(7, latent_dim).to(device)
        z = z * z_std + z_mean
        op, ad = model.decoder(z.unsqueeze(0))

        op = op.squeeze(0).detach().cpu()
        ad = ad.squeeze(0).detach().cpu()
        max_idx = torch.argmax(op, dim=-1)
        one_hot = torch.zeros_like(op)

        for i in range(one_hot.shape[0]):
            one_hot[i][max_idx[i]] = 1

        one_hot = one_hot.numpy()

        op_decode = to_ops_nasbench101(max_idx)
        ad_decode = (ad > 0.5).int().triu(1).numpy()
        ad_decode_list = np.ndarray.tolist(ad_decode)

        spec = api.ModelSpec(matrix=ad_decode_list, ops=op_decode)
        if nasbench.is_valid(spec):
            validity_counter += 1
            fingerprint = graph_util.hash_module(ad_decode, one_hot.tolist())

            if fingerprint not in buckets:
                buckets[fingerprint] = (ad_decode_list, one_hot.astype('int8').tolist())

    validity = validity_counter / n_latent_points
    uniqueness = len(buckets) / (validity_counter + 1e-8)
    return validity, uniqueness


def eval_validation_accuracy(model, val_dataset, config=4, device=None):
    model.eval()
    if isinstance(config, int):
        config = configs[config]

    correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave, acc_ave = 0, 0, 0, 0, 0
    n_validation = len(val_dataset)

    for i, (adj, ops) in enumerate(val_dataset):
        adj, ops = adj.to(device), ops.to(device)

        # preprocessing
        adj, ops, prep_reverse = preprocessing(adj, ops, **config['prep'])

        # forward
        ops_recon, adj_recon, mu, logvar, _ = model.forward(ops, adj)

        # reverse preprocessing
        adj_recon, ops_recon = prep_reverse(adj_recon, ops_recon)
        adj, ops = prep_reverse(adj, ops)
        correct_ops, mean_correct_adj, mean_false_positive_adj, correct_adj = get_accuracy((ops_recon, adj_recon),
                                                                                           (ops, adj))
        # average stats
        fraction = len(adj) / n_validation

        correct_ops_ave += correct_ops * fraction
        mean_correct_adj_ave += mean_correct_adj * fraction
        mean_false_positive_adj_ave += mean_false_positive_adj * fraction
        correct_adj_ave += correct_adj * fraction

    return correct_ops_ave, mean_correct_adj_ave, mean_false_positive_adj_ave, correct_adj_ave

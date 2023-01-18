# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Things that don't belong anywhere else
"""

import hashlib
import json
import os
import re
import copy

import itertools
import sys
from shutil import copyfile
from collections import OrderedDict, defaultdict
from numbers import Number
import operator

import numpy as np
import torch
import tqdm


def _detect_folders(args):
    folders = [os.path.join(args.aux_dir, path) for path in os.listdir(args.aux_dir)
               if not path.startswith(args.dataset)]
    folders = sorted([folder for folder in folders if os.path.isdir(folder)])
    print(f"Discover {len(folders)} auxiliary folders")
    # assert 20 % (len(folders) + 1) == 0
    return folders

def _detect_path_in_subfolder(args, folder):
    subfolders = [os.path.join(folder, path) for path in os.listdir(folder)]
    subfolders = sorted([subfolder for subfolder in subfolders if os.path.isdir(subfolder) and "done" in os.listdir(subfolder) and "featurizer_last.pkl" in os.listdir(subfolder)])
    assert len(subfolders), f"No subfolders found with finished auxiliary trainings in {folder}"
    index_subfolder = args.hparams_seed % len(subfolders)
    return os.path.join(subfolders[index_subfolder], "featurizer_last.pkl")

def get_featurizer_aux(args):
    folders = _detect_folders(args)
    index_folder = args.hparams_seed % (len(folders) + 1)
    if index_folder == 0:
        # case of directly transferred from ImageNet, so no need of any auxiliary trainings
        return "imagenet"
    return _detect_path_in_subfolder(args, folders[index_folder-1])

def get_list_featurizers_aux(args):
    folders = _detect_folders(args)
    list_aux_featurizers = ["imagenet"]
    for folder in folders:
        list_aux_featurizers.append(_detect_path_in_subfolder(args, folder))
    return list_aux_featurizers

def get_name_waparameters(featurizers_aux, lambdas_aux):
    weights = {}
    list_gen_named_params = [featurizer.named_parameters() for featurizer in featurizers_aux]
    for name_0, param_0 in featurizers_aux[0].named_parameters():
        named_params = [next(gen_named_params) for gen_named_params in list_gen_named_params]
        new_data = torch.zeros_like(param_0.data)
        for i in range(len(featurizers_aux)):
            name_i, param_i = named_params[i]
            assert name_0 == name_i
            new_data = new_data + lambdas_aux[i] * param_i
        weights[name_0] = new_data
    return weights

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def get_score(results, test_envs, metric_key="acc"):
    val_env_keys = []
    for i in itertools.count():
        acc_key = f'env{i}_out_' + metric_key
        if acc_key in results:
            if i not in test_envs:
                val_env_keys.append(acc_key)
        else:
            break
    assert i > 0
    return np.mean([results[key] for key in val_env_keys])

class MergeDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        super(MergeDataset, self).__init__()
        self.datasets = datasets

    def __getitem__(self, key):
        count = 0
        for d in self.datasets:
            if key - count >= len(d):
                count += len(d)
            else:
                return d[key - count]
        raise ValueError(key)

    def __len__(self):
        return sum([len(d) for d in self.datasets])

def pdb():
    sys.stdout = sys.__stdout__
    import pdb
    print("Launching PDB, enter 'n' to step to parent function.")
    pdb.set_trace()

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)

def print_separator():
    print("="*80)

def print_row(row, colwidth=10, latex=False):
    if latex:
        sep = " & "
        end_ = "\\\\"
    else:
        sep = "  "
        end_ = ""

    def format_val(x):
        if np.issubdtype(type(x), np.floating):
            x = "{:.10f}".format(x)
        return str(x).ljust(colwidth)[:colwidth]
    print(sep.join([format_val(x) for x in row]), end_)


class _SplitDataset(torch.utils.data.Dataset):
    """Used by split_dataset"""
    def __init__(self, underlying_dataset, keys):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
    def __getitem__(self, key):
        return self.underlying_dataset[self.keys[key]]
    def __len__(self):
        return len(self.keys)

def split_dataset(dataset, n, seed=0):
    """
    Return a pair of datasets corresponding to a random split of the given
    dataset, with n datapoints in the first dataset and the rest in the last,
    using the given random seed
    """
    assert(n <= len(dataset))
    keys = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(keys)
    keys_1 = keys[:n]
    keys_2 = keys[n:]
    return _SplitDataset(dataset, keys_1), _SplitDataset(dataset, keys_2)

def random_pairs_of_minibatches(minibatches):
    perm = torch.randperm(len(minibatches)).tolist()
    pairs = []

    for i in range(len(minibatches)):
        j = i + 1 if i < (len(minibatches) - 1) else 0

        xi, yi = minibatches[perm[i]][0], minibatches[perm[i]][1]
        xj, yj = minibatches[perm[j]][0], minibatches[perm[j]][1]

        min_n = min(len(xi), len(xj))

        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def split_meta_train_test(minibatches, num_meta_test=1):
    n_domains = len(minibatches)
    perm = torch.randperm(n_domains).tolist()
    pairs = []
    meta_train = perm[:(n_domains-num_meta_test)]
    meta_test = perm[-num_meta_test:]

    for i,j in zip(meta_train, cycle(meta_test)):
        xi, yi = minibatches[i][0], minibatches[i][1]
        xj, yj = minibatches[j][0], minibatches[j][1]

        min_n = min(len(xi), len(xj))
        pairs.append(((xi[:min_n], yi[:min_n]), (xj[:min_n], yj[:min_n])))

    return pairs

def accuracy(network, loader, device):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            p = network.predict(x)
            batch_weights = torch.ones(len(x)).to(device)
            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float() * batch_weights.view(-1, 1)).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float() * batch_weights).sum().item()
            total += batch_weights.sum().item()
    network.train()

    return correct / total

class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

class ParamDict(OrderedDict):
    """Code adapted from https://github.com/Alok/rl_implementations/tree/master/reptile.
    A dictionary where the values are Tensors, meant to represent weights of
    a model. This subclass lets you perform arithmetic on weights directly."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, *kwargs)

    def _prototype(self, other, op):
        if isinstance(other, Number):
            return ParamDict({k: op(v, other) for k, v in self.items()})
        elif isinstance(other, dict):
            return ParamDict({k: op(self[k], other[k]) for k in self})
        else:
            raise NotImplementedError

    def __add__(self, other):
        return self._prototype(other, operator.add)

    def __rmul__(self, other):
        return self._prototype(other, operator.mul)

    __mul__ = __rmul__

    def __neg__(self):
        return ParamDict({k: -v for k, v in self.items()})

    def __rsub__(self, other):
        # a- b := a + (-b)
        return self.__add__(other.__neg__())

    __sub__ = __rsub__

    def __truediv__(self, other):
        return self._prototype(other, operator.truediv)

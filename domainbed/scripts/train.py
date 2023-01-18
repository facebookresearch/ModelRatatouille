# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import math

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir, args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        out, in_ = misc.split_dataset(env,
            int(len(env)*args.holdout_fraction),
            misc.seed_hash(args.trial_seed, env_i))
        in_splits.append(in_)
        out_splits.append(out)
    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, env in enumerate(in_splits)
        if i not in args.test_envs]
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env in in_splits + out_splits]
    eval_loader_names = ['env{}_in'.format(i)
        for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
        for i in range(len(out_splits))]

    dict_featurizers_aux = {}
    if args.aux_dir not in ["", "none"]:
        if args.fusing_range >= 0:
            # interpolating multiple featurizers
            list_featurizers_aux = misc.get_list_featurizers_aux(args)
            list_kappas_aux = [math.exp(args.fusing_range * random.random()) for _ in list_featurizers_aux]
            dict_featurizers_aux = {
                featurizer_aux: kappa_aux / sum(list_kappas_aux)
                for featurizer_aux, kappa_aux in zip(list_featurizers_aux, list_kappas_aux)
                }
        else:
            # selecting only one single auxiliary featurizer
            dict_featurizers_aux = {misc.get_featurizer_aux(args): 1.}
        print(f"Dictionnary mapping featurizers to interpolating lambda: {dict_featurizers_aux}")

    algorithm = algorithms.get_algorithm_class(args.algorithm)(
            input_shape=dataset.input_shape,
            num_classes=dataset.num_classes,
            hparams=hparams,
            what_is_trainable=args.what_is_trainable,
            path_init=args.path_init,
            dict_featurizers_aux=dict_featurizers_aux)
    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env in in_splits])
    n_steps = dataset.N_STEPS

    def save_checkpoint(results=None, suffix="best"):
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        if results is not None:
            save_dict["results"] = results
        torch.save(save_dict, os.path.join(args.output_dir, "model_" + suffix + ".pkl"))
        torch.save(algorithm.network.state_dict(), os.path.join(args.output_dir, "network_" + suffix + ".pkl"))

    best_score = -float("inf")
    last_results_keys = None
    for step in range(0, n_steps):
        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device)) for x,y in next(train_minibatches_iterator)]
        step_vals = algorithm.update(minibatches_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        if (step % dataset.CHECKPOINT_FREQ == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            for name, loader, in zip(eval_loader_names, eval_loaders):
                acc = misc.accuracy(algorithm, loader, device)
                results[name+'_acc'] = acc

            results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            with open(os.path.join(args.output_dir, 'results.jsonl'), 'a') as f:
                f.write(json.dumps(results, sort_keys=True, default=misc.np_encoder) + "\n")

            ## DiWA ##
            current_score = misc.get_score(results, args.test_envs)
            if current_score > best_score:
                best_score = current_score
                print(f"Saving new best score at step: {step} at path: model_best.pkl")
                save_checkpoint(
                    results=json.dumps(results, sort_keys=True, default=misc.np_encoder),
                )
                algorithm.to(device)

            checkpoint_vals = collections.defaultdict(lambda: [])

    # saving the last featurizer's weights
    torch.save(algorithm.featurizer.state_dict(), os.path.join(args.output_dir, "featurizer_last.pkl"))
    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')
    algorithm.cpu()

def parse_args(raw_args=None):
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--test_envs', type=int, nargs='+', default=[])
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--holdout_fraction', type=float, default=0.2)

    # New args for ratatouille
    parser.add_argument('--what_is_trainable', type=str, default="all")
    parser.add_argument('--path_init', type=str, default="")
    parser.add_argument('--aux_dir', type=str, default="")
    parser.add_argument('--fusing_range', type=float, default=-1)
    args = parser.parse_args(raw_args)
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

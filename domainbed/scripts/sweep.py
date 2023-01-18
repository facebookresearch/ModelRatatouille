# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Run sweeps
"""

import argparse
import copy
import hashlib
import json
import os
import shutil
import gc
import torch
import numpy as np

from domainbed.lib import misc
from domainbed import command_launchers
from domainbed.scripts import train as train_script

import tqdm
import shlex

class Job:
    NOT_LAUNCHED = 'Not launched'
    INCOMPLETE = 'Incomplete'
    DONE = 'Done'

    def __init__(self, train_args, sweep_output_dir):
        args_str = json.dumps(train_args, sort_keys=True)

        args_hash = hashlib.md5(args_str.encode('utf-8')).hexdigest()
        self.output_dir = os.path.join(sweep_output_dir, args_hash)

        self.train_args = copy.deepcopy(train_args)
        self.train_args['output_dir'] = self.output_dir
        command = ['python', '-m', 'domainbed.scripts.train']
        for k, v in sorted(self.train_args.items()):
            if isinstance(v, list):
                v = ' '.join([str(v_) for v_ in v])
            elif isinstance(v, str):
                v = shlex.quote(v)
            command.append(f'--{k} {v}')
        self.command_str = ' '.join(command)

        if os.path.exists(os.path.join(self.output_dir, 'done')):
            self.state = Job.DONE
        elif os.path.exists(self.output_dir):
            self.state = Job.INCOMPLETE
        else:
            self.state = Job.NOT_LAUNCHED

    def __str__(self):
        job_info = (self.train_args['dataset'],
            self.train_args['algorithm'],
            self.train_args['test_envs'],
            self.train_args['hparams_seed'])
        return '{}: {} {}'.format(
            self.state,
            self.output_dir,
            job_info)

    @staticmethod
    def launch(jobs, launcher_fn):
        print('Launching...')
        jobs = jobs.copy()
        np.random.shuffle(jobs)
        print('Making job directories:')
        for job in tqdm.tqdm(jobs, leave=False):
            os.makedirs(job.output_dir, exist_ok=True)
        commands = [job.command_str for job in jobs]
        launcher_fn(commands)
        print(f'Launched {len(jobs)} jobs!')

    @staticmethod
    def delete(jobs):
        print('Deleting...')
        for job in jobs:
            shutil.rmtree(job.output_dir)
        print(f'Deleted {len(jobs)} jobs!')

def all_test_env_combinations(n):
    """
    For a dataset with n >= 3 envs, return all combinations of 1 and 2 test
    envs.
    """
    assert(n >= 3)
    for i in range(n):
        yield [i]
        for j in range(i+1, n):
            yield [i, j]

def make_args_lp(args):
    dict_args_lp = {}
    dict_args_lp['output_dir'] = args.output_dir_lp
    dict_args_lp['dataset'] = args.dataset
    dict_args_lp['algorithm'] = args.algorithm
    dict_args_lp['test_envs'] = args.test_env
    dict_args_lp['data_dir'] = args.data_dir
    dict_args_lp["what_is_trainable"] = "classifier"
    raw_args_lp = [
        item
        for key, value in dict_args_lp.items()
        for item in ["--" + key, str(value)]
    ]
    return raw_args_lp


def make_args_list(args):
    args_list = []
    for trial_seed in range(args.n_trials):
        for hparams_seed in range(args.n_hparams_from, args.n_hparams):
            train_args = {}
            train_args['trial_seed'] = trial_seed
            train_args['path_init'] = args.path_init
            train_args['dataset'] = args.dataset
            train_args['algorithm'] = args.algorithm
            train_args['test_envs'] = [args.test_env]
            train_args['hparams_seed'] = hparams_seed
            train_args['data_dir'] = args.data_dir
            train_args['aux_dir'] = args.aux_dir
            train_args['fusing_range'] = args.fusing_range

            train_args['seed'] = misc.seed_hash(
                args.dataset,
                args.algorithm,
                [args.test_env],
                hparams_seed,
                trial_seed)
            args_list.append(train_args)
    return args_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a sweep')
    parser.add_argument('command', choices=['launch', 'delete'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--test_env', type=int, required=True)
    parser.add_argument('--algorithm', default="ERM")
    parser.add_argument('--n_hparams_from', type=int, default=0)
    parser.add_argument('--n_hparams', type=int, default=20)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_trials', type=int, default=3)

    # New args for ratatouille
    parser.add_argument('--aux_dir', type=str, default="")
    parser.add_argument('--fusing_range', type=float, default=-1)
    parser.add_argument('--output_dir_lp', type=str, default=None)

    args = parser.parse_args()

    # 1. LP procedure to initialize a shared classifier
    if args.output_dir_lp is None:
        args.path_init = ""
    else:
        if os.path.exists(os.path.join(args.output_dir_lp, 'done')):
            print('LP already done.')
        # elif os.path.isdir(args.output_dir_lp):
        #     print("incomplete")
        else:
            print('Do LP.')
            raw_args_lp = make_args_lp(args)
            train_args_lp = train_script.parse_args(raw_args_lp)
            train_script.main(train_args_lp)
            print('Done LP.')
            gc.collect()
            torch.cuda.empty_cache()
            # be sure to free gpus memory
        args.path_init = os.path.join(args.output_dir_lp, "network_best.pkl")
        assert os.path.exists(args.path_init)
        print('Init path is ready.')

    # 2. Hyperparamerer sweep
    args_list = make_args_list(args)
    jobs = [Job(train_args, args.output_dir) for train_args in args_list]
    for job in jobs:
        print(job)
    print("{} jobs: {} done, {} incomplete, {} not launched.".format(
        len(jobs),
        len([j for j in jobs if j.state == Job.DONE]),
        len([j for j in jobs if j.state == Job.INCOMPLETE]),
        len([j for j in jobs if j.state == Job.NOT_LAUNCHED]))
    )

    if args.command == 'launch':
        to_launch = [j for j in jobs if j.state in [Job.NOT_LAUNCHED, Job.INCOMPLETE]]
        print(f'About to launch {len(to_launch)} jobs.')
        Job.launch(to_launch, launcher_fn = command_launchers.multi_gpu_launcher)

    elif args.command == 'delete':
        to_delete = [j for j in jobs if j.state == Job.INCOMPLETE]
        print(f'About to delete {len(to_delete)} jobs.')
        Job.delete(to_delete)

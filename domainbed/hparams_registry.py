# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from domainbed.lib import misc


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.
    _hparam('data_augmentation', True, lambda r: True)
    _hparam('resnet_dropout', 0., lambda r: r.choice([0., 0.1, 0.5]))

    ## Mild hyperparameter ranges as first defined in SWAD (https://arxiv.org/abs/2102.08604) and DiWA
    _hparam('lr', 5e-5, lambda r: r.choice([1e-5, 3e-5, 5e-5]))
    _hparam('weight_decay', 0, lambda r: r.choice([1e-4, 1e-6]))
    _hparam('batch_size', 32, lambda r: 32)
    _hparam('lp_steps', 200., lambda r: r.choice([200]))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from domainbed.lib import misc
from domainbed import networks

ALGORITHMS = [
    'ERM',
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, hparams):
        super(Algorithm, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)
    """

    def __init__(self, input_shape, num_classes, hparams, what_is_trainable=False, path_init="", dict_featurizers_aux={}):

        super(ERM, self).__init__(input_shape, num_classes, hparams)
        self._what_is_trainable = what_is_trainable
        self._dict_featurizers_aux = dict_featurizers_aux
        self._create_network()
        self._load_network(path_init)
        self.register_buffer('update_count', torch.tensor([0]))
        self._init_optimizer()

    def _create_network(self):
        self.featurizer = networks.Featurizer(self.input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs,
            self.num_classes)

        self.network = nn.Sequential(self.featurizer, self.classifier)

    def _load_network(self, path_init):
        if path_init:
            assert os.path.exists(path_init)
            self.network.load_state_dict(torch.load(path_init), strict=True)

        if self._dict_featurizers_aux:
            list_featurizers_aux = []
            list_lambdas_aux = []
            for path_featurizer_aux, lambda_aux in self._dict_featurizers_aux.items():
                featurizer_aux = networks.Featurizer(self.input_shape, self.hparams)
                if path_featurizer_aux != 'imagenet':
                    featurizer_aux.load_state_dict(torch.load(path_featurizer_aux), strict=True)
                list_featurizers_aux.append(featurizer_aux)
                list_lambdas_aux.append(lambda_aux)

            if len(list_featurizers_aux) == 1:
                wa_weights = {k:v for k, v in list_featurizers_aux[0].named_parameters()}
            else:
                # for fusing at initialization
                wa_weights = misc.get_name_waparameters(
                    list_featurizers_aux,
                    list_lambdas_aux)
            for name, param in self.featurizer.named_parameters():
                param.data = wa_weights[name]

    def _need_lp(self):
        return len([key for key in self._dict_featurizers_aux.keys() if key != "imagenet"])

    def _get_training_parameters(self):
        if self._need_lp():
            # apply another lp linear probe only when the featurizer is not transferred directly from ImageNet
            if self.update_count == self.hparams["lp_steps"]:
                print(f"Now back to update {self._what_is_trainable}")
                what_is_trainable = self._what_is_trainable
            else:
                assert self.update_count == 0
                what_is_trainable = "classifier"
        else:
            what_is_trainable = self._what_is_trainable

        if what_is_trainable in ["all"]:
            training_parameters = self.network.parameters()
        else:
            assert what_is_trainable in ["classifier"]
            training_parameters = self.classifier.parameters()
        return training_parameters

    def _init_optimizer(self):
        training_parameters = self._get_training_parameters()
        self.optimizer = torch.optim.Adam(
            training_parameters,
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )

    def update(self, minibatches):
        all_x = torch.cat([x for x,y in minibatches])
        all_y = torch.cat([y for x,y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self._need_lp() and self.update_count == self.hparams["lp_steps"]:
            self._init_optimizer()
        self.update_count += 1

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)

    def save_path_for_future_init(self, path_init):
        assert not os.path.exists(path_init), "The initialization has already been saved"
        torch.save(self.network.state_dict(), path_init)

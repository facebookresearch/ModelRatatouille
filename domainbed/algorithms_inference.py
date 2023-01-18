# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
from domainbed import algorithms

class ERM(algorithms.ERM):

    def __init__(self, input_shape, num_classes, hparams):
        algorithms.Algorithm.__init__(self, input_shape, num_classes, hparams)
        algorithms.ERM._create_network(self)

class WA(algorithms.ERM):

    def __init__(self, input_shape, num_classes):
        """
        """
        algorithms.Algorithm.__init__(self, input_shape, num_classes, hparams={})
        self.network_wa = None
        self.global_count = 0

    def add_weights(self, network):
        if self.network_wa is None:
            self.network_wa = copy.deepcopy(network)
        else:
            for param_q, param_k in zip(network.parameters(), self.network_wa.parameters()):
                param_k.data = (param_k.data * self.global_count + param_q.data) / (1. + self.global_count)
        self.global_count += 1

    def predict(self, x):
        return self.network_wa(x)

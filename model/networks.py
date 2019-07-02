"""Network models
Used as value function for molecular generation MDP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerNetwork(nn.Module):
    """Simple feed forward network"""

    def __init__(self, hparams):
        super(MultiLayerNetwork, self).__init__()
        self.hparams = hparams
        self.dense = nn.Sequential()

        hparams_layers = self.hparams['dense_layers']
        # The input length is the size of Morgan fingerprint plus steps
        input_size = self.hparams['fingerprint_length']
        output_size = self.hparams['num_bootstrap_heads']
        hparams_layers = [input_size + 1] + hparams_layers + [output_size]

        for i in range(1, len(hparams_layers)):
            self.dense.add_module('dense_%i' % i, nn.Linear(hparams_layers[i - 1], hparams_layers[i]))
            if i != len(hparams_layers) - 1:
                if self.hparams['batch_norm']:
                    self.dense.add_module('BN_%i' % i, nn.BatchNorm1d(1))
                self.dense.add_module('%s_%i' % (self.hparams['activation'], i), getattr(nn, self.hparams['activation'])())

    def forward(self, x):
        # x = self._get_input(x)
        x = self.dense(x)
        return x


if __name__ == '__main__':
    from utils.functions import get_hparams, get_fingerprint, get_fingerprint_with_stpes_left

    hparams = get_hparams('./configs/naive_dqn.json')
    net = MultiLayerNetwork(hparams)
    print(net)
    mol = 'CCOCC'
    x = torch.randn(5, 1, 2049)
    print(net(x))

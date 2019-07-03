"""Network models
Used as value function for molecular generation MDP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


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

    def get_action(self, state, observations, epsilon):

        if random.random() > epsilon:
            q_value = torch.squeeze(self.forward(state))
            action = observations[q_value.argmax().item()]

        else:
            action_space_n = len(observations)
            rand_num = random.randrange(action_space_n)
            action = observations[rand_num]

        return action


"""Molecule Embedding Networks"""


def get_fingerprint(smiles, hparams):
    """Get Morgan Fingerprint of a specific SMIELS string"""

    radius = hparams['fingerprint_radius']
    length = hparams['fingerprint_length']

    if smiles is None:
        return np.zeros((length, ))

    molecule = Chem.MolFromSmiles(smiles)

    if molecule is None:
        return np.zeros((length, ))

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, radius, length)

    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def mol2fp(smiles, step, hparams):
    if isinstance(smiles, str):
        smiles = [smiles]

    state_tensor = torch.Tensor(np.vstack([
        np.append(get_fingerprint(act, hparams), step)
        for act in smiles
    ]))
    state_tensor = torch.unsqueeze(state_tensor, 1)
    return state_tensor


if __name__ == '__main__':
    import json

    def get_hparams(path=None):
        hparams = {
            'atom_types': ['C', 'O', 'N'],
            'max_steps_per_episode': 40,
            'allow_removal': True,
            'allow_no_modification': True,
            'allow_bonds_between_rings': False,
            'allowed_ring_sizes': [3, 4, 5, 6],
            'replay_buffer_size': 1000000,
            'learning_rate': 1e-4,
            'learning_rate_decay_steps': 10000,
            'learning_rate_decay_rate': 0.8,
            'num_episodes': 5000,
            'batch_size': 64,
            'learning_frequency': 4,
            'update_frequency': 20,
            'grad_clipping': 10.0,
            'gamma': 0.9,
            'double_q': True,
            'num_bootstrap_heads': 12,
            'prioritized': False,
            'prioritized_alpha': 0.6,
            'prioritized_beta': 0.4,
            'prioritized_epsilon': 1e-6,
            'fingerprint_radius': 3,
            'fingerprint_length': 2048,
            'dense_layers': [1024, 512, 128, 32],
            'activation': 'relu',
            'optimizer': 'Adam',
            'batch_norm': False,
            'save_frequency': 1000,
            'max_num_checkpoints': 100,
            'discount_factor': 0.7
        }
        if path is not None:
            with open(path, 'r') as f:
                hparams.update(json.load(f))
        return hparams

    hparams = get_hparams('./naive_dqn.json')
    net = MultiLayerNetwork(hparams)
    print(net)

    obs = ['CCOCC', 'C']
    step = 3
    e = 1

    print(net.get_action(obs, step, e))
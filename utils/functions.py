import json
import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem


def get_hparams(*args):
    """Function to read hyper parameters"""
    # Default setting
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
        'activation': 'ReLU',
        'optimizer': 'Adam',
        'batch_norm': False,
        'save_frequency': 1000,
        'max_num_checkpoints': 100,
        'discount_factor': 0.7
    }
    with open(args[0], 'r') as f:
        hparams.update(json.load(f))
    return hparams


def get_fingerprint_with_stpes_left(smiles, steps_left, hparams):
    fingerprint = get_fingerprint(smiles, hparams)
    return np.append(fingerprint, steps_left)


def get_fingerprint(mol, hparams):
    """Get Morgan fingerprint for a specific molecule"""

    length = hparams['fingerprint_length']
    radius = hparams['fingerprint_radius']

    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)

    if mol is None:
        return np.zeros((length, ))

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr

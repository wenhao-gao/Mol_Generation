"""Run the DQN"""
import torch
import torch.optim as optim
from utils.functions import get_hparams
from environments.envs import OptLogPMolecule
from model.networks import MultiLayerNetwork
from model.DQN import DQLearning
from logger import Logger
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trigger the DQN training process.')
    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('--hparams', default='./configs/naive_dqn.json',
                        help='The JSON file define teh hyper parameters.')
    parser.add_argument('-g', '--gen_path', default='./gen_mol',
                        help='The file to store the generated molecules.')
    args = parser.parse_args()

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.hparams is not None:
        hparams = get_hparams(args.hparams)
    else:
        hparams = get_hparams()

    env = OptLogPMolecule(
        atom_types=set(hparams['atom_types']),
        allow_removal=hparams['allow_removal'],
        allow_no_modification=hparams['allow_no_modification'],
        allow_bonds_between_rings=hparams['allow_bonds_between_rings'],
        allowed_ring_sizes=set(hparams['allowed_ring_sizes']),
        max_steps=hparams['max_steps_per_episode']
    )

    model = MultiLayerNetwork(hparams)
    if args.parameters is not None:
        model.load_state_dict(torch.load(args.parameters, map_location=DEVICE))

    optimizer = optim.Adam(model.parameters())
    logger = Logger()

    dqn = DQLearning(
        q_fn=model,
        environment=env,
        optimizer=optimizer,
        logger=logger,
        hparams=hparams,
        gen_epsilon=0.1,
        gen_file='./mol_gen.csv',
        gen_num_episode=50
    )
    dqn.generation()

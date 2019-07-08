"""Run the DQN"""
import os
import argparse
import torch
import environments.envs as envs
from utils.functions import get_hparams
from model.networks import MultiLayerNetwork
from model.DQN import DQLearning
from tensorboardX import SummaryWriter


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

    env = envs.OptLogPMolecule(
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

    log_path = os.path.join(args.gen_path, args.task)
    writer = SummaryWriter(log_path)

    if not os.path.exists(args.gen_path):
        os.makedirs(args.gen_path)
    gen_file = os.path.join(args.gen_path, 'mol_gen.csv')

    dqn = DQLearning(
        q_fn=model,
        environment=env,
        writer=writer,
        hparams=hparams,
        gen_epsilon=hparams['gen_epsilon'],
        gen_file=gen_file,
        gen_num_episode=hparams['gen_number']
    )
    dqn.generation()

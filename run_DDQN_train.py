"""Run the DQN"""
import os
import torch
import torch.optim as optim
from utils.functions import get_hparams
from environments.envs import ScaleOptLogPMolecule
from model.networks import MultiLayerNetwork
from model.DQN import DQLearning
import argparse
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trigger the DQN training process.')

    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('-t', '--task', default='test',
                        help='The task name.')
    parser.add_argument('--hparams', default='./configs/naive_dqn.json',
                        help='The JSON file define teh hyper parameters.')

    args = parser.parse_args()

    if args.hparams is not None:
        hparams = get_hparams(args.hparams)
    else:
        hparams = get_hparams()

    env = ScaleOptLogPMolecule(
        atom_types=set(hparams['atom_types']),
        allow_removal=hparams['allow_removal'],
        allow_no_modification=hparams['allow_no_modification'],
        allow_bonds_between_rings=hparams['allow_bonds_between_rings'],
        allowed_ring_sizes=set(hparams['allowed_ring_sizes']),
        max_steps=hparams['max_steps_per_episode']
    )

    model = MultiLayerNetwork(hparams)
    if args.parameters is not None:
        model.load_state_dict(torch.load(args.parameters))

    optimizer = optim.Adam(
        params=model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    log_path = os.path.join('./checkpoints/', args.task)
    writer = SummaryWriter(log_path)

    dqn = DQLearning(
        task=args.task,
        q_fn=model,
        environment=env,
        optimizer=optimizer,
        writer=writer,
        hparams=hparams,
        double=True,
        model_path=log_path,
        gen_epsilon=0.01,
        gen_file='./mol_gen.csv',
        gen_num_episode=100
    )

    dqn.train()

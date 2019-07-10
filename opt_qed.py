"""Run the DQN"""
import os
import argparse
import torch
import torch.optim as optim
import environments.envs as envs
from utils.functions import get_hparams
from model.networks import MultiLayerNetwork
from model.DQN import DQLearning
from tensorboardX import SummaryWriter


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Trigger the DQN training process.')

    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('-t', '--task', default='test',
                        help='The task name.')
    parser.add_argument('--hparams', default='./configs/test.json',
                        help='The JSON file define teh hyper parameters.')
    parser.add_argument('-i', '--init_mol', default='C',
                        help='The initial molecule to start with.')
    parser.add_argument('-o', '--out_path', default='./checkpoints/',
                        help='path to put output files.')

    args = parser.parse_args()

    if args.hparams is not None:
        hparams = get_hparams(args.hparams)
    else:
        hparams = get_hparams()

    env = envs.OptQEDMolecule(
        discount_factor=hparams['discount_factor'],
        init_mol=args.init_mol,
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
        lr=hparams['learning_rate'],
        betas=(hparams['adam_beta_1'], hparams['adam_beta_2']),
        eps=1e-08,
        weight_decay=0,
        amsgrad=False
    )

    lr_schedule = optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer,
        gamma=hparams['learning_rate_decay_rate']
    )

    log_path = os.path.join(args.out_path, args.task)
    writer = SummaryWriter(log_path)

    dqn = DQLearning(
        task=args.task,
        q_fn=model,
        environment=env,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
        writer=writer,
        hparams=hparams,
        model_path=log_path,
    )

    dqn.train()

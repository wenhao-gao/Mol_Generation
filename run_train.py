import torch.optim as optim
from utils.functions import get_hparams
from environments.envs import OptLogPMolecule
from model.networks import MultiLayerNetwork, mol2fp
from model.DQN import DQLearning
from logger import Logger


if __name__ == '__main__':
    hparams = get_hparams('./configs/naive_dqn.json')

    env = OptLogPMolecule(
        atom_types=set(hparams['atom_types']),
        allow_removal=hparams['allow_removal'],
        allow_no_modification=hparams['allow_no_modification'],
        allow_bonds_between_rings=hparams['allow_bonds_between_rings'],
        allowed_ring_sizes=set(hparams['allowed_ring_sizes']),
        max_steps=hparams['max_steps_per_episode']
    )

    net = MultiLayerNetwork(hparams)
    optimizer = optim.Adam(net.parameters())
    logger = Logger()

    dqn = DQLearning(
        q_fn=net,
        environment=env,
        optimizer=optimizer,
        logger=logger,
        hparams=hparams
    )
    dqn.train()

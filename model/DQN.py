"""The Deep Q Learning Process"""

import numpy as np
import time
import os
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import matplotlib.pyplot as plt
from rdkit.Chem import Draw
from rdkit import Chem

from environments.envs import OptLogPMolecule
from model.networks import MultiLayerNetwork, mol2fp
from utils import replay_buffer
from utils import schedules
from utils.functions import get_hparams


class DQLearning:

    def __init__(self,
                 task,
                 q_fn,
                 environment,
                 optimizer,
                 hparams,
                 writer,
                 keep=10,
                 double=True,
                 model_path='./checkpoints',
                 gen_epsilon=0.01,
                 gen_file='./mol_gen.csv',
                 gen_num_episode=100):

        self.task = task
        self.hparams = hparams
        self.env = environment
        self.optimizer = optimizer
        self.writer = writer
        self.model_path = model_path
        self.double = double

        if not os.path.exists(self.model_path):
            os.makedirs(model_path)

        self.num_episodes = hparams['num_episodes']
        self.batch_size = hparams['batch_size']
        self.gamma = hparams['gamma']
        self.prioritized = hparams['prioritized']
        self.save_frequency = hparams['save_frequency']
        self.max_steps_per_episode = hparams['max_steps_per_episode']
        self.learning_rate = hparams['learning_rate']
        self.learning_frequency = hparams['learning_frequency']
        self.learning_rate_decay_steps = hparams['learning_rate_decay_steps']
        self.learning_rate_decay_rate = hparams['learning_rate_decay_rate']
        self.update_frequency = hparams['update_frequency']
        self.replay_buffer_size = hparams['replay_buffer_size']
        self.prioritized_alpha = hparams['prioritized_alpha']
        self.prioritized_beta = hparams['prioritized_beta']
        self.prioritized_epsilon = hparams['prioritized_epsilon']

        self.losses = []
        self.all_rewards = []
        self.memory = None
        self.beta_schedule = None
        self.smiles = []
        self.keep_criterion = -99999.9
        self.keep = keep

        # generation options
        self.gen_epsilon = gen_epsilon
        self.gen_file = gen_file
        self.gen_num_episode = gen_num_episode

        # epsilon-greedy exploration
        self.exploration = schedules.PiecewiseSchedule(
            [(0, 1.0), (int(self.num_episodes / 2), 0.1), (self.num_episodes, 0.01)],
            outside_value=0.01
        )

        self.DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device: {self.DEVICE}')

        self.q_fn = q_fn.to(self.DEVICE)
        if self.double:
            self.q_fn_target = copy.deepcopy(self.q_fn)

    def train(self):

        global_step = 0

        if self.prioritized:
            self.memory = replay_buffer.PrioritizedReplayBuffer(self.replay_buffer_size, self.prioritized_alpha)
            self.beta_schedule = schedules.LinearSchedule(self.num_episodes,
                                                          initial_p=self.prioritized_beta, final_p=0)
        else:
            self.memory = replay_buffer.ReplayBuffer(self.replay_buffer_size)
            self.beta_schedule = None

        for episode in range(1, self.num_episodes + 1):

            global_step = self._episode(
                episode,
                global_step
            )

            # Save checkpoint
            if episode % self.save_frequency == 0:
                model_name = 'dqn_checkpoint_' + str(episode) + '.pth'
                torch.save(self.q_fn.state_dict(), os.path.join(self.model_path, model_name))

    def _episode(self,
                 episode,
                 global_step):

        episode_start_time = time.time()
        epsilon = self.exploration.value(episode)

        state_mol, state_step = self.env.reset()
        state = mol2fp(state_mol, state_step, self.hparams)

        for step in range(self.max_steps_per_episode):

            state, state_mol, state_step, reward, done = self._step(
                state,
                state_step,
                epsilon
            )

            if done:

                print('Episode %d/%d took %gs' % (episode, self.num_episodes, time.time() - episode_start_time))
                print('SMIELS: %s' % state_mol)
                print('The reward is: %s\n' % str(reward))

                self.all_rewards.append(reward)

                # Keep track the result
                if reward > self.keep_criterion:
                    self.update_tracking(state_mol, reward, global_step)

                # Log result
                self.writer.add_scalar('reward', reward, global_step)

            if len(self.memory) > self.batch_size and (global_step % self.learning_frequency == 0):
                td_error = self._compute_td_loss(self.batch_size)
                self.losses.append(td_error)
                print('Current TD error: %.4f' % np.mean(np.abs(td_error)))
                # Log result
                self.writer.add_scalar('td_error', td_error, global_step)

                if self.double:
                    if global_step % self.learning_frequency == 0:
                        self.q_fn_target.load_state_dict(self.q_fn.state_dict())

            global_step += 1

        return global_step

    def _step(self,
              state,
              state_step,
              epsilon,
              gen=False):

        # Get valid actions
        observations = list(self.env.get_valid_actions())

        # State Embedding
        observation_tensor = mol2fp(observations, state_step + 1, self.hparams).to(self.DEVICE)
        action = self.q_fn.get_action(observation_tensor, observations, epsilon)

        next_state_mol, next_state_step, reward, done = self.env.step(action)
        next_state = mol2fp(next_state_mol, next_state_step, self.hparams)

        if not gen:
            self.memory.add(
                obs_t=state,
                action=0,
                reward=reward,
                obs_tp1=next_state,
                done=float(done)
            )

        return next_state, next_state_mol, next_state_step, reward, done

    def _compute_td_loss(self, batch_size):

        if self.prioritized:
            state, _, reward, next_state, done, weight, indices = \
                self.memory.sample(batch_size, beta=self.beta_schedule)
        else:
            state, _, reward, next_state, done = self.memory.sample(batch_size)
            weight = np.ones(reward.shape)
            indices = 0

        state = Variable(torch.FloatTensor(np.float32(state))).squeeze(1).to(self.DEVICE)
        next_state = Variable(torch.FloatTensor(np.float32(next_state))).squeeze(1).to(self.DEVICE)
        reward = Variable(torch.FloatTensor(reward)).to(self.DEVICE)
        done = Variable(torch.FloatTensor(done)).to(self.DEVICE)

        q_value = self.q_fn(state).squeeze()

        if self.double:
            next_q_value = self.q_fn_target(next_state)
        else:
            next_q_value = self.q_fn(next_state)

        # q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        # next_q_value = next_q_values.max(1)[0]
        td_target = reward + self.gamma * next_q_value.squeeze() * (1 - done)
        td_target = Variable(td_target.data)

        td_error = (q_value - td_target).pow(2).mean()

        loss = F.smooth_l1_loss(q_value, td_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.prioritized:
            self.memory.update_priorities(indices, np.abs(np.squeeze(td_error) + self.prioritized_epsilon).tolist())

        return td_error.data.item()

    def generation(self):
        with open(self.gen_file, 'wt') as f:
            print('SMILES,reward', file=f)
            for episode in range(1, self.gen_num_episode + 1):

                episode_start_time = time.time()

                state_mol, state_step = self.env.reset()
                state = mol2fp(state_mol, state_step, self.hparams)

                for step in range(self.max_steps_per_episode):

                    state, state_mol, state_step, reward, done = self._step(
                        state,
                        state_step,
                        self.gen_epsilon,
                        gen=True
                    )

                    if done:
                        print('Episode %d/%d took %gs' % (episode, self.gen_num_episode, time.time() - episode_start_time))
                        print('SMIELS: %s' % state_mol)
                        print('The reward is: %s\n' % str(reward))

                        print(str(state_mol) + ',' + str(reward), file=f)

    def update_tracking(self, mol, reward, step):

        if len(self.smiles) == self.keep:
            del self.smiles[0]

        if len(self.smiles) == 0:
            self.smiles = [(mol, reward)]
        else:
            for i, sample in enumerate(self.smiles):
                smiles, r = sample
                if reward > r:
                    self.smiles = self.smiles[:i] + [(mol, reward)] + self.smiles[i:]
                    break

        self.keep_criterion = self.smiles[-1][1]

        smiles_ = []
        r_ = []
        for sample in self.smiles:
            smiles, r = sample
            smiles_.append(Chem.MolFromSmiles(smiles))
            r_.append(str(r))
            self.writer.add_text('SMILES reward', smiles + ' Reward: ' + str(r), step)

        # img = Draw.MolsToGridImage(smiles_, molsPerRow=5, subImgSize=(200, 200), legends=r_)
        # img = np.array(img)
        # self.writer.add_figure('Molecules generated', img, step)

        return None


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

    dqn = DQLearning(
        q_fn=net,
        environment=env,
        optimizer=optimizer,
        hparams=hparams
    )
    dqn.train()

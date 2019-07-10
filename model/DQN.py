"""The Deep Q Learning Process"""

import numpy as np
import time
import os
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
                 hparams,
                 writer,
                 optimizer=None,
                 lr_schedule=None,
                 keep=5,
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
        self.lr_schedule = lr_schedule
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

        if not os.path.exists(self.model_path):
            os.makedirs(model_path)

        self.double = hparams['double_q']
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
        self.grad_clipping = hparams['grad_clipping']
        self.num_bootstrap_heads = hparams['num_bootstrap_heads']

        # epsilon-greedy exploration schedule
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
        head = np.random.randint(self.num_bootstrap_heads)

        for step in range(self.max_steps_per_episode):

            state, state_mol, state_step, reward, done = self._step(
                state_step,
                epsilon,
                head
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

            if (episode > min(50, self.num_episodes / 10)) and (global_step % self.learning_frequency == 0):
                if (global_step % self.learning_rate_decay_steps == 0) and (self.lr_schedule is not None):
                    self.lr_schedule.step()
                td_error = self._compute_td_loss(self.batch_size, episode)
                self.losses.append(td_error)
                print('Current TD error: %.4f' % np.mean(np.abs(td_error)))
                # Log result
                self.writer.add_scalar('td_error', td_error, global_step)

                if self.double and (episode % self.update_frequency == 0):
                    self.q_fn_target.load_state_dict(self.q_fn.state_dict())

            global_step += 1

        return global_step

    def _step(self,
              state_step,
              epsilon,
              head,
              gen=False):

        # Get valid actions
        observations = list(self.env.get_valid_actions())

        # State Embedding
        observation_tensor = mol2fp(observations, state_step, self.hparams).to(self.DEVICE)
        action = self.q_fn.get_action(observation_tensor, observations, epsilon, head, self.DEVICE)

        state_tensor = mol2fp(action, state_step + 1, self.hparams)

        next_state_mol, next_state_step, reward, done = self.env.step(action)
        next_state = mol2fp(next_state_mol, next_state_step, self.hparams).to(self.DEVICE)

        next_observations = list(self.env.get_valid_actions())
        next_observation_tensor = mol2fp(next_observations, next_state_step, self.hparams)

        if not gen:
            self.memory.add(
                obs_t=state_tensor.numpy(),
                action=0,
                reward=reward,
                obs_tp1=next_observation_tensor.numpy(),
                done=float(done)
            )

        return next_state, next_state_mol, next_state_step, reward, done

    def _compute_td_loss(self, batch_size, episode):

        if self.prioritized:
            state, _, reward, next_states, done, weight, indices = \
                self.memory.sample(batch_size, beta=self.beta_schedule.value(episode))
        else:
            state, _, reward, next_states, done = self.memory.sample(batch_size)
            weight = np.ones(reward.shape)
            indices = 0

        state = Variable(torch.FloatTensor(np.float32(state))).squeeze(1).to(self.DEVICE)
        next_states = [
            Variable(torch.FloatTensor(np.float32(next_state))).to(self.DEVICE)
            for next_state in next_states
        ]
        reward = Variable(torch.FloatTensor(reward)).to(self.DEVICE)
        done = Variable(torch.FloatTensor(done)).to(self.DEVICE)

        q_value = self.q_fn(state).squeeze()  # tensor batch_size*num_head

        q_tp1_online = [self.q_fn(state).squeeze() for state in next_states]  # batch list of num_observation*num_heads

        if self.double:

            q_tp1 = [self.q_fn_target(state).squeeze() for state in next_states]  # batch list of num_observation*num_heads

            q_tp1_online_idx = [
                torch.stack(
                    [torch.argmax(q, dim=0), torch.range(0, self.num_bootstrap_heads - 1, dtype=torch.int64)],
                    dim=1
                ) for q in q_tp1_online
            ]

            next_q_value = torch.stack(
                [q[idx[:, 0], idx[:, 1]] for q, idx in zip(q_tp1, q_tp1_online_idx)],
                dim=0
            )

        else:
            next_q_value = torch.stack(
                [q.gather(0, torch.max(q, 0)[1].unsqueeze(0)) for q in q_tp1_online],
                dim=0
            )

        done_mask = 1 - done
        masked_next_q = next_q_value * done_mask.unsqueeze(1)
        td_target = Variable(reward.unsqueeze(1) + self.gamma * masked_next_q)
        td_error = (q_value - td_target).pow(2)
        td_error = td_error.mean()

        loss = F.smooth_l1_loss(q_value, td_target, reduction='none')

        if self.num_bootstrap_heads > 1:
            head_mask = torch.Tensor(np.random.binomial(1, 0.6, self.num_bootstrap_heads))
            loss = loss * head_mask
            loss = loss.mean(1)

        prios = loss.data + self.prioritized_epsilon

        loss = loss.mul(torch.FloatTensor(weight)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.q_fn.parameters(), self.grad_clipping)
        self.optimizer.step()

        if self.prioritized:
            self.memory.update_priorities(
                indices, prios
            )

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

        add = False
        for i, sample in enumerate(self.smiles):
            smi, rwd = sample
            if reward > rwd:
                self.smiles = self.smiles[:i] + [(mol, reward)] + self.smiles[i:]
                add = True
                break
        if not add:
            self.smiles.append((mol, reward))

        if len(self.smiles) > self.keep:
            self.smiles = self.smiles[:self.keep]

        self.keep_criterion = self.smiles[-1][1]

        for sample in self.smiles:
            smiles, r = sample
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

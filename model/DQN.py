"""The Deep Q Learning Process"""

import numpy as np
import math, random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from utils.functions import get_fingerprint


class DQLearning:

    def __init__(self,
                 hparams,
                 q_fn,
                 environment,
                 epsilon=0.2):

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'Device: {self.device}')
        self.hparams = hparams
        self.lr = self.hparams['learning_rate']
        self.optimizer = getattr(optim, self.hparams['optimizer'])(self.q_fn.parameters(), self.lr)
        self.q_fn = q_fn.to(self.device)
        self.q_fn_q = q_fn.to(self.device)
        self.env = environment
        self.epsilon = epsilon
        self.gamma = hparams['gamma']

    def get_action(self,
                   observations,
                   stochastic=True,
                   update_epsilon=None):
        """Function that choose an action given the observations
        Argument
        ------------
            - observations. np.array. shape = [num_actions, fingerprint_length].
                The next states.
            - stochastic. Boolean.
                If set to True, all the actions are always deterministic.
            - head. Int.
                The number of bootstrap heads.
            - update_epsilon. Float or None.
                Update the epsilon to a new value.
        Return
            - action.
        """
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        if stochastic and np.random.uniform() < self.epsilon:
            return np.random.randint(0, observations.shape[0])
        else:
            return self._run_action(observations, self.hparams['num_bootstrap_head'])

    def _run_action(self, observations, head):
        """Run the network to get a result
        Then pick an action based on policy"""
        observations = self._get_tensor(observations)
        self.q_fn.eval()
        q_value = self.q_fn(observations)
        # TODO enable Bootstrap
        return int(q_value.argmax())

    def _get_tensor(self, observations):
        """Transfer a Mol observations into torch.Tensor"""
        observations = np.vstack([
            np.append(get_fingerprint(act, self.hparams), self.env.num_steps_taken)
            for act in observations
        ])
        return torch.Tensor(observations).to(device)






def compute_td_loss(batch_size):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True)
    action     = Variable(torch.LongTensor(action))
    reward     = Variable(torch.FloatTensor(reward))
    done       = Variable(torch.FloatTensor(done))

    q_values      = model(state)
    next_q_values = model(next_state)

    q_value          = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value     = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value * (1 - done)

    loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss

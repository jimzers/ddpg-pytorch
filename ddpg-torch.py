import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        # lets you call on the class
        # noise = OUActionNoise()
        # noise()
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * \
                self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_mem = np.zeros((self.mem_size, *input_shape))
        self.new_state_mem = np.zeros((self.mem_size, *input_shape))
        self.action_mem = np.zeros((self.mem_size, n_actions))
        self.reward_mem = np.zeros(self.mem_size)
        self.terminal_mem = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, new_state, done):
        idx = self.mem_cntr % self.mem_size # wrap around
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_memory[idx] = reward
        self.new_state_mem[idx] = new_state
        self.terminal_mem[idx] = 1 - done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size) # get batch indices

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states  = self.new_state_mem[batch]
        done = self.self.terminal_mem[batch]

        return states, actions, rewards, new_states, terminal


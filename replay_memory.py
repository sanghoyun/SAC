import random
import numpy as np
import torch
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))

class ReplayMemory():
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.device = device
        self.buffer = []

    def push(self, *data):
        self.buffer.append(Transition(*data))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size, state_dim, action_dim):
        batch = random.sample(self.buffer, batch_size)
        batch = Transition(*zip(*batch))

        states = torch.cat(batch.state).view(batch_size, state_dim).to(self.device)
        rewards = torch.cat(batch.reward).view(batch_size, 1).to(self.device)
        dones = torch.cat(batch.done).view(batch_size, 1).to(self.device)
        actions = torch.cat(batch.action).view(-1, action_dim).to(self.device)
        next_states = torch.cat(batch.next_state).view(batch_size, state_dim).to(self.device)

        return states, actions, rewards, dones, next_states

    def __len__(self):
        print("Length of Memory : {}".format(len(self.buffer)))
        return len(self.buffer)
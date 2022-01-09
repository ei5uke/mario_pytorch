"""
models.py

This file holds all the current and future models used in the main.py file. Currently, I will have added a CNN model that follows
the DQN policy, but I hope to add PPO, A2C, and other algorithms in the future.
"""

# The general imports from torch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Aside from numpy which is essentially a necessity, the rest are including as neat tricks to help fluidity
import math                             # Used to calculate the output of the CNN -> specifically math.floor()
import numpy as np
from typing import Any                  # An "any" object
from dataclasses import dataclass       # Nice way to impose a specific structure when passing transitions
from random import sample, random       # Easy way to sample from the Memory replay buffer

import wandb                            # Weights and Biases; helps to visualize the models
# import ipdb; ipdb.set_trace()         # Debugging tool;

# Forcing a specific style of passing arguments
@dataclass
class Transition:
    state: Any
    action: int
    reward: float
    state_: Any
    done: bool

# This is our CNN model, which I'm just calling Model for convenience
class Model(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(Model, self).__init__()
        self.input_shape = input_shape                        # ideally this is (N,C,H,W)
        self.num_actions = num_actions                        # this is just the # of actions we can take

        # This calculates the output h and w after each convolution, we can tinker with the numbers as long as we also change them in the convolution.
        # In the PyTorch Mario tutorial, they had this number "3136" that came out of nowhere and I was so confused; this is how one would get that.
        conv_size = self.calculate_output(84, 8, 0, 4)
        conv_size = self.calculate_output(conv_size, 4, 0, 2)
        conv_size = self.calculate_output(conv_size, 3, 0, 1)
        conv_size = conv_size * conv_size * 64

        # Our main Neural Net that follows DeepMind's DQN paper.
        # We make sure to pass in the number of channels and output the number of actions our agent can take.
        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape[1], 32, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.opt = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

    def forward(self, x):
        return self.net(x)

    # This follows equation given in the PyTorch Conv2d documentation under the shape section.
    def calculate_output(self, n, f, p, s):
        return math.floor(((n+2*p-(f-1)-1)/s + 1))
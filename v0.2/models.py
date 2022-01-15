"""
models.py

This file holds all the current and future models used in the main.py file. Currently, I will have added a CNN model that follows the DQN 
policy, but I hope to add PPO, A2C, and other algorithms in the future.
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
from random import sample, random       # Easy way to sample from the Memory replay buffer

import wandb                            # Weights and Biases; helps to visualize the models
# import ipdb; ipdb.set_trace()         # Debugging tool;

# This is our CNN model, which I'm just calling Model for convenience
class Model(nn.Module):
    def __init__(self, input_shape, num_actions, lr):
        super(Model, self).__init__()
        self.input_shape = input_shape                        # ideally this is (N,C,H,W)
        self.num_actions = num_actions                        # this is just the # of actions we can take

        # This calculates the output h and w after each convolution, we can tinker with the numbers as long as we also change them in the convolution.
        # In the PyTorch Mario tutorial, they had this number "3136" that came out of nowhere and I was so confused; this is how one would get that.
        conv_size = self.calculate_output(84, 8, 0, 4)
        conv_size = self.calculate_output(conv_size, 4, 0, 2)
        conv_size = self.calculate_output(conv_size, 3, 0, 1)
        #conv_size = self.calculate_output(conv_size, 7, 0, 1)
        conv_size = conv_size * conv_size * 64

        # Our main Neural Net that follows DeepMind's DQN paper.
        # We make sure to pass in the number of channels and output the number of actions our agent can take.
        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=(8,8), stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=(4,4), stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3,3), stride=1),
            nn.ReLU(),
            # the next two lines are experimental
            #nn.Conv2d(64, 1024, kernel_size=(7,7), stride=1),
            #nn.ReLU(),
            nn.Flatten(),
            nn.Linear(conv_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
            
        )
        self.opt = optim.Adam(self.net.parameters(), lr=lr)

    def forward(self, x):
        return self.net(x/255.0) # normalizing values

    # This follows equation given in the PyTorch Conv2d documentation under the shape section.
    def calculate_output(self, n, f, p, s):
        return math.floor(((n+2*p-(f-1)-1)/s + 1))

def train(online, target, transitions, num_actions, device, gamma=0.99):
    # Get all necessary information, make them into a PyTorch tensor and send to the device    
    # currently i'm trying out a combination of Deep RL hands-on's method and brthor's
    # we first convert to nparrays because it's faster to change into tensors from nparrays compared to pyarrays
    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    states_ = np.asarray([t[4] for t in transitions])

    # we then change the nparrays to tensors to enable our device to compute
    # actions, rewards, dones are unsqueezed to enable hadamard products and addition of same dimension tensors
    states = torch.as_tensor(states, dtype=torch.float32).to(device)
    actions = torch.as_tensor(actions, dtype=torch.int64).to(device) # readd .unsqueeze(0) before to(device) if it doesn't work
    rewards = torch.as_tensor(rewards, dtype=torch.float32).to(device) # here and 
    dones = torch.as_tensor(dones, dtype=torch.float32).to(device) # here also
    states_ = torch.as_tensor(states_, dtype=torch.float32).to(device)

    # Essentially from here on, we want to calculate squared error of our q_values and our optimal q_values
    # Disable gradient calculation because we aren't going to update it later -> makes it more memory efficient
    with torch.no_grad():
        # DQN
        #qvals_opt = target(states_).max(dim=1, keepdim=True)[0] # Calculate the "Bellman" equation using the target net

        # DDQN
        qvals_target = online(states_) # get next q values, get best action essentially
        best_action = torch.argmax(qvals_target, axis=1)
        qvals_next = target(states_)[np.arange(0, 32), best_action]

    # in PyTorch, multiplying tensors with * has in-place value multiplcation aka hadamard product

    # DQN y value
    # y = rewards + gamma * (1 - dones) * qvals_opt # Get our y value
    #qvals = torch.gather(input=qvals, dim=1, index=actions) # get our qvalues based on our actions

    # DDQN y value
    y = rewards + gamma * (1 - dones) * qvals_next
    qvals = online(states)[np.arange(0, 32), actions] # get our current qvalues passing them to the online model

    #loss = nn.MSELoss()(qvals, y) # same as loss = ((qvals - y) ** 2).mean()
    loss = nn.HuberLoss()(qvals, y)
    #loss = nn.SmoothL1Loss()(qvals, y)
    online.opt.zero_grad()
    loss.backward()
    online.opt.step()
    return loss

class Memory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, transition):
        self.buffer[self.idx % self.buffer_size] = transition
        self.idx += 1

    def sample(self, batch_size):
        assert batch_size < min(self.idx, self.buffer_size), "Has to sample less than number of memories"
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], batch_size)
        return sample(self.buffer, batch_size)
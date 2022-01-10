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

def train(online, target, transitions, num_actions, device, gamma=0.99):
    # Get all necessary information, make them into a PyTorch tensor and send to the device    

    # this follows Jack of Some's tutorial
    # states = torch.stack([torch.Tensor(s.state) for s in transition]).to(device)
    # rewards = torch.stack([torch.Tensor([s.reward]) for s in transition]).to(device)
    # dones = torch.stack([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in transition]).to(device)
    # states_ = torch.stack([torch.Tensor(s.state_) for s in transition]).to(device)
    # actions = [s.action for s in transition] # exception because actions are just ints and don't have to be tensors

    # i'm trying out a combination of Deep RL hands-on's method and brthor's
    states = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    states_ = np.asarray([t[4] for t in transitions])

    states = torch.as_tensor(states, dtype=torch.float32).to(device)
    actions = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1).to(device)
    rewards = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1).to(device)
    dones = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1).to(device)
    states_ = torch.as_tensor(states_, dtype=torch.float32).to(device)

    # Essentially from here we want to calculate squared error of our q_values and our optimal q_values
    # Disable gradient calculation because we aren't going to update it later -> makes it more memory efficient
    with torch.no_grad():
        qvals_opt = target(states_).max(dim=1, keepdim=True)[0] # Calculate the "Bellman" equation using the target net

    y = rewards + gamma * (1 - dones) * qvals_opt # Get our y value
    online.opt.zero_grad() # Clear our data to not use past gradients; unsure if i should place this here or later on
    qvals = online(states) # get our current qvalues 

    # clipping the error term between -1 and 1 inclusive; this is based off of the paper but it's not used much in other tutorials idk why
    # also, while the paper uses this clipping method, other tutorials use huber loss or smoothl1 but i'm just strictly following the paper for today
    #loss = qvals - y 
    #loss = loss if -1<=loss<=1 else (-1 if loss < -1 else 1)
    loss = nn.MSELoss()(qvals, y) # same as loss = ((qvals - y) ** 2).mean()

    loss.backward()
    model.opt.step()
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
        assert num_samples < min(self.idx, self.buffer_size), "Has to sample less than number of memories"
        if self.idx < self.buffer_size:
            return sample(self.buffer[:self.idx], batch_size)
        return sample(self.buffer, batch_size)
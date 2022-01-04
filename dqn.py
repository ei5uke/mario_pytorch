import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy

# the DQN algorithm that the agent class will wrap around
class DQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(DQN, self).__init__()
        c, h, w = inputs
        self.net = nn.Sequential( #modules will be added in the order passed to the constructor; more concise
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4), #kernel_size=sidelength of kernel cuz it's a square
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), #stride is how far kernel is moved every step
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, outputs)
        )

    def forward(self, input):
        return self.net(input)
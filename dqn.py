import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy

class DQN(nn.Module):
    def __init__(self, channels, output_dim): #in_channels just means number of channels aka depth
        super(DQN, self).__init__()
        self.pi = nn.Sequential( #modules will be added in the order passed to the constructor; more concise
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4), #kernel_size=sidelength of kernel cuz it's a square
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), #stride is how far kernel is moved every step
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim) # output_dim = action_space
        )
        self.target = copy.deepcopy(self.pi)
        for p in self.target.parameters():
            p.requires_grad = False
        #self.target.eval()
        #self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        #self.to(self.device)

    def forward(self, input, model):
        if model == 'pi':
            return self.pi(input)
        return self.target(input)
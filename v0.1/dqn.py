import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import copy

class DQN(nn.Module):
    def __init__(self, env): #in_channels just means number of channels aka depth
        super(DQN, self).__init__()
        self.net = nn.Sequential( #modules will be added in the order passed to the constructor; more concise
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4), #kernel_size=sidelength of kernel cuz it's a square
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2), #stride is how far kernel is moved every step
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n) # output_dim = action_space
        )

    def forward(self, input):
        return self.net(input)

    def act(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32)
        q_values = self(state_t.unsqueeze(0))

        max_q_index = torch.argmax(q_values, dim=1)[0]
        action = max_q_index.detach().item()

        return action
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from dqn import DQN
from collections import deque

class Agent():
    def __init__(self, channels, action_dim, device):
        self.channels = channels         # the dimension of the state space
        self.action_dim = action_dim       # the dimension of the action space
        self.device = device

        self.gamma = 0.99                  # discount rate
        self.epsilon = 1.0                 # exploration rate
        self.eps_end = 0.1                 # minimum exploration rate
        self.eps_dec = 0.99999975          # exploration decay rate

        self.batch_size = 32               # batch size 
        self.curr_mem = 0                  # current memory index
        self.curr_step = 0                 # agent's current step in the environment
        self.memory = deque(maxlen=100000) # deque of all the memories

        self.net = DQN(self.channels, self.action_dim).float()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00025) # change lr
        self.loss = nn.SmoothL1Loss() # CHANGE IF I CAN FIND SOMETHING BETTER

        # not sure what this does yet
        self.min_experience = 1e4
        self.learn_every = 3
        self.sync_every = 1e4
    
    def act(self, state):
        if np.random.random() > self.epsilon:       # exploit action
            state = torch.tensor([state]).to(self.net.device) 
            action_values = self.net(state.unsqueeze(0), model='pi')
            action = torch.argmax(action_values, axis=1).item()
        else:                                       # exploration action
            action = np.random.choice(self.action_dim)

        # update exploration rate
        self.epsilon *= self.eps_dec
        self.epsilon = max(self.epsilon, self.eps_end)

        self.curr_step += 1
        return action

    def cache(self, state, state_, action, reward, done):
        # change everything to tensors
        # state = torch.tensor([state])
        # state_ = torch.tensor([state_])
        # action = torch.tensor([action]) # do we really need to wrap with brackets
        # reward = torch.tensor([reward])
        # done = torch.tensor([done])

        # add to our memory
        experience = (state, state_, action, reward, done)
        if len(self.memory) < 100000:
            self.memory.append(experience)
        else:
            self.memory[self.curr_mem % 100000] = experience
        self.curr_mem += 1

    def sample(self):
        return random.sample(self.memory, batch_size)

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()
        # if self.curr_step % self.save_every == 0:
        #     self.save()
        if self.curr_step < self.min_experience:
            return None, None
        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.sample()

        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)

        loss = self.update_Q_pi(td_est, td_tgt)

        return (td_est.mean().item(), loss)


    def td_estimate(self, state, action):
        current_Q = self.net(state, model='pi')[
            np.arange(0, self.batch_size), action
        ]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state_Q = self.net(next_state, model="pi")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_pi(self, td_estimate, td_target):
        loss = self.loss(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.pi.state_dict())
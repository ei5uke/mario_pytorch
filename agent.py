import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from dqn import DQN

class Agent(): #doesn't derive from anything in this tutorial but does clearly in mario
    def __init__(self, state_dim, action_dim): #gamma discount rate, epsilon e-greedy
        ## CHANGE FROM HERE
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = 0.99
        self.epsilon = 1.0
        self.eps_end = 0.1
        self.eps_dec = 5e-4

        self.batch_size = 32
        #self.mem_size = 100000 #idt it's necessary
        self.curr_mem = 0 #keep track the position of first available memory
        self.memory = deque(maxlen=100000)
        ## TO HERE

        self.net = DQN(self.state_dim, self.action_dim).float() #originally Q_eval
        self.optimizer = optim.Adam(self.parameters(), lr=0.00025) # CHANGE LR
        self.loss = nn.SmoothL1Loss() # CHANGE IF I CAN FIND SOMETHING BETTER

        # not sure what this does yet
        self.min_experience = 1e4
        self.learn_every = 3
        self.sync_every = 1e4
    
    def store_transition(self, state, action, reward, state_, done): #this stores memory
        index = self.mem_cntr % self.mem_size 
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def act(self, state):
        if np.random.random() > self.epsilon:
            state = T.tensor([state]).to(self.net.device) # note to future self, make sure to alt-tab to look at the pytorch dqn tutorial cuz they put these next few steps in the main.py not a method so it's confusing
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_dim)

        return action

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
        self.Q_eval.optimizer.zero_grad() #sets gradient of all model parameters to zero
        #self.Q_eval.optimizer.zero_grad(set_to_none=True) #sets all gradients to None instead of zero, better performance but error-prone

        max_mem = min(self.mem_cntr, self.mem_size) #calculate pos of max memory to only select all memories up to the last filled memory
        batch = np.random.choice(max_mem, self.batch_size, replace=False) #replace=False bc we don't want to select same memories more than once
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) #convert numpy array to tensor
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0] #T.max returns value and index so we only want value so [0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
                        else self.eps_min
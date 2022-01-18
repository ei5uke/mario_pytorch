# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# nice libraries to use
import numpy as np
import random
from arguments import parse_args

# following Costa Huang's video, need to see why we really do this
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, env, args):
        super(Agent, self).__init__()
        # this follows the DeepMind DQN network
        self.env = env
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, kernel_size=(8,8), stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=(4,4), stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=(3,3), stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(3136, 512)),
            nn.ReLU(),
        )
        self.actor = layer_init(nn.Linear(512, env.action_space.n))
        self.critic = layer_init(nn.Linear(512, 1))
        self.args = args

    # def get_value(self, x):
    #     return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action.squeeze()), probs.entropy(), self.critic(hidden)

    def rollout(self):
        # our set of trajectories with already made sizes to promote efficiency
        obs = torch.zeros(self.args.num_steps, 4, 84, 84).to(self.args.device)
        actions = torch.zeros(self.args.num_steps, 1).to(self.args.device)
        logprobs = torch.zeros(self.args.num_steps, 1).to(self.args.device)
        rewards = torch.zeros(self.args.num_steps, 1).to(self.args.device)
        dones = torch.zeros(self.args.num_steps, 1).to(self.args.device)
        values = torch.zeros(self.args.num_steps, 1).to(self.args.device)

        next_obs = torch.Tensor(self.env.reset()).to(self.args.device)

        for step in range(0, self.args.num_steps):
            self.args.global_step += 1

            obs[step] = next_obs
            with torch.no_grad():
                action, logprob, _, value = self.get_action_and_value(next_obs.unsqueeze(0))
            values[step] = value.flatten()
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.args.device), torch.Tensor(np.array(done)).to(self.args.device)
            rewards[step] = torch.tensor(reward).to(self.args.device).view(-1) #idk what the view is here for
            actions[step] = action
            logprobs[step] = logprob
            dones[step] = next_done
        
        return obs, actions, logprobs, rewards, dones, values

    def compute_returns(self, rewards):
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(self.args.device)
            discounted_reward = 0
            idx = 0
            for reward in reversed(rewards):
                discounted_reward = reward + discounted_reward * self.args.gamma
                returns[idx] = discounted_reward
                idx += 1
        return returns

    def learn(self):
        while self.args.global_step <= self.args.total_timesteps:
            obs, actions, logprobs, rewards, dones, values = self.rollout()

            returns = self.compute_returns(rewards)
            action, log_prob, entropy, values = self.get_action_and_value(obs, actions)
            
            #import ipdb; ipdb.set_trace()

            # not following GAE here, will change in the future
            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-10)

            ratio = torch.exp(log_prob - logprobs.squeeze()).unsqueeze(1)
            actor_loss = (-torch.min(ratio * advantage, torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip) * advantage)).mean()
            critic_loss = nn.MSELoss()(values, returns)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()
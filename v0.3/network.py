# torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# nice libraries to use
import numpy as np
import random
from arguments import parse_args

import wandb

# following Costa Huang's video, need to see why we really do this
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs, args, writer):
        super(Agent, self).__init__()
        # this follows the DeepMind DQN network
        self.envs = envs
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
        self.args = args
        self.writer = writer

        self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1)

    # def get_value(self, x):
    #     return self.critic(self.network(x / 255.0))
    
    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / 255.0)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def rollout(self):
        # our set of trajectories with already made sizes to promote efficiency
        obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(self.args.device)
        actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(self.args.device)
        logprobs = torch.zeros(self.args.num_steps, self.args.num_envs).to(self.args.device)
        rewards = torch.zeros(self.args.num_steps, self.args.num_envs).to(self.args.device)
        dones = torch.zeros(self.args.num_steps, self.args.num_envs).to(self.args.device)
        values = torch.zeros(self.args.num_steps, self.args.num_envs).to(self.args.device)

        next_obs = torch.Tensor(self.envs.reset()).to(self.args.device)
        next_done = torch.zeros(self.args.num_envs).to(self.args.device)
        for step in range(0, self.args.num_steps):
            self.args.global_step += 1 * self.args.num_envs

            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, entropy, value = self.get_action_and_value(next_obs)
            values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done, info = self.envs.step(action.cpu().numpy())
            next_obs, next_done = torch.Tensor(next_obs).to(self.args.device), torch.Tensor(done).to(self.args.device)
            rewards[step] = torch.tensor(reward).to(self.args.device).view(-1)
            global_step = self.args.global_step
            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    wandb.log({"Episodic Return": item["episode"]["r"]}, step=self.args.global_step )
                    break

        return obs, actions, logprobs, rewards, dones, values

    def calculate_advantage(self, rewards, values):
        advantages = torch.zeros_like(rewards).to(self.args.device)
        lastgaelam = 0
        for step in reversed(range(self.args.num_steps-1)):
            delta = rewards[step] + self.args.gamma * values[step+1] - values[step]
            advantages[step] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * lastgaelam
        returns = advantages + values
        return advantages, returns

    def learn(self, optim):
        while self.args.global_step <= self.args.total_timesteps:
            # add some anneal rate
            self.args.alpha = 1.0 - self.args.global_step * (1.0 / self.args.total_timesteps)
            optim.param_groups[0]["lr"] = self.args.lr * self.args.alpha

            obs, actions, logprobs, rewards, dones, values = self.rollout()

            advantages, returns = self.calculate_advantage(rewards, values)

            batch_idxs = np.arange(self.args.batch_size)

            obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            logprobs = logprobs.reshape(-1)
            actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            advantages = advantages.reshape(-1)
            returns = returns.reshape(-1)
            values = values.reshape(-1)

            for epoch in range(self.args.update_epochs):
                np.random.shuffle(batch_idxs)

                for start in range(0, self.args.batch_size, self.args.minibatch_size):
                    end = start + self.args.minibatch_size
                    minibatch_idxs = batch_idxs[start:end]

                    throw, minibatch_log_prob, minibatch_entropy, minibatch_values = self.get_action_and_value(obs[minibatch_idxs], actions.int()[minibatch_idxs])
                
                    minibatch_adv = advantages[minibatch_idxs]
                    
                    minibatch_adv = (minibatch_adv - minibatch_adv.mean()) / (minibatch_adv.std() + 1e-10)

                    ratio = torch.exp(minibatch_log_prob - logprobs[minibatch_idxs])
                    actor_loss = (torch.max(ratio * -minibatch_adv, torch.clamp(ratio, 1 - self.args.clip * self.args.alpha, 1 + self.args.clip * self.args.alpha) * -minibatch_adv)).mean()
                    critic_loss = nn.MSELoss()(minibatch_values.view(-1), returns[minibatch_idxs])

                    entropy_loss = minibatch_entropy.mean()
                    #loss = actor_loss - critic_loss * self.args.vf + self.args.ent * entropy_loss
                    loss = actor_loss + critic_loss * self.args.vf - self.args.ent * entropy_loss # signs are flipped because adam minimizes so we do the opposite
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
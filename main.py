import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import random
import itertools
from collections import deque

#from agent import Agent
from dqn import DQN
from transformation import GrayScaleObservation, SkipFrame, ResizeObservation

# HYPERPARAMETERS
GAMMA=0.9
BATCH_SIZE=32
MEMORY_SIZE=100000
MIN_REPLAY_SIZE=1000
EPSILON_START=1.0
EPSILON_END=0.02
EPSILON_DECAY=100000
TARGET_UPDATE_FREQ=1000
LEARNING_RATE=5e-4

# Apply Wrappers to environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

memory = deque(maxlen=MEMORY_SIZE)
reward_buffer = deque([0.0], maxlen=100)

episode_reward = 0.0

online_net = DQN(env)
target_net = DQN(env)
target_net.load_state_dict(online_net.state_dict())
optimizer = optim.Adam(online_net.parameters(), lr=LEARNING_RATE)

obs = env.reset()
for _ in range(MIN_REPLAY_SIZE):
    action = env.action_space.sample()

    obs_, reward, done, _ = env.step(action)
    transition = (obs, action, reward, done, obs_)
    memory.append(transition)
    obs = obs_

    if done:
        obs = env.reset()

obs = env.reset()
for step in itertools.count():
    epsilon = np.interp(step, [0, EPSILON_DECAY], [EPSILON_START, EPSILON_END]) # for EPSILON_DECAY steps, it will be between start and end, and be at end after that

    rnd_sample = random.random()

    if rnd_sample <= epsilon: #this is our exploration vs greedy method
        action = env.action_space.sample()
    else:
        action = online_net.act(np.array(obs)) # note that i chnaged this

    obs_, reward, done, _ = env.step(action)
    #print(obs_.size)
    transition = (obs, action, reward, done, obs_)
    memory.append(transition)
    obs = obs_

    episode_reward += reward

    if done:
        obs = env.reset()

        reward_buffer.append(episode_reward)
        episode_reward = 0.0

    # After solved, watch it play
    if len(reward_buffer) >= 10:
        if np.mean(reward_buffer) >= 1000: # probably want to change this to a higher score
            while True:
                action = online_net.act(np.array(obs)) # i also changed this

                obs, _, done, _ = env.step(action)
                env.render()
                if done:
                    env.reset()

    # START GRADIENT STEP

    transitions = random.sample(memory, BATCH_SIZE)
    
    # pytorch is must faster to make a tensor from a numpy array compared to a python array
    obses = np.asarray([t[0] for t in transitions])
    actions = np.asarray([t[1] for t in transitions])
    rewards = np.asarray([t[2] for t in transitions])
    dones = np.asarray([t[3] for t in transitions])
    obses_ = np.asarray([t[4] for t in transitions])

    obses_t = torch.as_tensor(obses, dtype=torch.float32)
    actions_t = torch.as_tensor(actions, dtype=torch.int64).unsqueeze(-1) # this is an index so it's an int
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1) # these are already batches so we add a dimension at the end
    dones_t = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    obses_t_ = torch.as_tensor(obses_, dtype=torch.float32)

    # Compute targets
    target_q_values = target_net(obses_t_)
    max_target_q_values = target_q_values.max(dim=1, keepdim=True)[0]

    targets = rewards_t + GAMMA * (1 - dones_t) * max_target_q_values

    # Compute loss
    q_values = online_net(obses_t)
    action_q_values = torch.gather(input=q_values, dim=1, index=actions_t) #applies index action_t to dim 1 to all the q_values
    loss = nn.functional.smooth_l1_loss(action_q_values, targets)

    # Gradient descent
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update target network
    if step % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(online_net.state_dict())

    # logging
    if step % 1000 == 0:
        print()
        print('Step', step)
        print('Avg Reward', np.mean(reward_buffer))
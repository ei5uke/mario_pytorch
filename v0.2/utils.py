"""
utils.py

This file will have the transformations to modify the environment. Ideally we add framestacking, grayscale, and also frameskipping following
the scientific journal. 
"""

import gym
from gym.spaces import Box
import cv2
import numpy as np
from random import randint

# I don't really see what this does cuz it's not really skipping frames rather just making increments of 4
class SkipFrames(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.env = env
        self.skip = skip

    def step(self, action):
        total_reward = 0
        done = False
        for i in range(self.skip):
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return state, total_reward, done, info

class ModifyObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.shape = (84, 84)
        self.observation_space = Box(low=0, high=255, shape=self.shape, dtype=np.uint8)

    # modify our image size and grayscale
    def modify(self, state):
        state = np.transpose(state, (2, 0, 1)) # change [H, W, C] to [C, H, W]
        state = cv2.resize(state, (self.w, self.h)) # change dimensions to (84, 84)
        state = cv2.cvtColor(state, cv2.COLOR_RGB2GRAY) # convert to grayscale
        return state

    # return the modified observation
    def observation(self, observation):
        return self.modify(observation)
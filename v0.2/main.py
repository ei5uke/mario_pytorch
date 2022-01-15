"""
main.py

To run:
python3 main.py

This is where we'll run the program. Currently we'll have the DQN policy implemented.
"""


# gym and super mario bros libraries
import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from gym.wrappers import FrameStack

import torch # Import it to instantiate the device
import time # To pause between frames to visualize it
from random import random
import numpy as np
from torch.nn import parameter

# allows us to visualize our models, progress, and how efficient our device is doing computations
import wandb

# Importing stuff from other files
from models import Model, Memory, train
from utils import SkipFrames, ModifyObservation

# Nice debugging tool
# import ipdb; ipdb.set_trace()

def main(test, device, chkpt=None):
    # HYPERPARAMETERS, following the DQN paper
    GAMMA=0.99
    BATCH_SIZE=32
    MEMORY_SIZE=1000000
    MIN_REPLAY_SIZE=50000
    EPSILON=1.0
    EPSILON_DECAY=0.999999 # this doesn't follow bc I don't like the DQN paper's method so
    EPSILON_END=0.1
    TARGET_UPDATE_FREQ=10000
    LEARNING_RATE=0.00025 #LEARNING_RATE=0.00001

    # If it's training, used wandb to check training data
    if not test:
        #wandb.init(project='pytorch-mario', name='dqn-1')
        wandb.init(project='breakout-v0', name='ddqn-12')

    # Instantiate the mario env
    #env = gym_super_mario_bros.make('SuperMarioBros-v0')
    #env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # env.observation_space.shape returns (240, 256, 3) so we want to convert that to (4, 84, 84)
    
    env = gym.make('Breakout-v0')

    env = SkipFrames(env)
    env = ModifyObservation(env)
    env = FrameStack(env, num_stack=4)

    online = Model((4, 84, 84), env.action_space.n, LEARNING_RATE) # Instantiate the online model
    if chkpt is not None: # If it's not training, give the online model the gradients
        online.load_state_dict(torch.load(chkpt))
    target = Model((4, 84, 84), env.action_space.n, LEARNING_RATE) # Instatiante the target model
    target.load_state_dict(online.state_dict())

    memory = Memory(MEMORY_SIZE) # Instantiate the replay buffer

    #steps_to_train = 0 # Number of steps we are currently at until we train
    #step_num = -1 * MIN_REPLAY_SIZE # still figuring what exactly this is
    #step_num = 0
    #step_num = -1 * MIN_REPLAY_SIZE
    step_num = 0
    rewardbuffer = [] # Instantiate the reward buffer that'll keep track of the rewards 
    rolling_reward = 0 # Store the rewards passed in a single episode
    epochs = 0
    parameter_updates = 1

    #import ipdb; ipdb.set_trace()
    state = env.reset() # Get the initial state
     
     # only for breakout the fire action, so we'll take this out for mario
    state_, reward, done, info = env.step(1)
    memory.insert((np.array(state), 1, reward, done, np.array(state_)))
    rolling_reward = reward
    state = state_

    try:
        # the main loop that'll consist of the training and loading a past model
        while True:
            if test:
                env.render('human')
                time.sleep(0.1)
            #EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_END)
            #EPSILON = EPSILON_DECAY**(step_num)
            #EPSILON = step_num * EPSILON_DECAY/1000000
            EPSILON = max(1 - (step_num * 0.9/1000000), EPSILON_END)

            if test:
                EPSILON = 0
            if random() < EPSILON:
                action = env.action_space.sample()
            else:
                action = online(torch.Tensor(np.array(state)).unsqueeze(0).to(device)).max(-1)[-1].item()
            
            state_, reward, done, info = env.step(action)
            #state_, reward, done, info, terminal_life_lost = env.step(action)
            rolling_reward += reward
            
            # reward clipping
            reward = 1 if reward > 0 else (0 if reward == 0 else -1)

            #memory.insert((np.array(state), action, reward, done, np.array(state_)))

            if done and info['ale.lives'] == 0:
                memory.insert((np.array(state), action, -1, done, np.array(state)))
            else:
                memory.insert((np.array(state), action, reward, done, np.array(state_)))

            state = state_

            if done:
                rewardbuffer.append(rolling_reward)
                if test:
                    print(rolling_reward)

                state = env.reset()
                state_, reward, done, info = env.step(1)
                memory.insert((np.array(state), 1, reward, done, np.array(state_)))
                rolling_reward = reward
                state = state_

            #steps_to_train += 1
            step_num += 1

            # episode = sequence of actions and states that end with terminal state
            # one epoch = one forward pass + one backward pass but our case we're doing 50000 of those
            if (not test) and memory.idx > MIN_REPLAY_SIZE:
                if step_num % 4 == 0:
                    #EPSILON = max(EPSILON - (step_num * EPSILON_DECAY/1000000), EPSILON_END)
                    loss = train(online, target, memory.sample(BATCH_SIZE), env.action_space.n, device, GAMMA)
                    #wandb.log({'Epsilon': EPSILON, 'Average Reward': np.mean(rewardbuffer[-100:]), 'Loss': loss.detach().cpu().item()}, step=step_num)
                    parameter_updates += 1
                    if parameter_updates % 50001 == 0:
                        print('hi')
                    #wandb.log({'Epsilon': EPSILON, 'Average Reward': np.mean(rewardbuffer[-100:]), 'Loss': loss.detach().cpu().item()}, step=epochs)
                    # if parameter_updates % 50000 == 0:
                    #     parameter_updates = 0
                    #     epochs += 1
                    #     wandb.log({'Epsilon': EPSILON, 'Average Reward': np.mean(rewardbuffer[-100:]), 'Loss': loss.detach().cpu().item()}, step=epochs)
                    # rewardbuffer = []
                    #if steps_to_train > TARGET_UPDATE_FREQ:
                if step_num % TARGET_UPDATE_FREQ == 0:
                    print(f'updating target model, Epoch {epochs}')
                    #print(EPSILON, step_num)
                    target.load_state_dict(online.state_dict())
                    #wandb.log({'Epsilon': EPSILON, 'Average Reward': np.mean(rewardbuffer[-100:]), 'Loss': loss.detach().cpu().item()}, step=epochs)
                    #torch.save(target.state_dict(), f'models/mario{step_num}')

                if parameter_updates % 50001 == 0:
                    #print(epochs)
                    parameter_updates = 0
                    epochs += 1
                    wandb.log({'Epsilon': EPSILON, 'Average Reward': np.mean(rewardbuffer[-100:]), 'Loss': loss.detach().cpu().item()}, step=epochs)
    except KeyboardInterrupt:
        #torch.save(target.state_dict(), f'models/mario{step_num}')
        torch.save(target.state_dict(), f'models/breakout{step_num}')
        env.close()
    env.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run this to train the model
    main(False, device)

    # Run this to test a saved model; replace __ with any of the saved models
    #main(True, device, 'models/breakout1707396')
    #main(True, device, 'models/breakout587312')
    #main(True, device, 'models/breakout1689352')
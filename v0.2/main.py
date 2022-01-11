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

# Import it to instantiate the device
import torch
from others.rl.pytorch.agent import update_tgt_model

# allows us ot visualize our models, progress, and how efficient our device is doing computations
import wandb
from tqdm import tdqm

# Importing stuff from other files
from models import Model, Memory, train
# from utils


# HYPERPARAMETERS, following the DQN paper
GAMMA=0.99
BATCH_SIZE=32
MEMORY_SIZE=1000000
MIN_REPLAY_SIZE=50000
EPSILON_DECAY=0.99999 # this doesn't follow bc I don't like the DQN paper's method so
EPSILON_END=0.1
TARGET_UPDATE_FREQ=10000
LEARNING_RATE=0.00025

def main(test, device, chkpt=None):
    # If it's training, used wandb to check training data
    if not test:
        wandb.init(project='pytorch-mario', name='dqn-1')

    # Instantiate the mario env
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    online = Model((4, 84, 84), env.action_space.n) # Instantiate the online model
    if chkpt is not None: # If it's not training, give the online model the gradients
        online.load_state_dict(torch.load(chkpt))
    target = Model((4, 84, 84), env.action_space.n) # Instatiante the target model
    target.load_state_dict(online.state_dict())

    memory = Memory(MEMORY_SIZE) # Instantiate the replay buffer

    steps_to_train = 0 # Number of steps we are currently at until we train
    step_num = -1 * MIN_REPLAY_SIZE # still figuring what exactly this is
    rewardbuffer = [] # Instantiate the reward buffer that'll keep track of the rewards 
    rolling_reward = 0 # Store the rewards passed in a single episode

    state = env.reset() # Get the initial state
    tq = tqdm() # Instantiate the progress bar

    # the main loop that'll consist of the training and loading a past model
    try:
        pass
    except KeyboardInterrupt:
        pass
    env.close()

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Run this to train the model
    main(False, device)

    # Run this to test a saved model; replace __ with any of the saved models
    # main(True, device, 'models/__')
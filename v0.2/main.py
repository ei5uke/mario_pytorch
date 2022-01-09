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

# Importing stuff from other files
from models import Model
# from utils
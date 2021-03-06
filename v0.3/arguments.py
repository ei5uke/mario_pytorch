import argparse
import os
from distutils.util import strtobool
import torch

def parse_args():
    parser = argparse.ArgumentParser()

    # arguments to set up
    parser.add_argument("--wandb-project-name", type=str, default="pytorch-mario",
        help="our wandb project name")
    parser.add_argument("--exp-name", type=str, default="mario-ppo10",
        help="the name of this project")
    parser.add_argument("--gym-id", type=str, default="SuperMarioBros-v0",
        help="name of gym environment")
    parser.add_argument('--seed', type=int, default=1,
        help='seed of the experiment')
    parser.add_argument('--torch-deterministic', type=bool, default=True, nargs='?', const=True,
        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="our device used for calculations")
    parser.add_argument("--track", type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="True -> track with wandb")
    parser.add_argument("--eval", type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
        help="True -> evaluate our bot and render")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="entity/team of wandb project")
    # parser.add_argument("--capture-video", )

    # arguments for our learning
    # parser.add_argument("--lr", type=float, default=2.5e-4,
    #     help="learning rate of our optimizer") # for breakout
    parser.add_argument("--lr", type=float, default=1.0e-4,
        help="learning rate of our optimizer") # for mario
    # parser.add_argument("--total-timesteps", type=int, default=10000000,
    #     help="total length of learning") # for breakout
    parser.add_argument("--total-timesteps", type=int, default=5000000,
        help="total length of learning") # for mario
    # parser.add_argument("--num-steps", type=int, default=128,
    #     help="number of steps to run in each rollout") # breakout
    parser.add_argument("--num-steps", type=int, default=512,
        help="number of steps to run in each rollout") # mario
    parser.add_argument("--global-step", type=int, default=0,
        help="keep track of current step")
    parser.add_argument("--num-envs", type=int, default=8,
        help="number of parallel game environments") 
    parser.add_argument("--alpha", type=float, default=1.0,
        help="Annealing rate for our stepsize and clip coef")
    parser.add_argument("--gae", type=bool, default=True,
        help="True -> Use GAE for advantage estimation")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="lambda for GAE")
    # parser.add_argument("--gamma", type=float, default=0.99,
    #     help="our discount rate") # breakout
    parser.add_argument("--gamma", type=float, default=0.9,
        help="our discount rate") # mario
    # parser.add_argument("--num-minibatches", type=int, default=4,
    #     help="number of mini-batches") # breakout
    parser.add_argument("--num-minibatches", type=int, default=16,
        help="number of mini-batches") # mario
    # parser.add_argument("--update-epochs", type=int, default=4,
    #     help="number of epochs to update policy") # breakout
    parser.add_argument("--update-epochs", type=int, default=10,
        help="number of epochs to update policy") # mario
    parser.add_argument("--clip", type=float, default=0.1,
        help="our clip constant")
    parser.add_argument("--ent", type=float, default=0.01,
        help="entropy constant")
    parser.add_argument("--vf", type=float, default=1.0,
        help="value function constant")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)